import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensordict import TensorDict
from decision_transformer_improved import DecisionTransformer, loadModel
from decision_transformer_config import DecisionTransformerConfig
from generate_trajectories import generateInstanceData, generateTrajectory, addTrajectoryToTrainingData, TRAJECTORY_LENGTH, FORECAST_LENGTH, MAX_LEAD_TIME
from data_scaler import compute_scaling_params_from_training_data

DEBUG_TEST_SAVED_COUNT = 0

def save_test_debug_input(td, step, realAction, realReturnToGo):
    """
    Guarda todos los inputs del modelo durante el test para debugging.
    Se ejecuta dos veces para análisis del primer y segundo paso.
    
    Args:
        td: TensorDict con todos los datos del estado
        step: Paso actual del test
        realAction: Acción real para comparación
        realReturnToGo: Return to go real para comparación
    """
    global DEBUG_TEST_SAVED_COUNT
    
    if DEBUG_TEST_SAVED_COUNT >= 2:
        return
    
    debug_data = {
        "step_info": {
            "step": step,
            "currentTimestep": td["currentTimestep"].cpu().tolist() if "currentTimestep" in td else None,
        },
        "state_data": {
            "onHandLevel": td["onHandLevel"].cpu().tolist(),
            "inTransitStock": td["inTransitStock"].cpu().tolist(),
            "forecast": td["forecast"].cpu().tolist(),
            "demand": td["demand"].cpu().tolist(),
        },
        "cost_data": {
            "holdingCost": td["holdingCost"].cpu().tolist(),
            "orderingCost": td["orderingCost"].cpu().tolist(),
            "stockOutPenalty": td["stockOutPenalty"].cpu().tolist(),
            "unitRevenue": td["unitRevenue"].cpu().tolist(),
            "leadTime": td["leadTime"].cpu().tolist(),
        },
        "returns_data": {
            "returnsToGo": td["returnsToGo"].cpu().tolist(),
            "benefit": td["benefit"].cpu().tolist() if "benefit" in td else None,
        },
        "comparison_data": {
            "realAction": realAction,
            "realReturnToGo": realReturnToGo,
        },
        "embedding_data": {
            "statesEmbedding_shape": list(td["statesEmbedding"].shape),
            "actionsEmbedding_shape": list(td["actionsEmbedding"].shape),
            "returnsToGoEmbedding_shape": list(td["returnsToGoEmbedding"].shape),
        }
    }
    
    DEBUG_TEST_SAVED_COUNT += 1
    output_path = f"debug_test_input_step{DEBUG_TEST_SAVED_COUNT}.json"
    with open(output_path, 'w') as f:
        json.dump(debug_data, f, indent=4)


def getProjectDirectory():
    """Obtiene el directorio del proyecto."""
    return str(Path(__file__).resolve().parent)


def loadTrainedModel(modelPath, configPath=None, dataPath=None):
    """
    Carga un modelo Decision Transformer ya entrenado con parámetros de escalado.
    
    Args:
        modelPath: Ruta al archivo del modelo entrenado (.pt)
        configPath: Ruta al archivo de configuración (opcional)
        dataPath: Ruta a los datos de entrenamiento para calcular parámetros de escalado (opcional)
    
    Returns:
        Modelo Decision Transformer cargado
    """
    if configPath is None:
        config = DecisionTransformerConfig()
    else:
        config = torch.load(configPath)
    
    # Calcular parámetros de escalado desde los datos de entrenamiento
    scaling_params = None
    if dataPath:
        try:
            scaling_params = compute_scaling_params_from_training_data([dataPath])
            print(f"Parámetros de escalado calculados desde {dataPath}")
        except Exception as e:
            print(f"Advertencia: No se pudieron calcular los parámetros de escalado: {e}")
            print("El modelo funcionará sin escalado")
    
    model = loadModel(modelPath, config, scaling_params=scaling_params)
    model.eval()
    
    return model


def getTestProblem(dataPath, problemIndex=0):
    """
    Obtiene un problema de los datos de entrenamiento para testing.
    
    Args:
        dataPath: Ruta a los datos de entrenamiento
        problemIndex: Índice del problema a usar (por defecto 0)
    
    Returns:
        TensorDict con los datos del problema
    """
    trainingData = torch.load(dataPath, weights_only=False)
    
    problemData = trainingData['states'][problemIndex]
    actionsData = trainingData['actions'][problemIndex]
    returnsToGoData = trainingData['returnsToGo'][problemIndex]
    
    problem = TensorDict({
        'onHandLevel': problemData['onHandLevel'].unsqueeze(0),
        'inTransitStock': problemData['inTransitStock'].unsqueeze(0),
        'forecast': problemData['forecast'].unsqueeze(0),
        'demand': problemData['demand'].unsqueeze(0),
        'holdingCost': problemData['holdingCost'].unsqueeze(0),
        'orderingCost': problemData['orderingCost'].unsqueeze(0),
        'stockOutPenalty': problemData['stockOutPenalty'].unsqueeze(0),
        'unitRevenue': problemData['unitRevenue'].unsqueeze(0),
        'leadTime': problemData['leadTime'].unsqueeze(0),
        'returnsToGo': returnsToGoData.unsqueeze(0),
        'realActions': actionsData.unsqueeze(0),
        'realReturnsToGo': returnsToGoData
    })

    return problem


def tryDecisionTransformer(model, problem, maxSteps=None):
    """
    Ejecuta el Decision Transformer en modo autoregresivo.
    El modelo genera predicciones basándose en sus propias predicciones anteriores.
    
    Args:
        model: Modelo Decision Transformer entrenado
        problem: TensorDict con los datos del problema
        maxSteps: Número máximo de pasos a ejecutar (por defecto TRAJECTORY_LENGTH)
    
    Returns:
        Lista de diccionarios con los resultados de cada paso
    """
    if maxSteps is None:
        maxSteps = TRAJECTORY_LENGTH
    
    trajectoryLength = problem['realActions'].size(1)
    maxSteps = min(maxSteps, trajectoryLength)
    
    td = {k: v.clone() for k, v in problem.items()}
    td["returnsToGo"] = torch.zeros_like(td["returnsToGo"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    td = model.initModel(td)
    
    results = []
    
    for step in range(maxSteps):
        realAction = problem['realActions'][0, step].item()
        realReturnToGo = problem['realReturnsToGo'].item()
        
        save_test_debug_input(td, step, realAction, realReturnToGo)
        
        with torch.no_grad():
            td = model.forward(td, nextOrderQuantity=None, is_test=True, update_only=False)
            predictedAction = td['predictedAction'][0, 0].item()
        
        currentState = {
            'onHandLevel': td['onHandLevel'][0].item(),
            'inTransitStock': td['inTransitStock'][0].tolist(),
            'demand': td['demand'][0, 0].item(),
            'forecast': td['forecast'][0, 0].tolist(),
            'holdingCost': td['holdingCost'][0].item(),
            'orderingCost': td['orderingCost'][0].item(),
            'stockOutPenalty': td['stockOutPenalty'][0].item(),
            'unitRevenue': td['unitRevenue'][0].item(),
            'leadTime': td['leadTime'][0].item(),
            'timeStep': float(step)
        }
        
        actionDifference = abs(predictedAction - realAction)
        actionError = actionDifference / max(realAction, 1e-6) if realAction > 0 else actionDifference
        
        stepResult = {
            'step': step,
            'window_start': step,
            'window_end': step + 1,
            'state': currentState,
            'realAction': realAction,
            'predictedAction': predictedAction,
            'realReturnToGo': realReturnToGo,
            'predictedReturnToGo': td['returnsToGo'][0].item(),
            'actionDifference': actionDifference,
            'actionError': actionError,
            'actionAccuracy': 1.0 - min(actionError, 1.0),
        }
        
        results.append(stepResult)
    
    return results


def generateTestReport(results, outputPath=None):
    """
    Genera un reporte JSON con los resultados del test.
    
    Args:
        results: Lista de resultados de cada paso
        outputPath: Ruta donde guardar el reporte (opcional)
    
    Returns:
        Diccionario con el reporte completo
    """
    totalSteps = len(results)
    totalActionError = sum(r['actionError'] for r in results)
    totalActionDifference = sum(r['actionDifference'] for r in results)
    averageActionError = totalActionError / totalSteps if totalSteps > 0 else 0
    averageActionDifference = totalActionDifference / totalSteps if totalSteps > 0 else 0
    averageAccuracy = sum(r['actionAccuracy'] for r in results) / totalSteps if totalSteps > 0 else 0
    
    report = {
        'summary': {
            'totalSteps': totalSteps,
            'averageActionError': averageActionError,
            'averageActionDifference': averageActionDifference,
            'averageAccuracy': averageAccuracy,
            'totalActionError': totalActionError,
            'totalActionDifference': totalActionDifference
        },
        'steps': results
    }
    
    if outputPath:
        with open(outputPath, 'w') as f:
            json.dump(report, f, indent=4)
    
    return report


def createActionComparisonPlot(results, outputPath=None):
    """
    Crea una gráfica comparando las acciones reales vs predichas usando datos de td.
    
    Args:
        results: Lista de resultados de cada paso
        outputPath: Ruta donde guardar la gráfica (opcional)
    
    Returns:
        None
    """
    steps = [r['step'] for r in results]
    realActions = [r['realAction'] for r in results]
    predictedActions = [r['predictedAction'] for r in results]
    
    plt.figure(figsize=(12, 8))
    
    plt.plot(steps, realActions, 'b-', label='Acción Real', linewidth=2, marker='o', markersize=4)
    plt.plot(steps, predictedActions, 'r--', label='Acción Predicha (DT)', linewidth=2, marker='s', markersize=4)
    
    plt.xlabel('Paso', fontsize=12)
    plt.ylabel('Cantidad a Ordenar', fontsize=12)
    plt.title('Comparación: Acciones Reales vs Decision Transformer', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    avgError = np.mean([abs(r['realAction'] - r['predictedAction']) for r in results])
    avgAccuracy = np.mean([r['actionAccuracy'] for r in results])
    
    plt.text(0.02, 0.98, f'Error Promedio: {avgError:.3f}\nPrecisión Promedio: {avgAccuracy:.3f}', 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if outputPath:
        plt.savefig(outputPath, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return plt.gcf()


def runDecisionTransformerTest(modelPath, dataPath, problemIndex=0, maxSteps=None, outputPath=None, plotOutputPath=None):
    """
    Función principal para ejecutar el test del Decision Transformer.
    
    Args:
        modelPath: Ruta al modelo entrenado
        dataPath: Ruta a los datos de entrenamiento
        problemIndex: Índice del problema a usar
        maxSteps: Número máximo de pasos
        outputPath: Ruta donde guardar el reporte JSON
        plotOutputPath: Ruta donde guardar la gráfica (opcional)
    
    Returns:
        Diccionario con el reporte completo
    """
    # Cargar modelo con parámetros de escalado calculados desde los datos
    model = loadTrainedModel(modelPath, dataPath=dataPath).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    problem = getTestProblem(dataPath, problemIndex).to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    results = tryDecisionTransformer(model, problem, maxSteps)
    report = generateTestReport(results, outputPath)
    createActionComparisonPlot(results, plotOutputPath)
    
    return report


if __name__ == "__main__":
    projectDir = getProjectDirectory()
    modelPath = os.path.join(projectDir, "training_models/decision_transformer_model", "training.pt")
    dataPath = os.path.join(projectDir, "data", "training_data2.pt")
    outputPath = os.path.join(projectDir, "test_results.json")
    plotOutputPath = os.path.join(projectDir, "action_comparison_plot.png")
    
    if not os.path.exists(modelPath):
        print(f"Error: No se encontró el modelo en {modelPath}")
        print("Asegúrate de que el modelo esté entrenado y guardado.")
        exit(1)
    
    if not os.path.exists(dataPath):
        print(f"Error: No se encontraron los datos en {dataPath}")
        print("Ejecuta generate_tajectories.py para generar los datos de entrenamiento.")
        exit(1)
    
    try:
        report = runDecisionTransformerTest(
            modelPath=modelPath,
            dataPath=dataPath,
            problemIndex=0,
            maxSteps=30,
            outputPath=outputPath,
            plotOutputPath=plotOutputPath
        )
        
        print(f"\nTest completado exitosamente!")
        print(f"Reporte detallado guardado en: {outputPath}")
        print(f"Gráfica de comparación guardada en: {plotOutputPath}")
        
    except Exception as e:
        print(f"Error durante el test: {e}")
        import traceback
        traceback.print_exc()
