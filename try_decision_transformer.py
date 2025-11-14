import os
import torch
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
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
        'realReturnsToGo': returnsToGoData,
        'realBenefits': problemData['benefit'].unsqueeze(0) if 'benefit' in problemData else None,
        'realCumulativeSales': problemData['cumulativeSales'].unsqueeze(0) if 'cumulativeSales' in problemData else None,
        'realCumulativeHoldingCost': problemData['cumulativeHoldingCost'].unsqueeze(0) if 'cumulativeHoldingCost' in problemData else None,
        'realCumulativeOrderingCost': problemData['cumulativeOrderingCost'].unsqueeze(0) if 'cumulativeOrderingCost' in problemData else None,
        'realCumulativeStockOutCost': problemData['cumulativeStockOutCost'].unsqueeze(0) if 'cumulativeStockOutCost' in problemData else None
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
        
        predictedBenefit = td['benefit'][0, -1].item() if 'benefit' in td and td['benefit'].size(1) > 0 else 0.0
        realBenefit = problem['realBenefits'][0, step].item() if problem.get('realBenefits') is not None else None
        
        predictedCumulativeSales = td['cumulativeSales'][0, -1].item() if 'cumulativeSales' in td and td['cumulativeSales'].size(1) > 0 else 0.0
        realCumulativeSales = problem['realCumulativeSales'][0, step].item() if problem.get('realCumulativeSales') is not None else 0.0
        
        predictedCumulativeHoldingCost = td['cumulativeHoldingCost'][0, -1].item() if 'cumulativeHoldingCost' in td and td['cumulativeHoldingCost'].size(1) > 0 else 0.0
        realCumulativeHoldingCost = problem['realCumulativeHoldingCost'][0, step].item() if problem.get('realCumulativeHoldingCost') is not None else None
        
        predictedCumulativeOrderingCost = td['cumulativeOrderingCost'][0, -1].item() if 'cumulativeOrderingCost' in td and td['cumulativeOrderingCost'].size(1) > 0 else 0.0
        realCumulativeOrderingCost = problem['realCumulativeOrderingCost'][0, step].item() if problem.get('realCumulativeOrderingCost') is not None else None
        
        predictedCumulativeStockOutCost = td['cumulativeStockOutCost'][0, -1].item() if 'cumulativeStockOutCost' in td and td['cumulativeStockOutCost'].size(1) > 0 else 0.0
        realCumulativeStockOutCost = problem['realCumulativeStockOutCost'][0, step].item() if problem.get('realCumulativeStockOutCost') is not None else None
        
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
            'predictedBenefit': predictedBenefit,
            'realBenefit': realBenefit,
            'predictedCumulativeSales': predictedCumulativeSales,
            'realCumulativeSales': realCumulativeSales,
            'predictedCumulativeHoldingCost': predictedCumulativeHoldingCost,
            'realCumulativeHoldingCost': realCumulativeHoldingCost,
            'predictedCumulativeOrderingCost': predictedCumulativeOrderingCost,
            'realCumulativeOrderingCost': realCumulativeOrderingCost,
            'predictedCumulativeStockOutCost': predictedCumulativeStockOutCost,
            'realCumulativeStockOutCost': realCumulativeStockOutCost,
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


def createCombinedPlots(results, outputPath=None):
    fig = make_subplots(
        rows=7, cols=1,
        subplot_titles=(
            'Comparación: Acciones Reales vs Decision Transformer',
            'Comparación: Benefits Reales vs Decision Transformer',
            'Inventario Disponible vs Demanda por Paso',
            'Unidades Vendidas Acumuladas: Real vs Decision Transformer',
            'Coste de Mantenimiento Acumulado: Real vs Decision Transformer',
            'Coste de Pedido Acumulado: Real vs Decision Transformer',
            'Coste de Stockout Acumulado: Real vs Decision Transformer'
        ),
        vertical_spacing=0.05,
        row_heights=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
    )
    
    steps = [r['step'] for r in results]
    
    realActions = [r['realAction'] for r in results]
    predictedActions = [r['predictedAction'] for r in results]
    avgError = np.mean([abs(r['realAction'] - r['predictedAction']) for r in results])
    avgAccuracy = np.mean([r['actionAccuracy'] for r in results])
    
    fig.add_trace(go.Scatter(x=steps, y=realActions, mode='lines+markers', name='Acción Real',
                             line=dict(color='blue', width=2), marker=dict(size=4, symbol='circle')), row=1, col=1)
    fig.add_trace(go.Scatter(x=steps, y=predictedActions, mode='lines+markers', name='Acción Predicha (DT)',
                             line=dict(color='red', width=2, dash='dash'), marker=dict(size=4, symbol='square')), row=1, col=1)
    
    predictedBenefits = [r['predictedBenefit'] for r in results]
    hasRealBenefits = any(r['realBenefit'] is not None for r in results)
    fig.add_trace(go.Scatter(x=steps, y=predictedBenefits, mode='lines+markers', name='Benefit Predicho (DT)',
                             line=dict(color='red', width=2, dash='dash'), marker=dict(size=4, symbol='square')), row=2, col=1)
    
    if hasRealBenefits:
        validResults = [r for r in results if r['realBenefit'] is not None]
        realSteps = [r['step'] for r in validResults]
        realBenefits = [r['realBenefit'] for r in validResults]
        fig.add_trace(go.Scatter(x=realSteps, y=realBenefits, mode='lines+markers', name='Benefit Real',
                                 line=dict(color='blue', width=2), marker=dict(size=4, symbol='circle')), row=2, col=1)
    
    onHandLevels = [r['state']['onHandLevel'] for r in results]
    demands = [r['state']['demand'] for r in results]
    fig.add_trace(go.Scatter(x=steps, y=onHandLevels, mode='lines+markers', name='Inventario Disponible',
                             line=dict(color='green', width=2), marker=dict(size=4, symbol='circle')), row=3, col=1)
    fig.add_trace(go.Scatter(x=steps, y=demands, mode='lines+markers', name='Demanda',
                             line=dict(color='orange', width=2, dash='dash'), marker=dict(size=4, symbol='square')), row=3, col=1)
    
    # Agregar gráfica de unidades vendidas acumuladas
    realCumulativeSales = [r['realCumulativeSales'] for r in results]
    predictedCumulativeSales = [r['predictedCumulativeSales'] for r in results]
    fig.add_trace(go.Scatter(x=steps, y=realCumulativeSales, mode='lines+markers', name='Ventas Acumuladas Real',
                             line=dict(color='blue', width=2), marker=dict(size=4, symbol='circle')), row=4, col=1)
    fig.add_trace(go.Scatter(x=steps, y=predictedCumulativeSales, mode='lines+markers', name='Ventas Acumuladas Predichas (DT)',
                             line=dict(color='red', width=2, dash='dash'), marker=dict(size=4, symbol='square')), row=4, col=1)
    
    # Actualizar ejes X
    for i in range(1, 8):
        fig.update_xaxes(title_text='Paso', row=i, col=1)
    
    # Actualizar ejes Y
    fig.update_yaxes(title_text='Cantidad a Ordenar', row=1, col=1)
    fig.update_yaxes(title_text='Benefit Acumulado', row=2, col=1)
    fig.update_yaxes(title_text='Cantidad', row=3, col=1)
    fig.update_yaxes(title_text='Unidades Vendidas', row=4, col=1)
    fig.update_yaxes(title_text='Coste Acumulado', row=5, col=1)
    fig.update_yaxes(title_text='Coste Acumulado', row=6, col=1)
    fig.update_yaxes(title_text='Coste Acumulado', row=7, col=1)
    
    fig.add_annotation(text=f'Error Promedio: {avgError:.3f}<br>Precisión Promedio: {avgAccuracy:.3f}',
                       xref='paper', yref='paper', x=0.02, y=0.98, xanchor='left', yanchor='top',
                       showarrow=False, font=dict(size=10), align='left',
                       bgcolor='rgba(245, 222, 179, 0.8)', bordercolor='rgba(0, 0, 0, 0.5)', borderwidth=1,
                       row=1, col=1)
    
    if hasRealBenefits:
        validResults = [r for r in results if r['realBenefit'] is not None]
        avgError = np.mean([abs(r['predictedBenefit'] - r['realBenefit']) for r in validResults])
        avgPredicted = np.mean([r['predictedBenefit'] for r in validResults])
        avgReal = np.mean([r['realBenefit'] for r in validResults])
        annotationText = f'Benefit Promedio Predicho: {avgPredicted:.3f}<br>Benefit Promedio Real: {avgReal:.3f}<br>Error Promedio: {avgError:.3f}'
    else:
        avgPredicted = np.mean(predictedBenefits)
        annotationText = f'Benefit Promedio Predicho: {avgPredicted:.3f}'
    
    fig.add_annotation(text=annotationText, xref='paper', yref='paper', x=0.02, y=0.98,
                       xanchor='left', yanchor='top', showarrow=False, font=dict(size=10), align='left',
                       bgcolor='rgba(245, 222, 179, 0.8)', bordercolor='rgba(0, 0, 0, 0.5)', borderwidth=1,
                       row=2, col=1)
    
    avgOnHand = np.mean(onHandLevels)
    avgDemand = np.mean(demands)
    minOnHand = np.min(onHandLevels)
    maxOnHand = np.max(onHandLevels)
    annotationText = f'Inventario Promedio: {avgOnHand:.3f}<br>Demanda Promedio: {avgDemand:.3f}<br>Inventario Mín: {minOnHand:.3f}<br>Inventario Máx: {maxOnHand:.3f}'
    
    fig.add_annotation(text=annotationText, xref='paper', yref='paper', x=0.02, y=0.98,
                       xanchor='left', yanchor='top', showarrow=False, font=dict(size=10), align='left',
                       bgcolor='rgba(245, 222, 179, 0.8)', bordercolor='rgba(0, 0, 0, 0.5)', borderwidth=1,
                       row=3, col=1)
    
    # Anotación para la gráfica de ventas acumuladas
    totalRealSales = realCumulativeSales[-1] if realCumulativeSales else 0
    totalPredictedSales = predictedCumulativeSales[-1] if predictedCumulativeSales else 0
    salesDifference = abs(totalRealSales - totalPredictedSales)
    salesError = salesDifference / max(totalRealSales, 1e-6) if totalRealSales > 0 else salesDifference
    annotationText = f'Ventas Totales Real: {totalRealSales:.2f}<br>Ventas Totales Predichas: {totalPredictedSales:.2f}<br>Diferencia: {salesDifference:.2f}<br>Error Relativo: {salesError*100:.2f}%'
    
    fig.add_annotation(text=annotationText, xref='paper', yref='paper', x=0.02, y=0.98,
                       xanchor='left', yanchor='top', showarrow=False, font=dict(size=10), align='left',
                       bgcolor='rgba(245, 222, 179, 0.8)', bordercolor='rgba(0, 0, 0, 0.5)', borderwidth=1,
                       row=4, col=1)
    
    # Gráfica de coste de mantenimiento acumulado
    predictedCumulativeHoldingCost = [r['predictedCumulativeHoldingCost'] for r in results]
    hasRealHoldingCost = any(r['realCumulativeHoldingCost'] is not None for r in results)
    fig.add_trace(go.Scatter(x=steps, y=predictedCumulativeHoldingCost, mode='lines+markers', name='Holding Cost Predicho (DT)',
                             line=dict(color='red', width=2, dash='dash'), marker=dict(size=4, symbol='square')), row=5, col=1)
    
    if hasRealHoldingCost:
        validResults = [r for r in results if r['realCumulativeHoldingCost'] is not None]
        realSteps = [r['step'] for r in validResults]
        realCumulativeHoldingCost = [r['realCumulativeHoldingCost'] for r in validResults]
        fig.add_trace(go.Scatter(x=realSteps, y=realCumulativeHoldingCost, mode='lines+markers', name='Holding Cost Real',
                                 line=dict(color='blue', width=2), marker=dict(size=4, symbol='circle')), row=5, col=1)
        
        totalReal = realCumulativeHoldingCost[-1] if realCumulativeHoldingCost else 0
        totalPredicted = predictedCumulativeHoldingCost[-1] if predictedCumulativeHoldingCost else 0
        costDifference = abs(totalReal - totalPredicted)
        costError = costDifference / max(totalReal, 1e-6) if totalReal > 0 else costDifference
        annotationText = f'Coste Total Real: {totalReal:.2f}<br>Coste Total Predicho: {totalPredicted:.2f}<br>Diferencia: {costDifference:.2f}<br>Error Relativo: {costError*100:.2f}%'
    else:
        totalPredicted = predictedCumulativeHoldingCost[-1] if predictedCumulativeHoldingCost else 0
        annotationText = f'Coste Total Predicho: {totalPredicted:.2f}'
    
    fig.add_annotation(text=annotationText, xref='paper', yref='paper', x=0.02, y=0.98,
                       xanchor='left', yanchor='top', showarrow=False, font=dict(size=10), align='left',
                       bgcolor='rgba(245, 222, 179, 0.8)', bordercolor='rgba(0, 0, 0, 0.5)', borderwidth=1,
                       row=5, col=1)
    
    # Gráfica de coste de pedido acumulado
    predictedCumulativeOrderingCost = [r['predictedCumulativeOrderingCost'] for r in results]
    hasRealOrderingCost = any(r['realCumulativeOrderingCost'] is not None for r in results)
    fig.add_trace(go.Scatter(x=steps, y=predictedCumulativeOrderingCost, mode='lines+markers', name='Ordering Cost Predicho (DT)',
                             line=dict(color='red', width=2, dash='dash'), marker=dict(size=4, symbol='square')), row=6, col=1)
    
    if hasRealOrderingCost:
        validResults = [r for r in results if r['realCumulativeOrderingCost'] is not None]
        realSteps = [r['step'] for r in validResults]
        realCumulativeOrderingCost = [r['realCumulativeOrderingCost'] for r in validResults]
        fig.add_trace(go.Scatter(x=realSteps, y=realCumulativeOrderingCost, mode='lines+markers', name='Ordering Cost Real',
                                 line=dict(color='blue', width=2), marker=dict(size=4, symbol='circle')), row=6, col=1)
        
        totalReal = realCumulativeOrderingCost[-1] if realCumulativeOrderingCost else 0
        totalPredicted = predictedCumulativeOrderingCost[-1] if predictedCumulativeOrderingCost else 0
        costDifference = abs(totalReal - totalPredicted)
        costError = costDifference / max(totalReal, 1e-6) if totalReal > 0 else costDifference
        annotationText = f'Coste Total Real: {totalReal:.2f}<br>Coste Total Predicho: {totalPredicted:.2f}<br>Diferencia: {costDifference:.2f}<br>Error Relativo: {costError*100:.2f}%'
    else:
        totalPredicted = predictedCumulativeOrderingCost[-1] if predictedCumulativeOrderingCost else 0
        annotationText = f'Coste Total Predicho: {totalPredicted:.2f}'
    
    fig.add_annotation(text=annotationText, xref='paper', yref='paper', x=0.02, y=0.98,
                       xanchor='left', yanchor='top', showarrow=False, font=dict(size=10), align='left',
                       bgcolor='rgba(245, 222, 179, 0.8)', bordercolor='rgba(0, 0, 0, 0.5)', borderwidth=1,
                       row=6, col=1)
    
    # Gráfica de coste de stockout acumulado
    predictedCumulativeStockOutCost = [r['predictedCumulativeStockOutCost'] for r in results]
    hasRealStockOutCost = any(r['realCumulativeStockOutCost'] is not None for r in results)
    fig.add_trace(go.Scatter(x=steps, y=predictedCumulativeStockOutCost, mode='lines+markers', name='Stockout Cost Predicho (DT)',
                             line=dict(color='red', width=2, dash='dash'), marker=dict(size=4, symbol='square')), row=7, col=1)
    
    if hasRealStockOutCost:
        validResults = [r for r in results if r['realCumulativeStockOutCost'] is not None]
        realSteps = [r['step'] for r in validResults]
        realCumulativeStockOutCost = [r['realCumulativeStockOutCost'] for r in validResults]
        fig.add_trace(go.Scatter(x=realSteps, y=realCumulativeStockOutCost, mode='lines+markers', name='Stockout Cost Real',
                                 line=dict(color='blue', width=2), marker=dict(size=4, symbol='circle')), row=7, col=1)
        
        totalReal = realCumulativeStockOutCost[-1] if realCumulativeStockOutCost else 0
        totalPredicted = predictedCumulativeStockOutCost[-1] if predictedCumulativeStockOutCost else 0
        costDifference = abs(totalReal - totalPredicted)
        costError = costDifference / max(totalReal, 1e-6) if totalReal > 0 else costDifference
        annotationText = f'Coste Total Real: {totalReal:.2f}<br>Coste Total Predicho: {totalPredicted:.2f}<br>Diferencia: {costDifference:.2f}<br>Error Relativo: {costError*100:.2f}%'
    else:
        totalPredicted = predictedCumulativeStockOutCost[-1] if predictedCumulativeStockOutCost else 0
        annotationText = f'Coste Total Predicho: {totalPredicted:.2f}'
    
    fig.add_annotation(text=annotationText, xref='paper', yref='paper', x=0.02, y=0.98,
                       xanchor='left', yanchor='top', showarrow=False, font=dict(size=10), align='left',
                       bgcolor='rgba(245, 222, 179, 0.8)', bordercolor='rgba(0, 0, 0, 0.5)', borderwidth=1,
                       row=7, col=1)
    
    fig.update_layout(height=5600, showlegend=True, hovermode='x unified')
    
    if outputPath:
        if outputPath.endswith('.html'):
            fig.write_html(outputPath)
        else:
            fig.write_image(outputPath, width=1200, height=5600, scale=2)
    
    fig.show()
    
    return fig


def runDecisionTransformerTest(modelPath, dataPath, problemIndex=0, maxSteps=None, outputPath=None, plotOutputPath=None):
    model = loadTrainedModel(modelPath, dataPath=dataPath).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    problem = getTestProblem(dataPath, problemIndex).to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    results = tryDecisionTransformer(model, problem, maxSteps)
    report = generateTestReport(results, outputPath)
    createCombinedPlots(results, plotOutputPath)
    
    return report


if __name__ == "__main__":
    projectDir = getProjectDirectory()
    modelPath = os.path.join(projectDir, "training_models/decision_transformer_model", "training.pt")
    dataPath = os.path.join(projectDir, "data", "training_data2.pt")
    outputPath = os.path.join(projectDir, "test_results.json")
    plotOutputPath = os.path.join(projectDir, "combined_plots.html")
    
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
            problemIndex=2,
            maxSteps=30,
            outputPath=outputPath,
            plotOutputPath=plotOutputPath
        )
        
        print(f"\nTest completado exitosamente!")
        print(f"Reporte detallado guardado en: {outputPath}")
        print(f"Gráficas combinadas guardadas en: {plotOutputPath}")
        
    except Exception as e:
        print(f"Error durante el test: {e}")
        import traceback
        traceback.print_exc()
