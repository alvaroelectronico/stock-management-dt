import os
import torch
import json
from pathlib import Path
from generate_tajectories import TRAJECTORY_LENGTH, FORECAST_LENGTH, MAX_LEAD_TIME


def getProjectDirectory():
    """Obtiene el directorio del proyecto."""
    return str(Path(__file__).resolve().parent)


def loadTrainingData(dataPath):
    """
    Carga los datos de entrenamiento que contienen las trayectorias.
    
    Args:
        dataPath: Ruta a los datos de entrenamiento
    
    Returns:
        TensorDict con todas las trayectorias
    """
    trainingData = torch.load(dataPath, weights_only=False)
    return trainingData


def extractTrajectoryInfo(trainingData, trajectoryIndex=0, startStep=0, numSteps=10):
    """
    Extrae la información detallada de una trayectoria específica.
    
    Args:
        trainingData: Datos de entrenamiento con todas las trayectorias
        trajectoryIndex: Índice de la trayectoria a analizar
        startStep: Paso inicial desde donde empezar a mostrar
        numSteps: Número de pasos a mostrar desde startStep
    
    Returns:
        Lista con la información de cada paso
    """
    # Obtener datos de la trayectoria seleccionada
    states = trainingData['states'][trajectoryIndex]
    actions = trainingData['actions'][trajectoryIndex]
    returnsToGo = trainingData['returnsToGo'][trajectoryIndex]
    
    # Calcular el paso final
    totalAvailableSteps = actions.size(0)
    endStep = min(startStep + numSteps, totalAvailableSteps)
    
    # Validar que startStep esté dentro del rango
    if startStep >= totalAvailableSteps:
        raise ValueError(f"startStep ({startStep}) debe ser menor que el total de pasos ({totalAvailableSteps})")
    
    trajectoryInfo = []
    
    for step in range(startStep, endStep):
        # Extraer información del estado
        stepInfo = {
            'step': step,
            'state': {
                'onHandLevel': states['onHandLevel'][step].item(),
                'inTransitStock': states['inTransitStock'][step].tolist(),
                'demand': states['demand'][step].item(),
                'forecast': states['forecast'][step].tolist(),
                'holdingCost': states['holdingCost'][step].item(),
                'orderingCost': states['orderingCost'][step].item(),
                'stockOutPenalty': states['stockOutPenalty'][step].item(),
                'unitRevenue': states['unitRevenue'][step].item(),
                'leadTime': states['leadTime'][step].item(),
                'timeStep': states['timesStep'][step].item()
            },
            'action': actions[step].item(),
            'returnToGo': returnsToGo[step].item()
        }
        
        trajectoryInfo.append(stepInfo)
    
    return trajectoryInfo


def displayTrajectoryInfo(trajectoryInfo, trajectoryIndex=0):
    """
    Muestra por consola la información de la trayectoria de forma legible.
    
    Args:
        trajectoryInfo: Lista con la información de cada paso
        trajectoryIndex: Índice de la trayectoria analizada
    """
    print(f"\n{'='*80}")
    print(f"INFORMACIÓN DE TRAYECTORIA #{trajectoryIndex}")
    print(f"{'='*80}\n")
    
    for stepData in trajectoryInfo:
        step = stepData['step']
        state = stepData['state']
        action = stepData['action']
        returnToGo = stepData['returnToGo']
        
        print(f"--- PASO {step} ---")
        print(f"\nEstado:")
        print(f"  • Stock Físico (On-Hand Level): {state['onHandLevel']:.2f}")
        print(f"  • Stock en Tránsito: {[f'{x:.2f}' for x in state['inTransitStock']]}")
        print(f"  • Demanda Actual: {state['demand']:.2f}")
        print(f"  • Forecast (próximos {FORECAST_LENGTH} períodos): {[f'{x:.2f}' for x in state['forecast']]}")
        print(f"  • Lead Time: {state['leadTime']:.0f}")
        print(f"  • Time Step: {state['timeStep']:.0f}")
        
        print(f"\nCostos y Beneficios:")
        print(f"  • Costo de Mantener Inventario: {state['holdingCost']:.2f}")
        print(f"  • Costo de Ordenar: {state['orderingCost']:.2f}")
        print(f"  • Penalización por Stockout: {state['stockOutPenalty']:.2f}")
        print(f"  • Ingreso por Unidad: {state['unitRevenue']:.2f}")
        
        print(f"\nDecisión y Objetivo:")
        print(f"  • Cantidad a Ordenar (Acción): {action:.2f}")
        print(f"  • Return-to-Go: {returnToGo:.2f}")
        
        print(f"\n{'-'*80}\n")


def saveTrajectoryInfoToJSON(trajectoryInfo, outputPath, trajectoryIndex=0):
    """
    Guarda la información de la trayectoria en un archivo JSON.
    
    Args:
        trajectoryInfo: Lista con la información de cada paso
        outputPath: Ruta donde guardar el archivo JSON
        trajectoryIndex: Índice de la trayectoria analizada
    """
    # Crear estructura del reporte
    report = {
        'trajectoryIndex': trajectoryIndex,
        'totalSteps': len(trajectoryInfo),
        'steps': trajectoryInfo
    }
    
    # Guardar en archivo JSON
    with open(outputPath, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"\nInformación de trayectoria guardada en: {outputPath}")


def analyzeTrajectory(dataPath, trajectoryIndex=0, startStep=0, numSteps=10, outputPath=None, showConsole=True):
    """
    Función principal para analizar y mostrar información de una trayectoria.
    
    Args:
        dataPath: Ruta a los datos de entrenamiento
        trajectoryIndex: Índice de la trayectoria a analizar
        startStep: Paso inicial desde donde empezar a mostrar
        numSteps: Número de pasos a mostrar desde startStep
        outputPath: Ruta donde guardar el reporte JSON (opcional)
        showConsole: Si True, muestra la información por consola
    
    Returns:
        Lista con la información de cada paso
    """
    print(f"\n=== Análisis de Trayectoria ===")
    print(f"Cargando datos desde: {dataPath}")
    
    # Cargar datos de entrenamiento
    trainingData = loadTrainingData(dataPath)
    
    # Obtener número total de trayectorias
    totalTrajectories = trainingData['states'].size(0)
    print(f"Total de trayectorias disponibles: {totalTrajectories}")
    print(f"Analizando trayectoria #{trajectoryIndex}")
    print(f"Mostrando pasos del {startStep} al {startStep + numSteps - 1}")
    
    # Extraer información de la trayectoria
    trajectoryInfo = extractTrajectoryInfo(trainingData, trajectoryIndex, startStep, numSteps)
    
    # Mostrar información por consola si se solicita
    if showConsole:
        displayTrajectoryInfo(trajectoryInfo, trajectoryIndex)
    
    # Guardar en archivo JSON si se especifica
    if outputPath:
        saveTrajectoryInfoToJSON(trajectoryInfo, outputPath, trajectoryIndex)
    
    return trajectoryInfo


if __name__ == "__main__":
    # Configuración por defecto
    projectDir = getProjectDirectory()
    dataPath = os.path.join(projectDir, "data", "training_data2.pt")
    outputPath = os.path.join(projectDir, "trajectory_analysis.json")
    
    # Verificar que los datos existen
    if not os.path.exists(dataPath):
        print(f"Error: No se encontraron los datos en {dataPath}")
        print("Ejecuta generate_tajectories.py para generar los datos de entrenamiento.")
        exit(1)
    
    # Analizar trayectoria
    try:
        trajectoryInfo = analyzeTrajectory(
            dataPath=dataPath,
            trajectoryIndex=0,  # Primera trayectoria
            startStep=0,  # Empezar desde el paso 10
            numSteps=20,  # Mostrar 10 pasos (del 10 al 19)
            outputPath=outputPath,
            showConsole=True
        )
        
        print(f"\n✓ Análisis completado exitosamente!")
        
    except Exception as e:
        print(f"Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()

