import numpy as np
from pathlib import Path
import torch
from tensordict import TensorDict
import scipy.stats as stats

MIN_LEAD_TIME = 1
MAX_LEAD_TIME = 10

UNIT_REVENUE = 100

MIN_UNIT_COST = 50
MAX_UNIT_COST = 95

MIN_HOLDING_COST = 0.2
MAX_HOLDING_COST = 5.0

MIN_STOCKOUT_PENALTY_ALPHA = 1.0
MAX_STOCKOUT_PENALTY_ALPHA = 4.0

MIN_ORDERING_COST = 10
MAX_ORDERING_COST = 500

MIN_DEMAND_MEAN = 10
MAX_DEMAND_MEAN = 20

MIN_DEMAND_STD = 1
MAX_DEMAND_STD = 2

TRAJECTORY_LENGTH = 30
FORECAST_LENGTH = 10 

RETURN_TO_GO_WINDOW = 10
DEMAND_FORECAST_WINDOW = 10

CSL = 0.95




def generateInstanceData():
    """
    Genera datos aleatorios para una instancia del problema de inventario.
    Usa rangos recomendados para stock management con valores continuos.
    """
    leadTime = np.random.randint(MIN_LEAD_TIME, MAX_LEAD_TIME + 1)  
    unitRevenue = UNIT_REVENUE
    
    unitCost = np.random.uniform(MIN_UNIT_COST, MAX_UNIT_COST)
    holdingCost = np.random.uniform(MIN_HOLDING_COST, MAX_HOLDING_COST)
    orderingCost = np.random.uniform(MIN_ORDERING_COST, MAX_ORDERING_COST)
    
    marginLost = unitRevenue - unitCost
    alpha = np.random.uniform(MIN_STOCKOUT_PENALTY_ALPHA, MAX_STOCKOUT_PENALTY_ALPHA)
    stockOutPenalty = marginLost * alpha
    
    inputData = {'leadtime': leadTime, 
                 'holdingCost': holdingCost,
                 'orderingCost': orderingCost, 
                 'stockOutPenalty': stockOutPenalty, 
                 'unitRevenue': unitRevenue,
                 'unitCost': unitCost,
                 'inTransitStock': np.zeros(leadTime, dtype=int)} 
    return inputData

def generateTrajectory(inputData, trajectoryLength=TRAJECTORY_LENGTH):
    """
    Genera trayectoria con warm-up hasta que llega el primer pedido.
    Guarda trajectoryLength pasos desde el momento en que llega el primer pedido (incluido ese paso).
    """
    leadTime = inputData['leadtime'] 
    holdingCost = inputData['holdingCost']
    inTransitStock = inputData['inTransitStock'] 
    orderingCost = inputData['orderingCost'] 
    stockOutPenalty = inputData['stockOutPenalty'] 
    unitRevenue = inputData['unitRevenue'] 

    totalHoldingCost = 0
    totalOrderingCost = 0
    totalStockOutCost = 0
    totalIncome = 0
    totalBenefit = 0

    demand_mean = np.random.uniform(MIN_DEMAND_MEAN, MAX_DEMAND_MEAN)
    demand_std = np.random.uniform(MIN_DEMAND_STD, MAX_DEMAND_STD)

    eoq = int(np.ceil(np.sqrt(2*orderingCost*demand_mean/holdingCost)))

    k = stats.norm.ppf(CSL)
    safetyStock = int(np.ceil(k * demand_std * np.sqrt(leadTime)))
    reorderPoint = int(np.ceil(demand_mean*leadTime + safetyStock))
    onHandLevel = int(np.ceil(eoq/2 + safetyStock))

    reward = 0
    trajectory = []
    
    currentForecast = np.ceil(np.random.normal(demand_mean, demand_std, size=FORECAST_LENGTH)).astype(int)
    
    startRecording = False
    stepsRecorded = 0
    
    maxIterations = trajectoryLength + leadTime + 10
    t = 0

    while stepsRecorded < trajectoryLength and t < maxIterations:
        currentDemand = max(0, int(np.ceil(np.random.normal(demand_mean, demand_std))))
        
        if t > 0:
            currentForecast = np.roll(currentForecast, -1)
            currentForecast[-1] = int(np.ceil(np.random.normal(demand_mean, demand_std)))
        
        if not startRecording and inTransitStock[0] > 0:
            startRecording = True
        
        if startRecording:
            state = {
                'onHandLevel': onHandLevel,
                'inTransitStock': inTransitStock.copy(),
                'forecast': currentForecast,
                'demand': currentDemand,
                'orderingCost': orderingCost,
                'holdingCost': holdingCost,
                'stockOutPenalty': stockOutPenalty,
                'unitRevenue': unitRevenue,
                'leadTime': leadTime,
                'timesStep': stepsRecorded
            }
        
        onHandLevel = int(onHandLevel + inTransitStock[0])
        inTransitStock = np.roll(inTransitStock, -1)
        inTransitStock[-1] = 0
        
        inventoryPosition = int(onHandLevel + sum(inTransitStock))
        
        current_order_quantity = 0
        if inventoryPosition <= reorderPoint:
            current_order_quantity = int(eoq)
            totalOrderingCost += orderingCost
        
        inTransitStock[-1] = current_order_quantity
        
        totalHoldingCost += holdingCost * onHandLevel
        
        totalStockOutCost += stockOutPenalty * max(0, currentDemand - onHandLevel)
        totalIncome += unitRevenue * min(currentDemand, onHandLevel)
        onHandLevel = max(0, int(onHandLevel - currentDemand))
        
        if startRecording:
            totalBenefit += totalIncome - totalHoldingCost - totalStockOutCost - totalOrderingCost
            
            trajectory.append({
                'state': state, 
                'action': current_order_quantity,
                'returnToGo': 0.0,
                'benefit': totalBenefit
            })
            
            stepsRecorded += 1
        
        t += 1
    
    if stepsRecorded < trajectoryLength:
        print(f"Advertencia: Solo se grabaron {stepsRecorded} de {trajectoryLength} pasos")

    reward = (totalIncome - totalHoldingCost - totalStockOutCost - totalOrderingCost) / stepsRecorded if stepsRecorded > 0 else 0
    
    for i in range(stepsRecorded):
        if i >= RETURN_TO_GO_WINDOW:
            #BenefitToAdd = totalBenefit[i-RETURN_TO_GO_WINDOW] - totalBenefit[i-RETURN_TO_GO_WINDOW-1]
            #benefitToSubstract = totalBenefit[i] - totalBenefit[i-1]
            #returnToGo = (reward*RETURN_TO_GO_WINDOW + BenefitToAdd - benefitToSubstract) / RETURN_TO_GO_WINDOW
            #trajectory[i]['returnToGo'] = returnToGo
            trajectory[i]['returnToGo'] = 0
        else:
            #trajectory[i]['returnToGo'] = reward
            trajectory[i]['returnToGo'] = 0

        
    return trajectory



def addTrajectoryToTrainingData(trajectory, trainingData):
    """
    Añade una trayectoria al conjunto de datos de entrenamiento con padding apropiado.
    """
    def addPaddingToTransitStock(inTransitStock):
        """
        Añade padding al array de stock en tránsito para igualar al máximo lead time.
        """
        padded = np.zeros(MAX_LEAD_TIME)
        padded[:len(inTransitStock)] = inTransitStock
        return padded
    
    first_state = trajectory[0]['state']
    
    new_trajectory = TensorDict({
        'states': TensorDict({
            'onHandLevel': torch.tensor(first_state['onHandLevel'], dtype=torch.float),
            'inTransitStock': torch.stack([torch.tensor(addPaddingToTransitStock(t['state']['inTransitStock']), dtype=torch.float) for t in trajectory]),
            'demand': torch.stack([torch.tensor(t['state']['demand'], dtype=torch.float) for t in trajectory]),
            'forecast': torch.stack([torch.tensor(t['state']['forecast'], dtype=torch.float) for t in trajectory]),
            'leadTime': torch.tensor(first_state['leadTime'], dtype=torch.float),
            'holdingCost': torch.tensor(first_state['holdingCost'], dtype=torch.float),
            'orderingCost': torch.tensor(first_state['orderingCost'], dtype=torch.float),
            'stockOutPenalty': torch.tensor(first_state['stockOutPenalty'], dtype=torch.float),
            'unitRevenue': torch.tensor(first_state['unitRevenue'], dtype=torch.float),
            'timesStep': torch.stack([torch.tensor(t['state']['timesStep'], dtype=torch.float) for t in trajectory]),
        }),
        'actions': torch.stack([torch.tensor(t['action'], dtype=torch.float) for t in trajectory]),
        'returnsToGo': torch.tensor(trajectory[0]['returnToGo'], dtype=torch.float),
        'benefit': torch.tensor(trajectory[-1]['benefit'], dtype=torch.float)
    })
    
    if len(trainingData.keys()) == 0:
        return new_trajectory.unsqueeze(0)
    
    return torch.cat([trainingData, new_trajectory.unsqueeze(0)], dim=0)





if __name__ == "__main__":
    noTrajectories = 60000
    trainingData = TensorDict({})
    
    
    for i in range(noTrajectories):
        inputData = generateInstanceData()
        trajectory = generateTrajectory(inputData)
        trainingData = addTrajectoryToTrainingData(trajectory, trainingData)

    def getProjectDirectory():
        """
        Obtiene el directorio raíz del proyecto.
        """
        return str(Path(__file__).resolve().parent)

    torch.save(trainingData,
               getProjectDirectory() + "/data/training_data2.pt")
    
    


