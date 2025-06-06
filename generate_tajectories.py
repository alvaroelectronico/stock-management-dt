import numpy as np
from pathlib import Path
import torch.random
from tensordict import TensorDict
import scipy.stats as stats # para calcular la k de la distribución normal




MIN_LEAD_TIME = 10
MAX_LEAD_TIME = 20

MIN_HOLDING_COST = 1
MAX_HOLDING_COST = 10

MIN_ORDERING_COST = 1 # TODO: Change this to a consistent value
MAX_ORDERING_COST = 10 # TODO: Change this to a consistent value

MIN_STOCKOUT_PENALTY = 1 # TODO: Change this to a consistent value
MAX_STOCKOUT_PENALTY = 10 # TODO: Change this to a consistent value

MIN_UNIT_REVENUE = 1 # TODO: Change this to a consistent value
MAX_UNIT_REVENUE = 10 # TODO: Change this to a consistent value

MIN_DEMAND_MEAN = 10
MAX_DEMAND_MEAN = 20

MIN_DEMAND_STD = 1
MAX_DEMAND_STD = 2

TRAJECTORY_LENGTH = 200
FORECAST_LENGTH = 10 

RETURN_TO_GO_WINDOW = 10 #TODO: Change this to a consisten value
DEMAND_FORECAST_WINDOW = 10

CSL = 0.95 # Probabilidad de que no haya stockout durante un ciclo




def generateInstanceData():
    leadTime = np.random.randint(MIN_LEAD_TIME, MAX_LEAD_TIME)
    holdingCost = np.random.randint(MIN_HOLDING_COST, MAX_HOLDING_COST)
    onHandLevel = 0 # TODO: Change this to a consisten value
    orderingCost = np.random.randint(MIN_ORDERING_COST, MAX_ORDERING_COST)  
    stockOutPenalty = np.random.randint(MIN_STOCKOUT_PENALTY, MAX_STOCKOUT_PENALTY)  
    unitRevenue = np.random.randint(MIN_UNIT_REVENUE, MAX_UNIT_REVENUE)  
    
    inputData = {'leadtime': leadTime, 
                 'holdingCost': holdingCost, 
                 'onHandLevel': onHandLevel,
                 'orderingCost': orderingCost, 
                 'stockOutPenalty': stockOutPenalty, 
                 'unitRevenue': unitRevenue, 
                 'inTransitStock': np.zeros(leadTime-1)} #  list with the pending orders 
    print(inputData)
    return inputData

def generateTrajectory(inputData, trajectoryLength=TRAJECTORY_LENGTH):
    print("\n=== Iniciando generación de trayectoria ===")

    # Extracting input data
    leadTime = inputData['leadtime'] 
    holdingCost = inputData['holdingCost']
    onHandLevel = inputData['onHandLevel']
    inTransitStock = inputData['inTransitStock'] 
    orderingCost = inputData['orderingCost'] 
    stockOutPenalty = inputData['stockOutPenalty'] 
    unitRevenue = inputData['unitRevenue'] 

    # Defining system variables
    totalHoldingCost = 0
    totalOrderingCost = 0
    totalStockOutCost = 0
    totalIncome = 0
    totalBenefit = np.zeros(trajectoryLength)
    orderQuantity = 0
    noOrders = 0
     

    demand_mean = np.random.randint(MIN_DEMAND_MEAN, MAX_DEMAND_MEAN)
    demand_std = np.random.randint(MIN_DEMAND_STD, MAX_DEMAND_STD)

    print(f"Demand mean: {demand_mean}, Demand std: {demand_std}")
    # Calulating EOQ
    
    eoq = np.sqrt(2*orderingCost*demand_mean/holdingCost)
    print(f"EOQ calculado: {eoq:.2f}")

    # Calculating ROP for a given CSL
    k = stats.norm.ppf(CSL)
    safetyStock = k * demand_std * np.sqrt(leadTime)  #generalizar  CSL probabilidad que durante un ciclo haya rotura para definir la k
    reorderPoint = demand_mean*leadTime + safetyStock
    onHandLevel = eoq/2 + safetyStock

    print(f"Reorder point: {reorderPoint}, On hand level: {onHandLevel}")

    reward = 0   # En realidad, esto tiene que guardar el benficio medio para el conjunto de la trayectoria.
    trajectory = []

    for t in range(trajectoryLength):
        #generate demand and forecast for the current period
        currentDemand = np.random.normal(demand_mean, demand_std)
        currentForecast = np.random.normal(demand_mean, demand_std, size=FORECAST_LENGTH)
        
        print(f"\n--- Período {t} ---")
        print(f"Current demand: {currentDemand}, Current forecast: {currentForecast}")
        print(f"Stock actual: {onHandLevel:.2f}")
        print(f"Stock en tránsito: {sum(inTransitStock)}")
        
        #If the inventory level is updated with the amount received in the period t.
        onHandLevel = onHandLevel + inTransitStock[0]  #the first element of the tensor is the amount that will arrive in the period t
        inTransitStock = np.roll(inTransitStock, -1)  #shift the tensor to the left
        inTransitStock[-1] = 0  #set to 0 the last element of the tensor which are not used

        print(f"Stock actual: {onHandLevel:.2f}")
        print(f"Stock en tránsito: {sum(inTransitStock)}")

        #Define the current state of the system
        state ={
            'onHandLevel': onHandLevel,
            'inTransitStock': inTransitStock,
            'forecast': currentForecast,
            'demand': currentDemand,
            'orderingCost': orderingCost,  # Fixed cost when placing an order, regardless of the amount ordered
            'holdingCost': holdingCost,
            'stockOutPenalty':  stockOutPenalty,
            'unitRevenue': unitRevenue,
            'leadTime':  leadTime,
            'timesStep': t  # Add the current time step
        }
            

        #Update the stock on hand and calculate the stockOut cost and the income
        totalStockOutCost += stockOutPenalty *max(0, currentDemand - onHandLevel) 
        totalIncome += unitRevenue * min(currentDemand, onHandLevel) 
        onHandLevel = max(0, onHandLevel - currentDemand )
        print(f"Stock actual: {onHandLevel:.2f}")
        print(f"StockOut cost: {totalStockOutCost:.2f}")
        print(f"Income: {totalIncome:.2f}")
        
        #Calculate the total stock, which is the physical stock plus the stock in transit.
        inventoryPosition = onHandLevel + sum(inTransitStock)  #inventory position = on hand/inventoyr level + on order/in transit
        print(f"Inventory position: {inventoryPosition:.2f}")

        #decide the amount to order and calculate the costs 
        current_order_quantity = 0  # Inicializar la cantidad a ordenar en este período
        if inventoryPosition <= reorderPoint:
            noOrders += 1
            current_order_quantity = eoq  # Usar el EOQ calculado
            inTransitStock[-1] = current_order_quantity
            totalOrderingCost += orderingCost
            print(f"Ordenando {current_order_quantity:.2f} unidades (EOQ)")
        else:
            print(f"No se ordena - Inventory Position ({inventoryPosition:.2f}) > Reorder Point ({reorderPoint:.2f})")
        
        print(f"No orders: {noOrders}")
        print(f"Stock en tránsito: {inTransitStock}")
        print(f"Ordering cost: {totalOrderingCost:.2f}")

        #Calculate the holding cost and the benefit
        totalHoldingCost += holdingCost * onHandLevel
        print(f"Holding cost: {totalHoldingCost:.2f}")
        totalBenefit[t]=totalIncome - totalHoldingCost - totalStockOutCost - totalOrderingCost 
        print(f"Benefit: {totalBenefit[t]:.2f}")
    
         #Add the trajectory con la cantidad actual a ordenar
        trajectory.append({
            'state': state, 
            'action': current_order_quantity,  # Usar la cantidad actual a ordenar
            'returnToGo': 0.0})
        
        #Calculate the reward

    reward = (totalIncome - totalHoldingCost - totalStockOutCost - totalOrderingCost)/trajectoryLength 
    print(f"Reward: {reward:.2f}")
    
    #Hacerlo fuera
    #Calculate the return-to-go
    for t in range(trajectoryLength):
        if t>RETURN_TO_GO_WINDOW: # if the time step is greater than the return-to-go window
            #add the benefit from the previous period to the window
            BenefitToAdd= totalBenefit[t-RETURN_TO_GO_WINDOW]-totalBenefit[t-RETURN_TO_GO_WINDOW-1] # the total benefit is the accumulated benefit, we need to subtract the accumulated benefit from the previous period to the one we are adding
            #subtract the benefit from the current period from the window
            benefitToSubstract= totalBenefit[t]-totalBenefit[t-1] # the benefit is the accumulated benefit, we need to subtract the accumulated benefit from the previous period to the one we are substracting
            returnToGo=(reward*RETURN_TO_GO_WINDOW+BenefitToAdd-benefitToSubstract)/RETURN_TO_GO_WINDOW
            trajectory[t]['returnToGo'] = returnToGo
            print(f"Calculo del return-to-go en el periodo {t}:")
            print(f"Benefit to add: {BenefitToAdd:.2f}")
            print(f"Benefit to substract: {benefitToSubstract:.2f}")
        else:
            trajectory[t]['returnToGo']=reward

        
    return trajectory



def addTrajectoryToTrainingData(trajectory, trainingData):
    def addPaddingToTransitStock(inTransitStock):
        padded = np.zeros(MAX_LEAD_TIME - 1)
        padded[:len(inTransitStock)] = inTransitStock
        return padded
    
    # Convertir la trayectoria a TensorDict
    new_trajectory = TensorDict({
        'states': TensorDict({
            'onHandLevel': torch.stack([torch.tensor(t['state']['onHandLevel'], dtype=torch.float) for t in trajectory]),
            'inTransitStock': torch.stack([torch.tensor(addPaddingToTransitStock(t['state']['inTransitStock']), dtype=torch.float) for t in trajectory]),
            'demand': torch.stack([torch.tensor(t['state']['demand'], dtype=torch.float) for t in trajectory]),
            'forecast': torch.stack([torch.tensor(t['state']['forecast'], dtype=torch.float) for t in trajectory]),
            'leadTime': torch.stack([torch.tensor(t['state']['leadTime'], dtype=torch.float) for t in trajectory]),
            'holdingCost': torch.stack([torch.tensor(t['state']['holdingCost'], dtype=torch.float) for t in trajectory]),
            'orderingCost': torch.stack([torch.tensor(t['state']['orderingCost'], dtype=torch.float) for t in trajectory]),
            'stockOutPenalty': torch.stack([torch.tensor(t['state']['stockOutPenalty'], dtype=torch.float) for t in trajectory]),
            'unitRevenue': torch.stack([torch.tensor(t['state']['unitRevenue'], dtype=torch.float) for t in trajectory]),
            'timesStep': torch.stack([torch.tensor(t['state']['timesStep'], dtype=torch.float) for t in trajectory]),
        }),
        'actions': torch.stack([torch.tensor(t['action'], dtype=torch.float) for t in trajectory]),
        'returnsToGo': torch.stack([torch.tensor(t['returnToGo'], dtype=torch.float) for t in trajectory])
    })
    
    # Si es la primera trayectoria, inicializar trainingData
    if len(trainingData.keys()) == 0:
        return new_trajectory.unsqueeze(0)
    
    # Concatenar la nueva trayectoria con las existentes
    return torch.cat([trainingData, new_trajectory.unsqueeze(0)], dim=0)





if __name__ == "__main__":
    noTrajectories = 50 #era 100
    # Inicializar trainingData como un TensorDict vacío
    trainingData = TensorDict({})
    
    print(f"Iniciando generación de {noTrajectories} trayectorias...")
    
    for i in range(noTrajectories):
        print(f"\n{'='*50}")
        print(f"Generando trayectoria {i+1}/{noTrajectories}")
        inputData = generateInstanceData()
        trajectory = generateTrajectory(inputData)
        trainingData = addTrajectoryToTrainingData(trajectory, trainingData)

    # Save the training data to a file
    def getProjectDirectory():
        return str(Path(__file__).resolve().parent)

    torch.save(trainingData,
               getProjectDirectory() + "/data/training_data.pt")
    
    print(f"\nGeneración de trayectorias completada.")
    print(f"Tamaño de datos de entrenamiento: {len(trainingData)}")
    print("\nEstructura detallada de los datos:")
    print("Forma general de trainingData:", trainingData.shape)
    print("\nFormas de cada componente:")
    print("States:")
    for key, value in trainingData['states'].items():
        print(f"  {key}: {value.shape}")
    print(f"\nActions: {trainingData['actions'].shape}")
    print(f"Returns to go: {trainingData['returnsToGo'].shape}")
    
    print("\nPrimera trayectoria completa:")
    trayectoria_idx = 0  # Cambiar para ver otras trayectorias
    for timestep in range(TRAJECTORY_LENGTH):
        print(f"\nTimestep {timestep}:")
        for key in trainingData['states'].keys():
            print(f"{key}: {trainingData['states'][key][trayectoria_idx][timestep]}")
        print(f"Action: {trainingData['actions'][trayectoria_idx][timestep]}")
        print(f"Return to go: {trainingData['returnsToGo'][trayectoria_idx][timestep]}")

    print(trainingData)
    print(trainingData.shape)

