import numpy as np
import torch.random
from tensordict import TensorDict



MIN_LEAD_TIME = 10
MAX_LEAD_TIME = 20

MIN_HOLDING_COST = 1
MAX_HOLDING_COST = 10

MIN_ORDERING_COST = 1 # TODO: Change this to a consisten value
MAX_ORDERING_COST = 10 # TODO: Change this to a consisten value

MIN_STOCKOUT_PENALTY = 1 # TODO: Change this to a consisten value
MAX_STOCKOUT_PENALTY = 10 # TODO: Change this to a consisten value

MIN_UNIT_REVENUE = 1 # TODO: Change this to a consisten value
MAX_UNIT_REVENUE = 10 # TODO: Change this to a consisten value

MIN_DEMAND_MEAN = 10
MAX_DEMAND_MEAN = 20

MIN_DEMAND_STD = 1
MAX_DEMAND_STD = 2

TRAYECTORY_LENGHT = 50 
FORECAST_LENGHT = 10

RETURN_TO_GO_WINDOW = 10
DEMAND_FORECAST_WINDOW = 5




def generateInstanceData():
    leadTime = np.random.randint(MIN_LEAD_TIME, MAX_LEAD_TIME)
    holdingCost = np.random.randint(MIN_HOLDING_COST, MAX_HOLDING_COST)
    onHandLevel = 0 # TODO: Change this to a consisten value
    orderingCost = np.random.randint(MIN_ORDERING_COST, MAX_ORDERING_COST)  
    stockoutPenalty = np.random.randint(MIN_STOCKOUT_PENALTY, MAX_STOCKOUT_PENALTY)  
    unitRevenue = np.random.randint(MIN_UNIT_REVENUE, MAX_UNIT_REVENUE)  
    
    inputData = {'leadtime': leadTime, 
                 'holdingCost': holdingCost, 
                 'onHandLevel': onHandLevel,
                 'orderingCost': orderingCost, 
                 'stockoutPenalty': stockoutPenalty, 
                 'unitRevenue': unitRevenue, 
                 'inTransitStock': {} } # stock en tránsito inicialmente 0 # En realidad, esto debe ser un diccionario, con las llegadas pendientes de recepción, donde las keys son los días y los value son las cantidades que llegarán.
    
    return inputData

def generateTrajectory(inputData, trajectoryLength=TRAYECTORY_LENGHT):
    # Extracting input data
    leadTime = inputData['leadtime'] 
    holdingCost = inputData['holdingCost']
    onHandLevel = inputData['onHandLevel']
    inTransitStock = inputData['inTransitStock'] 
    orderingCost = inputData['orderingCost'] 
    stockoutPenalty = inputData['stockoutPenalty'] 
    unitRevenue = inputData['unitRevenue'] 

    # Defining system variables
    totalHoldingCost = 0
    totalOrderingCost = 0
    totalStockoutCost = 0
    totalIncome = 0
    totalBenefit = np.zeros(trajectoryLength)
    orderQuantity = 0
    noOrders = 0
    

    demand_mean = np.random.randint(MIN_DEMAND_MEAN, MAX_DEMAND_MEAN)
    demand_std = np.random.randint(MIN_DEMAND_STD, MAX_DEMAND_STD)

    currentDemandInitial = np.random.normal(demand_mean, demand_std)
    currentForecastInitial = np.random.normal(demand_mean, demand_std, size=FORECAST_LENGHT)
    
    # Calulating EOQ
    
    meanDemandLT = (currentDemandInitial + sum(currentForecastInitial[:leadTime]))/(leadTime+1) #calculate the mean demand during the lead time
    orderQuantity = np.sqrt(2*orderingCost*meanDemandLT/holdingCost)

    # Calculating ROP for a given CSL
    
    safetyStock = 3 * demand_std * np.sqrt(leadTime) 
    reorderPoint = meanDemandLT + safetyStock
    onHandLevel = orderQuantity/2 + safetyStock

    reward = 0   # En realidad, esto tiene que guardar el benficio medio para el conjunto de la trayectoria.
    trajectory = [] # Esto hay que revisarlo, porque lo que necesitamos es almacenar para cada paso de la trayectoria: estado, acción, return-to-go

    
    for t in range(trajectoryLength):
        #generate demand and forecast for the current period
        currentDemand = np.random.normal(demand_mean, demand_std)
        currentForecast = np.random.normal(demand_mean, demand_std, size=FORECAST_LENGHT)
        
        if t in inTransitStock.keys():
            #If t is a day of arrival of a pending order, then the inventory level is updated with the amount received in the period t.
            onHandLevel = onHandLevel + inTransitStock[t] 
            del inTransitStock[t] #delete the amount received from the stock in transit 
        
        #Define the current state of the system
        state = {
            'onHandLevel': onHandLevel,
            'inTransitStock': dict(inTransitStock),
            'forecast': currentForecast.tolist(),
            'demand': currentDemand,
            #'orderingCost': orderingCost,  # Fixed cost when placing an order, regardless of the amount ordered
            #'holdingCost': holdingCost,
            #'stockoutPenalty': stockoutPenalty,
            #'unitRevenue': unitRevenue,
            #'leadTime': leadTime,
            'time_step': t  # Add the current time step
        }
            

        #Update the stock on hand
        totalStockoutCost += stockoutPenalty *max(0, currentDemand - onHandLevel) # hay que calcularlos antes de actualizar el stock físico
        totalIncome += unitRevenue * min(currentDemand, onHandLevel) #hay que calcularlo antes de actualizar el stock físico 
        onHandLevel = max(0, onHandLevel - currentDemand )
        
        #Calculate the total stock, which is the physical stock plus the stock in transit.
        inventoryPosition = onHandLevel + sum(inTransitStock.values())  #inventory position = on hand/inventoyr level +  on order/in transit

        #decide the amount to order and calculate the costs 
        if inventoryPosition <= reorderPoint:
            noOrders += 1
            inTransitStock[t+leadTime]= orderQuantity  # save the order in transit that will arrive in the period t+leadTime
            totalOrderingCost += orderingCost
        

        #Calculamos los costes totales
        #totalStockoutCost += stockoutPenalty *max(0, currentDemand - onHandLevel)        
        totalHoldingCost += holdingCost * onHandLevel
        #totalIncome += unitRevenue * min(currentDemand, onHandLevel)# el ingreso es el beneficio por la cantidad vendida, que es la demanda actual o el stock físico, lo que sea menor
        totalBenefit[t]=totalIncome - totalHoldingCost - totalStockoutCost - totalOrderingCost 
    
    #No se si calcular el beneficio medio por pedido o el beneficio medio de la trayectoria incluyendo los periodos que no hay pedido

        reward = (totalIncome - totalHoldingCost - totalStockoutCost - totalOrderingCost)/noOrders

        #if t==0:
        #    reward = (totalIncome - totalHoldingCost - totalStockoutCost - totalOrderingCost)
        #else:
        #   reward = (totalIncome - totalHoldingCost - totalStockoutCost - totalOrderingCost)/t

        
        #Calculate the return-to-go
        if t>=RETURN_TO_GO_WINDOW: # if the time step is greater than the return-to-go window
            #add the benefit from the previous period to the window
            BenefitToAdd= totalBenefit[t-RETURN_TO_GO_WINDOW]-totalBenefit[t-RETURN_TO_GO_WINDOW-1] # the total benefit is the accumulated benefit, we need to subtract the acumulated benefit from the previos period to the one we are adding
            #subtract the benefit from the current period from the window
            benefitToSubstract= totalBenefit[t]-totalBenefit[t-1] # the benefit is the accumulated benefit, we need to subtract the acumulated benefit from the previos period to the one we are substracting
            reward=(reward*RETURN_TO_GO_WINDOW+BenefitToAdd-benefitToSubstract)/RETURN_TO_GO_WINDOW
            
            
        #Add the trajectory
        trajectory.append({
            'state':state, 
            'action':orderQuantity, 
            'returnToGo':reward})
        
    return trajectory



def addTrajectoryToTrainingData(trajectory, trainingData):
    return trainingData.append(trajectory)




if __name__ == "__main__":
    noTrajectories = 100
    trainingData = TensorDict()
    
    print(f"Iniciando generación de {noTrajectories} trayectorias...")
    
    for i in range(noTrajectories):
        print(f"\n{'='*50}")
        print(f"Generando trayectoria {i+1}/{noTrajectories}")
        inputData = generateInstanceData()
        trajectory = generateTrajectory(inputData)
        addTrajectoryToTrainingData(trajectory, trainingData)
    
    print(f"\nGeneración de trayectorias completada.")
    print(f"Tamaño de datos de entrenamiento: {len(trainingData)}")


