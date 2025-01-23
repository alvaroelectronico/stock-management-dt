import numpy as np
import torch.random
from tesordict import TensorDict



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
                 'inTransitStock': 0 } # stock en tránsito inicialmente 0 # En realidad, esto debe ser un diccionario, con las llegadas pendientes de recepción, donde las keys son los días y los value son las cantidades que llegarán.
    
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
    orderQuantity = 0
    noOrders = 0
    meanRewards = 0

    rewards = np.zeros(trajectoryLength)  # Rewards for each period # En realidad, esto tiene que guardar el benficio medio para el conjunto de la trayectoria.
    trajectory = np.zeros(trajectoryLength) # Esto hay que revisarlo, porque lo que necesitamos es almacenar para cada paso de la trayectoria: estado, acción, return-to-go

    # Por sencillo que pueda ser, comenta cada bloquecito de código (quizá no haga falta siempre cada línea, pero sí cada bloque, en lo que sigue de código hace falta comentar más
    demand_mean = np.random.randint(MIN_DEMAND_MEAN, MAX_DEMAND_MEAN)
    demand_std = np.random.randint(MIN_DEMAND_STD, MAX_DEMAND_STD)
    
    for t in range(trajectoryLength):

        currentDemand = np.random.normal(demand_mean, demand_std)
        currentForecast = np.random.normal(demand_mean, demand_std, size=FORECAST_LENGHT)

        if t >= leadTime:
            onHandLevel = onHandLevel + trajectory[t-leadTime] # Esto no lo entiendo, no sé a qué responde. En algunas de las siguientes filas me pasa algo parecido, porfa comento y reviso.

        inTransitStock = sum(trajectory[max(0, t-leadTime+1):t])
        projected_stock = onHandLevel + inTransitStock
        
        if onHandLevel < currentDemand: # if the stock is less than the demand, we stockout
            totalStockoutCost += stockoutPenalty * (currentDemand - onHandLevel)
            orderQuantity = max(0,currentDemand - projected_stock)
            trajectory[t] = orderQuantity  # save the order quantity
            noOrders += 1
            inTransitStock += orderQuantity  # save the order in transit
        else:
            if projected_stock < sum(currentForecast): # if the stock is less than the forecast, we order
                orderQuantity = sum(currentForecast) - projected_stock
                trajectory[t] = orderQuantity  # save the order quantity
                noOrders += 1
                inTransitStock += orderQuantity  # save the order in transit
            else:
                trajectory[t] = 0  # no order
        
        onHandLevel = max(0, onHandLevel - currentDemand)

        totalOrderingCost += orderingCost * orderQuantity
        totalHoldingCost += holdingCost * onHandLevel
        totalIncome += unitRevenue * min(currentDemand, onHandLevel)
        rewards[t] = totalIncome - totalHoldingCost - totalStockoutCost - totalOrderingCost

    meanRewards = sum(rewards)/noOrders

    return trajectory



def addTrajectoryToTrainingData(trajectory, trainingData):
    return trainingData.append(trajectory)




if __name__ == "__main__":
    noTrajectories = 100
    trainingData = TensorDict()

    for i in range(noTrajectories):
        inputData = generateInstanceData()
        trajectory = generateTrajectory(inputData)
        addTrajectoryToTrainingData(trajectory, trainingData)


