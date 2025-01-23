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
    orderQuantity = 0
    noOrders = 0

    reward = 0   # En realidad, esto tiene que guardar el benficio medio para el conjunto de la trayectoria.
    trajectory = [] # Esto hay que revisarlo, porque lo que necesitamos es almacenar para cada paso de la trayectoria: estado, acción, return-to-go

    # Por sencillo que pueda ser, comenta cada bloquecito de código (quizá no haga falta siempre cada línea, pero sí cada bloque, en lo que sigue de código hace falta comentar más
    demand_mean = np.random.randint(MIN_DEMAND_MEAN, MAX_DEMAND_MEAN)
    demand_std = np.random.randint(MIN_DEMAND_STD, MAX_DEMAND_STD)
    
    for t in range(trajectoryLength):
        #Generamos la demanda actual y la demanda prevista
        currentDemand = np.random.normal(demand_mean, demand_std)
        currentForecast = np.random.normal(demand_mean, demand_std, size=FORECAST_LENGHT)

        if t in inTransitStock:
            #Si t es un día de llegada de un pedido en tránsito, entonces se actualiza el nivel de stock en mano con la cantidad que se ha recibido en el período t.
            onHandLevel = onHandLevel + inTransitStock[t] # Esto no lo entiendo, no sé a qué responde. En algunas de las siguientes filas me pasa algo parecido, porfa comento y reviso.
            del inTransitStock[t] #eliminamos la cantidad que se ha recibido del stock en tránsito
        
        #Definimos el estado del sistema actual
        state = {
            'onHandLevel': onHandLevel,
            'inTransitStock': dict(inTransitStock),
            'forecast': currentForecast.tolist(),
            'demand': currentDemand,
            'orderingCost': orderingCost,
            'holdingCost': holdingCost,
            'stockoutPenalty': stockoutPenalty,
            'unitRevenue': unitRevenue,
            'leadTime': leadTime,
            'time_step': t  # Añadimos el paso de tiempo actual
        }

        #Calculamos el stock total, que es el stock físico más el stock en tránsito.
        projectedStock = onHandLevel + sum(inTransitStock.values())
        
        #decidimos la cantidad a pedir y calculamos los costes si hay stockout
        if onHandLevel < currentDemand: # if the stock is less than the demand, we stockout
            totalStockoutCost += stockoutPenalty * (currentDemand - onHandLevel) #coste de stockout * la cantidad que falta para cubrir la demanda
            orderQuantity = max(0,currentDemand - projectedStock) #la cantidad a pedir será el máximo entre 0 y la demanda menos el stock total proyectado (aunque no tengamos stock físico si con el stock en tránsito podemos cubrir la demanda no sería necesario volver a pedir)
            noOrders += 1
            inTransitStock[t+leadTime]= orderQuantity  # save the order in transit que llegará en el período t+leadTime que es lo que tarda en llegar
        else:
            if projectedStock < sum(currentForecast): # if the stock is less than the forecast, we order
                orderQuantity = sum(currentForecast) - projectedStock # la cantidad pedida sería la demanda prevista menos el stock que tenemos mas el que está por llegar
                noOrders += 1
                inTransitStock[t+leadTime]= orderQuantity  # save the order in transit que llegará en el período t+leadTime que es lo que tarda en llegar
            else:
                orderQuantity = 0  # no order
        
        #Actualizamos el stock físico
        onHandLevel = onHandLevel - currentDemand 

        #Calculamos los costes totales
        totalOrderingCost += orderingCost * orderQuantity
        totalHoldingCost += holdingCost * onHandLevel
        totalIncome += unitRevenue * min(currentDemand, onHandLevel)
        reward = (totalIncome - totalHoldingCost - totalStockoutCost - totalOrderingCost)/noOrders

        #Añadimos la trayectoria
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

    for i in range(noTrajectories):
        inputData = generateInstanceData()
        trajectory = generateTrajectory(inputData)
        addTrajectoryToTrainingData(trajectory, trainingData)


