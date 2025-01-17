import numpy as np
import torch.random
from tesordict import TensorDict



MIN_LEAD_TIME = 10
MAX_LEAD_TIME = 20

MIN_HOLDING_COST = 1
MAX_HOLDING_COST = 10

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

    inputData = {'leadtime': leadTime, 
                 'holdingCost': holdingCost, 
                 'onHandLevel': onHandLevel}
    
    return inputData

def generateTrajectory(inputData, trajectoryLength=TRAYECTORY_LENGHT):
    # Extracting input data
    leadTime = inputData['leadtime']    
    holdingCost = inputData['holdingCost']
    onHandLevel = inputData['onHandLevel']
    

    # Defining system variables
    totalHoldingCost = 0
    totalOrderingCost = 0
    totalStockoutCost = 0
    totalIncome = 0
    
    trajectory = np.zeros(trajectoryLength)
    
    for i in range(trajectoryLength):
        currentDemand = np.random.normal(MIN_DEMAND_MEAN, MAX_DEMAND_MEAN)

        currentForecast = 0
    

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


