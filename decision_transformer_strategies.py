import torch
import math
from generate_trajectories import generateInstanceData, generateTrajectory, addTrajectoryToTrainingData
from abc import abstractmethod
from tensordict import TensorDict

class TrainingStrategy:
    def __init__(self):
        pass

    @abstractmethod
    def getTrainingData(self, batchSize):
        return None

class DTTrainingStrategy(TrainingStrategy):

    def __init__(self, dataPath : list, shuffle=True):
        """
        Inicializa la estrategia de entrenamiento.
        
        Args:
            dataPath: Lista de rutas a los archivos de datos de entrenamiento
            shuffle: Si True, mezcla los datos
        """
        super().__init__()
        self.shuffle = shuffle
        self.dataPath = dataPath

        allProblemData = []
        allOrderQuantityData = []
        allReturnsToGoData = []

        for i, path in enumerate(self.dataPath):
            element = torch.load(path, weights_only=False)
            
            allProblemData.append(element["states"])
            allOrderQuantityData.append(element["actions"])
            allReturnsToGoData.append(element["returnsToGo"])

        self.problemData = torch.cat(allProblemData, dim=0)
        self.orderQuantityData = torch.cat(allOrderQuantityData, dim=0)
        self.returnsToGoData = torch.cat(allReturnsToGoData, dim=0)

        self.lengthData = self.problemData.batch_size[0]
        self.dataIndices = torch.arange(self.lengthData)
        self.currentIndex = 0

        self.resetData()
    
    def resetData(self):
        if self.shuffle:
            self.dataIndices = self.dataIndices[torch.randperm(self.lengthData)]
        self.currentIndex = 0

    def getTrainingData(self, batchSize):
 
        startIndex = self.currentIndex
        endIndex = startIndex + batchSize
        if endIndex > self.lengthData:
            endIndex = self.lengthData
        batchIndices = self.dataIndices[startIndex:endIndex]

        batch = self.problemData[batchIndices]
        orderQuantity = self.orderQuantityData[batchIndices]
        returnsToGo = self.returnsToGoData[batchIndices]

        self.currentIndex += batchSize
        if self.currentIndex >= self.lengthData:
            self.resetData()


        return (batch.clone(), orderQuantity.clone(), returnsToGo.clone())
    
    def getValidationData(self, batchSize):
        """
        Genera datos de validación.
        """
        validationData = TensorDict({}, batch_size=[1])
        
        inputData = generateInstanceData()
        trajectory = generateTrajectory(inputData)
        validationData = addTrajectoryToTrainingData(trajectory, validationData)
        
        problemData = validationData['states']
        orderQuantityData = validationData['actions']
        returnsToGoData = validationData['returnsToGo']
        
        return (problemData.clone(), orderQuantityData.clone(), returnsToGoData.clone())

    def to_dict(self):
        """
        Convierte la estrategia en un diccionario serializable.
        
        Returns:
            Diccionario con la configuración de la estrategia
        """
        return {
            'dataPath': self.dataPath,
            'shuffle': self.shuffle,
        }
    



       
