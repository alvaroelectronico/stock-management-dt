import torch
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

    def __init__(self, dataPath: list, shuffle=True):
        """
        Inicializa la estrategia de entrenamiento.
        
        Args:
            dataPath: Lista de rutas a los archivos de datos de entrenamiento
            shuffle: Si True, mezcla los datos
            num_workers: Número de workers para carga paralela (0 = sin workers)
            pin_memory: Si True, usa memoria pinned para transferencias GPU más rápidas
        """
        super().__init__()
        self.shuffle = shuffle
        self.dataPath = dataPath
        
        allProblemData = []
        allOrderQuantityData = []
        allReturnsToGoData = []

        for path in dataPath:
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
        """
        Reinicia los índices y mezcla los datos si es necesario.
        """
        if self.shuffle:
            self.dataIndices = self.dataIndices[torch.randperm(self.lengthData)]
        self.currentIndex = 0

    def getTrainingData(self, batchSize):
        """
        Obtiene el siguiente batch de datos de entrenamiento.
        
        Args:
            batchSize: Tamaño del batch
            
        Returns:
            Tupla con (estados, acciones, returns-to-go)
        """
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

        return (batch, orderQuantity, returnsToGo)
    
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
    



       
