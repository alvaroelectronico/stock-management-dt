import torch
import math
from generate_tajectories import generateInstanceData, generateTrajectory, addTrajectoryToTrainingData
from abc import abstractmethod
from tensordict import TensorDict

#Clase base para estrategias de entrenamiento
class TrainingStrategy:
    def __init__(self):
        pass

    @abstractmethod
    def getTrainingData(self, batchSize):
        return None

#Estrategia de entrenamiento para Decision Transformer
class DTTrainingStrategy(TrainingStrategy):

    def __init__(self, dataPath : list, trainPercentage : list = None, shuffle=True, augment=False):
        super().__init__()
        self.shuffle = shuffle
        self.dataPath = dataPath
        self.trainPercentage = trainPercentage

        self.problemData = None
        self.orderQuantityData = None
        self.returnsToGoData = None
        print(f"dataPath: {self.dataPath}")

        #Cargar los datos de los archivos
        for i in range(len(self.dataPath)): 
            element = torch.load(self.dataPath[i])
            
            #print(element["states"])
            #batchSize = element["states"].batch_size[0]
            #if hasattr(element["states"], "batch_dims"):
            #    print("Dimensiones de batch:", element["states"].batch_dims)
            #    print("Tamaño de batch:", element["states"].batch_size)
    
            print("Estructura de datos cargada:", element.keys())
        
            # Acceder correctamente a los estados
            states_dict = element["states"]
            print("Estados disponibles:", states_dict.keys())
        
            batchSize = states_dict.batch_size[0]

            #Calcular el porcentaje de datos de entrenamiento
            percentage = 1 if trainPercentage is None else trainPercentage[i]
            length = math.floor(batchSize * percentage) #element["problemData"].batch_size[0] * percentage cuando modifique el generate trajectories
            
            #Si es el primer archivo, inicializar los datos
            if i == 0:
                #esto me da error porque el generate trajectories lo tengo hecho para un solo batch y los datos no tienen la dimension de batch
                #self.problemData = element["states"][:length]
                self.problemData = states_dict
                #self.orderQuantityData = element["actions"][:length]
                #self.returnsToGoData = element["returnsToGo"][:length]
                self.orderQuantityData = element["actions"]
                self.returnsToGoData = element["returnsToGo"]
            else:
                #Concatenar los datos de los archivos
                #self.problemData = torch.cat((self.problemData, element["states"][:length]), dim=0)
                #self.orderQuantityData = torch.cat((self.orderQuantityData, element["orderQuantity"][:length]), dim=0)
                #self.returnsToGoData = torch.cat((self.returnsToGoData, element["returnsToGo"][:length]), dim=0)

                self.problemData = torch.cat((self.problemData, states_dict), dim=0)
                self.orderQuantityData = torch.cat([self.orderQuantityData, element["actions"]], dim=0)
                self.returnsToGoData = torch.cat([self.returnsToGoData, element["returnsToGo"]], dim=0)


        self.lengthData = self.problemData.batch_size[0]
        self.dataIndices = torch.arange(self.lengthData)
        self.currentIndex = 0

        self.resetData()  

    def resetData(self):
        if self.shuffle:
            self.dataIndices = self.dataIndices[torch.randperm(self.lengthData)]  # Barajar índices
        self.currentIndex = 0  # Reiniciar posición actual


    #Obtener los datos de entrenamiento
    def getTrainingData(self, batchSize):

        #Augmentar los datos si es necesario
        #if self.augment:
        #    batchSize = batchSize // 8
        
        #Calcular los índices de inicio y fin del batch
        startIndex = self.currentIndex
        endIndex = startIndex + batchSize
        if endIndex > self.lengthData:
            endIndex = self.lengthData
        batchIndices = self.dataIndices[startIndex:endIndex]

        #Seleccionar los datos del batch
        batch = self.problemData[batchIndices]
        orderQuantity = self.orderQuantityData[batchIndices]
        returnsToGo = self.returnsToGoData[batchIndices]

        #Actualizar el índice actual 
        self.currentIndex += batchSize
        if self.currentIndex >= self.lengthData:
            self.resetData()

       
        #Entrenar DT
        #if self.augment:
        #    instance = Instance(data=batch)
        #    instance.augment()

        #    batch = instance.currentData.clone()
        #    orderQuantity = repeatBatchTensordict(TensorDict(
        #        {
        #            "orderQuantity": orderQuantity
        #        }, batch_size=endIndex - startIndex), 8)
        #    orderQuantity = orderQuantity["orderQuantity"]

        #    returnsToGo = repeatBatchTensordict(TensorDict(
        #        {
        #            "returnsToGo": returnsToGo
        #        }, batch_size=endIndex - startIndex), 8)
        #    returnsToGo = returnsToGo["returnsToGo"]

        #
        #    instance = Instance(data=batch2)
        #    instance.augment()

        return (batch.clone(), orderQuantity.clone(), returnsToGo.clone())


    def getValidationData(self, batchSize):
        # Generar datos de validación usando las funciones existentes
        validationData = TensorDict({}, batch_size=[1])
        
        # Generar una trayectoria de validación
        inputData = generateInstanceData()
        trajectory = generateTrajectory(inputData)
        validationData = addTrajectoryToTrainingData(trajectory, validationData)
        
        # Extraer los datos necesarios
        problemData = validationData['states']
        orderQuantityData = validationData['actions']
        returnsToGoData = validationData['returnsToGo']
        
        return (problemData.clone(), orderQuantityData.clone(), returnsToGoData.clone())

    def to_dict(self):
        """Convierte la estrategia en un diccionario serializable"""
        return {
            'dataPath': self.dataPath,
            'trainPercentage': self.trainPercentage,
            'shuffle': self.shuffle,
            'augment': False  # No serializamos los datos cargados
        }



       
