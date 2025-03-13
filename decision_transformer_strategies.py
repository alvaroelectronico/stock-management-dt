import torch
import math
from generate_tajectories import generateInstanceData
from abc import abstractmethod

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

        self.problemData = None
        self.orderQuantityData = None
        self.returnsToGoData = None

        #Cargar los datos de los archivos
        for i in range(len(self.dataPath)): 
            element = torch.load(self.dataPath[i])
            if hasattr(element["states"], "batch_dims"):
                print("Dimensiones de batch:", element["states"].batch_dims)
                print("Tamaño de batch:", element["states"].batch_size)
    

            #Calcular el porcentaje de datos de entrenamiento
            percentage = 1 if trainPercentage is None else trainPercentage[i]
            length = math.floor(element["states"]['batchSize'][0] * percentage) #element["problemData"].batch_size[0] * percentage cuando modifique el generate trajectories
            
            #Si es el primer archivo, inicializar los datos
            if i == 0:
                print(element["states"]['batchSize'][0])
                #esto me da error porque el generate trajectories lo tengo hecho para un solo batch y los datos no tienen la dimension de batch
                self.problemData = element["states"][:length]
                self.orderQuantityData = element["orderQuantity"][:length]
                self.returnsToGoData = element["returnsToGo"][:length]
            else:
                #Concatenar los datos de los archivos
                self.problemData = torch.cat((self.problemData, element["states"][:length]), dim=0)
                self.orderQuantityData = torch.cat((self.orderQuantityData, element["orderQuantity"][:length]), dim=0)
                self.returnsToGoData = torch.cat((self.returnsToGoData, element["returnsToGo"][:length]), dim=0)


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

if __name__ == "__main__":
    # Ruta de ejemplo a tus archivos de datos
    data_paths = ["./data/training_data.pt"]  # Reemplaza con tu ruta real
    train_percentages = [0.8]  # 80% de los datos para entrenamiento
    
    # Crear una instancia de la estrategia
    training_strategy = DTTrainingStrategy(
        dataPath=data_paths,
        trainPercentage=train_percentages,
        shuffle=True,
        augment=False
    )
    
    print("\n=== Información de la Estrategia de Entrenamiento ===")
    print(f"Tamaño total de datos: {training_strategy.lengthData}")
    print(f"Shuffle activado: {training_strategy.shuffle}")
    print(f"Augment activado: {training_strategy.augment}")
    
    # Probar getTrainingData
    batch_size = 2
    print(f"\n=== Probando getTrainingData con batch_size={batch_size} ===")
    
    # Obtener un batch de datos
    batch, order_quantity, returns_to_go = training_strategy.getTrainingData(batch_size)
    
    print(f"Forma del batch: {batch.shape}")
    print(f"Forma de order_quantity: {order_quantity.shape}")
    print(f"Forma de returns_to_go: {returns_to_go.shape}")
     # Probar el reset de datos
    print("\n=== Probando reset de datos ===")
    training_strategy.resetData()
    print(f"Índice actual después de reset: {training_strategy.currentIndex}")
    
    # Probar múltiples batches
    print("\n=== Probando múltiples batches ===")
    for i in range(3):
        batch, _, _ = training_strategy.getTrainingData(batch_size)
        print(f"Batch {i+1} - Índice actual: {training_strategy.currentIndex}")