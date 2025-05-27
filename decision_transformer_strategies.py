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

    def __init__(self, dataPath : list, shuffle=True):
        super().__init__()
        self.shuffle = shuffle
        self.dataPath = dataPath
        
        print(f"\nCargando datos de entrenamiento...")
        print(f"Archivos a cargar: {self.dataPath}")

        # Inicializar listas para almacenar todos los datos
        allProblemData = []
        allOrderQuantityData = []
        allReturnsToGoData = []

        #LO QUE TENIA ANTES
        #self.problemData = None
        #self.orderQuantityData = None
        #self.returnsToGoData = None

        #Cargar los datos de los archivos
        #for i in range(len(self.dataPath)): 
        #    element = torch.load(self.dataPath[i])
            
            #print("Estructura de datos cargada:", element.keys())
        
            # Acceder correctamente a los estados
            #states_dict = element["states"]
            #print("Estados disponibles:", states_dict.keys())
            # Cargar y concatenar todos los datos
        for i, path in enumerate(self.dataPath):
            print(f"\nCargando archivo {i+1}/{len(self.dataPath)}: {path}")
            element = torch.load(path, weights_only=False)
            
            print(f"\nDEBUG - Dimensiones del archivo {i+1}:")
            print(f"States batch_size: {element['states'].batch_size}")
            print(f"Actions shape: {element['actions'].shape}")
            print(f"Returns shape: {element['returnsToGo'].shape}")
            
            # Añadir los datos a las listas
            allProblemData.append(element["states"])
            allOrderQuantityData.append(element["actions"])
            allReturnsToGoData.append(element["returnsToGo"])

        # Concatenar todos los datos
        print("\nConcatenando todos los datos...")
        self.problemData = torch.cat(allProblemData, dim=0)
        self.orderQuantityData = torch.cat(allOrderQuantityData, dim=0)
        self.returnsToGoData = torch.cat(allReturnsToGoData, dim=0)

        print("\nDEBUG - Dimensiones finales después de concatenar:")
        print(f"problemData batch_size: {self.problemData.batch_size}")
        print(f"orderQuantityData shape: {self.orderQuantityData.shape}")
        print(f"returnsToGoData shape: {self.returnsToGoData.shape}")
        print(f"Total de trayectorias: {self.problemData.batch_size[0]}")

        # Inicializar índices y contador
        self.lengthData = self.problemData.batch_size[0]
        self.dataIndices = torch.arange(self.lengthData)
        self.currentIndex = 0

        # Mezclar los datos si es necesario
        self.resetData()

    def resetData(self):
        if self.shuffle:
            self.dataIndices = self.dataIndices[torch.randperm(self.lengthData)]  # Barajar índices
        self.currentIndex = 0  # Reiniciar posición actual


    #Obtener los datos de entrenamiento
    def getTrainingData(self, batchSize):
 
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
            'shuffle': self.shuffle,
        }

if __name__ == "__main__":
    print("\n=== Probando carga de datos de entrenamiento ===")
    
    # Configurar la ruta a los datos
    data_path = ["data/training_data.pt"]
    
    print("\nInicializando DTTrainingStrategy...")
    strategy = DTTrainingStrategy(dataPath=data_path)
    
    print("\nProbando getTrainingData con batch_size=4...")
    batch, order_quantity, returns = strategy.getTrainingData(batchSize=4)
    
    print("\nDimensiones del batch obtenido:")
    print(f"Batch shape: {batch.batch_size}")
    print(f"Order quantity shape: {order_quantity.shape}")
    print(f"Returns shape: {returns.shape}")
    
    print("\nProbando getValidationData...")
    val_batch, val_order_quantity, val_returns = strategy.getValidationData(batchSize=1)
    
    print("\nDimensiones de los datos de validación:")
    print(f"Validation batch shape: {val_batch.batch_size}")
    print(f"Validation order quantity shape: {val_order_quantity.shape}")
    print(f"Validation returns shape: {val_returns.shape}")



       
