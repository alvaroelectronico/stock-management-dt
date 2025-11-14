import torch
from torch.utils.data import Dataset, DataLoader
from generate_trajectories import generateInstanceData, generateTrajectory, addTrajectoryToTrainingData
from abc import abstractmethod
from tensordict import TensorDict

def tensordict_collate(batch):
    """
    Función de collate personalizada para TensorDict.
    
    Args:
        batch: Lista de tuplas (problemData, orderQuantityData, returnsToGoData)
        
    Returns:
        Tupla con los tensores concatenados
    """
    problemData, orderQuantityData, returnsToGoData = zip(*batch)
    problemData_batch = torch.stack(problemData, dim=0)
    orderQuantityData_batch = torch.stack(orderQuantityData, dim=0)
    returnsToGoData_batch = torch.stack(returnsToGoData, dim=0)
    
    return (problemData_batch, orderQuantityData_batch, returnsToGoData_batch)


class TrainingStrategy:
    def __init__(self):
        pass

    @abstractmethod
    def getTrainingData(self, batchSize):
        return None


class DTDataset(Dataset):
    """
    Dataset personalizado para Decision Transformer.
    """
    
    def __init__(self, dataPath: list):
        """
        Inicializa el dataset cargando datos de múltiples archivos.
        
        Args:
            dataPath: Lista de rutas a los archivos de datos de entrenamiento
        """
        super().__init__()
        
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

    def __len__(self):
        """
        Retorna el número de muestras en el dataset.
        """
        return self.problemData.batch_size[0]

    def __getitem__(self, idx):
        """
        Retorna una muestra del dataset en el índice especificado.
        
        Args:
            idx: Índice de la muestra
            
        Returns:
            Tupla con (estados, acciones, returns-to-go)
        """
        return (
            self.problemData[idx],
            self.orderQuantityData[idx],
            self.returnsToGoData[idx]
        )


class DTTrainingStrategy(TrainingStrategy):

    def __init__(self, dataPath: list, shuffle=True, num_workers=0, pin_memory=True):
        """
        Inicializa la estrategia de entrenamiento con DataLoader optimizado.
        
        Args:
            dataPath: Lista de rutas a los archivos de datos de entrenamiento
            shuffle: Si True, mezcla los datos en cada época
            num_workers: Número de workers para carga paralela de datos
            pin_memory: Si True, usa memoria pinned para transferencias GPU más rápidas
        """
        super().__init__()
        self.shuffle = shuffle
        self.dataPath = dataPath
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self.dataset = DTDataset(dataPath)
        self.dataloader = None
        self.dataloader_iter = None
        
        self.problemData = self.dataset.problemData
        self.orderQuantityData = self.dataset.orderQuantityData
        self.returnsToGoData = self.dataset.returnsToGoData
        self.lengthData = len(self.dataset)

    def _create_dataloader(self, batchSize):
        """
        Crea un DataLoader con el tamaño de batch especificado.
        
        Args:
            batchSize: Tamaño del batch
            
        Returns:
            DataLoader configurado
        """
        return DataLoader(
            self.dataset,
            batch_size=batchSize,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=tensordict_collate
        )

    def getTrainingData(self, batchSize):
        """
        Obtiene el siguiente batch de datos de entrenamiento.
        
        Args:
            batchSize: Tamaño del batch
            
        Returns:
            Tupla con (estados, acciones, returns-to-go)
        """
        if self.dataloader is None or self.dataloader.batch_size != batchSize:
            self.dataloader = self._create_dataloader(batchSize)
            self.dataloader_iter = iter(self.dataloader)
        
        try:
            batch = next(self.dataloader_iter)
        except StopIteration:
            self.dataloader_iter = iter(self.dataloader)
            batch = next(self.dataloader_iter)
        
        return batch
    
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
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
        }
    



       
