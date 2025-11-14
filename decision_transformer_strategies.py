import torch
from torch.utils.data import Dataset, DataLoader
from generate_trajectories import generateInstanceData, generateTrajectory, addTrajectoryToTrainingData
from abc import abstractmethod
from tensordict import TensorDict

class TrainingStrategy:
    def __init__(self):
        pass

    @abstractmethod
    def getTrainingData(self, batchSize):
        return None


def dt_collate_fn(batch):
    """
    Función de collate personalizada para combinar muestras en un batch.
    
    Args:
        batch: Lista de tuplas (problemData, orderQuantityData, returnsToGoData)
        
    Returns:
        Tupla con (problemData_batch, orderQuantityData_batch, returnsToGoData_batch)
    """
    problemData_list, orderQuantityData_list, returnsToGoData_list = zip(*batch)

    problemData_batch = torch.stack(problemData_list, dim=0) 
    orderQuantityData_batch = torch.stack(orderQuantityData_list, dim=0)
    returnsToGoData_batch = torch.stack(returnsToGoData_list, dim=0)
    
    return (problemData_batch, orderQuantityData_batch, returnsToGoData_batch)


class DTDataset(Dataset):
    """
    Dataset personalizado para los datos de entrenamiento del Decision Transformer.
    """
    
    def __init__(self, problemData, orderQuantityData, returnsToGoData):
        """
        Inicializa el dataset.
        
        Args:
            problemData: Tensor con los estados del problema
            orderQuantityData: Tensor con las acciones (cantidades de pedido)
            returnsToGoData: Tensor con los retornos a futuro
        """
        self.problemData = problemData
        self.orderQuantityData = orderQuantityData
        self.returnsToGoData = returnsToGoData
        self.lengthData = self.problemData.batch_size[0]
    
    def __len__(self):
        """
        Retorna el tamaño del dataset.
        """
        return self.lengthData
    
    def __getitem__(self, idx):
        """
        Retorna un elemento del dataset.
        
        Args:
            idx: Índice del elemento
            
        Returns:
            Tupla con (estados, acciones, retornos_a_futuro)
        """
        return (
            self.problemData[idx],
            self.orderQuantityData[idx],
            self.returnsToGoData[idx]
        )


class DTTrainingStrategy(TrainingStrategy):

    def __init__(self, dataPath : list, shuffle=True, num_workers=0):
        """
        Inicializa la estrategia de entrenamiento usando DataLoader.
        
        Args:
            dataPath: Lista de rutas a los archivos de datos de entrenamiento
            shuffle: Si True, mezcla los datos
            num_workers: Número de workers para el DataLoader (0 = main process)
        """
        super().__init__()
        self.shuffle = shuffle
        self.dataPath = dataPath
        self.num_workers = num_workers

        allProblemData = []
        allOrderQuantityData = []
        allReturnsToGoData = []

        for path in self.dataPath:
            element = torch.load(path, weights_only=False)
            
            allProblemData.append(element["states"])
            allOrderQuantityData.append(element["actions"])
            allReturnsToGoData.append(element["returnsToGo"])
            
        problemData = torch.cat(allProblemData, dim=0)
        orderQuantityData = torch.cat(allOrderQuantityData, dim=0)
        returnsToGoData = torch.cat(allReturnsToGoData, dim=0)


        self.dataset = DTDataset(problemData, orderQuantityData, returnsToGoData)
        
        self.problemData = problemData
        self.orderQuantityData = orderQuantityData
        self.returnsToGoData = returnsToGoData
        self.lengthData = self.dataset.lengthData
        
        self.dataloader = None
        self.dataloader_iter = None
        self.batch_size = None
    
    def _get_dataloader(self, batch_size):
        """
        Crea o retorna el DataLoader con el tamaño de batch especificado.
        
        Args:
            batch_size: Tamaño del batch
            
        Returns:
            DataLoader configurado
        """
        # Solo recrear el DataLoader si el batch_size cambió
        if self.dataloader is None or self.batch_size != batch_size:
            self.batch_size = batch_size
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                pin_memory=False,
                collate_fn=dt_collate_fn
            )
            self.dataloader_iter = iter(self.dataloader)
        
        return self.dataloader, self.dataloader_iter

    def getTrainingData(self, batchSize):
        """
        Obtiene un batch de datos de entrenamiento usando DataLoader.
        
        Args:
            batchSize: Tamaño del batch
            
        Returns:
            Tupla con (estados, acciones, retornos_a_futuro)
        """
        self._get_dataloader(batchSize)
        
        try:
            # Intentar obtener el siguiente batch
            batch = next(self.dataloader_iter)
            problemData, orderQuantityData, returnsToGoData = batch
            return (problemData, orderQuantityData, returnsToGoData)
        except StopIteration:
            # Si se acabaron los datos, reiniciar el iterador
            self.dataloader_iter = iter(self.dataloader)
            batch = next(self.dataloader_iter)
            problemData, orderQuantityData, returnsToGoData = batch
            return (problemData, orderQuantityData, returnsToGoData)

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
