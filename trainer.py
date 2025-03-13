import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ConstantLR
import platform
from decision_transformer_strategies import TrainingStrategy

def checkCompileSupport():
    # Verificar si el sistema operativo es Linux
    if platform.system() != "Linux":
        return False

    # Verificar si Triton está disponible
    try:
        import triton  # Intentar importar Triton
    except ImportError:
        return False

    # Verificar si hay una GPU disponible
    if not torch.cuda.is_available():
        return False

    return True


#Configuracion del trainer
class TrainerConfig:

    def __init__(self, nBatch, nVal, stepsPerEpoch, trainStrategy=None, optimizer=None, lr_scheduler=None):
        
        self.nBatch = nBatch
        self.nVal = nVal
        self.stepsPerEpoch = stepsPerEpoch 
        self.trainStrategy = trainStrategy
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
    

class Trainer:

    def __init__(self, savePath, name, model, trainerConfig):
        self.directoryModels = savePath + name + "/"
        self.directoryProgress = savePath + name + "/"

        #self.trainingSavePath = self.directoryModels + "/training.pt" guarda el estado actual del modelo durante el entrenamiento (checkpoints), lo dejo como comentario porque no he definido los checkpoints
        #self.baselineSavePath = self.directoryModels + "/best.pt" guarda el mejor modelo, lo dejo como comentario porque no he definido el mejor modelo
        self.trackPath = self.directoryProgress + "/track.json" #guarda el historial de entrenamiento
        #self.jsonPath = self.directoryProgress + "/validation.json" guarda los resultados de la validacion, lo dejo como comentario porque no he definido la validacion

        #crea los directorios si no existen
        if not os.path.exists(self.directoryModels):
            os.makedirs(self.directoryModels)
        if not os.path.exists(self.directoryProgress):
            os.makedirs(self.directoryProgress)

        #configuracion del trainer y de la estrategia de entrenamiento
        self.trainerConfig = trainerConfig
        self.trainStrategy = trainerConfig.trainStrategy
        self.optimizer = trainerConfig.optimizer
        self.lr_scheduler = trainerConfig.lr_scheduler

        self.nBatch = trainerConfig.nBatch
        self.nVal = trainerConfig.nVal
        self.stepsPerEpoch = trainerConfig.stepsPerEpoch

        #no es necesario porque de momento no tengo las funciones para ver si qué modelo es mejor
        #self.bestLR = 0
        #self.currentEpoch = 0
        #self.bestEpoch = 0
        #self.bestAverageReward = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        #optimizador Adam
        self.optimizer =  None
        #tasa de aprendizaje constante
        self.lr_scheduler = None

        # verifica que el sistema operativo sea Linux, que Triton esté disponible y que haya una GPU disponible
        if checkCompileSupport():
            self.model = torch.compile(self.model)

        # Estas variables se utilizarán cuando se implemente la lógica de seguimiento del mejor modelo
        #self.bestModel = None
        #self.bestResults = None


if __name__ == "__main__":
    # Crear un modelo simple para pruebas
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 2)
            
        def forward(self, x):
            return self.linear(x)
    
    # Configurar parámetros de prueba
    model = SimpleModel()
    config = TrainerConfig(
        nBatch=32,
        nVal=100,
        stepsPerEpoch=2000,
        trainStrategy=TrainingStrategy(),
        
    )
    
    # Crear instancia del trainer
    trainer = Trainer(
        savePath="./test_models/",
        name="test_run",
        model=model,
        trainerConfig=config
    )
     # Imprimir información de configuración
    print("\n=== Configuración del Trainer ===")
    print(f"Dispositivo: {trainer.device}")
    print(f"Directorio de modelos: {trainer.directoryModels}")
    print(f"Tamaño del batch: {trainer.nBatch}")
    print(f"Pasos por época: {trainer.stepsPerEpoch}")
    print(f"Soporte para compilación: {checkCompileSupport()}")
    
    # Verificar que el optimizador está configurado
    print("\n=== Configuración del Optimizador ===")
    print(f"Tipo de optimizador: {type(trainer.optimizer).__name__}")
    print(f"Learning rate inicial: {trainer.optimizer.param_groups[0]['lr']}")
    
    print("\n=== Rutas de guardado ===")
    print(f"Ruta de entrenamiento: {trainer.trainingSavePath}")
    print(f"Ruta de mejor modelo: {trainer.baselineSavePath}")