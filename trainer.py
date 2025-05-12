import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ConstantLR
import platform
from decision_transformer_strategies import TrainingStrategy
from abc import abstractmethod
import logging
from logger.logger_setup import setup_logging
import json
from pathlib import Path

# Configurar logging solo una vez
setup_logging()

def getProjectDirectory():
        return str(Path(__file__).resolve().parent)

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
        
        if trainStrategy is None:
            trainStrategy = {
                "strategy": "DTTrainingStrategy",
                "strategyArgs": {
                    "dataPath": [getProjectDirectory() + "/data/training_data.pt"],
                }
            }
        self.nBatch = nBatch
        self.nVal = nVal
        self.stepsPerEpoch = stepsPerEpoch 
        self.trainStrategy = trainStrategy
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
    
    def to_dict(self):
       # Si se quiere poner el optimizer y el scheduler,
       # habría que añadirlo en un formato compatible con JSON 
       dict = {
           "nBatch": self.nBatch,
           "nVal": self.nVal,
           "stepsPerEpoch": self.stepsPerEpoch,
           "trainStrategy": self.trainStrategy,
       }
       return dict

class Trainer:

    def __init__(self, savePath, name, model, trainerConfig):
        self.directoryModels = savePath + name + "/"
        self.directoryProgress = savePath + name + "/"

        self.trainingSavePath = self.directoryModels + "/training.pt" #guarda el estado actual del modelo durante el entrenamiento (checkpoints), lo dejo como comentario porque no he definido los checkpoints
        self.baselineSavePath = self.directoryModels + "/best.pt" #guarda el mejor modelo, lo dejo como comentario porque no he definido el mejor modelo
        self.trackPath = self.directoryProgress + "/track.json" #guarda el historial de entrenamiento
        print(self.trackPath)
        print(f"existe: {os.path.exists(self.trackPath)}")
        #self.jsonPath = self.directoryProgress + "/validation.json" #guarda los resultados de la validacion

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

        #self.bestLR = 0
        #self.currentEpoch = 0
        #self.bestEpoch = 0
        #self.bestAverageReward = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.content = None
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

    @abstractmethod
    def createModel(self):
        pass

    #@abstractmethod
    #def testModel(self, model, validationData):
    #    pass


    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def getTrainingStrategyModule(self):
        pass

    @abstractmethod
    def getModelConfig(self):
        pass

    def initTraining(self):

        trackPath = self.trackPath
        #setup_logging(self.directoryModels)

        # Crear el documento para guardar los pasos del entrenamiento
        if not os.path.isfile(trackPath):
            print(" no existe")
            self.content = {
                "NUMBER PARAMETERS": f"{sum(t.numel() for t in self.model.parameters()) / 1000 ** 2:.1f}M",
                "MODEL INFO": self.getModelConfig().__dict__,
                "TRAINING INFO": self.trainerConfig.to_dict(),
                "EPOCHS": {}
            }

            self.updateTrackFile()
            
        else:
            print(" existe")
            print(self.trainerConfig.to_dict())
            self.content = self.JSONtoDict(trackPath)
            if "EPOCHS" not in self.content:
                self.content["EPOCHS"] = {}

        # Obtenemos el checkpoint, que es nulo si es el principio del entrenamiento
        checkpoint, self.optimizer, self.lr_scheduler = \
            self.loadModelFromFile()
        self.strategy = self.getStrategy()

        # Iniciamos las cosas. Si el checkpoint no es nulo las leemos de ahi

        self.currentEpoch = 0
        self.bestAverageReward = 0
        if checkpoint is not None:
            self.currentEpoch = checkpoint["start_epochs"]

        # Si estamos en la primera epoch, hacemos que la baseline sea el modelo
        # para que compita consigo mismo. Si no, hacemos que compita con el modelo
        # pero fijado
        #if self.currentEpoch == 0:
        #    self.bestModel = self.model
        #else:
        #    self.loadBestModelFromFile()
        #    self.bestModel = self.model

        # Generamos datos de validacion
        self.train()

    def getStrategy(self):
        if isinstance(self.trainStrategy, dict):
            strategyClass = getattr(self.getTrainingStrategyModule(), self.trainStrategy["strategy"])
            strategy = strategyClass(**self.trainStrategy["strategyArgs"])
        else:
            strategy = self.trainStrategy
        return strategy
    
    def updateTrackFile(self):
        #Guardar el archivo JSON con el nuevo contenido
        if hasattr(self, 'content') and isinstance(self.content, dict):
            # Si el contenido tiene una estrategia de entrenamiento, convertirla a diccionario
            if 'TRAINING INFO' in self.content and 'trainStrategy' in self.content['TRAINING INFO']:
                train_strategy = self.content['TRAINING INFO']['trainStrategy']
                if hasattr(train_strategy, 'to_dict'):
                    self.content['TRAINING INFO']['trainStrategy'] = train_strategy.to_dict()
    
        with open(self.trackPath, 'w') as f:
            json.dump(self.content, f, indent=4)


    #def JSONtoDict(self, filePath):
    #    with open(filePath, 'r') as f:
    #        fileData = json.load(f)
    #    return fileData
    
    def JSONtoDict(self, trackPath):
        try:
            # Si el archivo no existe, crear uno nuevo con estructura válida
            if not os.path.exists(trackPath):
                initial_content = {
                    "NUMBER PARAMETERS": f"{sum(t.numel() for t in self.model.parameters()) / 1000 ** 2:.1f}M",
                    "MODEL INFO": self.getModelConfig().__dict__,
                    "TRAINING INFO": {
                        "nBatch": self.trainerConfig.nBatch,
                        "nVal": self.trainerConfig.nVal,
                        "stepsPerEpoch": self.trainerConfig.stepsPerEpoch,
                        "trainStrategy": self.trainerConfig.trainStrategy.to_dict() if hasattr(self.trainerConfig.trainStrategy, 'to_dict') else str(self.trainerConfig.trainStrategy)
                    },
                    "EPOCHS": {}
                }
                with open(trackPath, 'w') as f:
                    json.dump(initial_content, f, indent=4)
                return initial_content

            # Si el archivo existe, intentar leerlo
            with open(trackPath, 'r') as f:
                content = f.read()
                
            # Si el archivo está vacío, crear contenido inicial
            if not content.strip():
                initial_content = {
                    "NUMBER PARAMETERS": f"{sum(t.numel() for t in self.model.parameters()) / 1000 ** 2:.1f}M",
                    "MODEL INFO": self.getModelConfig().__dict__,
                    "TRAINING INFO": {
                        "nBatch": self.trainerConfig.nBatch,
                        "nVal": self.trainerConfig.nVal,
                        "stepsPerEpoch": self.trainerConfig.stepsPerEpoch,
                        "trainStrategy": self.trainerConfig.trainStrategy.to_dict() if hasattr(self.trainerConfig.trainStrategy, 'to_dict') else str(self.trainerConfig.trainStrategy)
                    },
                    "EPOCHS": {}
                }
                with open(trackPath, 'w') as f:
                    json.dump(initial_content, f, indent=4)
                return initial_content

            # Intentar parsear el JSON
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                print(f"\nError en el archivo JSON: {e}")
                print(f"Línea: {e.lineno}, Columna: {e.colno}")
                print("Creando nuevo archivo de seguimiento...")
                
                # Crear nuevo archivo con estructura válida
                initial_content = {
                    "NUMBER PARAMETERS": f"{sum(t.numel() for t in self.model.parameters()) / 1000 ** 2:.1f}M",
                    "MODEL INFO": self.getModelConfig().__dict__,
                    "TRAINING INFO": {
                        "nBatch": self.trainerConfig.nBatch,
                        "nVal": self.trainerConfig.nVal,
                        "stepsPerEpoch": self.trainerConfig.stepsPerEpoch,
                        "trainStrategy": self.trainerConfig.trainStrategy.to_dict() if hasattr(self.trainerConfig.trainStrategy, 'to_dict') else str(self.trainerConfig.trainStrategy)
                    },
                    "EPOCHS": {}
                }
                with open(trackPath, 'w') as f:
                    json.dump(initial_content, f, indent=4)
                return initial_content

        except Exception as e:
            print(f"\nError inesperado al procesar el archivo JSON: {e}")
            print("Creando nuevo archivo de seguimiento...")
            
            # Crear nuevo archivo con estructura válida
            initial_content = {
                "NUMBER PARAMETERS": f"{sum(t.numel() for t in self.model.parameters()) / 1000 ** 2:.1f}M",
                "MODEL INFO": self.getModelConfig().__dict__,
                "TRAINING INFO": {
                    "nBatch": self.trainerConfig.nBatch,
                    "nVal": self.trainerConfig.nVal,
                    "stepsPerEpoch": self.trainerConfig.stepsPerEpoch,
                    "trainStrategy": self.trainerConfig.trainStrategy.to_dict() if hasattr(self.trainerConfig.trainStrategy, 'to_dict') else str(self.trainerConfig.trainStrategy)
                },
                "EPOCHS": {}
            }
            with open(trackPath, 'w') as f:
                json.dump(initial_content, f, indent=4)
            return initial_content
    
    def saveModel(self):
        torch.save({'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'lr_scheduler_state': self.lr_scheduler.state_dict(),
                    'start_epochs': self.currentEpoch,
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else 0,
                    #"best_lr": self.bestLR,
                    #"best_epoch": self.bestEpoch
                    },
                   self.trainingSavePath,
                   )
    
    #def loadBestModelFromFile(self):
    #    self.bestModel = self.createModel()
    #    if checkCompileSupport():
    #        self.bestModel = torch.compile(self.bestModel)

    #    fileExists = os.path.isfile(self.baselineSavePath)
    #    if fileExists:
    #        checkpoint = torch.load(self.baselineSavePath)
    #        self.bestModel.load_state_dict(checkpoint["model_state"])

    # Generamos los datos de validacion
    #def getValidationData(self):
    #    print("=== Obteniendo datos de validación ===")
    #    validationData = self.strategy.getValidationData(self.nBatch, self.nVal)
    #    print("=== Datos de validación obtenidos ===")
    #    return validationData


    def loadModelFromFile(self):
        #comprueba si el archivo existe
        fileExists = os.path.isfile(self.trainingSavePath)
        checkpoint = None
        #usar el optimizador y el scheduler que se han definido en el trainerConfig
        optimizer = self.trainerConfig.optimizer
        lrScheduler = self.trainerConfig.lr_scheduler

        if fileExists:
            checkpoint = torch.load(self.trainingSavePath)
            #cargar el estado del modelo
            self.model.load_state_dict(checkpoint["model_state"])
            #cargar el estado del optimizador
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            #cargar el estado del scheduler
            lrScheduler.load_state_dict(checkpoint["lr_scheduler_state"])
            #cargar el estado del generador de numeros aleatorios
            torch.set_rng_state(checkpoint["rng_state"])
            #guardar el mejor learning rate
            #self.bestLR = checkpoint["best_lr"]
            #guardar el numero de epoch actual
            self.currentEpoch = checkpoint["start_epochs"]
            #guardar el numero de epoch con el mejor resultado
            #self.bestEpoch = checkpoint["best_epoch"]
            #si hay una GPU, cargar el estado del generador de numeros aleatorios de la GPU
            if torch.cuda.is_available(): torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])

        return checkpoint, optimizer, lrScheduler



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
        nBatch=2,
        nVal=100,
        stepsPerEpoch=20,
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
    
