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

setup_logging()

def getProjectDirectory():
        return str(Path(__file__).resolve().parent)

def checkCompileSupport():
    if platform.system() != "Linux":
        return False

    try:
        import triton
    except ImportError:
        return False

    if not torch.cuda.is_available():
        return False

    return True

class TrainerConfig:

    def __init__(self, nBatch, nVal, stepsPerEpoch, trainStrategy=None, optimizer=None, lr_scheduler=None, testDataPath=None, mixed_precision=None):
        """
        Configuración del entrenador.
        
        Args:
            nBatch: Tamaño del batch
            nVal: Tamaño de validación
            stepsPerEpoch: Pasos por época
            trainStrategy: Estrategia de entrenamiento
            optimizer: Optimizador
            lr_scheduler: Scheduler de learning rate
            testDataPath: Ruta de datos de test
            mixed_precision: Tipo de precisión mixta. Puede ser None (precisión completa), "bfloat16" o "float16"
        """
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
        self.testDataPath = testDataPath
        self.mixed_precision = mixed_precision
        self.use_bfloat16 = (mixed_precision == "bfloat16")
    
    def to_dict(self):
       dict = {
           "nBatch": self.nBatch,
           "nVal": self.nVal,
           "stepsPerEpoch": self.stepsPerEpoch,
           "trainStrategy": self.trainStrategy,
           "mixed_precision": self.mixed_precision,
       }
       return dict

class Trainer:

    def __init__(self, savePath, name, model, trainerConfig):
        self.directoryModels = savePath + name + "/"
        self.directoryProgress = savePath + name + "/"

        self.trainingSavePath = self.directoryModels + "/training.pt"
        self.baselineSavePath = self.directoryModels + "/best.pt"
        self.trackPath = self.directoryProgress + "/track.json"
        print(self.trackPath)
        print(f"existe: {os.path.exists(self.trackPath)}")

        if not os.path.exists(self.directoryModels):
            os.makedirs(self.directoryModels)
        if not os.path.exists(self.directoryProgress):
            os.makedirs(self.directoryProgress)

        self.trainerConfig = trainerConfig
        self.trainStrategy = trainerConfig.trainStrategy
        self.optimizer = trainerConfig.optimizer
        self.lr_scheduler = trainerConfig.lr_scheduler

        self.nBatch = trainerConfig.nBatch
        self.nVal = trainerConfig.nVal
        self.stepsPerEpoch = trainerConfig.stepsPerEpoch

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.content = None

        if checkCompileSupport():
            #self.model = torch.compile(self.model)
            pass

    @abstractmethod
    def createModel(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def getTrainingStrategyModule(self):
        pass

    @abstractmethod
    def getModelConfig(self):
        pass

    def getFilteredModelInfo(self):
        """
        Obtiene solo los campos relevantes de la configuración del modelo para guardar en track.json.
        
        Returns:
            dict: Diccionario con solo los campos filtrados de la configuración del modelo
        """
        model_config = self.getModelConfig().__dict__
        fields_to_keep = [
            "hidden_size",
            "n_layer",
            "n_head",
            "n_inner",
            "activation_function",
            "resid_pdrop",
            "embd_pdrop",
            "attn_pdrop",
            "layer_norm_epsilon"
        ]
        filtered_info = {key: model_config.get(key) for key in fields_to_keep if key in model_config}
        # Añadir información de precisión mixta del trainerConfig
        if hasattr(self.trainerConfig, 'mixed_precision'):
            filtered_info["mixed_precision"] = self.trainerConfig.mixed_precision
        elif hasattr(self.trainerConfig, 'use_bfloat16'):
            # Compatibilidad hacia atrás
            filtered_info["mixed_precision"] = "bfloat16" if self.trainerConfig.use_bfloat16 else None
        return filtered_info

    def initTraining(self):

        trackPath = self.trackPath

        if not os.path.isfile(trackPath):
            print(" no existe")
            self.content = {
                "NUMBER PARAMETERS": sum(t.numel() for t in self.model.parameters()),
                "MODEL INFO": self.getFilteredModelInfo(),
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

        checkpoint, self.optimizer, self.lr_scheduler = \
            self.loadModelFromFile()
        self.strategy = self.getStrategy()

        self.currentEpoch = 0
        self.bestAverageReward = 0
        if checkpoint is not None:
            self.currentEpoch = checkpoint["start_epochs"]

        self.model = self.model.to(self.device)
        self.train()

    def getStrategy(self):
        if isinstance(self.trainStrategy, dict):
            strategyClass = getattr(self.getTrainingStrategyModule(), self.trainStrategy["strategy"])
            strategy = strategyClass(**self.trainStrategy["strategyArgs"])
        else:
            strategy = self.trainStrategy
        return strategy
    
    def updateTrackFile(self):
        if hasattr(self, 'content') and isinstance(self.content, dict):
            if 'TRAINING INFO' in self.content and 'trainStrategy' in self.content['TRAINING INFO']:
                train_strategy = self.content['TRAINING INFO']['trainStrategy']
                if hasattr(train_strategy, 'to_dict'):
                    self.content['TRAINING INFO']['trainStrategy'] = train_strategy.to_dict()
    
        with open(self.trackPath, 'w') as f:
            json.dump(self.content, f, indent=4)

    def JSONtoDict(self, trackPath):
        try:
            if not os.path.exists(trackPath):
                initial_content = {
                    "NUMBER PARAMETERS": sum(t.numel() for t in self.model.parameters()),
                    "MODEL INFO": self.getFilteredModelInfo(),
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

            with open(trackPath, 'r') as f:
                content = f.read()
                
            if not content.strip():
                initial_content = {
                    "NUMBER PARAMETERS": sum(t.numel() for t in self.model.parameters()),
                    "MODEL INFO": self.getFilteredModelInfo(),
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

            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                print(f"\nError en el archivo JSON: {e}")
                print(f"Línea: {e.lineno}, Columna: {e.colno}")
                print("Creando nuevo archivo de seguimiento...")
                
                initial_content = {
                    "NUMBER PARAMETERS": sum(t.numel() for t in self.model.parameters()),
                    "MODEL INFO": self.getFilteredModelInfo(),
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
            
            initial_content = {
                "NUMBER PARAMETERS": sum(t.numel() for t in self.model.parameters()),
                "MODEL INFO": self.getFilteredModelInfo(),
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
                    },
                   self.trainingSavePath,
                   )

    def loadModelFromFile(self):
        fileExists = os.path.isfile(self.trainingSavePath)
        checkpoint = None
        optimizer = self.trainerConfig.optimizer
        lrScheduler = self.trainerConfig.lr_scheduler

        if fileExists:
            checkpoint = torch.load(self.trainingSavePath)
            self.model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            lrScheduler.load_state_dict(checkpoint["lr_scheduler_state"])
            torch.set_rng_state(checkpoint["rng_state"])
            self.currentEpoch = checkpoint["start_epochs"]
            if torch.cuda.is_available(): torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])

        return checkpoint, optimizer, lrScheduler



if __name__ == "__main__":
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 2)
            
        def forward(self, x):
            return self.linear(x)
    
    model = SimpleModel()
    config = TrainerConfig(
        nBatch=2,
        nVal=100,
        stepsPerEpoch=20,
        trainStrategy=TrainingStrategy(),
        
    )
    
    trainer = Trainer(
        savePath="./test_models/",
        name="test_run",
        model=model,
        trainerConfig=config
    )
    print("\n=== Configuración del Trainer ===")
    print(f"Dispositivo: {trainer.device}")
    print(f"Directorio de modelos: {trainer.directoryModels}")
    print(f"Tamaño del batch: {trainer.nBatch}")
    print(f"Pasos por época: {trainer.stepsPerEpoch}")
    print(f"Soporte para compilación: {checkCompileSupport()}")
    
    print("\n=== Configuración del Optimizador ===")
    
