from torch import nn, optim
from decision_transformer import DecisionTransformer
from trainer import Trainer, TrainerConfig
import torch
from decision_transformer_strategies import TrainingStrategy
from decision_transformer_config import DecisionTransformerConfig
from decision_transformer_strategies import DTTrainingStrategy



class DecisionTransformerTrainer(Trainer):

    def __init__(self, savePath, name, model, trainerConfig):
        super().__init__(savePath, name, model, trainerConfig)
       
    #Entrenamiento del modelo
    def train(self):
        epoch = 0
        
        #Bucle de entrenamiento
        while True:  # Falta condición de parada 
            epochLoss = 0
            currentStep = 1
            #Bucle de pasos de entrenamiento de un epoch
            while currentStep <= self.stepsPerEpoch:
                #Obtener datos de entrenamiento
                dtData = self.trainStrategy.getTrainingData(self.nBatch)  # Usar la estrategia de entrenamiento
                problemData, orderQuantityData, returnsToGoData = dtData
                
                #Poner el modelo en modo entrenamiento
                self.model.train() 
                #Convertir los datos a la GPU si está disponible
                orderQuantityData = orderQuantityData.to(self.device)
                returnsToGoData = returnsToGoData.to(self.device)
                td = problemData.to(self.device)
                
                #inicializar el modelo con los datos del problema revisar
                self.model.setInitalReturnToGo(td, returnsToGoData) 
                td = self.model.initModel(td)

                #forward pass y calculo de perdidas
                predictedAction = self.model(td)["orderQuantity"]
                loss = nn.MSELoss()(predictedAction, orderQuantityData)

                #backward pass
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step() 

                #Sumar la pérdida del paso actual al total del epoch
                epochLoss += loss.item()

                #Incrementar el contador de pasos
                currentStep += 1
            
            avgEpochLoss = epochLoss / self.stepsPerEpoch
            print(f"Epoch {epoch}, Pérdida promedio: {avgEpochLoss:.4f}")
            
            #Incrementar el contador de epoch
            epoch += 1

           
            

                
if __name__ == "__main__":
    print("\n=== Iniciando prueba del DecisionTransformerTrainer ===")
    
    # Configuración básica
    config = TrainerConfig(
    nBatch=32,
    nVal=100,  # Ajusta este valor según tus necesidades
    stepsPerEpoch=10,
    trainStrategy=DTTrainingStrategy(dataPath=["data/training_data.pt"], trainPercentage=[0.8]),
    learningRate=1e-4
)

    
    print(f"\nConfiguración:")
    print(f"Learning rate: {config.learningRate}")
    print(f"Steps per epoch: {config.stepsPerEpoch}")
    print(f"Batch size: {config.nBatch}")
    # Nota: device ya no es parte de TrainerConfig, se determina en la clase Trainer
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    try:
        # Crear el modelo
        print("\nCreando modelo Decision Transformer...")
        model = DecisionTransformer(
            decisionTransformerConfig=DecisionTransformerConfig())
        print("Modelo creado exitosamente")
        
         # Crear el trainer
        print("\nCreando trainer...")
        trainer = DecisionTransformerTrainer(
            savePath="./checkpoints",
            name="dt_test",
            model=model,
            trainerConfig=config
        )
        print("Trainer creado exitosamente")
        
        # Intentar un paso de entrenamiento
        print("\n=== Iniciando entrenamiento de prueba ===")
        print("Realizando un epoch de entrenamiento...")
        
        # Entrenar por un epoch
        trainer.train()
        
        print("\n=== Entrenamiento completado exitosamente ===")
        
    except Exception as e:
        print("\n=== Error durante la ejecución ===")
        print(f"Error: {str(e)}")
        raise e
    
    print("\n=== Prueba completada ===")
                
