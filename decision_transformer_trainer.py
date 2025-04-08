from torch import nn
from decision_transformer import DecisionTransformer
from trainer import Trainer, TrainerConfig
import torch
from decision_transformer_strategies import TrainingStrategy
from decision_transformer_config import DecisionTransformerConfig
from decision_transformer_strategies import DTTrainingStrategy
import decision_transformer_strategies
import numpy as np
from generate_tajectories import TRAYECTORY_LENGHT
import copy
from tensordict import TensorDict



class DecisionTransformerTrainer(Trainer):

    def __init__(self, savePath, name, model, trainerConfig):
        super().__init__(savePath, name, model, trainerConfig)

    def createModel(self):
        return DecisionTransformer(self.getModelConfig())
    
    #def testModel(self, model, validationData):
    #    model.eval()
    #    averageReward = 0
    #    results = []

        # Evitamos los calculos de los gradientes
    #    with torch.inference_mode():
    #        for traj in range(validationData['states'].batch_size[0]):
    #            # Reiniciar el estado para esta trayectoria
    #            current_td = {
    #                'states': {k: v[traj:traj+1] for k, v in validationData['states'].items()},
    #                'actions': validationData['actions'][traj:traj+1],
    #                'returnsToGo': validationData['returnsToGo'][traj:traj+1]
    #            }
    #            current_td = TensorDict(current_td, batch_size=[1])
            
                # Inicializar esta trayectoria específica
    #            trajectory_td = model.initModel(current_td)
            
                # Simular la trayectoria completa
    #            for _ in range(TRAYECTORY_LENGHT):
    #                trajectory_td = model(trajectory_td)
            
                # Obtener el return-to-go final de esta trayectoria
    #            reward = trajectory_td['returnsToGo']
    #            results.extend(reward.cpu().tolist())
    #            averageReward += reward.mean()

        # Calcular el promedio sobre todas las trayectorias
    #    averageReward = averageReward / validationData['states'].batch_size[0]

    #    print("\n=== Resultados de la Validación ===")
    #    print(f"Número de trayectorias evaluadas: {validationData['states'].batch_size[0]}")
    #    print(f"Return-to-go promedio: {averageReward:.2f}")
    
    #    return averageReward.cpu().item(), np.array(results), 0

    def getModelConfig(self):
        return self.model.decisionTransformerConfig
    
    def getTrainingStrategyModule(self):
        return decision_transformer_strategies
       
    #Entrenamiento del modelo
    def train(self):
        #validationData = self.getValidationData()
        #if self.currentEpoch != 0:
        #    self.bestAverageReward, self.bestResults, _ = self.testModel(self.bestModel, validationData)
        #    print(f"El modelo actual tiene de validación: {self.bestAverageReward}.")

        epoch = 0

        
        #Bucle de entrenamiento
        while True:  # Falta condición de parada 
            epochLoss = 0
            currentStep = 1
            #self.content["EPOCHS"][self.currentEpoch] = [{
            #    "Starting Validation Reward": self.bestAverageReward
            #}]
            #Bucle de pasos de entrenamiento de un epoch
            while currentStep <= self.stepsPerEpoch:
                #Obtener datos de entrenamiento
                dtData = self.trainStrategy.getTrainingData(self.nBatch)  # Usar la estrategia de entrenamiento
                problemData, orderQuantityData, returnsToGoData = dtData
                print(f"problemData: {problemData}")
                print(f"orderQuantityData: {orderQuantityData}")
                print(f"returnsToGoData: {returnsToGoData}")
                
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

            # Validación del modelo
            #print("\n=== Iniciando validación del modelo ===")
            #averageNewReward, newResults, _ = self.testModel(self.model, validationData)
            #print(f"Reward de validación actual: {averageNewReward:.2f}")
            #print(f"Mejor reward hasta ahora: {self.bestAverageReward:.2f}")

            # Comprobar si el nuevo modelo es mejor
            #if self.bestAverageReward < averageNewReward:
            #    pValue = 0
            #    saveModel = False

                #if self.currentEpoch == 0:
            #       saveModel = True
            #       self.bestResults = newResults
                #else:
                    # Realizar test estadístico
                    #from scipy.stats import ttest_rel
                    #t, p = ttest_rel(newResults, self.bestResults)
                    #pValue = p / 2
                    #print(f"P-valor: {pValue}")
                    #saveModel = pValue < 0.05

                #if saveModel:
                #    print("Guardando nuevo mejor modelo...")
                #    modelStateDict = copy.deepcopy(self.model.state_dict())
                #    self.bestModel.load_state_dict(modelStateDict)
                #    self.saveBestModel()
                
                    # Actualizar mejores resultados
                    #self.bestAverageReward, self.bestResults, _ = self.testModel(self.bestModel, validationData)

                #self.content["EPOCHS"][self.currentEpoch].append({
                #    "Step": currentStep - 1,
                #    "Validation Reward": self.bestAverageReward,
                #    "P-value": float(pValue),
                #    "Saved": True
                #})
                #else:
                #    self.content["EPOCHS"][self.currentEpoch].append({
                #        "Step": currentStep - 1,
                #        "Validation Reward": averageNewReward,
                #        "P-value": float(pValue),
                #        "Saved": False
                #    })
            #else:
            #    print("Modelo no guardado - No hay mejora")
            #    self.content["EPOCHS"][self.currentEpoch].append({
            #        "Step": currentStep - 1,
            #        "Validation Reward": averageNewReward,
            #        "Saved": False
            #    })

            # Actualizar información final de la época
            #self.content["EPOCHS"][self.currentEpoch].append({
            #    "Ending Validation Reward": self.bestAverageReward,
            #    "Mean Loss": avgEpochLoss,
            #})

            # Guardar progreso
            #self.updateTrackFile()
        
            # Guardar checkpoint del modelo actual
            #self.saveModel()
        
            # Preparar siguiente época
            #self.currentEpoch += 1
                
           
            

                
if __name__ == "__main__":
    print("\n=== Iniciando prueba del DecisionTransformerTrainer ===")
    
    # Configuración básica
    config = TrainerConfig(
    nBatch=32,
    nVal=100,  # Ajusta este valor según tus necesidades
    stepsPerEpoch=10,
    trainStrategy=DTTrainingStrategy(dataPath=["C:/Users/elood/Desktop/DTgestionStock/Stock_management_dt/stock-management-dt/data/training_data.pt"], trainPercentage=[0.8]),
    lr_scheduler=1e-4
)

    
    print(f"\nConfiguración:")
    print(f"Learning rate: {config.lr_scheduler}")
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
                
