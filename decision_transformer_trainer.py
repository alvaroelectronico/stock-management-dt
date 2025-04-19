from torch import nn
from decision_transformer import DecisionTransformer
from trainer import Trainer, TrainerConfig
import torch
from decision_transformer_strategies import TrainingStrategy
from decision_transformer_config import DecisionTransformerConfig
from decision_transformer_strategies import DTTrainingStrategy
import decision_transformer_strategies
import numpy as np
from generate_tajectories import TRAJECTORY_LENGTH
import copy
from tensordict import TensorDict
import os
import json
from datetime import datetime



class DecisionTransformerTrainer(Trainer):

    def __init__(self, savePath, name, model, trainerConfig):
        super().__init__(savePath, name, model, trainerConfig)

    def createModel(self):
        return DecisionTransformer(self.getModelConfig())
    
    def getModelConfig(self):
        return self.model.decisionTransformerConfig
    
    def getTrainingStrategyModule(self):
        return decision_transformer_strategies
       
    def saveTrainingMetrics(self):
        """Guarda las métricas de entrenamiento en un archivo JSON"""
        metrics_path = os.path.join(self.directoryProgress, "training_metrics.json")
        
        # Preparar los datos para guardar
        metrics_data = {
            'epoch_losses': self.training_metrics['epoch_losses'],
            'validation_losses': self.training_metrics['validation_losses'],
            'final_learning_rate': self.optimizer.param_groups[0]['lr'],
            'ordered_quantities': []  # Añadir las cantidades ordenadas
        }
        
        # Procesar y guardar las cantidades ordenadas
        for batch_data in self.training_metrics['ordered_quantities']:
            epoch_data = {
                'epoch': batch_data['epoch'],
                'training_step': batch_data['training_step'],
                'trajectory': []
            }
            
            # Procesar cada paso de la trayectoria
            for t, (pred, real) in enumerate(zip(batch_data['allPredictedActions'], batch_data['allRealActions'])):
                trajectory_step = {
                    'step': t,
                    'real_quantity': real.cpu().detach().numpy().tolist(),
                    'predicted_quantity': pred.cpu().detach().numpy().tolist()
                }
                epoch_data['trajectory'].append(trajectory_step)
            
            metrics_data['ordered_quantities'].append(epoch_data)
        
        # Guardar en el archivo
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=4)
        
        print(f"Métricas guardadas en: {metrics_path}")

    #Entrenamiento del modelo
    def train(self):
        epoch = 0
        
        print("\n=== Iniciando entrenamiento ===")
        print(f"Pasos por época: {self.stepsPerEpoch}")
        print(f"Tamaño del batch: {self.nBatch}")
        
        # Inicializar métricas
        self.training_metrics = {
            'epoch_losses': [],
            'validation_losses': [],
            'ordered_quantities': []  # Nueva métrica para cantidades ordenadas
        }
        
        #Bucle de entrenamiento
        while True:
            print(f"\n=== Comenzando época {epoch} ===")
            epochLoss = 0
            currentStep = 1
            
            #Bucle de pasos de entrenamiento de un epoch
            while currentStep <= self.stepsPerEpoch:
                print(f"\nStep {currentStep}/{self.stepsPerEpoch}")
                
                #Obtener datos de entrenamiento
                dtData = self.trainStrategy.getTrainingData(self.nBatch)
                problemData, orderQuantityData, returnsToGoData = dtData
                
                #Poner el modelo en modo entrenamiento
                self.model.train() 
                
                # Convertir los datos a la GPU si está disponible
                orderQuantityData = orderQuantityData.to(self.device)
                returnsToGoData = returnsToGoData.to(self.device)
                td = problemData.to(self.device)
                
                #inicializar el modelo con los datos del problema
                self.model.setInitalReturnToGo(td, returnsToGoData) 
                td = self.model.initModel(td)

                # Variables para almacenar predicciones y valores reales de la trayectoria
                allPredictedActions = []
                allRealActions = []
                
                # Procesar la trayectoria completa
                trajectoryLength = orderQuantityData.size(1)  # Longitud de la trayectoria
                print(f"Procesando trayectoria de longitud {trajectoryLength}")
                
                for t in range(trajectoryLength):
                    print(f"  Paso {t+1}/{trajectoryLength} de la trayectoria")
                    
                    # Forward pass
                    td = self.model.forward(td)
                    predictedAction = td["orderQuantity"]
                    
                    # Guardar predicción y valor real
                    allPredictedActions.append(predictedAction)
                    allRealActions.append(orderQuantityData[:, t:t+1])
                    
                    print(f"  Predicción: {predictedAction.item():.2f}, Real: {orderQuantityData[:, t].item():.2f}")
                
                # Calcular la pérdida para toda la trayectoria
                predictedTensor = torch.cat(allPredictedActions, dim=1)
                realTensor = torch.cat(allRealActions, dim=1)
                loss = nn.MSELoss()(predictedTensor, realTensor)
                
                # Guardar las cantidades de la trayectoria
                batch_quantities = {
                    'epoch': epoch,
                    'training_step': currentStep,
                    'allPredictedActions': allPredictedActions,
                    'allRealActions': allRealActions
                }
                self.training_metrics['ordered_quantities'].append(batch_quantities)
                
                # Backward pass
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
                
                # Acumular la pérdida
                epochLoss += loss.item()
                print(f"Pérdida de la trayectoria: {loss.item():.4f}")
                
                # Incrementar el contador de steps
                currentStep += 1
            
            avgEpochLoss = epochLoss / self.stepsPerEpoch
            self.training_metrics['epoch_losses'].append(avgEpochLoss)
            print(f"\nResumen de época {epoch}:")
            print(f"Pérdida promedio: {avgEpochLoss:.4f}")
            
            # Validación del modelo
            print("\nIniciando validación...")
            validation_loss = self.validate_model()
            self.training_metrics['validation_losses'].append(validation_loss)
            print(f"Pérdida de validación: {validation_loss:.4f}")
            
            # Actualizar mejor modelo si es necesario
            #if validation_loss < best_validation_loss:
            #    best_validation_loss = validation_loss
            #    self.training_metrics['best_validation_loss'] = validation_loss
            #    self.training_metrics['best_epoch'] = epoch
            #    epochs_without_improvement = 0
            #    print("¡Nuevo mejor modelo encontrado! Guardando...")
            #    self.saveBestModel()
            #else:
            #    epochs_without_improvement += 1
            #    print(f"Épocas sin mejora: {epochs_without_improvement}/{patience}")
            
            # Guardar métricas en el archivo de seguimiento
            print("Guardando métricas en track.json...")
            self.content["EPOCHS"][epoch] = {
                "training_loss": avgEpochLoss,
                "validation_loss": validation_loss,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            }
            self.updateTrackFile()
            print("Métricas guardadas exitosamente")
            
            #Incrementar el contador de epoch
            epoch += 1
            
        # Guardar el modelo final y las métricas
            print("\n=== Guardando modelo final y métricas ===")
            self.saveModel()
            self.saveTrainingMetrics()
            print("Modelo y métricas guardados exitosamente")

    def validate_model(self):
        print("\nIniciando validación del modelo...")
        self.model.eval()  # Poner el modelo en modo evaluación
        total_loss = 0
        n_val = self.nVal  # Número de muestras de validación (10 en tu caso)
        
        with torch.no_grad():
            for _ in range(n_val):
                # Obtener datos de validación
                dtData = self.trainStrategy.getValidationData(self.nBatch)
                problemData, orderQuantityData, returnsToGoData = dtData
                
                # Convertir los datos a la GPU si está disponible
                orderQuantityData = orderQuantityData.to(self.device)
                returnsToGoData = returnsToGoData.to(self.device)
                td = problemData.to(self.device)
                
                # Inicializar el modelo
                self.model.setInitalReturnToGo(td, returnsToGoData)
                td = self.model.initModel(td)
                
                # Forward pass
                predictedAction = self.model(td)["orderQuantity"]
                
                # Calcular pérdida
                loss = nn.MSELoss()(predictedAction, orderQuantityData)
                
                total_loss += loss.item()
        
        self.model.train()  # Volver al modo de entrenamiento
        return total_loss / n_val

           
        
                
if __name__ == "__main__":
    print("\n=== Iniciando prueba del DecisionTransformerTrainer ===")
    
    # Configuración básica
    config = TrainerConfig(
        nBatch=1,#32
        nVal=100,  # Ajusta este valor según tus necesidades
        stepsPerEpoch=10,
        trainStrategy=DTTrainingStrategy(dataPath=["C:/Users/elood/Desktop/DTgestionStock/Stock_management_dt/stock-management-dt/data/training_data.pt"]),
        lr_scheduler=1e-4
    )
    
    print(f"\nConfiguración:")
    print(f"Learning rate: {config.lr_scheduler}")
    print(f"Steps per epoch: {config.stepsPerEpoch}")
    print(f"Batch size: {config.nBatch}")
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
            savePath="./training_models",  # Cambiado a training_models
            name="decision_transformer_model",  # Nombre más descriptivo
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
        print(f"Modelo guardado en: ./training_models/decision_transformer_model")
        
    except Exception as e:
        print("\n=== Error durante la ejecución ===")
        print(f"Error: {str(e)}")
        raise e
    
    print("\n=== Prueba completada ===")
                
