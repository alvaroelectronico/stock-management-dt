from torch import nn
from decision_transformer_improved import DecisionTransformer
from trainer import Trainer, TrainerConfig
import torch
from decision_transformer_strategies import TrainingStrategy
from decision_transformer_config import DecisionTransformerConfig
from decision_transformer_strategies import DTTrainingStrategy
import decision_transformer_strategies
import numpy as np
from generate_trajectories import TRAJECTORY_LENGTH
import copy
from tensordict import TensorDict
import os
import json
from datetime import date
import random
import time
from data_scaler import compute_scaling_params_from_training_data

DEBUG_TRAINING_SAVED_COUNT = 0

def save_training_debug_input(td, orderQuantityData, currentStep, cont, window_start, window_end, index=1):
    """
    Guarda todos los inputs del modelo durante el entrenamiento para debugging.
    Solo guarda los datos del primer problema del batch (índice 0).
    Se ejecuta dos veces para análisis del primer y segundo paso.
    
    Args:
        td: TensorDict con todos los datos del estado
        orderQuantityData: Tensor con las acciones reales
        currentStep: Paso actual de entrenamiento
        cont: Contador de ventana
        window_start: Inicio de la ventana de predicción
        window_end: Fin de la ventana de predicción
    """
    global DEBUG_TRAINING_SAVED_COUNT
    
    if DEBUG_TRAINING_SAVED_COUNT >= 2:
        return
    
    debug_data = {
        "step_info": {
            "currentStep": currentStep,
            "cont": cont,
            "window_start": window_start,
            "window_end": window_end,
            "currentTimestep": td["currentTimestep"][index].cpu().tolist()
        },
        "state_data": {
            "onHandLevel": td["onHandLevel"][index].cpu().tolist(),
            "inTransitStock": td["inTransitStock"][index].cpu().tolist(),
            "forecast": td["forecast"][index].cpu().tolist(),
            "demand": td["demand"][index].cpu().tolist(),
        },
        "cost_data": {
            "holdingCost": td["holdingCost"][index].cpu().item(),
            "orderingCost": td["orderingCost"][index].cpu().item(),
            "stockOutPenalty": td["stockOutPenalty"][index].cpu().item(),
            "unitRevenue": td["unitRevenue"][index].cpu().item(),
            "leadTime": td["leadTime"][index].cpu().item(),
        },
        "returns_data": {
            "returnsToGo": td["returnsToGo"][index].cpu().tolist(),
            "benefit": td["benefit"][index].cpu().tolist() if "benefit" in td else None,
        },
        "action_data": {
            "nextOrderQuantity": orderQuantityData[index, cont].cpu().item(),
            "orderQuantity": td["orderQuantity"][index].cpu().tolist() if "orderQuantity" in td else None,
        },
        "embedding_data": {
            "statesEmbedding_shape": list(td["statesEmbedding"][index].shape),
            "actionsEmbedding_shape": list(td["actionsEmbedding"][index].shape),
            "returnsToGoEmbedding_shape": list(td["returnsToGoEmbedding"][index].shape),
        }
    }
    
    DEBUG_TRAINING_SAVED_COUNT += 1
    output_path = f"debug_training_input_step{DEBUG_TRAINING_SAVED_COUNT}.json"
    with open(output_path, 'w') as f:
        json.dump(debug_data, f, indent=4)



class DecisionTransformerTrainer(Trainer):

    def __init__(self, savePath, name, model, trainerConfig):
        super().__init__(savePath, name, model, trainerConfig)

    def createModel(self):
        """
        Crea el modelo con parámetros de escalado calculados desde los datos de entrenamiento.
        """
        # Obtener las rutas de datos del trainStrategy
        data_paths = self.trainStrategy.dataPath if hasattr(self.trainStrategy, 'dataPath') else []
        
        # Calcular parámetros de escalado si hay rutas de datos
        scaling_params = {}
        if data_paths:
            try:
                scaling_params = compute_scaling_params_from_training_data(data_paths)
                print(f"Parámetros de escalado calculados desde {len(data_paths)} archivo(s) de datos")
            except Exception as e:
                print(f"Advertencia: No se pudieron calcular los parámetros de escalado: {e}")
                print("El modelo funcionará sin escalado")
        
        return DecisionTransformer(self.getModelConfig(), scaling_params=scaling_params)
    
    def getModelConfig(self):
        return self.model.decisionTransformerConfig
    
    def getTrainingStrategyModule(self):
        return decision_transformer_strategies

    def saveModel(self):
        super().saveModel()

    def evaluate_benefits(self):
        batch = self.nBatch

        self.model.eval()
        all_test_benefits = []
        all_real_benefits = []
        with torch.no_grad():
            all_problem_data = self.trainStrategy.problemData
            num_batches = (self.trainStrategy.lengthData) // batch
            for i in range(num_batches):
                start_idx = i * batch
                end_idx = min(start_idx + batch, self.trainStrategy.lengthData)
                batch_indices = torch.arange(start_idx, end_idx)
                test_problem = all_problem_data[batch_indices]
                real_benefits = test_problem["benefit"]
                test_td = {k: v.clone() for k, v in test_problem.items()}
                test_td = {k: v.to(self.device) for k, v in test_td.items()}
                test_td["returnsToGo"] = torch.zeros((batch), device=self.device)
                test_td = self.model.initModel(test_td)
                trajectory_length = test_td['demand'].size(1)

                for step in range(trajectory_length):
                    test_td = self.model.forward(test_td, nextOrderQuantity=None, is_test=True, update_only=False)
                all_test_benefits.append(test_td["benefit"].cpu())
                all_real_benefits.append(real_benefits)
        self.model.train()
        test_mean = torch.cat(all_test_benefits).mean().item()
        real_mean = torch.cat(all_real_benefits).mean().item()
        return test_mean, real_mean

    def train(self):
        epoch = getattr(self, 'currentEpoch', -1) + 1

        self.training_metrics = {
            'epoch_losses': [],
            'validation_losses': [],
            'validation_cost_metrics': []
        }
        
        while True:
            epoch_start_time = time.time()
            epochLoss = 0
            currentStep = 1
            progress_checkpoints = [int(self.stepsPerEpoch * p / 100) for p in range(10, 101, 10)]
            
            while currentStep <= self.stepsPerEpoch:

                dtData = self.trainStrategy.getTrainingData(self.nBatch)
                problemData, orderQuantityData, returnsToGoData = dtData
                
                orderQuantityData = orderQuantityData.to(self.device)
                returnsToGoData = torch.zeros_like(returnsToGoData)
                returnsToGoData = returnsToGoData.to(self.device)
                td = {k: v.to(self.device) for k, v in problemData.items()}
                
                self.model.setInitalReturnToGo(td, returnsToGoData) 
                td = self.model.initModel(td)
                
                trajectoryLength = orderQuantityData.size(1)
                
                startPoint = random.randint(0, max(0, trajectoryLength-self.model.maxSeqLength))
                window_start = startPoint
                window_end = min(window_start + self.model.maxSeqLength, trajectoryLength)
                    
                self.model.train()
                cont = 0
                while cont < window_end:
                    end = min(cont + self.model.maxSeqLength, window_end)

                    td["currentTimestep"] = torch.zeros((orderQuantityData.size(0), 1), device=self.device)
                    
                    if cont >= window_start:
                        save_training_debug_input(td, orderQuantityData, currentStep, cont, window_start, window_end)
                    
                    td = self.model.forward(td, nextOrderQuantity=orderQuantityData[:, cont].unsqueeze(-1), is_test=False, update_only=cont<window_start)
                    cont += 1


                self.model.non_constructive_forward(td)
                predictedAction = td["predictedAction"]
                predictedOrderDecision = td["predictedOrderDecision"]
                realActions = orderQuantityData[:, window_start:window_end]
                
                realOrderDecision = (realActions > 0).float()
                
                decisionLoss = nn.BCEWithLogitsLoss()(predictedOrderDecision.squeeze(-1), realOrderDecision)
                
                predictedActionScaled = self.model._scale_field(predictedAction.squeeze(-1), "orderQuantity")
                realActionsScaled = self.model._scale_field(realActions, "orderQuantity")
                
                orderMask = (realActions > 0).float()
                quantityLoss = nn.MSELoss(reduction='none')(predictedActionScaled, realActionsScaled)
                quantityLoss = (quantityLoss * orderMask).sum() / (orderMask.sum() + 1e-8)
                
                loss = decisionLoss + quantityLoss

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                epochLoss += loss.detach().item()
                
                if currentStep in progress_checkpoints:
                    progress_pct = int((currentStep / self.stepsPerEpoch) * 100)
                    avg_loss_so_far = epochLoss / currentStep
                    avg_decision_loss = decisionLoss.detach().item()
                    avg_quantity_loss = quantityLoss.detach().item()
                    print(f"Epoch {epoch} - Progreso: {progress_pct}% | Loss total: {avg_loss_so_far:.6f} | Decision loss: {avg_decision_loss:.6f} | Quantity loss: {avg_quantity_loss:.6f}")
                
                currentStep += 1
            
            avgEpochLoss = epochLoss / self.stepsPerEpoch
            self.training_metrics['epoch_losses'].append(avgEpochLoss)
            
            self.lr_scheduler.step()
            
            test_benefit, real_benefit = self.evaluate_benefits()

            validation_loss, cost_metrics = 0, 0
            self.training_metrics['validation_losses'].append(validation_loss)
            self.training_metrics['validation_cost_metrics'].append(cost_metrics)
            
            self.content["EPOCHS"][epoch] = {
                "training_loss": avgEpochLoss,
                "validation_loss": validation_loss,
                "mean_test_total_cost": 0,
                "mean_real_total_cost": 0,
                "mean_cost_difference": 0,
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "context_windows": {
                    "trajectory_length": trajectoryLength,
                    "max_seq_length": self.model.maxSeqLength,
                    "num_windows": trajectoryLength - self.model.maxSeqLength,
                }
            }

            self.updateTrackFile()
            
            epoch_time = time.time() - epoch_start_time
            test_benefit_str = f" | Test benefit: {test_benefit:.2f}" if test_benefit is not None else ""
            real_benefit_str = f" | Real benefit: {real_benefit:.2f}" if real_benefit is not None else ""
            print(f"Epoch {epoch} completada - Tiempo: {epoch_time:.2f} segundos{test_benefit_str}{real_benefit_str}")
            
            self.currentEpoch = epoch
            self.saveModel()
            
            epoch += 1

    def calculate_trajectory_cost(self, td):
        """
        DEPRECATED: Esta función asume que onHandLevel es una secuencia temporal,
        pero ahora onHandLevel es un escalar que se actualiza en cada forward pass.
        Esta función ya no es compatible con la nueva estructura de datos.
        
        Calcula los costes de una trayectoria usando el estado actual del sistema.
        
        Args:
            td: TensorDict con el estado actual del sistema
        
        Returns:
            dict: Diccionario con los costes y métricas de la trayectoria
        """
        return {
            'holding_cost': 0.0,
            'ordering_cost': 0.0,
            'stockout_cost': 0.0,
            'sales_revenue': 0.0,
            'total_cost': 0.0,
            'avg_on_hand': 0.0,
            'avg_in_transit': 0.0,
            'total_orders': 0,
            'total_stockouts': 0
        }
                
if __name__ == "__main__":
    
    # Configuración básica
    config = TrainerConfig(
        nBatch=32,#32
        nVal=1000,  # Ajusta este valor según tus necesidades
        stepsPerEpoch=2000,
        trainStrategy=DTTrainingStrategy(dataPath=["data/training_data2.pt"]),
        lr_scheduler=1e-4
    )
    
    
    try:
        # Rutas de datos de entrenamiento
        data_paths = ["data/training_data2.pt"]
        
        # Calcular parámetros de escalado desde los datos de entrenamiento
        scaling_params = compute_scaling_params_from_training_data(data_paths)
        print(f"Parámetros de escalado calculados desde {len(data_paths)} archivo(s) de datos")
        
        # Crear el modelo con los parámetros de escalado
        model = DecisionTransformer(
            decisionTransformerConfig=DecisionTransformerConfig(),
            scaling_params=scaling_params)
        
        # Crear el trainer con AdamW y learning rate más alto
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        
        # Learning rate scheduler más conservador
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,  # Factor inicial (5e-4)
            end_factor=0.05,    # Factor final (2.5e-4 / 5e-4 = 0.5)
            total_iters=200    # Mucho menos agresivo
        )
        
        config = TrainerConfig(
            nBatch=64,
            nVal=1000, 
            stepsPerEpoch=64000//64,
            trainStrategy=DTTrainingStrategy(dataPath=data_paths),
            lr_scheduler=lr_scheduler,
            optimizer=optimizer
        )
        trainer = DecisionTransformerTrainer(
            savePath="./training_models/",  # Cambiado a training_models
            name="decision_transformer_model",  # Nombre más descriptivo
            model=model,
            trainerConfig=config
        )
        
        trainer.initTraining()
        
        
    except Exception as e:
        raise e
    
                
