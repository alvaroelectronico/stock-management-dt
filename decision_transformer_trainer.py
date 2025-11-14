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
        self.testStrategy = None
        if hasattr(trainerConfig, 'testDataPath') and trainerConfig.testDataPath is not None:
            self.testStrategy = DTTrainingStrategy(dataPath=trainerConfig.testDataPath, shuffle=False)
        self.best_test_benefit = float('-inf')

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
    
    def saveBestModel(self):
        """
        Guarda el mejor modelo basado en el beneficio de test en best.pt.
        """
        torch.save({'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'lr_scheduler_state': self.lr_scheduler.state_dict(),
                    'start_epochs': self.currentEpoch,
                    'best_test_benefit': self.best_test_benefit,
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else 0,
                    },
                   self.baselineSavePath,
                   )
    
    def loadBestModel(self):
        """
        Carga el mejor beneficio de test desde best.pt si existe.
        """
        if os.path.isfile(self.baselineSavePath):
            try:
                checkpoint = torch.load(self.baselineSavePath)
                if 'best_test_benefit' in checkpoint:
                    self.best_test_benefit = checkpoint['best_test_benefit']
                    print(f"Mejor beneficio de test cargado: {self.best_test_benefit:.2f}")
            except Exception as e:
                print(f"Advertencia: No se pudo cargar el mejor beneficio desde best.pt: {e}")
    
    def initTraining(self):
        """
        Inicializa el entrenamiento y carga el mejor beneficio de test si existe.
        """ 
        self.loadBestModel() 
        super().initTraining()

    def evaluate_benefits(self):
        """
        Evalúa el modelo en datos de test y calcula beneficios y costes acumulados finales medios.
        
        Returns:
            tuple: (test_benefit_mean, real_benefit_mean, test_holding_cost_mean, real_holding_cost_mean,
                   test_stockout_cost_mean, real_stockout_cost_mean, test_ordering_cost_mean, real_ordering_cost_mean)
        """
        batch = self.nBatch

        self.model.eval()
        all_test_benefits = []
        all_real_benefits = []
        all_test_holding_costs = []
        all_real_holding_costs = []
        all_test_stockout_costs = []
        all_real_stockout_costs = []
        all_test_ordering_costs = []
        all_real_ordering_costs = []
        with torch.no_grad():
            if self.testStrategy is not None:
                all_problem_data = self.testStrategy.problemData
                length_data = self.testStrategy.lengthData
            else:
                all_problem_data = self.trainStrategy.problemData
                length_data = self.trainStrategy.lengthData
            
            num_batches = length_data // batch
            for i in range(num_batches):
                start_idx = i * batch
                end_idx = min(start_idx + batch, length_data)
                batch_indices = torch.arange(start_idx, end_idx)
                test_problem = all_problem_data[batch_indices]
                
                if test_problem["benefit"].dim() > 1:
                    real_benefits = test_problem["benefit"][:, -1]
                else:
                    real_benefits = test_problem["benefit"][-1].unsqueeze(0)
                
                if test_problem["cumulativeHoldingCost"].dim() > 1:
                    real_holding_costs = test_problem["cumulativeHoldingCost"][:, -1]
                else:
                    real_holding_costs = test_problem["cumulativeHoldingCost"][-1].unsqueeze(0)
                
                if test_problem["cumulativeStockOutCost"].dim() > 1:
                    real_stockout_costs = test_problem["cumulativeStockOutCost"][:, -1]
                else:
                    real_stockout_costs = test_problem["cumulativeStockOutCost"][-1].unsqueeze(0)
                
                if test_problem["cumulativeOrderingCost"].dim() > 1:
                    real_ordering_costs = test_problem["cumulativeOrderingCost"][:, -1]
                else:
                    real_ordering_costs = test_problem["cumulativeOrderingCost"][-1].unsqueeze(0)
                
                test_td = {k: v.clone() for k, v in test_problem.items()}
                test_td = {k: v.to(self.device) for k, v in test_td.items()}
                test_td["returnsToGo"] = torch.zeros((batch), device=self.device)
                test_td = self.model.initModel(test_td)
                trajectory_length = test_td['demand'].size(1)

                for step in range(trajectory_length):
                    test_td = self.model.forward(test_td, nextOrderQuantity=None, is_test=True, update_only=False)
                
                all_test_benefits.append(test_td["benefit"][:, -1].cpu())
                all_real_benefits.append(real_benefits.cpu())
                all_test_holding_costs.append(test_td["cumulativeHoldingCost"][:, -1].cpu())
                all_real_holding_costs.append(real_holding_costs.cpu())
                all_test_stockout_costs.append(test_td["cumulativeStockOutCost"][:, -1].cpu())
                all_real_stockout_costs.append(real_stockout_costs.cpu())
                all_test_ordering_costs.append(test_td["cumulativeOrderingCost"][:, -1].cpu())
                all_real_ordering_costs.append(real_ordering_costs.cpu())
        
        self.model.train()
        test_benefit_mean = torch.cat(all_test_benefits).mean().item()
        real_benefit_mean = torch.cat(all_real_benefits).mean().item()
        test_holding_cost_mean = torch.cat(all_test_holding_costs).mean().item()
        real_holding_cost_mean = torch.cat(all_real_holding_costs).mean().item()
        test_stockout_cost_mean = torch.cat(all_test_stockout_costs).mean().item()
        real_stockout_cost_mean = torch.cat(all_real_stockout_costs).mean().item()
        test_ordering_cost_mean = torch.cat(all_test_ordering_costs).mean().item()
        real_ordering_cost_mean = torch.cat(all_real_ordering_costs).mean().item()
        
        return (test_benefit_mean, real_benefit_mean, test_holding_cost_mean, real_holding_cost_mean,
                test_stockout_cost_mean, real_stockout_cost_mean, test_ordering_cost_mean, real_ordering_cost_mean)

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

                if self.trainerConfig.use_bfloat16 and self.device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
                else:
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
            
            (test_benefit, real_benefit, test_holding_cost, real_holding_cost,
             test_stockout_cost, real_stockout_cost, test_ordering_cost, real_ordering_cost) = self.evaluate_benefits()
 
            if test_benefit > self.best_test_benefit:
                self.best_test_benefit = test_benefit
                self.saveBestModel()
                print(f"¡Nuevo mejor modelo guardado! Beneficio de test: {test_benefit:.2f}")

            validation_loss, cost_metrics = 0, 0
            self.training_metrics['validation_losses'].append(validation_loss)
            self.training_metrics['validation_cost_metrics'].append(cost_metrics)
            
            self.content["EPOCHS"][epoch] = {
                "training_loss": avgEpochLoss,
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "mean_test_benefit": f"({test_benefit:.2f}/{real_benefit:.2f})",
                "mean_test_holding_cost": f"({test_holding_cost:.2f}/{real_holding_cost:.2f})",
                "mean_test_stockout_cost": f"({test_stockout_cost:.2f}/{real_stockout_cost:.2f})",
                "mean_test_ordering_cost": f"({test_ordering_cost:.2f}/{real_ordering_cost:.2f})",
                "best_test_benefit": self.best_test_benefit
            }

            self.updateTrackFile()
            
            epoch_time = time.time() - epoch_start_time
            benefit_str = f" | Benefit: ({test_benefit:.2f}/{real_benefit:.2f})" if test_benefit is not None and real_benefit is not None else ""
            holding_cost_str = f" | Holding cost: ({test_holding_cost:.2f}/{real_holding_cost:.2f})" if test_holding_cost is not None and real_holding_cost is not None else ""
            stockout_cost_str = f" | Stockout cost: ({test_stockout_cost:.2f}/{real_stockout_cost:.2f})" if test_stockout_cost is not None and real_stockout_cost is not None else ""
            ordering_cost_str = f" | Ordering cost: ({test_ordering_cost:.2f}/{real_ordering_cost:.2f})" if test_ordering_cost is not None and real_ordering_cost is not None else ""
            best_benefit_str = f" | Mejor beneficio: {self.best_test_benefit:.2f}"
            print(f"Epoch {epoch} completada - Tiempo: {epoch_time:.2f} segundos{benefit_str}{holding_cost_str}{stockout_cost_str}{ordering_cost_str}{best_benefit_str}")
            
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
            nBatch=32,
            nVal=1000, 
            stepsPerEpoch=2,
            trainStrategy=DTTrainingStrategy(dataPath=data_paths),
            lr_scheduler=lr_scheduler,
            optimizer=optimizer,
            testDataPath=["data/test_data.pt"]
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
    
                
