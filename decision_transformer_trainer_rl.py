from torch import nn
from decision_transformer_rl import DecisionTransformer
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
from torch.distributions import Normal, Bernoulli

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
        
        self.ppo_clip = getattr(trainerConfig, 'ppo_clip', 0.2)
        self.ppo_epochs = getattr(trainerConfig, 'ppo_epochs', 2)
        self.entropy_coef = getattr(trainerConfig, 'entropy_coef', 0.01)

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
        with torch.no_grad():
            if self.testStrategy is not None:
                all_problem_data = self.testStrategy.problemData
                length_data = self.testStrategy.lengthData
            else:
                all_problem_data = self.trainStrategy.problemData
                length_data = self.trainStrategy.lengthData
            
            num_batches = length_data // batch
            
            # Preasignar tensors
            all_test_benefits = torch.zeros(num_batches * batch, device='cpu')
            all_real_benefits = torch.zeros(num_batches * batch, device='cpu')
            all_test_holding_costs = torch.zeros(num_batches * batch, device='cpu')
            all_real_holding_costs = torch.zeros(num_batches * batch, device='cpu')
            all_test_stockout_costs = torch.zeros(num_batches * batch, device='cpu')
            all_real_stockout_costs = torch.zeros(num_batches * batch, device='cpu')
            all_test_ordering_costs = torch.zeros(num_batches * batch, device='cpu')
            all_real_ordering_costs = torch.zeros(num_batches * batch, device='cpu')
            
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
                test_td = self.model.initModel(test_td)
                trajectory_length = test_td['demand'].size(1)

                for step in range(trajectory_length):
                    test_td = self.model.forward(test_td)
                
                # Escribir directamente en los tensors preasignados
                all_test_benefits[start_idx:end_idx] = test_td["benefit"][:, -1].cpu()
                all_real_benefits[start_idx:end_idx] = real_benefits.cpu()
                all_test_holding_costs[start_idx:end_idx] = test_td["cumulativeHoldingCost"][:, -1].cpu()
                all_real_holding_costs[start_idx:end_idx] = real_holding_costs.cpu()
                all_test_stockout_costs[start_idx:end_idx] = test_td["cumulativeStockOutCost"][:, -1].cpu()
                all_real_stockout_costs[start_idx:end_idx] = real_stockout_costs.cpu()
                all_test_ordering_costs[start_idx:end_idx] = test_td["cumulativeOrderingCost"][:, -1].cpu()
                all_real_ordering_costs[start_idx:end_idx] = real_ordering_costs.cpu()
        
        self.model.train()
        test_benefit_mean = all_test_benefits.mean().item()
        real_benefit_mean = all_real_benefits.mean().item()
        test_holding_cost_mean = all_test_holding_costs.mean().item()
        real_holding_cost_mean = all_real_holding_costs.mean().item()
        test_stockout_cost_mean = all_test_stockout_costs.mean().item()
        real_stockout_cost_mean = all_real_stockout_costs.mean().item()
        test_ordering_cost_mean = all_test_ordering_costs.mean().item()
        real_ordering_cost_mean = all_real_ordering_costs.mean().item()
        
        return (test_benefit_mean, real_benefit_mean, test_holding_cost_mean, real_holding_cost_mean,
                test_stockout_cost_mean, real_stockout_cost_mean, test_ordering_cost_mean, real_ordering_cost_mean)

    def compute_ppo_loss(self, old_log_probs, new_log_probs, advantages,
                         old_order_log_probs, new_order_log_probs,
                         old_quantity_log_probs, new_quantity_log_probs,
                         order_decisions, order_entropies, quantity_entropies):
        """
        Calcula la pérdida PPO usando las log_probs combinadas de ambas cabezas.
        Las log_probs ya vienen sumadas correctamente: log_prob = order_log_prob + order_decision * quantity_log_prob
        """ 
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages.unsqueeze(-1)
        surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * advantages.unsqueeze(-1)
        policy_loss = torch.min(surr1, surr2).mean()
         
        order_entropy = order_entropies.mean()
        quantity_entropy = quantity_entropies.mean()
        entropy = order_entropy + quantity_entropy
        
        loss = -policy_loss - self.entropy_coef * entropy
        
        return loss, policy_loss, entropy

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
            epoch_policy_loss = 0
            epoch_entropy = 0
            currentStep = 1
            progress_checkpoints = [int(self.stepsPerEpoch * p / 100) for p in range(10, 101, 10)]
            
            while currentStep <= self.stepsPerEpoch:
                dtData = self.trainStrategy.getTrainingData(self.nBatch)
                problemData, orderQuantityData, returnsToGoData = dtData
                
                returnsToGoData = returnsToGoData.to(self.device)
                td = {k: v.to(self.device) for k, v in problemData.items()}
                
                td = self.model.initModel(td)
                
                trajectoryLength = td['demand'].size(1)
                batch_size = td['onHandLevel'].size(0)
                
                # Preasignar tensors para todos los timesteps
                states_emb_buffer = torch.zeros(batch_size, trajectoryLength, td["statesEmbedding"].size(-1), 
                                               device=self.device, dtype=td["statesEmbedding"].dtype)
                actions_emb_buffer = torch.zeros(batch_size, trajectoryLength, td["actionsEmbedding"].size(-1), 
                                                device=self.device, dtype=td["actionsEmbedding"].dtype)
                old_log_probs = torch.zeros(batch_size, trajectoryLength, 1, device=self.device)
                old_order_log_probs = torch.zeros(batch_size, trajectoryLength, 1, device=self.device)
                old_quantity_log_probs = torch.zeros(batch_size, trajectoryLength, 1, device=self.device)
                order_decisions = torch.zeros(batch_size, trajectoryLength, 1, device=self.device)
                quantity_values = torch.zeros(batch_size, trajectoryLength, 1, device=self.device)
                benefits = torch.zeros(batch_size, trajectoryLength, device=self.device)
                
                self.model.train()
                
                for step in range(trajectoryLength):
                    td = self.model.forward(td)
                    
                    order_decision = td["orderDecision"].detach()
                    quantity_value = td["quantityValue"].detach()
                    
                    old_order_log_prob = td["orderDistribution"].log_prob(order_decision).detach()
                    old_quantity_log_prob = td["quantityDistribution"].log_prob(quantity_value).detach()
                    
                    # Escribir directamente en los buffers preasignados
                    states_emb_buffer[:, step, :] = td["statesEmbedding"].detach()
                    actions_emb_buffer[:, step, :] = td["actionsEmbedding"].detach()
                    old_log_probs[:, step, :] = td["actionLogProb"].detach()
                    old_order_log_probs[:, step, :] = old_order_log_prob
                    old_quantity_log_probs[:, step, :] = old_quantity_log_prob
                    quantity_values[:, step, :] = quantity_value
                    order_decisions[:, step, :] = order_decision
                    benefits[:, step] = td["benefit"][:, -1].squeeze(-1).detach()
                
                # Normalizar benefits para usarlos como advantages
                advantages = (benefits - benefits.mean()) / (benefits.std() + 1e-8)
                
                # Fase 2: PPO epochs - recalcular log_probs para cada timestep
                total_loss = 0
                total_policy_loss = 0
                total_entropy = 0
                
                # Preasignar tensors para nuevos log_probs
                new_log_probs = torch.zeros(batch_size, trajectoryLength, 1, device=self.device)
                new_order_log_probs = torch.zeros(batch_size, trajectoryLength, 1, device=self.device)
                new_quantity_log_probs = torch.zeros(batch_size, trajectoryLength, 1, device=self.device)
                order_entropies = torch.zeros(batch_size, trajectoryLength, 1, device=self.device)
                quantity_entropies = torch.zeros(batch_size, trajectoryLength, 1, device=self.device)
                
                for ppo_epoch in range(self.ppo_epochs):
                    # Recalcular log_probs para cada timestep usando embeddings guardados
                    for step in range(trajectoryLength):
                        # Obtener nuevas distribuciones usando embeddings de este timestep
                        order_dist, quantity_dist, order_logit = self.model.forward_from_embeddings(
                            states_emb_buffer[:, step, :], 
                            actions_emb_buffer[:, step, :]
                        )
                        
                        # Recalcular log_probs separados de la acción TOMADA en este timestep
                        new_order_log_prob = order_dist.log_prob(order_decisions[:, step, :])
                        new_quantity_log_prob = quantity_dist.log_prob(quantity_values[:, step, :])
                        new_log_prob = new_order_log_prob + order_decisions[:, step, :] * new_quantity_log_prob
                        
                        # Escribir directamente en los tensors preasignados
                        new_log_probs[:, step, :] = new_log_prob
                        new_order_log_probs[:, step, :] = new_order_log_prob
                        new_quantity_log_probs[:, step, :] = new_quantity_log_prob
                        order_entropies[:, step, :] = order_dist.entropy()
                        quantity_entropies[:, step, :] = quantity_dist.entropy()
                    
                    loss, policy_loss, entropy = self.compute_ppo_loss(
                        old_log_probs, new_log_probs, advantages,
                        old_order_log_probs, new_order_log_probs,
                        old_quantity_log_probs, new_quantity_log_probs,
                        order_decisions, order_entropies, quantity_entropies
                    )
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    total_loss += loss.detach().item()
                    total_policy_loss += policy_loss.detach().item()
                    total_entropy += entropy.detach().item()
                
                avg_loss = total_loss / self.ppo_epochs
                avg_policy_loss = total_policy_loss / self.ppo_epochs
                avg_entropy = total_entropy / self.ppo_epochs
                
                epochLoss += avg_loss
                epoch_policy_loss += avg_policy_loss
                epoch_entropy += avg_entropy
                
                if currentStep in progress_checkpoints:
                    progress_pct = int((currentStep / self.stepsPerEpoch) * 100)
                    avg_loss_so_far = epochLoss / currentStep
                    avg_policy_loss_so_far = epoch_policy_loss / currentStep
                    avg_entropy_so_far = epoch_entropy / currentStep
                    print(f"Epoch {epoch} - Progreso: {progress_pct}% | Loss: {avg_loss_so_far:.6f} | Policy loss: {avg_policy_loss_so_far:.6f} | Entropy: {avg_entropy_so_far:.6f}")
                
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
        trainStrategy=DTTrainingStrategy(dataPath=["data/training_data.pt"]),
        lr_scheduler=1e-4
    )
    
    
    try:
        # Rutas de datos de entrenamiento
        data_paths = ["data/training_data.pt"]
        
        # Calcular parámetros de escalado desde los datos de entrenamiento
        scaling_params = compute_scaling_params_from_training_data(data_paths)
        print(f"Parámetros de escalado calculados desde {len(data_paths)} archivo(s) de datos")
        
        # Crear el modelo con los parámetros de escalado
        model = DecisionTransformer(
            decisionTransformerConfig=DecisionTransformerConfig(
                hidden_size=96,
                n_head=2,
            ),
            scaling_params=scaling_params)
        
        # Crear el trainer con AdamW y learning rate más alto
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        
        # Learning rate scheduler más conservador
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,  # Factor inicial (5e-4)
            end_factor=0.01,    # Factor final (2.5e-4 / 5e-4 = 0.5)
            total_iters=50    # Mucho menos agresivo
        )
        
        config = TrainerConfig(
            nBatch=16,
            nVal=1000, 
            stepsPerEpoch=100000//128 * 3,
            trainStrategy=DTTrainingStrategy(dataPath=data_paths),
            lr_scheduler=lr_scheduler,
            optimizer=optimizer,
            testDataPath=["data/test_data.pt"],
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
    
                
