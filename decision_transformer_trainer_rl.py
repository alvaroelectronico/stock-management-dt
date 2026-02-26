from torch import nn
from decision_transformer_rl import DecisionTransformer
from trainer import Trainer, TrainerConfig, getProjectDirectory
import torch
import torch.nn.functional as F
from decision_transformer_strategies import TrainingStrategy
from decision_transformer_config import DecisionTransformerConfig
from decision_transformer_strategies import DTTrainingStrategy
import decision_transformer_strategies
import numpy as np
from generate_trajectories import TRAJECTORY_LENGTH, MAX_LEAD_TIME, FORECAST_LENGTH, MAX_LEAD_TIME, FORECAST_LENGTH
import copy
from tensordict import TensorDict
import os
import json
from datetime import date
import random
import time
import signal
import sys
from data_scaler import compute_scaling_params_from_training_data
from torch.distributions import Normal, Bernoulli
import math

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
        self.save_rl_trajectories = getattr(trainerConfig, 'save_rl_trajectories', False)
        self.rl_trajectories_per_problem = {}
         
        self.save_rtg_min = getattr(trainerConfig, 'save_rtg_min', True)  # Guardar RTG=0 (peor)
        self.save_rtg_max = getattr(trainerConfig, 'save_rtg_max', True)  # Guardar RTG=1 (mejor) 
        self.rtg_intervals = getattr(trainerConfig, 'rtg_intervals', [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)])
        
        self.ppo_clip = getattr(trainerConfig, 'ppo_clip', 0.2)
        self.ppo_epochs = getattr(trainerConfig, 'ppo_epochs', 1)
        self.entropy_coef = getattr(trainerConfig, 'entropy_coef', 0.001)
        self.value_coef = getattr(trainerConfig, 'value_coef', 0.5)
        self.gamma = getattr(trainerConfig, 'gamma', 1)

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
        Inicializa el entrenamiento RL.
        """ 
        self.loadBestModel()

        trackPath = self.trackPath

        if not os.path.isfile(trackPath):
            self.content = {
                "NUMBER PARAMETERS": sum(t.numel() for t in self.model.parameters()),
                "MODEL INFO": self.getFilteredModelInfo(),
                "TRAINING INFO": self.trainerConfig.to_dict(),
                "EPOCHS": {},
            }

            self.updateTrackFile()
        else:
            self.content = self.JSONtoDict(trackPath)
            if "EPOCHS" not in self.content:
                self.content["EPOCHS"] = {}

        checkpoint, self.optimizer, self.lr_scheduler = self.loadModelFromFile()
        self.strategy = self.getStrategy()

        self.currentEpoch = 0
        self.bestAverageReward = 0
        if checkpoint is not None:
            self.currentEpoch = checkpoint["start_epochs"]

        self.model = self.model.to(self.device)

        self.train()

    def _store_rl_trajectories_for_batch(self, batch_indices, test_td, trajectory_length, mark_as_initial):
        """
        Construye y guarda trayectorias RL por problema a partir de un batch de evaluación.
        Guarda en el mismo formato que generate_trajectories.py.
        """
        batch_size = test_td["onHandLevel"].size(0)

        # Guardar secuencias temporales durante la ejecución.
        saved_on_hand = []
        saved_in_transit = []
        saved_demand = []
        saved_forecast = []
        saved_actions = []
        saved_benefit = []
        saved_cumulative_sales = []
        saved_cumulative_holding = []
        saved_cumulative_ordering = []
        saved_cumulative_stockout = []

        # Guardar estado inicial.
        saved_on_hand.append(test_td["onHandLevel"].detach().cpu().clone())
        saved_in_transit.append(test_td["inTransitStock"].detach().cpu().clone())
        saved_demand.append(test_td["demand"][:, 0].detach().cpu().clone())
        saved_forecast.append(test_td["forecast"][:, 0, :].detach().cpu().clone())

        # Ejecutar la trayectoria completa del batch guardando datos en cada paso.
        for step in range(trajectory_length):
            test_td = self.model.forward(test_td)
            
            # Guardar datos después de cada paso.
            saved_on_hand.append(test_td["onHandLevel"].detach().cpu().clone())
            saved_in_transit.append(test_td["inTransitStock"].detach().cpu().clone())
            saved_demand.append(test_td["demand"][:, 0].detach().cpu().clone())
            saved_forecast.append(test_td["forecast"][:, 0, :].detach().cpu().clone())
            
            # Extraer acciones y métricas.
            action = test_td["orderQuantity"].squeeze(-1).detach().cpu().clone()
            saved_actions.append(action)
            
            benefit_val = test_td["benefit"][:, -1]
            if benefit_val.dim() > 1:
                benefit_val = benefit_val.squeeze(-1)
            saved_benefit.append(benefit_val.detach().cpu().clone())
            
            saved_cumulative_sales.append(test_td["cumulativeSales"][:, -1].detach().cpu().clone())
            
            holding_val = test_td["cumulativeHoldingCost"][:, -1]
            if holding_val.dim() > 1:
                holding_val = holding_val.squeeze(-1)
            saved_cumulative_holding.append(holding_val.detach().cpu().clone())
            
            ordering_val = test_td["cumulativeOrderingCost"][:, -1]
            if ordering_val.dim() > 1:
                ordering_val = ordering_val.squeeze(-1)
            saved_cumulative_ordering.append(ordering_val.detach().cpu().clone())
            
            stockout_val = test_td["cumulativeStockOutCost"][:, -1]
            if stockout_val.dim() > 1:
                stockout_val = stockout_val.squeeze(-1)
            saved_cumulative_stockout.append(stockout_val.detach().cpu().clone())

        # Extraer beneficios finales.
        final_benefits = saved_benefit[-1]

        # Construir trayectorias en formato generate_trajectories.py.
        for local_idx in range(batch_size):
            problem_idx = int(batch_indices[local_idx].item())

            # Función para hacer padding de inTransitStock.
            def add_padding_to_transit_stock(in_transit):
                padded = torch.zeros(MAX_LEAD_TIME, dtype=in_transit.dtype)
                actual_len = min(len(in_transit), MAX_LEAD_TIME)
                padded[:actual_len] = in_transit[:actual_len]
                return padded

            # Construir secuencias temporales (excluyendo el estado inicial extra).
            in_transit_seq = torch.stack([add_padding_to_transit_stock(saved_in_transit[i][local_idx]) 
                                         for i in range(1, len(saved_in_transit))])
            demand_seq = torch.stack([saved_demand[i][local_idx] for i in range(1, len(saved_demand))])
            forecast_seq = torch.stack([saved_forecast[i][local_idx] for i in range(1, len(saved_forecast))])
            actions_seq = torch.stack([saved_actions[i][local_idx] for i in range(len(saved_actions))])
            benefit_seq = torch.stack([saved_benefit[i][local_idx] for i in range(len(saved_benefit))])
            cumulative_sales_seq = torch.stack([saved_cumulative_sales[i][local_idx] for i in range(len(saved_cumulative_sales))])
            cumulative_holding_seq = torch.stack([saved_cumulative_holding[i][local_idx] for i in range(len(saved_cumulative_holding))])
            cumulative_ordering_seq = torch.stack([saved_cumulative_ordering[i][local_idx] for i in range(len(saved_cumulative_ordering))])
            cumulative_stockout_seq = torch.stack([saved_cumulative_stockout[i][local_idx] for i in range(len(saved_cumulative_stockout))])
            times_seq = torch.arange(trajectory_length, dtype=torch.float32)

            # Escalares del primer estado.
            first_on_hand = saved_on_hand[0][local_idx].item()
            lead_time = test_td["leadTime"][local_idx].detach().cpu().item()
            holding_cost = test_td["holdingCost"][local_idx].detach().cpu().item()
            ordering_cost = test_td["orderingCost"][local_idx].detach().cpu().item()
            stockout_penalty = test_td["stockOutPenalty"][local_idx].detach().cpu().item()
            unit_revenue = test_td["unitRevenue"][local_idx].detach().cpu().item()

            # Construir TensorDict en formato generate_trajectories.py.
            states_td = TensorDict({
                "onHandLevel": torch.tensor(first_on_hand, dtype=torch.float32),
                "inTransitStock": in_transit_seq,
                "demand": demand_seq,
                "forecast": forecast_seq,
                "leadTime": torch.tensor(lead_time, dtype=torch.float32),
                "holdingCost": torch.tensor(holding_cost, dtype=torch.float32),
                "orderingCost": torch.tensor(ordering_cost, dtype=torch.float32),
                "stockOutPenalty": torch.tensor(stockout_penalty, dtype=torch.float32),
                "unitRevenue": torch.tensor(unit_revenue, dtype=torch.float32),
                "timesStep": times_seq,
                "benefit": benefit_seq,
                "cumulativeSales": cumulative_sales_seq,
                "cumulativeHoldingCost": cumulative_holding_seq,
                "cumulativeOrderingCost": cumulative_ordering_seq,
                "cumulativeStockOutCost": cumulative_stockout_seq,
            }, batch_size=[])

            trajectory_td = TensorDict({
                "states": states_td,
                "actions": actions_seq,
                "returnsToGo": torch.tensor([0.0], dtype=torch.float32),  # Se sobrescribirá después.
            }, batch_size=[])

            if problem_idx not in self.rl_trajectories_per_problem:
                self.rl_trajectories_per_problem[problem_idx] = []

            self.rl_trajectories_per_problem[problem_idx].append(
                {
                    "trajectory": trajectory_td,
                    "final_benefit": float(final_benefits[local_idx].item()),
                    "is_initial": bool(mark_as_initial),
                }
            )

    def _select_and_save_rl_trajectories(self):
        """
        Selecciona trayectorias por problema según RTG normalizado y las guarda a disco.
        La selección se basa en la configuración: save_rtg_min, save_rtg_max y rtg_intervals.
        """
        if not self.rl_trajectories_per_problem:
            return

        training_data = None
        eps = 1e-8

        for problem_idx in sorted(self.rl_trajectories_per_problem.keys()):
            runs = self.rl_trajectories_per_problem[problem_idx]
            if not runs:
                continue

            benefits = torch.tensor([r["final_benefit"] for r in runs], dtype=torch.float32)
            min_benefit = benefits.min().item()
            max_benefit = benefits.max().item()

            if max_benefit - min_benefit < eps:
                rtg_values = torch.full_like(benefits, 0.5)
            else:
                rtg_values = (benefits - min_benefit) / (max_benefit - min_benefit)

            selected_indices = set()
            idx_min = None
            idx_max = None
            
            # Guardar RTG=0 (peor beneficio) si está configurado
            if self.save_rtg_min:
                initial_indices = [i for i, r in enumerate(runs) if r.get("is_initial", False)]
                if initial_indices:
                    idx_min = initial_indices[0]
                else:
                    idx_min = int(torch.argmin(benefits).item())
                selected_indices.add(idx_min)
            
            # Guardar RTG=1 (mejor beneficio) si está configurado
            if self.save_rtg_max:
                idx_max = int(torch.argmax(benefits).item())
                selected_indices.add(idx_max)

            # Guardar trayectorias de intervalos intermedios si están configurados
            if self.rtg_intervals:
                for low, high in self.rtg_intervals:
                    # Excluir exactamente 0 y 1 para los intervalos.
                    mask = (rtg_values > low + eps) & (rtg_values <= high + eps) & (rtg_values < 1.0 - eps)
                    candidates = torch.nonzero(mask, as_tuple=False).view(-1).tolist()
                    candidates = [c for c in candidates if c not in selected_indices]

                    if candidates:
                        chosen = random.choice(candidates)
                        selected_indices.add(chosen)
            
            for idx in selected_indices:
                # Asignar RTG=0 si es el índice seleccionado para el mínimo
                if idx_min is not None and idx == idx_min:
                    rtg_initial = 0.0
                # Asignar RTG=1 si es el índice seleccionado para el máximo
                elif idx_max is not None and idx == idx_max:
                    rtg_initial = 1.0
                # Para el resto, usar el RTG normalizado calculado
                else:
                    rtg_initial = float(rtg_values[idx].item())
                
                traj_td = runs[idx]["trajectory"].clone()
                traj_td["returnsToGo"] = torch.tensor([rtg_initial], dtype=torch.float32)

                if training_data is None:
                    training_data = traj_td.unsqueeze(0)
                else:
                    training_data = torch.cat([training_data, traj_td.unsqueeze(0)], dim=0)

        if training_data is None:
            return

        # Guardar en el mismo formato que generate_trajectories.py.
        project_dir = getProjectDirectory()
        data_dir = os.path.join(project_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        output_path = os.path.join(data_dir, "rl_solution_trajectories.pt")
        torch.save(training_data, output_path)

    def _run_batch_trajectory(self, test_td, trajectory_length):
        """
        Ejecuta la trayectoria completa para un batch de problemas.

        Args:
            test_td: TensorDict inicializado con el estado del batch.
            trajectory_length: Número de pasos de la trayectoria.

        Returns:
            TensorDict actualizado tras ejecutar todos los pasos.
        """
        for _ in range(trajectory_length):
            test_td = self.model.forward(test_td)
        return test_td

    def _extract_real_metrics(self, test_problem):
        """
        Extrae las métricas reales finales de un batch de problemas.

        Args:
            test_problem: TensorDict con los datos originales del problema.

        Returns:
            Tuple con (real_benefits, real_holding, real_stockout, real_ordering).
        """
        # Extraer beneficio real final.
        if test_problem["benefit"].dim() > 1:
            real_benefits = test_problem["benefit"][:, -1]
        else:
            real_benefits = test_problem["benefit"][-1].unsqueeze(0)

        # Extraer coste de holding real final.
        if test_problem["cumulativeHoldingCost"].dim() > 1:
            real_holding_costs = test_problem["cumulativeHoldingCost"][:, -1]
        else:
            real_holding_costs = test_problem["cumulativeHoldingCost"][-1].unsqueeze(0)

        # Extraer coste de stockout real final.
        if test_problem["cumulativeStockOutCost"].dim() > 1:
            real_stockout_costs = test_problem["cumulativeStockOutCost"][:, -1]
        else:
            real_stockout_costs = test_problem["cumulativeStockOutCost"][-1].unsqueeze(0)

        # Extraer coste de ordering real final.
        if test_problem["cumulativeOrderingCost"].dim() > 1:
            real_ordering_costs = test_problem["cumulativeOrderingCost"][:, -1]
        else:
            real_ordering_costs = test_problem["cumulativeOrderingCost"][-1].unsqueeze(0)

        return real_benefits, real_holding_costs, real_stockout_costs, real_ordering_costs

    def evaluate_benefits(self, store_trajectories=False, mark_as_initial=False):
        """
        Evalúa el modelo en datos de test y calcula beneficios y costes acumulados finales medios.

        Args:
            store_trajectories: Si True, guarda las trayectorias completas en memoria.
            mark_as_initial: Si True, marca las trayectorias como iniciales.

        Returns:
            Tuple con las medias de (test_benefit, real_benefit, test_holding, real_holding,
                                     test_stockout, real_stockout, test_ordering, real_ordering).
        """
        batch_size = self.nBatch

        self.model.eval()
        with torch.no_grad():
            if self.testStrategy is not None:
                all_problem_data = self.testStrategy.problemData
                length_data = self.testStrategy.lengthData
            else:
                all_problem_data = self.trainStrategy.problemData
                length_data = self.trainStrategy.lengthData

            test_benefits_list = []
            real_benefits_list = []
            test_holding_costs_list = []
            real_holding_costs_list = []
            test_stockout_costs_list = []
            real_stockout_costs_list = []
            test_ordering_costs_list = []
            real_ordering_costs_list = []

            # Iterar sobre los datos en batches, incluyendo el último batch parcial.
            for start_idx in range(0, length_data, batch_size):
                end_idx = min(start_idx + batch_size, length_data)
                current_batch_size = end_idx - start_idx
                batch_indices = torch.arange(start_idx, end_idx)
                test_problem = all_problem_data[batch_indices]

                # Extraer métricas reales del batch actual.
                real_benefits, real_holding_costs, real_stockout_costs, real_ordering_costs = \
                    self._extract_real_metrics(test_problem)

                # Preparar TensorDict para el batch actual.
                test_td = {k: v.clone() for k, v in test_problem.items()}
                test_td = {k: v.to(self.device) for k, v in test_td.items()}
                test_td = self.model.initModel(test_td)
                trajectory_length = test_td['demand'].size(1)

                if store_trajectories:
                    # Guardar trayectorias completas en memoria (ejecuta forward internamente).
                    self._store_rl_trajectories_for_batch(
                        batch_indices=batch_indices,
                        test_td=test_td,
                        trajectory_length=trajectory_length,
                        mark_as_initial=mark_as_initial,
                    )
                else:
                    # Ejecutar trayectoria completa para el batch.
                    test_td = self._run_batch_trajectory(test_td, trajectory_length)

                # Recopilar métricas del batch (funciona para batches completos y parciales).
                test_benefits_list.append(test_td["benefit"][:current_batch_size, -1].cpu())
                real_benefits_list.append(real_benefits[:current_batch_size].cpu())
                test_holding_costs_list.append(test_td["cumulativeHoldingCost"][:current_batch_size, -1].cpu())
                real_holding_costs_list.append(real_holding_costs[:current_batch_size].cpu())
                test_stockout_costs_list.append(test_td["cumulativeStockOutCost"][:current_batch_size, -1].cpu())
                real_stockout_costs_list.append(real_stockout_costs[:current_batch_size].cpu())
                test_ordering_costs_list.append(test_td["cumulativeOrderingCost"][:current_batch_size, -1].cpu())
                real_ordering_costs_list.append(real_ordering_costs[:current_batch_size].cpu())

        self.model.train()

        if len(test_benefits_list) == 0:
            print(f"No hay datos de test para evaluar beneficios (length_data={length_data}, batch={batch_size}).")
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Concatenar todos los resultados de los batches.
        all_test_benefits = torch.cat(test_benefits_list)
        all_real_benefits = torch.cat(real_benefits_list)
        all_test_holding_costs = torch.cat(test_holding_costs_list)
        all_real_holding_costs = torch.cat(real_holding_costs_list)
        all_test_stockout_costs = torch.cat(test_stockout_costs_list)
        all_real_stockout_costs = torch.cat(real_stockout_costs_list)
        all_test_ordering_costs = torch.cat(test_ordering_costs_list)
        all_real_ordering_costs = torch.cat(real_ordering_costs_list)

        # Calcular medias.
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

    def compute_ppo_loss(
        self,
        old_log_probs,
        new_log_probs,
        advantages,
        old_order_log_probs,
        new_order_log_probs,
        old_quantity_log_probs,
        new_quantity_log_probs,
        order_decisions,
        order_entropies,
        quantity_entropies,
        new_values,
        returns,
    ):
        """
        Calcula la pérdida PPO con ventajas por paso y pérdida de valor escalar.
        
        Args:
            old_log_probs: Log-probs antiguas de las acciones
            new_log_probs: Log-probs nuevas de las mismas acciones
            advantages: Ventajas por paso ya calculadas
            old_order_log_probs: Log-probs antiguas de la cabeza de decisión de pedido
            new_order_log_probs: Log-probs nuevas de la cabeza de decisión de pedido
            old_quantity_log_probs: Log-probs antiguas de la cabeza de cantidad
            new_quantity_log_probs: Log-probs nuevas de la cabeza de cantidad
            order_decisions: Decisiones binarias de pedido
            order_entropies: Entropías de la cabeza de decisión de pedido
            quantity_entropies: Entropías de la cabeza de cantidad
            new_values: Predicciones de valor por paso
            returns: Retornos acumulados por paso usados como objetivo de valor
        
        Returns:
            tuple: (loss_total, policy_loss, entropy, value_loss)
        """
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * advantages
        policy_loss = torch.min(surr1, surr2).mean()

        mask = order_decisions.float()
        order_entropy = order_entropies.mean()
        quantity_entropy = (quantity_entropies * mask).sum() / (mask.sum() + 1e-8)
        entropy = order_entropy + quantity_entropy

        value_loss = F.mse_loss(new_values, returns)

        loss = -policy_loss - self.entropy_coef * entropy + self.value_coef * value_loss

        return loss, policy_loss, entropy, value_loss

    def train(self):
        """
        Ejecuta el bucle principal de entrenamiento PPO sobre trayectorias generadas online.
        """
        epoch = getattr(self, 'currentEpoch', -1) + 1

        self.training_metrics = {
            'epoch_losses': [],
            'validation_losses': [],
            'validation_cost_metrics': []
        }
        
        try:
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
                    
                    old_log_probs = torch.zeros(batch_size, trajectoryLength, 1, device=self.device)
                    old_order_log_probs = torch.zeros(batch_size, trajectoryLength, 1, device=self.device)
                    old_quantity_log_probs = torch.zeros(batch_size, trajectoryLength, 1, device=self.device)
                    order_decisions = torch.zeros(batch_size, trajectoryLength, 1, device=self.device)
                    quantity_values = torch.zeros(batch_size, trajectoryLength, 1, device=self.device)
                    
                    self.model.train()
                    with torch.no_grad():
                        for step in range(trajectoryLength):
                            td = self.model.forward(td)

                            order_decision = td["orderDecision"]
                            quantity_value = td["quantityValue"]

                            old_order_log_prob = td["orderDistribution"].log_prob(order_decision)
                            old_quantity_log_prob = td["quantityDistribution"].log_prob(quantity_value)

                            old_log_probs[:, step, :] = td["actionLogProb"]
                            old_order_log_probs[:, step, :] = old_order_log_prob
                            old_quantity_log_probs[:, step, :] = old_quantity_log_prob
                            quantity_values[:, step, :] = quantity_value
                            order_decisions[:, step, :] = order_decision

                    rewards = td["returns"].clone()
                    gamma = self.gamma
                    T = trajectoryLength
                    time_idx = torch.arange(T, device=self.device, dtype=rewards.dtype)
                    power = time_idx.unsqueeze(1) - time_idx.unsqueeze(0)
                    discount_matrix = torch.tril(torch.pow(gamma, power))
                    rewards_2d = rewards.squeeze(-1)
                    returns_2d = rewards_2d @ discount_matrix
                    returns = returns_2d.unsqueeze(-1) / 40000
                    values = td["values"]
                    if values.dim() == 2:
                        values = values.unsqueeze(-1)
                    values = values
                    advantages = (returns - values).detach()
                    advantages_mean = advantages.mean()
                    advantages_std = advantages.std() + 1e-8
                    advantages = (advantages - advantages_mean) / advantages_std
                    returns = returns.detach()
                    
                    # Fase 2: PPO epochs - recalcular log_probs para cada timestep
                    total_loss = 0
                    total_policy_loss = 0
                    total_entropy = 0
                    total_value_loss = 0
                    self.model.train()
                    for ppo_epoch in range(self.ppo_epochs):
                        new_log_probs = torch.zeros(batch_size, trajectoryLength, 1, device=self.device)
                        new_order_log_probs = torch.zeros(batch_size, trajectoryLength, 1, device=self.device)
                        new_quantity_log_probs = torch.zeros(batch_size, trajectoryLength, 1, device=self.device)
                        order_entropies = torch.zeros(batch_size, trajectoryLength, 1, device=self.device)
                        quantity_entropies = torch.zeros(batch_size, trajectoryLength, 1, device=self.device)
                        new_values = torch.zeros(batch_size, trajectoryLength, 1, device=self.device)

                        # Recalcular distribuciones y valores para toda la trayectoria con gradiente
                        order_dist_seq, quantity_dist_seq, values_seq = self.model.non_constructive_forward(td)

                        new_order_log_probs = order_dist_seq.log_prob(order_decisions)
                        new_quantity_log_probs = quantity_dist_seq.log_prob(quantity_values)
                        new_log_probs = new_order_log_probs + order_decisions * new_quantity_log_probs

                        order_entropies = order_dist_seq.entropy()
                        quantity_entropies = quantity_dist_seq.entropy()
                        new_values = values_seq
                        
                        loss, policy_loss, entropy, value_loss = self.compute_ppo_loss(
                            old_log_probs, new_log_probs, advantages,
                            old_order_log_probs, new_order_log_probs,
                            old_quantity_log_probs, new_quantity_log_probs,
                            order_decisions, order_entropies, quantity_entropies,
                            new_values, returns
                        )
                        
                        loss.backward()
                        grad=torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        
                        total_loss += loss.detach().item()
                        total_policy_loss += policy_loss.detach().item()
                        total_entropy += entropy.detach().item()
                        total_value_loss += value_loss.detach().item()
                    
                    avg_loss = total_loss / self.ppo_epochs
                    avg_policy_loss = total_policy_loss / self.ppo_epochs
                    avg_entropy = total_entropy / self.ppo_epochs
                    avg_value_loss = total_value_loss / self.ppo_epochs
                    
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
                 test_stockout_cost, real_stockout_cost, test_ordering_cost, real_ordering_cost) = self.evaluate_benefits(
                    store_trajectories=self.save_rl_trajectories,
                    mark_as_initial=False,
                )
     
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
        except KeyboardInterrupt:
            print("\n\nEntrenamiento interrumpido por el usuario (Ctrl+C)")
            if self.save_rl_trajectories and self.rl_trajectories_per_problem:
                print("Guardando trayectorias RL acumuladas...")
                self._select_and_save_rl_trajectories()
                print("Trayectorias guardadas exitosamente.")
            raise

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
    # Rutas de datos de entrenamiento
    data_paths = ["data/training_data.pt"]
    
    # Calcular parámetros de escalado desde los datos de entrenamiento
    scaling_params = compute_scaling_params_from_training_data(data_paths)
    print(f"Parámetros de escalado calculados desde {len(data_paths)} archivo(s) de datos")
    
    # Crear el modelo con los parámetros de escalado
    model = DecisionTransformer(
        decisionTransformerConfig=DecisionTransformerConfig(
            hidden_size=32,
            n_head=1,
        ),
        scaling_params=scaling_params)
    
    # Crear el trainer con AdamW y learning rate más alto
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    
    # Learning rate scheduler más conservador
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,  # Factor inicial (5e-4)
        end_factor=1e-5/2e-4,    # Factor final (2.5e-4 / 5e-4 = 0.5)
        total_iters=150    # Mucho menos agresivo
    )
    
    config = TrainerConfig(
        nBatch=64,
        nVal=1000, 
        stepsPerEpoch=500,
        trainStrategy=DTTrainingStrategy(dataPath=data_paths),
        lr_scheduler=lr_scheduler,
        optimizer=optimizer,
        testDataPath=["data/test_data.pt"],
        save_rl_trajectories=True,
        rtg_intervals=[(0.8, 1)],
        save_rtg_min=False,
        save_rtg_max=True,
    )
    trainer = DecisionTransformerTrainer(
        savePath="./training_models/",  # Cambiado a training_models
        name="decision_transformer_model_rl_2",  # Nombre más descriptivo
        model=model,
        trainerConfig=config
    )
    
    def signal_handler(sig, frame):
        print("\n\nSeñal de interrupción recibida. Guardando trayectorias...")
        if trainer.save_rl_trajectories and hasattr(trainer, 'rl_trajectories_per_problem') and trainer.rl_trajectories_per_problem:
            trainer._select_and_save_rl_trajectories()
            print("Trayectorias guardadas exitosamente.")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        trainer.initTraining()
    except KeyboardInterrupt:
        print("\n\nEntrenamiento interrumpido por el usuario (Ctrl+C)")
        if trainer.save_rl_trajectories and hasattr(trainer, 'rl_trajectories_per_problem') and trainer.rl_trajectories_per_problem:
            print("Guardando trayectorias RL acumuladas...")
            trainer._select_and_save_rl_trajectories()
            print("Trayectorias guardadas exitosamente.")
    except Exception as e:
        print(f"\n\nError durante el entrenamiento: {e}")
        if trainer.save_rl_trajectories and hasattr(trainer, 'rl_trajectories_per_problem') and trainer.rl_trajectories_per_problem:
            print("Intentando guardar trayectorias RL acumuladas...")
            try:
                trainer._select_and_save_rl_trajectories()
                print("Trayectorias guardadas exitosamente.")
            except Exception as save_error:
                print(f"Error al guardar trayectorias: {save_error}")
        raise e
