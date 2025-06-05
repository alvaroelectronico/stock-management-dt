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
from datetime import date



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
            'mean_test_total_cost': self.training_metrics['mean_test_total_cost'],
            'mean_real_total_cost': self.training_metrics['mean_real_total_cost'],
            'mean_cost_difference': self.training_metrics['mean_cost_difference'],
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
        print(f"Longitud de la ventana de contexto: {self.model.maxSeqLength}")

        # Guardar los parámetros iniciales del optimizador
        #self.saveOptimizerParams(epoch, 0)

        # Inicializar métricas
        self.training_metrics = {
            'epoch_losses': [],
            'validation_losses': [],
            'validation_cost_metrics': [],
            'ordered_quantities': []
        }
        
        #Bucle de entrenamiento
        while True:  # Condición de parada
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
                td = {k: v.to(self.device) for k, v in problemData.items()}
                #td = problemData.to(self.device)
                
                #inicializar el modelo con los datos del problema
                self.model.setInitalReturnToGo(td, returnsToGoData) 
                td = self.model.initModel(td)

                # Variables para almacenar predicciones y valores reales
                allPredictedActions = []
                allRealActions = []
                
                # Obtener la longitud total de la trayectoria
                trajectoryLength = orderQuantityData.size(1)
                print(f"\nProcesando trayectoria de longitud {trajectoryLength}")
                print(f"Tamaño de ventana de contexto: {self.model.maxSeqLength}")
                print(f"Número de ventanas a procesar: {trajectoryLength - self.model.maxSeqLength + 1}")
                
                # Procesar la trayectoria usando ventanas deslizantes
                for window_start in range(0, trajectoryLength, 1):
                    window_end = window_start + self.model.maxSeqLength
                    if window_end - window_start == self.model.maxSeqLength and window_end < trajectoryLength:
                        print(f"\n=== Ventana {window_start}-{window_end} ===")
                        print(f"Tamaño de la ventana: {window_end - window_start}")
                        # Recortar los tensores secuenciales a la ventana
                        for key in ["onHandLevel", "holdingCost", "orderingCost", "stockOutPenalty", "unitRevenue",
                                    "leadTime", "forecast", "inTransitStock", "returnsToGo", "actions"]:
                            if key in td and td[key].dim() > 1 and td[key].shape[1] >= window_end:
                                td[key] = td[key][:, window_start:window_end]

                        # Actualizar el timestep a cero
                        td["currentTimestep"] = torch.zeros((self.nBatch, 1), device=self.device)

                        print(f"Timestep actual:{td['currentTimestep'].squeeze().tolist()}")

                        nextAction = orderQuantityData[:, window_end:window_end+1]  # Acción real para comparar
                            
                        # Modo entrenamiento: usar acción real y actualizar pesos
                        self.model.train()
                        td = self.model.forward(td, nextOrderQuantity=nextAction, is_test=False)
                        predictedAction = td["predictedAction"]  # Usar la predicción para calcular pérdida
                            
                        # Guardar predicción y valor real
                        allPredictedActions.append(predictedAction)
                        allRealActions.append(nextAction)
                            
                        print(f"Predicción media: {predictedAction.mean().item():.2f}")
                        print(f"Valor real medio: {nextAction.mean().item():.2f}")
                        
                    # Si hemos llegado a la última ventana (que termina en 199), salir del bucle
                    if window_end >= trajectoryLength:
                        break
                
                # Calcular la pérdida para todas las predicciones
                predictedTensor = torch.cat(allPredictedActions, dim=1)
                realTensor = torch.cat(allRealActions, dim=1)
                loss = nn.MSELoss()(predictedTensor, realTensor)

                for i in range(len(allPredictedActions)):
                    allPredictedActions[i] = allPredictedActions[i].detach()
                for i in range(len(allRealActions)):
                    allRealActions[i] = allRealActions[i].detach()

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
                epochLoss += loss.detach().item()
                print(f"Pérdida de la trayectoria: {loss.detach().item():.4f}")
                
                # Incrementar el contador de steps
                currentStep += 1
            
            avgEpochLoss = epochLoss / self.stepsPerEpoch
            self.training_metrics['epoch_losses'].append(avgEpochLoss)
            print(f"\nResumen de época {epoch}:")
            print(f"Pérdida promedio: {avgEpochLoss:.4f}")
            
            # Validación del modelo
            print("\nIniciando validación...")
            validation_loss, cost_metrics = self.validate_model()
            self.training_metrics['validation_losses'].append(validation_loss)
            self.training_metrics['validation_cost_metrics'].append(cost_metrics)
            print(f"Pérdida de validación: {validation_loss:.4f}")
            print(f"Métricas de coste de validación:")
            print(f"  Coste predicho: {cost_metrics['test']['total_cost']:.2f}")
            print(f"  Coste real: {cost_metrics['real']['total_cost']:.2f}")
            print(f"  Diferencia de coste: {cost_metrics['difference']['total_cost']:.2f}")
            
            # Guardar métricas en el archivo de seguimiento
            print("Guardando métricas en track.json...")
            self.content["EPOCHS"][epoch] = {
                "training_loss": avgEpochLoss,
                "validation_loss": validation_loss,
                "mean_test_total_cost": cost_metrics['test']['total_cost'],
                "mean_real_total_cost": cost_metrics['real']['total_cost'],
                "mean_cost_difference": cost_metrics['difference']['total_cost'],
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "context_windows": {
                    "trajectory_length": trajectoryLength,
                    "max_seq_length": self.model.maxSeqLength,
                    "num_windows": trajectoryLength - self.model.maxSeqLength,
                    #"windows": []
                }
            }

            # Añadir información de cada ventana
            #for window_start in range(0, trajectoryLength - self.model.maxSeqLength-1, 1):
                #window_end = window_start + self.model.maxSeqLength
                #window_info = {
                    #"window_range": [window_start, window_end],
                    #"window_size": window_end - window_start,
                    #"predictions": {
                    #    "mean": float(predictedTensor[:, window_end].mean().item()),
                    #    "std": float(predictedTensor[:, window_end].std().item())
                    #},
                    #"real_values": {
                    #    "mean": float(realTensor[:, window_end].mean().item()),
                    #    "std": float(realTensor[:, window_end].std().item())
                    #}
                #}
                #self.content["EPOCHS"][epoch]["context_windows"]["windows"].append(window_info)

            self.updateTrackFile()
            print("Métricas guardadas exitosamente")
            
            #Incrementar el contador de epoch
            epoch += 1
            
        # Guardar el modelo final y las métricas
            print("\n=== Guardando modelo final y métricas ===")
            self.saveModel()
            self.saveTrainingMetrics()
            print("Modelo y métricas guardados exitosamente")

    def calculate_trajectory_cost(self, td):
        """
        Calcula los costes de una trayectoria usando el estado actual del sistema.
        
        Args:
            td: TensorDict con el estado actual del sistema
        
        Returns:
            dict: Diccionario con los costes y métricas de la trayectoria
        """
        batch_size = td["onHandLevel"].size(0)
        trajectory_length = td["forecast"].size(1)
        
        # Inicializar listas para almacenar métricas
        holding_costs = []
        ordering_costs = []
        stockout_costs = []
        sales_revenue = []
        total_costs = []
        on_hand_levels = []
        in_transit_levels = []
        
        # Para cada paso en la trayectoria
        for t in range(td["orderQuantity"].size(1)):
            # Obtener el estado actual
            current_stock = td["onHandLevel"][:, t]
            current_demand = td["forecast"][:, t, 0]
            current_order = td["orderQuantity"][:, t] 
            
            # Calcular costes y beneficios
            holding_cost = td["holdingCost"] * current_stock
            ordering_cost = torch.where(current_order > 0, td["orderingCost"], torch.zeros_like(td["orderingCost"]))
            sales = torch.min(current_demand, current_stock)
            stockout = torch.max(torch.zeros_like(current_demand), current_demand - current_stock)
            
            # Calcular ingresos y penalizaciones
            revenue = sales * td["unitRevenue"]
            stockout_penalty = stockout * td["stockOutPenalty"]
            
            # Calcular beneficio total
            total_cost = holding_cost + ordering_cost + stockout_penalty - revenue
            
            # Guardar métricas
            holding_costs.append(holding_cost.mean().item())
            ordering_costs.append(ordering_cost.mean().item())
            stockout_costs.append(stockout_penalty.mean().item())
            sales_revenue.append(revenue.mean().item())
            total_costs.append(total_cost.mean().item())
            on_hand_levels.append(current_stock.mean().item())
            in_transit_levels.append(td["inTransitStock"][:, t].mean().item())
        
        return {
            'holding_cost': np.mean(holding_costs),
            'ordering_cost': np.mean(ordering_costs),
            'stockout_cost': np.mean(stockout_costs),
            'sales_revenue': np.mean(sales_revenue),
            'total_cost': np.mean(total_costs),
            'avg_on_hand': np.mean(on_hand_levels),
            'avg_in_transit': np.mean(in_transit_levels),
            'total_orders': sum(1 for x in ordering_costs if x > 0),
            'total_stockouts': sum(1 for x in stockout_costs if x > 0)
        }

    def validate_model(self):
        print("\nIniciando validación del modelo...")
        total_loss = 0
        n_val = self.nVal
        
        with torch.no_grad():
            for val_step in range(n_val):
                print(f"\n=== Validación {val_step + 1}/{n_val} ===")
                
                # Obtener datos de validación
                dtData = self.trainStrategy.getValidationData(self.nBatch)
                problemData, orderQuantityData, returnsToGoData = dtData
                
                # Convertir los datos a la GPU si está disponible
                orderQuantityData = orderQuantityData.to(self.device)
                returnsToGoData = returnsToGoData.to(self.device)
                td = {k: v.to(self.device) for k, v in problemData.items()}
                
                # Inicializar el modelo
                self.model.setInitalReturnToGo(td, returnsToGoData)
                td = self.model.initModel(td)
                
                # Variables para almacenar predicciones y valores reales
                allPredictedActions = []
                allRealActions = []
                
                # Obtener la longitud total de la trayectoria
                trajectoryLength = orderQuantityData.size(1)
                
                # Crear copias del estado para simular ambas trayectorias
                td_real = {k: v.clone() for k, v in td.items()}
                td_test = {k: v.clone() for k, v in td.items()}

                # Inicializar orderQuantity para toda la trayectoria
                #td_real["orderQuantity"] = torch.zeros(self.nBatch, trajectoryLength, device=self.device)
                #td_test["orderQuantity"] = torch.zeros(self.nBatch, trajectoryLength, device=self.device)
                print(f"\nDimensiones iniciales de orderQuantity: {td_real['orderQuantity'].shape}")

                # Procesar la trayectoria usando ventanas deslizantes
                for window_start in range(0, trajectoryLength, 1):
                    window_end = window_start + self.model.maxSeqLength
                    
                    if window_end - window_start == self.model.maxSeqLength and window_end < trajectoryLength:
                        # Recortar los tensores secuenciales a la ventana
                        for key in ["onHandLevel", "holdingCost", "orderingCost", "stockOutPenalty", "unitRevenue", "leadTime", "forecast", "inTransitStock", "returnsToGo", "actions"]:
                            if key in td_real and td_real[key].dim() > 1 and td_real[key].shape[1] >= window_end:
                                td_real[key] = td_real[key][:, window_start:window_end]
                                td_test[key] = td_test[key][:, window_start:window_end]

                        # Reiniciar el timestep para esta ventana
                        td_real["currentTimestep"] = torch.zeros((self.nBatch, 1), device=self.device)
                        td_test["currentTimestep"] = torch.zeros((self.nBatch, 1), device=self.device)

                    
                        nextAction = orderQuantityData[:, window_end:window_end+1]
                            
                        # Modo validación: usar acción real sin actualizar pesos
                        td_real = self.model.forward(td_real, nextOrderQuantity=nextAction, is_test=False)
                            
                        # Modo test: usar predicciones del modelo
                        td_test = self.model.forward(td_test, nextOrderQuantity=None, is_test=True)
                            
                        # Guardar predicción y valor real
                        allPredictedActions.append(td_test["predictedAction"])
                        allRealActions.append(nextAction)
                
                # Calcular la pérdida para todas las predicciones
                predictedTensor = torch.cat(allPredictedActions, dim=1)
                realTensor = torch.cat(allRealActions, dim=1)
                loss = nn.MSELoss()(predictedTensor, realTensor)
                
                # Calcular métricas para ambas trayectorias
                real_metrics = self.calculate_trajectory_cost(td_real)
                test_metrics = self.calculate_trajectory_cost(td_test)
                
                # Calcular métricas promedio
                avg_metrics = {
                    'real': real_metrics,
                    'test': test_metrics,
                    'difference': {
                        'holding_cost': test_metrics['holding_cost'] - real_metrics['holding_cost'],
                        'ordering_cost': test_metrics['ordering_cost'] - real_metrics['ordering_cost'],
                        'stockout_cost': test_metrics['stockout_cost'] - real_metrics['stockout_cost'],
                        'sales_revenue': test_metrics['sales_revenue'] - real_metrics['sales_revenue'],
                        'total_cost': test_metrics['total_cost'] - real_metrics['total_cost']
                    }
                }
                
                total_loss += loss.item()
        
        # Calcular promedios finales
        avg_loss = total_loss / n_val
        
        print("\nMétricas de validación:")
        print(f"Pérdida promedio: {avg_loss:.4f}")
        print("\nMétricas de coste (Real vs Test):")
        print(f"Coste de almacenamiento: {avg_metrics['real']['holding_cost']:.2f} vs {avg_metrics['test']['holding_cost']:.2f}")
        print(f"Coste de pedido: {avg_metrics['real']['ordering_cost']:.2f} vs {avg_metrics['test']['ordering_cost']:.2f}")
        print(f"Coste de rotura: {avg_metrics['real']['stockout_cost']:.2f} vs {avg_metrics['test']['stockout_cost']:.2f}")
        print(f"Ingresos por ventas: {avg_metrics['real']['sales_revenue']:.2f} vs {avg_metrics['test']['sales_revenue']:.2f}")
        print(f"Coste total: {avg_metrics['real']['total_cost']:.2f} vs {avg_metrics['test']['total_cost']:.2f}")
        print("\nDiferencias (Test - Real):")
        print(f"Diferencia en coste de almacenamiento: {avg_metrics['difference']['holding_cost']:.2f}")
        print(f"Diferencia en coste de pedido: {avg_metrics['difference']['ordering_cost']:.2f}")
        print(f"Diferencia en coste de rotura: {avg_metrics['difference']['stockout_cost']:.2f}")
        print(f"Diferencia en ingresos: {avg_metrics['difference']['sales_revenue']:.2f}")
        print(f"Diferencia en coste total: {avg_metrics['difference']['total_cost']:.2f}")
        
        self.model.train()  # Volver al modo de entrenamiento

        # Lista de métricas por validación
        validation_cost_metrics = self.training_metrics['validation_cost_metrics']  # lista de dicts
        n_val = len(validation_cost_metrics)

        # Sumar y hacer la media
        mean_test_total_cost = sum(m['test']['total_cost'] for m in validation_cost_metrics) / n_val
        mean_real_total_cost = sum(m['real']['total_cost'] for m in validation_cost_metrics) / n_val
        mean_cost_difference = sum(m['difference']['total_cost'] for m in validation_cost_metrics) / n_val

        return avg_loss, {
            'test': {
                'total_cost': mean_test_total_cost,
                'real': {
                    'total_cost': mean_real_total_cost,
                    'difference': {
                        'total_cost': mean_cost_difference
                    }
                }
            }
        }

           
        
                
if __name__ == "__main__":
    print("\n=== Iniciando prueba del DecisionTransformerTrainer ===")
    
    # Configuración básica
    config = TrainerConfig(
        nBatch=4,#32
        nVal=1000,  # Ajusta este valor según tus necesidades
        stepsPerEpoch=2000,
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
                
