import torch.nn as nn
import torch
import numpy as np
from tensordict import TensorDict
from generate_trajectories import RETURN_TO_GO_WINDOW, FORECAST_LENGTH, MAX_LEAD_TIME, TRAJECTORY_LENGTH
from transformers import DecisionTransformerGPT2Model
from decision_transformer_config import DecisionTransformerConfig
from pathlib import Path
from typing import Dict, Optional, Tuple
from torch.distributions import Normal, Bernoulli
from data_scaler import (
    scale_on_hand_level,
    unscale_on_hand_level,
    scale_holding_cost,
    unscale_holding_cost,
    scale_ordering_cost,
    unscale_ordering_cost,
    scale_stock_out_penalty,
    unscale_stock_out_penalty,
    scale_unit_revenue,
    unscale_unit_revenue,
    scale_lead_time,
    unscale_lead_time,
    scale_forecast,
    unscale_forecast,
    scale_in_transit_stock,
    unscale_in_transit_stock,
    scale_demand,
    unscale_demand,
    scale_order_quantity,
    unscale_order_quantity,
    scale_returns_to_go,
    unscale_returns_to_go,
    scale_benefit,
    unscale_benefit,
)


def getTorchDevice():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #return torch.device("cpu")

def loadModel(path, decisionTransformerConfig, scaling_params=None):
    """
    Carga un modelo DecisionTransformer RL desde un checkpoint.

    Args:
        path: Ruta al archivo del checkpoint.
        decisionTransformerConfig: Configuración del modelo.
        scaling_params: Parámetros de escalado opcionales; si son None se intentan
            cargar desde el propio checkpoint.

    Returns:
        Modelo DecisionTransformer inicializado con los pesos y el escalado.
    """
    checkpoint = torch.load(path, weights_only=False)
    model = DecisionTransformer(decisionTransformerConfig, scaling_params=scaling_params)
    model.load_state_dict(checkpoint["model_state"])
    
    # Si no se proporcionaron scaling_params y están en el checkpoint, usarlos
    if scaling_params is None and "scaling_params" in checkpoint:
        model.scaling_params = checkpoint["scaling_params"]
    
    return model  


class DecisionTransformer(nn.Module):

    def __init__(self, decisionTransformerConfig, scaling_params: Optional[Dict[str, Tuple[float, float]]] = None):
        super().__init__()
        self.device = getTorchDevice()

        self.decisionTransformerConfig = decisionTransformerConfig
        self.embeddingDim = decisionTransformerConfig.hidden_size

        self.maxSeqLength = 30

        self.projectScalarData = nn.Linear(5, self.embeddingDim)
        self.projectStockInTransitData= nn.Linear(1, self.embeddingDim)
        self.projectDemandData = nn.Linear(1, self.embeddingDim)
        
        maxSeqTimeLength = self.maxSeqLength
        self.positionTimeEmbedding = nn.Embedding(maxSeqTimeLength, self.embeddingDim)
        self.demandTimeEmbedding = nn.Embedding(FORECAST_LENGTH, self.embeddingDim)
        self.stockInTransitTimeEmbedding = nn.Embedding(MAX_LEAD_TIME, self.embeddingDim) 

        self.mhaState = nn.MultiheadAttention(
            embed_dim=self.embeddingDim,
            num_heads=decisionTransformerConfig.n_head,
            batch_first=True
        )

        self.embeddingReturnsToGo = nn.Linear(1, self.embeddingDim)
        self.embeddingAction = nn.Linear(1, self.embeddingDim)  

        self.outputMean = nn.Linear(self.embeddingDim, 1)  
        self.outputStd = nn.Linear(self.embeddingDim, 1)
        self.outputOrder = nn.Linear(self.embeddingDim, 1)
        self.outputValue = nn.Linear(self.embeddingDim, 1)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
    
        # Parámetros de escalado
        self.scaling_params = scaling_params if scaling_params is not None else {}

        self.transformer = DecisionTransformerGPT2Model(decisionTransformerConfig)
        #self.outputOrder.bias.data.fill_(1.8)

    def non_constructive_forward(self, td: Dict[str, torch.Tensor]):
        """
        Calcula distribuciones y valores para toda la trayectoria de una vez.

        Usa los estados crudos guardados en los buffers saved_* para reconstruir
        los embeddings y pasar toda la secuencia por el transformer con gradiente.

        Args:
            td: TensorDict con las listas saved_* rellenas durante el rollout.

        Returns:
            tuple: (order_dist_seq, quantity_dist_seq, values) con shape [B, T, 1].
        """
        device = self.device
        batchSize = td["holdingCost"].size(0)

        onHandLevel = torch.stack(td["saved_onHandLevel"], dim=1)  # [B, T] o [B, T, 1]
        if onHandLevel.dim() == 3:
            onHandLevel = onHandLevel.squeeze(-1)

        inTransitStock = torch.stack(td["saved_inTransitStock"], dim=1)  # [B, T, MAX_LEAD_TIME]
        forecast = torch.stack(td["saved_forecast"], dim=1)  # [B, T, FORECAST_LENGTH]
        actions = torch.stack(td["saved_actions"], dim=1)  # [B, T, 1]
        if actions.dim() == 2:
            actions = actions.unsqueeze(-1)

        seqLen = forecast.size(1)

        # Escalado de entradas escalares y proyección para todos los timesteps.
        onHandLevelScaled = self._scale_field(onHandLevel, "onHandLevel")
        holdingCostScaled = self._scale_field(td["holdingCost"], "holdingCost")
        orderingCostScaled = self._scale_field(td["orderingCost"], "orderingCost")
        stockOutPenaltyScaled = self._scale_field(td["stockOutPenalty"], "stockOutPenalty")
        leadTimeScaled = self._scale_field(td["leadTime"], "leadTime")

        scalarData = torch.cat(
            [
                onHandLevelScaled.unsqueeze(-1),
                holdingCostScaled.unsqueeze(-1).unsqueeze(1).expand(-1, seqLen, -1),
                orderingCostScaled.unsqueeze(-1).unsqueeze(1).expand(-1, seqLen, -1),
                stockOutPenaltyScaled.unsqueeze(-1).unsqueeze(1).expand(-1, seqLen, -1),
                leadTimeScaled.unsqueeze(-1).unsqueeze(1).expand(-1, seqLen, -1),
            ],
            dim=-1,
        )

        scalarDataProjection = self.projectScalarData(scalarData)

        forecastScaled = self._scale_field(forecast, "forecast")
        forecastProjection = self.projectDemandData(forecastScaled.unsqueeze(-1))

        demandTimeIndices = torch.arange(FORECAST_LENGTH, device=device).long()
        demandTimeProjection = self.demandTimeEmbedding(demandTimeIndices)
        demandTimeProjectionExpanded = demandTimeProjection.unsqueeze(0).unsqueeze(0)
        demandTimeEmbedding = forecastProjection + demandTimeProjectionExpanded

        inTransitStockScaled = self._scale_field(inTransitStock, "inTransitStock")
        stockInTransitProjection = self.projectStockInTransitData(inTransitStockScaled.unsqueeze(-1))

        stockTimeIndices = torch.arange(MAX_LEAD_TIME, device=device).long()
        stockTimeProjection = self.stockInTransitTimeEmbedding(stockTimeIndices)
        stockTimeProjectionExpanded = stockTimeProjection.unsqueeze(0).unsqueeze(0)
        stockInTransitTimeEmbedding = stockInTransitProjection + stockTimeProjectionExpanded

        demandStockTimeEmbedding = torch.cat([demandTimeEmbedding, stockInTransitTimeEmbedding], dim=2)

        scalarDataProjectionReshaped = scalarDataProjection.reshape(batchSize * seqLen, 1, self.embeddingDim)
        demandStockTimeEmbeddingReshaped = demandStockTimeEmbedding.reshape(
            batchSize * seqLen,
            FORECAST_LENGTH + MAX_LEAD_TIME,
            self.embeddingDim,
        )

        leadTimes = td["leadTime"].long()
        leadTimesExpanded = leadTimes.unsqueeze(1).expand(-1, seqLen).reshape(-1)
        positions = torch.arange(FORECAST_LENGTH + MAX_LEAD_TIME, device=device).unsqueeze(0)
        valid_limit = FORECAST_LENGTH + leadTimesExpanded.unsqueeze(1)
        key_padding_mask = positions >= valid_limit

        mhaState, _ = self.mhaState(
            query=scalarDataProjectionReshaped,
            key=demandStockTimeEmbeddingReshaped,
            value=demandStockTimeEmbeddingReshaped,
            key_padding_mask=key_padding_mask,
        )

        statesEmbedding = mhaState.reshape(batchSize, seqLen, self.embeddingDim)

        # Embedding de acciones.
        actionsScaled = self._scale_field(actions, "orderQuantity")
        actionsEmbedding = self.embeddingAction(actionsScaled)

        # Positional embeddings.
        positions = torch.arange(seqLen, device=device)
        positionsEmbeddings = self.positionTimeEmbedding(positions).unsqueeze(0)

        statesEmbedding = statesEmbedding + positionsEmbeddings
        actionsEmbedding = actionsEmbedding + positionsEmbeddings

        stackedInputs = (
            torch.stack(
                (
                    statesEmbedding,
                    actionsEmbedding,
                ),
                dim=1,
            )
            .permute(0, 2, 1, 3)
            .reshape(batchSize, 2 * seqLen, self.embeddingDim)
        )

        output = self.transformer(inputs_embeds=stackedInputs)
        output = output["last_hidden_state"]
        output = (
            output.reshape(batchSize, seqLen, 2, self.embeddingDim)
            .permute(0, 2, 1, 3)[:, 0, :, :]
        )

        predictedMean = self.softplus(self.outputMean(output))
        predictedStd = self.softplus(self.outputStd(output))
        value = self.outputValue(output)

        orderLogit = self.outputOrder(output)
        orderActionProb = self.sigmoid(orderLogit)

        order_dist_seq = Bernoulli(orderActionProb)
        quantity_dist_seq = Normal(predictedMean, predictedStd)

        return order_dist_seq, quantity_dist_seq, value

    def _scale_field(self, data: torch.Tensor, field_name: str) -> torch.Tensor:
        """
        Escala un campo usando los parámetros almacenados.

        Args:
            data: Tensor a escalar.
            field_name: Nombre lógico del campo a escalar.

        Returns:
            Tensor escalado con la transformación adecuada o el tensor original.
        """
        if field_name not in self.scaling_params:
            return data
        
        params = self.scaling_params[field_name]
        
        # orderQuantity usa Min-Max (min_val, max_val) en lugar de Z-Score (mean, std)
        if field_name == 'orderQuantity':
            min_val, max_val = params
            return scale_order_quantity(data, min_val, max_val)
        
        mean, std = params
        scale_funcs = {
            'onHandLevel': scale_on_hand_level,
            'holdingCost': scale_holding_cost,
            'orderingCost': scale_ordering_cost,
            'stockOutPenalty': scale_stock_out_penalty,
            'unitRevenue': scale_unit_revenue,
            'leadTime': scale_lead_time,
            'forecast': scale_forecast,
            'inTransitStock': scale_in_transit_stock,
            'demand': scale_demand,
            'returnsToGo': scale_returns_to_go,
            'benefit': scale_benefit,
        }
        
        if field_name in scale_funcs:
            return scale_funcs[field_name](data, mean, std)
        return data

    def _unscale_field(self, data: torch.Tensor, field_name: str) -> torch.Tensor:
        """
        Desescala un campo usando los parámetros almacenados.

        Args:
            data: Tensor a desescalar.
            field_name: Nombre lógico del campo a desescalar.

        Returns:
            Tensor desescalado con la transformación adecuada o el tensor original.
        """
        if field_name not in self.scaling_params:
            return data
        
        params = self.scaling_params[field_name]
        
        # orderQuantity usa Min-Max (min_val, max_val) en lugar de Z-Score (mean, std)
        if field_name == 'orderQuantity':
            min_val, max_val = params
            return unscale_order_quantity(data, min_val, max_val)
        
        mean, std = params
        unscale_funcs = {
            'onHandLevel': unscale_on_hand_level,
            'holdingCost': unscale_holding_cost,
            'orderingCost': unscale_ordering_cost,
            'stockOutPenalty': unscale_stock_out_penalty,
            'unitRevenue': unscale_unit_revenue,
            'leadTime': unscale_lead_time,
            'forecast': unscale_forecast,
            'inTransitStock': unscale_in_transit_stock,
            'demand': unscale_demand,
            'returnsToGo': unscale_returns_to_go,
            'benefit': unscale_benefit,
        }
        
        if field_name in unscale_funcs:
            return unscale_funcs[field_name](data, mean, std)
        return data


    def initModel(self, td): 
        """
        Inicializa el TensorDict de RL con todos los campos necesarios del entorno.

        Args:
            td: TensorDict original con los datos estáticos y series temporales.

        Returns:
            Nuevo TensorDict con estado interno inicializado para RL.
        """
        batchSize = td["leadTime"].size(0)
        device = getTorchDevice()
        if hasattr(td, 'clone'):
            tdNew = td.clone()
        else:
            tdNew = {k: v.clone() for k, v in td.items()}

        tdNew["currentTimestep"] = torch.zeros((batchSize, 1), dtype=torch.long, device=device)
        tdNew["orderQuantity"] = torch.zeros((batchSize, 1), dtype=torch.float32, device=device)
        tdNew["onHandLevel"] = td["onHandLevel"]
        tdNew["inTransitStock"] = td["inTransitStock"][:, 0, :]
        tdNew["forecast"] = td["forecast"]
        tdNew["orderingCost"] = td["orderingCost"]
        tdNew["stockOutPenalty"] = td["stockOutPenalty"]
        tdNew["unitRevenue"] = td["unitRevenue"]
        tdNew["leadTime"] = td["leadTime"]
        tdNew["benefit"] = torch.zeros((batchSize, 0, 1), dtype=torch.float32, device=device)
        tdNew["returns"] = torch.zeros((batchSize, 0, 1), dtype=torch.float32, device=device)
        tdNew["cumulativeSales"] = torch.zeros((batchSize, 0), dtype=torch.float32, device=device)
        tdNew["cumulativeHoldingCost"] = torch.zeros((batchSize, 0, 1), dtype=torch.float32, device=device)
        tdNew["cumulativeOrderingCost"] = torch.zeros((batchSize, 0, 1), dtype=torch.float32, device=device)
        tdNew["cumulativeStockOutCost"] = torch.zeros((batchSize, 0, 1), dtype=torch.float32, device=device)
        tdNew["predictedAction"] = torch.zeros(batchSize, 1, dtype=torch.float32, device=device)
        tdNew["demand"] = td["demand"]

        tdNew["statesEmbedding"] = torch.zeros((batchSize, 0, self.embeddingDim), dtype=torch.float, device=device)
        tdNew["actionsEmbedding"] = torch.zeros((batchSize, 0, self.embeddingDim), dtype=torch.float, device=device)
        tdNew["returnsToGoEmbedding"] = torch.zeros((batchSize, 0, self.embeddingDim), dtype=torch.float, device=device)
        tdNew["values"] = torch.zeros((batchSize, 0, 1), dtype=torch.float32, device=device)
        tdNew["returns"] = torch.zeros((batchSize, 0, 1), dtype=torch.float32, device=device)

        # Buffers para guardar trayectorias completas por timestep (usados en PPO).
        tdNew["saved_onHandLevel"] = []
        tdNew["saved_inTransitStock"] = []
        tdNew["saved_forecast"] = []
        tdNew["saved_actions"] = []
        return tdNew

    def rollout_step(self, td: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Ejecuta un único paso de entorno con muestreo de acción para RL online.

        Esta función se usa durante el 'rollout' para generar trayectorias y
        rellenar los campos de recompensa y estado.

        Args:
            td: TensorDict con el estado actual.

        Returns:
            TensorDict actualizado tras aplicar la acción y avanzar el entorno.
        """
        batchSize = td["statesEmbedding"].size(0)

        # Guardar estado crudo para poder reconstruir la red de forma vectorizada.
        td["saved_onHandLevel"].append(td["onHandLevel"].clone())
        td["saved_inTransitStock"].append(td["inTransitStock"].clone())
        td["saved_forecast"].append(td["forecast"][:, 0, :].clone())
        td["saved_actions"].append(td["orderQuantity"].clone())

        # Escalar entradas escalares para el estado actual.
        onHandLevelScaled = self._scale_field(td["onHandLevel"], "onHandLevel")
        holdingCostScaled = self._scale_field(td["holdingCost"], "holdingCost")
        orderingCostScaled = self._scale_field(td["orderingCost"], "orderingCost")
        stockOutPenaltyScaled = self._scale_field(td["stockOutPenalty"], "stockOutPenalty")
        leadTimeScaled = self._scale_field(td["leadTime"], "leadTime")

        scalarData = torch.cat(
            [
                onHandLevelScaled.unsqueeze(-1),
                holdingCostScaled.unsqueeze(-1),
                orderingCostScaled.unsqueeze(-1),
                stockOutPenaltyScaled.unsqueeze(-1),
                leadTimeScaled.unsqueeze(-1),
            ],
            dim=-1,
        )

        forecastScaled = self._scale_field(td["forecast"][:, 0, :], "forecast")
        demandData = forecastScaled.unsqueeze(-1)

        inTransitStockScaled = self._scale_field(td["inTransitStock"].float(), "inTransitStock")
        stockInTransitData = inTransitStockScaled.unsqueeze(-1)

        scalarDataProjection = self.projectScalarData(scalarData)
        demandDataProjection = self.projectDemandData(demandData)
        stockInTransitDataProjection = self.projectStockInTransitData(stockInTransitData)

        demandTimeIndices = torch.arange(FORECAST_LENGTH, device=td["forecast"].device).long()
        demandTimeProjection = self.demandTimeEmbedding(demandTimeIndices)

        stockTimeIndices = torch.arange(MAX_LEAD_TIME, device=td["forecast"].device).long()
        stockTimeProjection = self.stockInTransitTimeEmbedding(stockTimeIndices)

        demandTimeEmbedding = demandDataProjection + demandTimeProjection.unsqueeze(0)
        stockInTransitTimeEmbedding = stockInTransitDataProjection + stockTimeProjection.unsqueeze(0)
        demandStockTimeEmbedding = torch.cat([demandTimeEmbedding, stockInTransitTimeEmbedding], dim=1)

        leadTimes = td["leadTime"].long()
        positions = torch.arange(FORECAST_LENGTH + MAX_LEAD_TIME, device=td["forecast"].device).unsqueeze(0)
        valid_limit = FORECAST_LENGTH + leadTimes.unsqueeze(1)
        key_padding_mask = positions >= valid_limit

        mhaState, _ = self.mhaState(
            query=scalarDataProjection.unsqueeze(1),
            key=demandStockTimeEmbedding,
            value=demandStockTimeEmbedding,
            key_padding_mask=key_padding_mask,
        )

        td["statesEmbedding"] = self.addSequenceData(td, td["statesEmbedding"], mhaState)

        statesEmbedding = td["statesEmbedding"]
        actionsEmbedding = td["actionsEmbedding"]

        positions = torch.arange(statesEmbedding.size(1), device=self.device)
        positionsEmbeddings = self.positionTimeEmbedding(positions)
        positionsEmbeddings = positionsEmbeddings.unsqueeze(0).expand(statesEmbedding.size(0), -1, -1)

        statesEmbedding = statesEmbedding + positionsEmbeddings
        actionsEmbedding = actionsEmbedding + positionsEmbeddings[:, : actionsEmbedding.size(1), :]
        stackedInputs = (
            torch.stack(
                (
                    statesEmbedding,
                    torch.cat(
                        (
                            actionsEmbedding,
                            torch.zeros(
                                batchSize,
                                statesEmbedding.size(1) - actionsEmbedding.size(1),
                                self.embeddingDim,
                                device=self.device,
                            ),
                        ),
                        dim=1,
                    ),
                ),
                dim=1,
            )
            .permute(0, 2, 1, 3)
            .reshape(batchSize, 2 * statesEmbedding.size(1), self.embeddingDim)
        )

        output = self.transformer(inputs_embeds=stackedInputs)
        output = output["last_hidden_state"]
        output = (
            output.reshape(batchSize, statesEmbedding.size(1), 2, self.embeddingDim)
            .permute(0, 2, 1, 3)[:, 0, -1, :]
        )

        predictedMean = self.softplus(self.outputMean(output))
        predictedStd = self.softplus(self.outputStd(output))
        value = self.outputValue(output)

        # Probabilidad de hacer pedido.
        orderLogit = self.outputOrder(output)
        td["orderLogit"] = orderLogit
        orderProb = self.sigmoid(orderLogit)

        # Distribuciones de política.
        order_dist = Bernoulli(orderProb)
        quantity_dist = Normal(predictedMean, predictedStd)

        if self.training:
            orderDecision = order_dist.sample()
            quantityValue = quantity_dist.sample()
        else:
            orderDecision = (orderProb >= 0.5).float()
            quantityValue = predictedMean

        predictedActionScaled = orderDecision * quantityValue

        orderLogProb = order_dist.log_prob(orderDecision)
        quantityLogProb = quantity_dist.log_prob(quantityValue)
        totalLogProb = orderLogProb + quantityLogProb * orderDecision

        predictedAction = self._unscale_field(predictedActionScaled, "orderQuantity")
        if not self.training:
            predictedAction = torch.ceil(predictedAction).long().float()
        orderQuantity = predictedAction

        # Guardar toda la información necesaria para PPO y análisis.
        td["orderQuantity"] = orderQuantity
        td["predictedAction"] = predictedAction
        td["actionLogProb"] = totalLogProb
        td["orderDistribution"] = order_dist
        td["quantityDistribution"] = quantity_dist
        td["orderDecision"] = orderDecision
        td["quantityValue"] = quantityValue
        td["values"] = self.addSequenceData(td, td["values"], value.unsqueeze(1))

        orderQuantityScaled = self._scale_field(orderQuantity, "orderQuantity")
        actionEmbedding = self.embeddingAction(orderQuantityScaled).unsqueeze(1)
        td["actionsEmbedding"] = self.addSequenceData(td, td["actionsEmbedding"], actionEmbedding)

        stockToArrive = td["inTransitStock"][:, 0]
        td["onHandLevel"] = td["onHandLevel"] + stockToArrive

        td["inTransitStock"] = torch.roll(td["inTransitStock"], shifts=-1, dims=-1)
        td["inTransitStock"][:, -1] = 0

        leadTimes = td["leadTime"].long()
        indexes = torch.arange(leadTimes.size(0), device=self.device)
        td["inTransitStock"][indexes, leadTimes - 1] = td["orderQuantity"].squeeze(-1)

        current_demand = td["demand"][:, 0]
        current_stock = td["onHandLevel"]

        holdingCost = (td["holdingCost"] * current_stock).unsqueeze(-1)

        sales = torch.minimum(current_demand, current_stock)
        stockout = torch.clamp(current_demand - current_stock, min=0).unsqueeze(-1)
        income = (td["unitRevenue"] * sales).unsqueeze(-1)

        if td["cumulativeSales"].size(1) == 0:
            td["cumulativeSales"] = sales.unsqueeze(-1)
        else:
            td["cumulativeSales"] = torch.cat(
                (td["cumulativeSales"], td["cumulativeSales"][:, -1].unsqueeze(-1) + sales),
                dim=1,
            )

        td["onHandLevel"] = current_stock - sales
        orderingCost = torch.where(
            td["orderQuantity"].squeeze(-1) > 0,
            td["orderingCost"],
            torch.zeros_like(td["orderingCost"]),
        ).unsqueeze(-1)

        stockoutPenalty = (td["stockOutPenalty"].unsqueeze(-1) * stockout)

        if td["cumulativeHoldingCost"].size(1) == 0:
            td["cumulativeHoldingCost"] = holdingCost
        else:
            td["cumulativeHoldingCost"] = torch.cat(
                (td["cumulativeHoldingCost"], td["cumulativeHoldingCost"][:, -1].unsqueeze(-1) + holdingCost),
                dim=1,
            )

        if td["cumulativeOrderingCost"].size(1) == 0:
            td["cumulativeOrderingCost"] = orderingCost
        else:
            td["cumulativeOrderingCost"] = torch.cat(
                (td["cumulativeOrderingCost"], td["cumulativeOrderingCost"][:, -1].unsqueeze(-1) + orderingCost),
                dim=1,
            )

        if td["cumulativeStockOutCost"].size(1) == 0:
            td["cumulativeStockOutCost"] = stockoutPenalty
        else:
            td["cumulativeStockOutCost"] = torch.cat(
                (
                    td["cumulativeStockOutCost"],
                    td["cumulativeStockOutCost"][:, -1].unsqueeze(-1) + stockoutPenalty,
                ),
                dim=1,
            )

        benefitUpdate = (income - holdingCost - stockoutPenalty - orderingCost).float()
        if td["benefit"].size(1) == 0:
            td["benefit"] = benefitUpdate
        else:
            td["benefit"] = torch.cat(
                (td["benefit"], td["benefit"][:, -1].unsqueeze(-1) + benefitUpdate),
                dim=1,
            )
        
        td["returns"] = self.addSequenceData(td, td["returns"], benefitUpdate)

        td["forecast"] = torch.roll(td["forecast"], shifts=-1, dims=1)
        td["demand"] = torch.roll(td["demand"], shifts=-1, dims=-1)

        return td

    def forward(self, td):
        """
        Forward pass del modelo para RL que genera acción, valor escalar y actualiza el estado.

        Args:
            td: TensorDict con el estado actual.
        """
        return self.rollout_step(td)

    def addSequenceData(self, td, tensor, data):
        """
        Añade nuevos datos a la secuencia manteniendo un máximo de maxSeqLength elementos.

        Args:
            td: TensorDict con los datos del estado actual.
            tensor: Tensor existente con la secuencia [batch_size, seq_len, embedding_dim].
            data: Nuevos datos a añadir [batch_size, 1, embedding_dim] o [batch_size, embedding_dim].

        Returns:
            Tensor actualizado con la nueva secuencia [batch_size, min(seq_len+1, maxSeqLength), embedding_dim]
        """
        
        if data.dim() == 2:
            data = data.unsqueeze(1)
        
        if tensor.size(1) == 0:
            return data
            
        if tensor.size(1) >= self.maxSeqLength:
            tensor = tensor[:, -self.maxSeqLength+1:, :]
            data = data[:, :1, :]
            result = torch.cat((tensor, data), dim=1)
        else:
            result = torch.cat((tensor, data), dim=1)
            
        return result
    
    
    def forward_from_embeddings(self, statesEmbedding, actionsEmbedding):
        """
        Calcula distribuciones de acción y valor escalar a partir de secuencias de embeddings.

        Args:
            statesEmbedding: Embeddings de estados con shape [batch_size, seq_len, embedding_dim].
            actionsEmbedding: Embeddings de acciones con shape [batch_size, seq_len_actions, embedding_dim].

        Returns:
            tuple: (orderDist, quantityDist, orderLogit, value) para el último estado de la secuencia
        """
        batchSize = statesEmbedding.size(0)
        positions = torch.arange(statesEmbedding.size(1), device=self.device)
        positionsEmbeddings = self.positionTimeEmbedding(positions)
        positionsEmbeddings = positionsEmbeddings.unsqueeze(0).expand(statesEmbedding.size(0), -1, -1)
        
        statesEmbedding = statesEmbedding + positionsEmbeddings
        actionsEmbedding = actionsEmbedding + positionsEmbeddings[:, :actionsEmbedding.size(1), :]
        
        stackedInputs = (
            torch.stack(
                (
                    statesEmbedding,
                    torch.cat(
                        (
                            actionsEmbedding,
                            torch.zeros(
                                batchSize,
                                statesEmbedding.size(1) - actionsEmbedding.size(1),
                                self.embeddingDim,
                                device=self.device,
                            ),
                        ),
                        dim=1,
                    ),
                ),
                dim=1,
            )
            .permute(0, 2, 1, 3)
            .reshape(batchSize, 2 * statesEmbedding.size(1), self.embeddingDim)
        )
    
        output = self.transformer(inputs_embeds=stackedInputs) 
        output = output["last_hidden_state"]
        output = output.reshape(batchSize, statesEmbedding.size(1), 2, self.embeddingDim).permute(0, 2, 1, 3)
        output = output[:, 0, -1, :]
        
        predictedMean = self.softplus(self.outputMean(output))
        predictedStd = self.softplus(self.outputStd(output))
        orderLogit = self.outputOrder(output)
        orderAction = self.sigmoid(orderLogit)
        value = self.outputValue(output)
        
        orderDist = Bernoulli(orderAction)
        quantityDist = Normal(predictedMean, predictedStd)
        
        return orderDist, quantityDist, orderLogit, value
