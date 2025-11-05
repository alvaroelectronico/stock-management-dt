import torch.nn as nn
import torch
import numpy as np
import sys
from tensordict import TensorDict
from generate_trajectories import RETURN_TO_GO_WINDOW, FORECAST_LENGTH, MAX_LEAD_TIME, TRAJECTORY_LENGTH
from transformers import DecisionTransformerGPT2Model
from decision_transformer_config import DecisionTransformerConfig
from pathlib import Path


def getTorchDevice():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loadModel(path, decisionTransformerConfig):
    checkpoint = torch.load(path)
    model = DecisionTransformer(decisionTransformerConfig)
    model.load_state_dict(checkpoint["model_state"])
    return model  


class DecisionTransformer(nn.Module):

    def __init__(self, decisionTransformerConfig):
        super().__init__()
        self.embeddingDim = 128
        self.device = getTorchDevice()

        self.decisionTransformerConfig = decisionTransformerConfig
        self.embeddingDim = decisionTransformerConfig.hidden_size

        self.maxSeqLength = 30

        self.projectScalarData= nn.Linear(6, self.embeddingDim)
        self.projectStockInTransitData= nn.Linear(1, self.embeddingDim)
        self.projectDemandData = nn.Linear(1, self.embeddingDim)
        
        maxSeqTimeLength = self.maxSeqLength
        self.positionTimeEmbedding = nn.Embedding(maxSeqTimeLength, self.embeddingDim)
        self.demandTimeEmbedding = nn.Embedding(FORECAST_LENGTH, self.embeddingDim)
        self.stockInTransitTimeEmbedding = nn.Embedding(MAX_LEAD_TIME, self.embeddingDim) 

        self.mhaState = nn.MultiheadAttention(
            embed_dim=self.embeddingDim,
            num_heads=4,
            batch_first=True
        )

        self.embeddingReturnsToGo = nn.Linear(1, self.embeddingDim)
        self.embeddingAction = nn.Linear(1, self.embeddingDim)  

        self.outputProjection = nn.Linear(self.embeddingDim, 1)  
        self.softplus = nn.Softplus()
    

        self.transformer = DecisionTransformerGPT2Model(decisionTransformerConfig)


    def initModel(self, td): 
        batchSize = td["leadTime"].size(0)   
        leadTime = int(torch.max(td["leadTime"]).item())
        device = getTorchDevice()
        if hasattr(td, 'clone'):
            tdNew = td.clone()
        else:
            tdNew = {k: v.clone() for k, v in td.items()}

        tdNew["currentTimestep"] = torch.zeros((batchSize, 1), dtype=torch.long)
        tdNew["orderQuantity"] = torch.zeros((batchSize, 1), dtype=torch.float32)
        tdNew["onHandLevel"] = td["onHandLevel"]
        tdNew["inTransitStock"] = td["inTransitStock"][:, 0, :]
        tdNew["forecast"] = td["forecast"]
        tdNew["orderingCost"] = td["orderingCost"]
        tdNew["stockOutPenalty"] = td["stockOutPenalty"]
        tdNew["unitRevenue"] = td["unitRevenue"]
        tdNew["leadTime"] = td["leadTime"]
        tdNew["benefit"] = torch.zeros(batchSize, RETURN_TO_GO_WINDOW, dtype=torch.float32)
        tdNew["returnsToGo"] = td["returnsToGo"]
        tdNew["predictedAction"] = torch.zeros(batchSize, 1, dtype=torch.float32)
        tdNew["demand"] = td["demand"]

        tdNew["statesEmbedding"] = torch.zeros((batchSize, 0, self.embeddingDim), dtype=torch.float, device=device)
        tdNew["actionsEmbedding"] = torch.zeros((batchSize, 0, self.embeddingDim), dtype=torch.float, device=device)
        tdNew["returnsToGoEmbedding"] = torch.zeros((batchSize, 0, self.embeddingDim), dtype=torch.float, device=device)
                             
        return tdNew
    
    def non_constructive_forward(self, td):
        returnsToGoEmbedding = td["returnsToGoEmbedding"]
        statesEmbedding = td["statesEmbedding"]
        actionsEmbedding = td["actionsEmbedding"]
        
        positions = torch.arange(statesEmbedding.size(1), device=self.device)
        positionsEmbeddings = self.positionTimeEmbedding(positions)
        positionsEmbeddings = positionsEmbeddings.unsqueeze(0).expand(statesEmbedding.size(0), -1, -1)
        
        statesEmbedding = statesEmbedding + positionsEmbeddings
        returnsToGoEmbedding = returnsToGoEmbedding + positionsEmbeddings
        actionsEmbedding = actionsEmbedding + positionsEmbeddings[:, :actionsEmbedding.size(1), :]
        batchSize = statesEmbedding.size(0)
        
        stackedInputs = (
            torch.stack((returnsToGoEmbedding, statesEmbedding,
                            torch.cat((actionsEmbedding,
                                    torch.zeros(batchSize, statesEmbedding.size(1) - actionsEmbedding.size(1), self.embeddingDim, device=self.device)), dim=1)),
                        dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batchSize, 3 * statesEmbedding.size(1), self.embeddingDim)
        )
    
        output = self.transformer(inputs_embeds=stackedInputs) 
        output = output["last_hidden_state"]
        output = output.reshape(batchSize, statesEmbedding.size(1), 3, self.embeddingDim).permute(0, 2, 1, 3)
        output = output[:, 1, :, :]
        predictedAction = self.outputProjection(output)
        predictedAction = self.softplus(predictedAction)
        td["orderQuantity"] = predictedAction
        td["predictedAction"] = predictedAction
        
   
    def forward(self, td, nextOrderQuantity=None, is_test=False, update_only=False):
        """
        Forward pass del modelo con tres modos:
        1. Entrenamiento (self.training=True, is_test=False): Usa acciones reales y actualiza pesos
        2. Validación (self.training=False, is_test=False): Usa acciones reales sin actualizar pesos
        3. Test (is_test=True): Usa predicciones del modelo sin actualizar pesos
        
        Args:
            td: TensorDict con el estado actual
            nextOrderQuantity: Acción real para el siguiente paso (usado en entrenamiento y validación)
            is_test: Si True, usa predicciones del modelo
            update_only: Si True, solo actualiza el estado sin calcular predicciones
        """
        batchSize = td["statesEmbedding"].size(0)
        leadTimeMax = MAX_LEAD_TIME
        
        if not is_test and nextOrderQuantity is None and not update_only:
            raise ValueError("nextOrderQuantity debe ser proporcionado cuando is_test=False y update_only=False")

        if not update_only:
            batch_size = batchSize
            currentTimeStep = td["currentTimestep"].long().squeeze(-1)
            
            scalarData = torch.cat([
                td["onHandLevel"].unsqueeze(-1),
                td["holdingCost"].unsqueeze(-1),
                td["orderingCost"].unsqueeze(-1),
                td["stockOutPenalty"].unsqueeze(-1),
                td["unitRevenue"].unsqueeze(-1),
                td["leadTime"].unsqueeze(-1)
                ], dim=-1)

            demandData = td["forecast"][:, 0, :].unsqueeze(-1)
            stockInTransitData = td["inTransitStock"].float().unsqueeze(-1)
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

            mhaState, _ = self.mhaState(
                query=scalarDataProjection.unsqueeze(1),
                key=demandStockTimeEmbedding,
                value=demandStockTimeEmbedding 
                )

            td["statesEmbedding"] = self.addSequenceData(td, td["statesEmbedding"], mhaState)
            returnsToGo = td["returnsToGo"].clone().float().unsqueeze(-1)
            
            embeddingsReturnsToGo = self.embeddingReturnsToGo(returnsToGo).unsqueeze(1)
            
            td["returnsToGoEmbedding"] = self.addSequenceData(td, td["returnsToGoEmbedding"],
                                                            embeddingsReturnsToGo)
            
            statesEmbedding = td["statesEmbedding"]
            returnsToGoEmbedding = td["returnsToGoEmbedding"]
            actionsEmbedding = td["actionsEmbedding"]

            if is_test:
                positions = torch.arange(statesEmbedding.size(1), device=self.device)
                positionsEmbeddings = self.positionTimeEmbedding(positions)
                positionsEmbeddings = positionsEmbeddings.unsqueeze(0).expand(statesEmbedding.size(0), -1, -1)

                statesEmbedding = statesEmbedding + positionsEmbeddings
                returnsToGoEmbedding = returnsToGoEmbedding + positionsEmbeddings
                actionsEmbedding = actionsEmbedding + positionsEmbeddings[:, :actionsEmbedding.size(1), :]
                stackedInputs = (
                    torch.stack((returnsToGoEmbedding, statesEmbedding,
                                torch.cat((actionsEmbedding,
                                            torch.zeros(batchSize, statesEmbedding.size(1) - actionsEmbedding.size(1), self.embeddingDim, device=self.device)), dim=1)),
                                dim=1)
                    .permute(0, 2, 1, 3)
                    .reshape(batchSize, 3 * statesEmbedding.size(1), self.embeddingDim)
                )
            
                output = self.transformer(inputs_embeds=stackedInputs)
                output = output["last_hidden_state"]
                output = output.reshape(batchSize, statesEmbedding.size(1), 3, self.embeddingDim).permute(0, 2, 1, 3)
                output = output[:, 1, -1, :]
                predictedAction = self.outputProjection(output)
                predictedAction = self.softplus(predictedAction)
                
                orderQuantity = predictedAction
            else:
                predictedAction = torch.zeros(batchSize, 1, device=self.device)
                orderQuantity = nextOrderQuantity

        if not update_only:
            td["orderQuantity"] = orderQuantity
            td["predictedAction"] = predictedAction
        
            actionEmbedding = self.embeddingAction(orderQuantity).unsqueeze(1)
            td["actionsEmbedding"] = self.addSequenceData(td, td["actionsEmbedding"], actionEmbedding)
        else:
            orderQuantity = nextOrderQuantity

        currentTimeStep = td["currentTimestep"].long()
        
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
        
        td["onHandLevel"] = current_stock - sales
        orderingCost = torch.where(
            orderQuantity.squeeze(-1) > 0,
            td["orderingCost"],
            torch.zeros_like(td["orderingCost"])
        ).unsqueeze(-1)

        stockoutPenalty = (td["stockOutPenalty"].unsqueeze(-1) * stockout)

        benefitUpdate = (income - holdingCost - stockoutPenalty - orderingCost).float()

        if currentTimeStep[0].item() >= RETURN_TO_GO_WINDOW:
            td["benefit"] = torch.roll(td["benefit"], shifts=-1, dims=-1)
            td["benefit"][..., -1] = benefitUpdate.squeeze(-1)
        else:
            td["benefit"][..., currentTimeStep] = benefitUpdate

        td["forecast"] = torch.roll(td["forecast"], shifts=-1, dims=1)
        td["demand"] = torch.roll(td["demand"], shifts=-1, dims=-1)

        if is_test:
            if not hasattr(self, 'test_metrics'):
                self.test_metrics = {
                    'holding_costs': [],
                    'ordering_costs': [],
                    'stockout_costs': [],
                    'sales_revenue': [],
                    'total_costs': [],
                    'on_hand_levels': [],
                    'in_transit_levels': []
                }
            
            self.test_metrics['holding_costs'].append(td["holdingCost"].mean().item())
            self.test_metrics['ordering_costs'].append(td["orderingCost"].mean().item())
            self.test_metrics['stockout_costs'].append(td["stockOutPenalty"].mean().item())
            self.test_metrics['sales_revenue'].append(td["unitRevenue"].mean().item())
            self.test_metrics['total_costs'].append((td["holdingCost"] * current_stock + td["stockOutPenalty"] * torch.max(torch.zeros_like(current_demand), current_demand - current_stock)).mean().item())
            self.test_metrics['on_hand_levels'].append(td["onHandLevel"].mean().item())
            self.test_metrics['in_transit_levels'].append(td["inTransitStock"].mean().item())

        return td

    def addSequenceData(self, td, tensor, data):
        """
        Añade nuevos datos a la secuencia manteniendo un máximo de maxSeqLength elementos.
        
        Args:
            td: TensorDict con los datos del estado actual
            tensor: Tensor existente con la secuencia [batch_size, seq_len, embedding_dim]
            data: Nuevos datos a añadir [batch_size, 1, embedding_dim] o [batch_size, embedding_dim]
        
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
    
    def setInitalReturnToGo(self, td, returnsToGo):
        td["returnsToGo"] = returnsToGo if returnsToGo is not None else torch.zeros(self.batchSize, device=self.device)
