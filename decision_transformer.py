import torch.nn as nn
import torch
import numpy as np
from tensordict import TensorDict
from generate_tajectories import RETURN_TO_GO_WINDOW, FORECAST_LENGTH, MIN_DEMAND_MEAN, MAX_DEMAND_MEAN, MIN_DEMAND_STD, MAX_DEMAND_STD, MAX_LEAD_TIME, TRAJECTORY_LENGTH
from transformers import DecisionTransformerGPT2Model
from decision_transformer_config import DecisionTransformerConfig
from pathlib import Path


demand_mean = np.random.randint(MIN_DEMAND_MEAN, MAX_DEMAND_MEAN)
demand_std = np.random.randint(MIN_DEMAND_STD, MAX_DEMAND_STD)


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

        self.maxSeqLength = 40 #provisionalmente 



        # Projections
        #projection for the scalar data
        self.projectScalarData= nn.Linear(6, self.embeddingDim) # 6 values: onHandLevel, holdingCost, OrderCost, StockOutPenalty, UnitRevenue, leadTime
        #projection for the stock in transit
        self.projectStockInTransitData= nn.Linear(1, self.embeddingDim)
        #projection for the demand data

        #2 OPCIONES:
        self.projectDemandData = nn.Linear(FORECAST_LENGTH, self.embeddingDim) # FORECAST_LENGTH es el número de períodos de prevision de demanda
        #self.projectDemandData = nn.Linear(1, self.embeddingDim)
        
        #projection for the time data
        maxTimeLength = max(TRAJECTORY_LENGTH, MAX_LEAD_TIME) #hago esto para que el embedding de tiempo tenga suficiente capacidad para almacenar los valores de tiempo de todos los datos (demanda y stock en tránsito)
        #self.maxSeqLength hiperparametro que determina cuantos pasos de historia se tienen en cuenta para la predicción
        self.projectTimeData = nn.Embedding(maxTimeLength, self.embeddingDim) 


        # MHA used in the forward
        self.mhaState = nn.MultiheadAttention(
            embed_dim=self.embeddingDim,
            num_heads=4,
            batch_first=True
        )

        # Embeddings for the returns to go and the actions
        self.embeddingReturnsToGo = nn.Linear(1, self.embeddingDim)
        self.embeddingAction = nn.Linear(1, self.embeddingDim)  

        # project the embedding to a scalar value and apply a ReLU activation function to avoid negative values
        self.outputProjection = nn.Linear(self.embeddingDim, 1)  
        self.softplus = nn.Softplus()
        #self.relu = nn.ReLU()
        #PROBAR TAMBIEN CON SOFTPLUS NN.SOFTPLUS
    

        self.transformer = DecisionTransformerGPT2Model(decisionTransformerConfig)


    def initModel(self, td): 
        batchSize = td["leadTime"].size(0)   
        print(f" este es Batch size: {batchSize}")
        print(f"Lead time: {td['leadTime'][0][0]}")
        leadTime = int(td["leadTime"][0][0])
        print(f"Lead time: {leadTime}")
        device = getTorchDevice()
        if hasattr(td, 'clone'):
            tdNew = td.clone()
        else:
            tdNew = {k: v.clone() for k, v in td.items()}

        #initialize some values of the new tensor dict
        tdNew["currentTimestep"] = torch.zeros((batchSize, 1), dtype=torch.long) #initially the current timestep is 0
        tdNew["orderQuantity"] = torch.zeros((batchSize, 1), dtype=torch.float32)  # Cambiado a float32
        tdNew["onHandLevel"] = td["onHandLevel"]
        tdNew["inTransitStock"] = torch.zeros(batchSize, leadTime-1, dtype=torch.float32)  # Cambiado a float32
        tdNew["forecast"] = td["forecast"]
        tdNew["orderingCost"] = td["orderingCost"]
        tdNew["stockOutPenalty"] = td["stockOutPenalty"]
        tdNew["unitRevenue"] = td["unitRevenue"]
        tdNew["leadTime"] = td["leadTime"]
        tdNew["benefit"] = torch.zeros(batchSize, RETURN_TO_GO_WINDOW, dtype=torch.float32)
        tdNew["returnsToGo"] = td["returnsToGo"]
        tdNew["predictedAction"] = torch.zeros(batchSize, 1, dtype=torch.float32)

        # Initialize embeddings
        tdNew["statesEmbedding"] = torch.zeros((batchSize, 0, self.embeddingDim), dtype=torch.float, device=device)
        tdNew["actionsEmbedding"] = torch.zeros((batchSize, 0, self.embeddingDim), dtype=torch.float, device=device)
        tdNew["returnsToGoEmbedding"] = torch.zeros((batchSize, 0, self.embeddingDim), dtype=torch.float, device=device)
                             
        return tdNew
   
    def forward(self, td, nextOrderQuantity=None, is_test=False):
        """
        Forward pass del modelo con tres modos:
        1. Entrenamiento (self.training=True, is_test=False): Usa acciones reales y actualiza pesos
        2. Validación (self.training=False, is_test=False): Usa acciones reales sin actualizar pesos
        3. Test (is_test=True): Usa predicciones del modelo sin actualizar pesos
        
        Args:
            td: TensorDict con el estado actual
            nextOrderQuantity: Acción real para el siguiente paso (usado en entrenamiento y validación)
            is_test: Si True, usa predicciones del modelo
        """
        batchSize = td["statesEmbedding"].size(0)  
        leadTime = td["leadTime"][0][0].item() 
        print(f"Lead time: {leadTime}")
        print("\nEstado Inicial:")
        print(f"Timestep actual: {td['currentTimestep']}")
        print(f"Stock físico: {td['onHandLevel']}")
        print(f"InTransitStock original shape: {td['inTransitStock'].shape}")  # [batch_size, leadTime-1]
    

        # group the scalar data and get the demand and stock in transit data
        scalarData = torch.cat([
            td["onHandLevel"].unsqueeze(-1),
            td["holdingCost"].unsqueeze(-1),
            td["orderingCost"].unsqueeze(-1),
            td["stockOutPenalty"].unsqueeze(-1),
            td["unitRevenue"].unsqueeze(-1),
            td["leadTime"].unsqueeze(-1)
            ], dim=-1)
        print(f"Scalar data shape: {scalarData.shape}")
        

        demandData = td["forecast"]    
        stockInTransitData = td["inTransitStock"].float().unsqueeze(-1)

        batch_size = td["forecast"].shape[0]
        print(f"Scalar data shape: {scalarData.shape}")
        scalarDataProjection = self.projectScalarData(scalarData)
        #demandDataProjection = self.projectDemandData(demandData.view(-1, 1)) #solo si pongo 1 arriba en la linear
        demandDataProjection = self.projectDemandData(demandData)
        # demandDataProjection = demandDataProjection.view(batch_size, TRAJECTORY_LENGTH, FORECAST_LENGTH, -1)
        print(f"DEMAND DATA PROJECTION SHAPE: {demandDataProjection.shape}") # [1,3,5] batch_size = 1, TRAJECTORY_LENGTH = 3, FORECAST_LENGTH = 5
        #demandDataProjection = demandDataProjection.sum(dim=1)
        print(f"DEMAND DATA PROJECTION SHAPE: {demandDataProjection.shape}")
        stockInTransitDataProjection = self.projectStockInTransitData(stockInTransitData)

        # Reorganizar demandDataProjection manteniendo las dimensiones originales del forecast
        batch_size = td["forecast"].shape[0]
        forecast_length = td["forecast"].shape[1]
        print(f"Demand projection shape before reshape: {demandDataProjection.shape}")

       # demandDataProjection = demandDataProjection.view(batch_size, forecast_length, self.embeddingDim)

        print("\nProyecciones:")
        print(f"Batch size: {batch_size}")
        print(f"Forecast length: {forecast_length}")
        print(f"Embedding dim: {self.embeddingDim}")
        print(f"Demand projection shape: {demandDataProjection.shape}")

        # project the time data
        timeIndices = torch.arange(max(self.maxSeqLength, leadTime-1), device=td["forecast"].device).long()
        timeDataProjection = self.projectTimeData(timeIndices)

        print(f"\nDebug dimensiones:")
        print(f"timeDataProjection shape: {timeDataProjection.shape}")
        print(f"stockInTransitDataProjection shape: {stockInTransitDataProjection.shape}")
        print(f"leadTime: {leadTime}, type: {type(leadTime)}")

        # add the time data to the demand and the stock in transit data
        # Para demanda, usar solo hasta TRAJECTORY_LENGTH
        demandTimeEmbedding = demandDataProjection + timeDataProjection[:self.maxSeqLength].unsqueeze(0)
        # Para stock en tránsito, usar solo hasta leadTime-1
        stockInTransitTimeEmbedding = stockInTransitDataProjection + timeDataProjection[:min(int(leadTime)-1,self.maxSeqLength)].unsqueeze(0).expand_as(stockInTransitDataProjection)

        print(f"\nDemand time embedding shape: {demandTimeEmbedding.shape}")
        print(f"Stock in transit time embedding shape: {stockInTransitTimeEmbedding.shape}")

        # concatenate the demand and the stock in transit data
        demandStockTimeEmbedding = torch.cat([demandTimeEmbedding, stockInTransitTimeEmbedding], dim=1)

        print(f"\nDemand stock time embedding shape: {demandStockTimeEmbedding.shape}")

        # apply the MHA to the scalar data and the concatenated demand and stock in transit data
        mhaState, _ = self.mhaState(
            query=scalarDataProjection,
            key=demandStockTimeEmbedding,
            value=demandStockTimeEmbedding 
            )
        print(f"MHA output shape: {mhaState.shape}")
    

        td["statesEmbedding"] = self.addSequenceData(td, td["statesEmbedding"], mhaState)
        print(f"States embedding shape: {td['statesEmbedding'].shape}")
        returnsToGo = td["returnsToGo"].clone().float()
        print(f"Returns to go shape: {returnsToGo.shape}")
        
        # Reshape returns to go para que cada elemento del batch tenga su propio embedding
        batch_size = returnsToGo.size(0)
        returnsToGo = returnsToGo.view(-1, 1)  # [batch_size * seq_len, 1]
        embeddingsReturnsToGo = self.embeddingReturnsToGo(returnsToGo)  # [batch_size * seq_len, embedding_dim]
        embeddingsReturnsToGo = embeddingsReturnsToGo.view(batch_size, -1, self.embeddingDim)  # [batch_size, seq_len, embedding_dim]
        
        print(f"Embeddings returns to go shape: {embeddingsReturnsToGo.shape}")
        td["returnsToGoEmbedding"] = self.addSequenceData(td, td["returnsToGoEmbedding"],
                                                          embeddingsReturnsToGo)
        
        statesEmbedding = td["statesEmbedding"]
        returnsToGoEmbedding = td["returnsToGoEmbedding"]
        actionsEmbedding = td["actionsEmbedding"]

        # Calcular embeddings posicionales locales para este timestep
        positions = torch.arange(statesEmbedding.size(1), device=self.device)
        positionsEmbeddings = self.projectTimeData(positions)
        positionsEmbeddings = positionsEmbeddings.unsqueeze(0).expand(statesEmbedding.size(0), -1, -1)

        # Aplicar embeddings posicionales localmente
        statesEmbedding = statesEmbedding + positionsEmbeddings
        returnsToGoEmbedding = returnsToGoEmbedding + positionsEmbeddings
        # Para acciones, solo usar las posiciones hasta la longitud actual de actionsEmbedding
        actionsEmbedding = actionsEmbedding + positionsEmbeddings[:, :actionsEmbedding.size(1), :]
        
        print(f"\nDimensiones en forward:")
        print(f"returnsToGo shape: {returnsToGo.shape}")
        print(f"embeddingsReturnsToGo shape: {embeddingsReturnsToGo.shape}")
        print(f"returnsToGoEmbedding shape: {td['returnsToGoEmbedding'].shape}")

        # stack the returns to go embedding, the states embedding and the actions embedding
        if not self.training or is_test:
            stackedInputs = (
                torch.stack((returnsToGoEmbedding, statesEmbedding,
                             torch.cat((actionsEmbedding,
                                        torch.zeros(batchSize, statesEmbedding.size(1) - actionsEmbedding.size(1), self.embeddingDim, device=self.device)), dim=1)),
                            dim=1)
                .permute(0, 2, 1, 3)
                .reshape(batchSize, 3 * statesEmbedding.size(1), self.embeddingDim)
            )
        
            #apply the transformer to the stacked inputs
            output = self.transformer(inputs_embeds=stackedInputs) 
            output = output["last_hidden_state"]
            output = output.reshape(batchSize, statesEmbedding.size(1), 3, self.embeddingDim).permute(0, 2, 1, 3)
            output = output[:, 1, -1, :] #output = output[:, 2, -1, :]
            predictedAction = self.outputProjection(output)  # [batchSize, 1]
            predictedAction = self.softplus(predictedAction)
            orderQuantity = predictedAction
        
        else:
            stackedInputs = (
                torch.stack((returnsToGoEmbedding, statesEmbedding,
                             torch.cat((actionsEmbedding,
                                        torch.zeros(batchSize, statesEmbedding.size(1) - actionsEmbedding.size(1), self.embeddingDim, device=self.device)), dim=1)),
                            dim=1)
                .permute(0, 2, 1, 3)
                .reshape(batchSize, 3 * statesEmbedding.size(1), self.embeddingDim)
            )
        
            #apply the transformer to the stacked inputs
            output = self.transformer(inputs_embeds=stackedInputs) 
            output = output["last_hidden_state"]
            output = output.reshape(batchSize, statesEmbedding.size(1), 3, self.embeddingDim).permute(0, 2, 1, 3)
            output = output[:, 1, -1, :] #output = output[:, 2, -1, :]
            predictedAction = self.outputProjection(output)  # [batchSize, 1]
            predictedAction = self.softplus(predictedAction)
            orderQuantity = nextOrderQuantity

        if nextOrderQuantity is None:
            orderQuantity = predictedAction # Asegura que la cantidad a ordenar sea no negativa, lo multiplico por 100 para ver los resultados
        else: 
            orderQuantity = nextOrderQuantity 

        td["orderQuantity"] = orderQuantity
        td["predictedAction"] = predictedAction  # Guardar la predicción para referencia
        print("\nDecisión de Orden:")
        print(f"Cantidad ordenada: {td['orderQuantity']}")
    

        # Añadir la acción al embedding sin posición temporal (se añadirá localmente en el siguiente forward)
        actionEmbedding = self.embeddingAction(orderQuantity).unsqueeze(1)
        td["actionsEmbedding"] = self.addSequenceData(td, td["actionsEmbedding"], actionEmbedding)
        print(f"Actions embedding shape: {td['actionsEmbedding'].shape}")

        #Actualizar todos los datos
        
        print("\nActualización del Sistema:")
        print("Antes de actualizar:")
        print(f"Stock físico shape: {td['onHandLevel'].shape}")
        print(f"InTransitStock shape: {td['inTransitStock'].shape}")
        print(f"Forecast shape: {td['forecast'][...,0:1].shape}")

        # Asegurar que las dimensiones coincidan antes de sumar
        batch_idx = torch.arange(td["onHandLevel"].size(0), device=self.device)
        
        # 1. Primero, actualizar el stock en tránsito existente
        print("\n=== ACTUALIZACIÓN DE STOCK EN TRÁNSITO ===")
        for i in range(min(3, batch_size)):
            print(f"\nElemento {i}:")
            print(f"1. Estado INICIAL del stock en tránsito:")
            print(f"   - Stock en tránsito actual: {td['inTransitStock'][i].tolist()}")
        
        # Desplazar el stock en tránsito una posición
        td["inTransitStock"] = torch.roll(td["inTransitStock"], shifts=-1, dims=-1)
        
        # 2. Añadir la nueva orden al final del stock en tránsito
        td["inTransitStock"][..., -1] = td["orderQuantity"].squeeze()
        
        for i in range(min(3, batch_size)):
            print(f"2. Después de añadir nueva orden:")
            print(f"   - Stock en tránsito actualizado: {td['inTransitStock'][i].tolist()}")
            print(f"   - Nueva orden añadida: {td['orderQuantity'][i].item():.2f}")

        # 3. Actualizar el stock físico con el stock en tránsito que llega
        print("\n=== ACTUALIZACIÓN DE STOCK FÍSICO ===")
        for i in range(min(3, batch_size)):
            print(f"\nElemento {i}:")
            print(f"1. Estado INICIAL:")
            print(f"   - Stock físico: {td['onHandLevel'][i, 0].item():.2f}")
            print(f"   - Stock en tránsito que llega: {td['inTransitStock'][i, 0].item():.2f}")
            print(f"   - Lead time: {td['leadTime'][i, 0].item():.0f}")
        
        # Añadir el stock en tránsito que llega al stock físico
        stockToArrive = td["inTransitStock"][batch_idx, 0].unsqueeze(-1)
        td["onHandLevel"] + stockToArrive
         
        for i in range(min(3, batch_size)):
            print(f"2. Después de añadir stock en tránsito:")
            print(f"   - Stock físico actualizado: {td['onHandLevel'][i, 0].item():.2f}")
            print(f"   - Stock en tránsito restante: {td['inTransitStock'][i, 1:].tolist()}")

        # 4. Satisfacer la demanda
        currentTimeStep = td["currentTimestep"].long()
        current_demand = td["forecast"][batch_idx, currentTimeStep.squeeze(-1), 0]
        current_stock = td["onHandLevel"][batch_idx, currentTimeStep.squeeze(-1)]
        
        print("\n=== SATISFACCIÓN DE DEMANDA ===")
        for i in range(min(3, batch_size)):
            print(f"\nElemento {i}:")
            print(f"1. Antes de satisfacer demanda:")
            print(f"   - Stock disponible: {current_stock[i].item():.2f}")
            print(f"   - Demanda actual: {current_demand[i].item():.2f}")
        
        # Calcular stockout y satisfacer demanda
        income = (td["unitRevenue"][batch_idx, currentTimeStep.squeeze(-1)] * torch.min(current_demand, current_stock)).unsqueeze(-1)  # [batch_size, 1]
        stockout = torch.max(torch.zeros_like(current_demand), current_demand - current_stock).unsqueeze(-1)
        td["onHandLevel"] = torch.clamp(torch.sub(td["onHandLevel"], td["forecast"][..., 0]), min=0)

        
        for i in range(min(3, batch_size)):
            print(f"2. Después de satisfacer demanda:")
            print(f"   - Stock físico final: {td['onHandLevel'][i, 0].item():.2f}")
            print(f"   - Stockout: {stockout[i].item():.2f}")
            print(f"   - Demanda satisfecha: {min(current_demand[i].item(), current_stock[i].item()):.2f}")

        # Calcular holding cost y ordering cost para el timestep actual
        holdingCost = (td["holdingCost"][batch_idx, currentTimeStep.squeeze(-1)] * current_stock).unsqueeze(-1)  # [batch_size, 1]
        print(f"Holding cost shape: {holdingCost.shape}")
        
        # Calcular coste de pedido usando la orderQuantity ya seleccionada
        orderingCost = torch.where(
            orderQuantity.squeeze(-1) > 0,
            td["orderingCost"][batch_idx, currentTimeStep.squeeze(-1)],
            torch.zeros_like(td["orderingCost"][batch_idx, currentTimeStep.squeeze(-1)])
        ).unsqueeze(-1)

        # Calcular stockout penalty
        print("\n=== DEBUG STOCKOUT PENALTY ===")
        print(f"td['stockOutPenalty'] shape: {td['stockOutPenalty'].shape}")
        print(f"batch_idx shape: {batch_idx.shape}")
        print(f"currentTimeStep shape: {currentTimeStep.shape}")
        print(f"currentTimeStep.squeeze(-1) shape: {currentTimeStep.squeeze(-1).shape}")
        print(f"stockout shape: {stockout.shape}")
        stockoutPenalty = (td["stockOutPenalty"][batch_idx, currentTimeStep.squeeze(-1)].unsqueeze(-1) * stockout)
        print(f"Stockout penalty shape: {stockoutPenalty.shape}")

        currentTimeStep = td["currentTimestep"].long()
        print(f"Current timestep: {currentTimeStep}")

        # Calcular el beneficio actual
        benefitUpdate = (income - holdingCost - stockoutPenalty - orderingCost).float()
        print("\n=== DEBUG BENEFIT UPDATE ===")
        print(f"Income shape: {income.shape}")
        print(f"Holding cost shape: {holdingCost.shape}")
        print(f"Stockout penalty shape: {stockoutPenalty.shape}")
        print(f"Ordering cost shape: {orderingCost.shape}")
        print(f"Benefit update shape: {benefitUpdate.shape}")
        print(f"Benefit update values: {benefitUpdate[:3]}")  # Mostrar primeros 3 elementos

        # Actualizar el beneficio
        mask = currentTimeStep >= RETURN_TO_GO_WINDOW
        currentReturns = td["returnsToGo"][batch_idx, currentTimeStep.squeeze(-1)].unsqueeze(-1)  # [batch_size, 1]
        oldBenefit = td["benefit"][batch_idx, 0].unsqueeze(-1)  # [batch_size, 1]
        
        print("\n=== DEBUG RETURNS TO GO ===")
        print(f"Returns to go shape: {td['returnsToGo'].shape}")
        # Calcular nuevos returns manteniendo las dimensiones correctas
        newReturnsToGo = torch.where(
            mask,
            (currentReturns * RETURN_TO_GO_WINDOW - benefitUpdate + oldBenefit) / RETURN_TO_GO_WINDOW,
            currentReturns
        )  # [batch_size, 1]
        td["returnsToGo"][batch_idx, currentTimeStep.squeeze(-1)] = newReturnsToGo.squeeze(-1)

        if mask.any():
            td["benefit"] = torch.roll(td["benefit"], shifts=-1, dims=-1)
            # Actualizar el último valor con el beneficio nuevo
            print(f"\nBenefit tensor shape: {td['benefit'].shape}")
            td["benefit"][..., -1] = benefitUpdate.squeeze(-1)  # Asegurar que es [batch_size]
            print(f"Nuevo beneficio: {td['benefit'][..., -1][:3]}")  # Mostrar primeros 3 elementos
        else:
            # Si aún no llegamos a RETURN_TO_GO_WINDOW, actualizar normalmente
            td["benefit"][..., currentTimeStep] = benefitUpdate

            


        # Actualizar timestep y forecast
        td["currentTimestep"] = td["currentTimestep"] + 1
        td["forecast"] = torch.roll(td["forecast"], shifts=-1, dims=-1)
        new_forecast = torch.normal(
            mean=demand_mean, 
            std=demand_std, 
            size=(td["forecast"].shape[0], 1)
        )
        td["forecast"][..., -1] = new_forecast

        # Si estamos en modo test, guardar métricas adicionales
        if is_test:
            if not hasattr(td, 'test_metrics'):
                td['test_metrics'] = {
                    'holding_costs': [],
                    'ordering_costs': [],
                    'stockout_costs': [],
                    'sales_revenue': [],
                    'total_costs': [],
                    'on_hand_levels': [],
                    'in_transit_levels': []
                }
            
            td['test_metrics']['holding_costs'].append(td["holdingCost"][batch_idx, currentTimeStep.squeeze(-1)].mean().item())
            td['test_metrics']['ordering_costs'].append(td["orderingCost"][batch_idx, currentTimeStep.squeeze(-1)].mean().item())
            td['test_metrics']['stockout_costs'].append(td["stockOutPenalty"][batch_idx, currentTimeStep.squeeze(-1)].mean().item())
            td['test_metrics']['sales_revenue'].append(td["unitRevenue"][batch_idx, currentTimeStep.squeeze(-1)].mean().item())
            td['test_metrics']['total_costs'].append((td["holdingCost"][batch_idx, currentTimeStep.squeeze(-1)] * current_stock + td["stockOutPenalty"][batch_idx, currentTimeStep.squeeze(-1)] * torch.max(torch.zeros_like(current_demand), current_demand - current_stock)).mean().item())
            td['test_metrics']['on_hand_levels'].append(td["onHandLevel"].mean().item())
            td['test_metrics']['in_transit_levels'].append(td["inTransitStock"].mean().item())

        print("\nEstado Final:")
        print(f"Nuevo timestep: {td['currentTimestep']}")
        print(f"Stock físico final: {td['onHandLevel']}")
    
        print("\n=== Fin Forward Pass ===\n")

        print("\nVerificación de valores:")
        for i in range(min(5, batch_size)):  # Mostrar primeros 5 elementos
            print(f"\nElemento {i}:")
            print(f"  orderQuantity original: {td['orderQuantity'][i].item():.2f}")
            print(f"  inTransitStock final: {td['inTransitStock'][i, -1].item():.2f}")
            print(f"  Acción real: {nextOrderQuantity[i].item() if nextOrderQuantity is not None else 'None'}")
            print(f"  Acción predicha: {td['predictedAction'][i].item():.2f}")

        # 6. Verificación final
        print("\n=== VERIFICACIÓN FINAL ===")
        for i in range(min(3, batch_size)):
            print(f"\nElemento {i}:")
            print(f"  Stock físico final: {td['onHandLevel'][i, 0].item():.2f}")
            print(f"  Stock en tránsito: {td['inTransitStock'][i].tolist()}")
            print(f"  Nueva orden: {td['orderQuantity'][i].item():.2f}")
            print(f"  Lead time: {td['leadTime'][i, 0].item():.0f}")
            print(f"  Timestep actual: {td['currentTimestep'][i, 0].item():.0f}")

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
        print(f"\nDebug addSequenceData:")
        print(f"Tensor original shape: {tensor.shape}")
        print(f"Data to add shape: {data.shape}")
        
        # Asegurar que data tenga la forma correcta [batch_size, 1, embedding_dim]
        if data.dim() == 2:
            data = data.unsqueeze(1)
        
        # Si el tensor está vacío, simplemente devolver los datos
        if tensor.size(1) == 0:
            return data
            
        # Si el tensor está lleno, mantener solo los últimos maxSeqLength-1 elementos
        if tensor.size(1) >= self.maxSeqLength:
            tensor = tensor[:, -self.maxSeqLength+1:, :]
            # Asegurar que data tenga la misma forma que tensor
            data = data[:, :1, :]  # Tomar solo el primer elemento
            result = torch.cat((tensor, data), dim=1)
        else:
            result = torch.cat((tensor, data), dim=1)
            
        print(f"Result shape: {result.shape}")
        return result
    
    def setInitalReturnToGo(self, td, returnsToGo):
        td["returnsToGo"] = returnsToGo if returnsToGo is not None else torch.zeros(self.batchSize, device=self.device)



if __name__ == "__main__":
    print("\n" + "="*50)
    print("INICIO DE LA SIMULACIÓN")
    print("="*50)
    
    # 1. Cargar datos
    print("\n1. CARGANDO DATOS DE ENTRENAMIENTO")
    print("-"*30)
    def getProjectDirectory():
        return str(Path(__file__).resolve().parent)
    
    training_data_path = getProjectDirectory() + "/data/training_data.pt"
    try:
        dtData = torch.load(training_data_path)
        print("✓ Datos cargados exitosamente")
        print(f"  - Tipo: {type(dtData)}")
        print(f"  - Keys: {list(dtData.keys())}")
        
        problemData = dtData['states']
        orderQuantityData = dtData['actions']
        returnsToGoData = dtData['returnsToGo']
        
        print("\nDimensiones de los datos:")
        print(f"  - Actions: {orderQuantityData.shape}")
        print(f"  - ReturnsToGo: {returnsToGoData.shape}")
        print("\nEstructura de states:")
        for k in problemData.keys():
            print(f"- {k}: {problemData[k].shape if hasattr(problemData[k], 'shape') else 'No shape'}")
            
    except Exception as e:
        print(f"❌ Error al cargar datos: {str(e)}")
        raise e

    # 2. Inicializar modelo
    print("\n2. INICIALIZANDO MODELO")
    print("-"*30)
    config = DecisionTransformerConfig()
    model = DecisionTransformer(config)
    model.eval()
    print("✓ Modelo inicializado en modo evaluación")

    # 3. Preparar datos
    print("\n3. PREPARANDO DATOS")
    print("-"*30)
    device = getTorchDevice()
    print(f"  - Dispositivo: {device}")
    
    orderQuantityData = orderQuantityData.to(device)
    returnsToGoData = returnsToGoData.to(device)
    td = {k: v.to(device) for k, v in problemData.items()}
    print("✓ Datos movidos al dispositivo")

    # 4. Inicializar modelo con datos
    print("\n4. INICIALIZANDO MODELO CON DATOS")
    print("-"*30)
    model.setInitalReturnToGo(td, returnsToGoData)
    td = model.initModel(td)
    print("✓ Modelo inicializado con datos")

    # 5. Forward pass con ventanas
    print("\n5. INICIANDO SIMULACIÓN CON VENTANAS")
    print("-"*30)
    trajectory_length = orderQuantityData.size(1)
    window_size = model.maxSeqLength
    batch_size = orderQuantityData.size(0)
    
    print(f"Parámetros de simulación:")
    print(f"  - Longitud trayectoria: {trajectory_length}")
    print(f"  - Tamaño ventana: {window_size}")
    print(f"  - Batch size: {batch_size}")
    
    for window_start in range(0, trajectory_length, 1):
        window_end = window_start + window_size
        if window_end > trajectory_length:
            break
            
        print(f"\n{'='*20} VENTANA {window_start}-{window_end} {'='*20}")
        
        # Estado inicial de la ventana
        print("\nESTADO INICIAL DE LA VENTANA:")
        print("\nValores para los primeros 3 elementos del batch:")
        for i in range(min(3, batch_size)):
            print(f"\nElemento {i}:")
            print(f"  Stock físico inicial: {td['onHandLevel'][i, 0].item():.2f}")
            print(f"  Stock en tránsito que llega: {td['inTransitStock'][i, 0].item():.2f}")
            print(f"  Demanda actual: {td['forecast'][i, 0, 0].item():.2f}")
        
        # Recortar tensores
        for key in ["onHandLevel", "holdingCost", "orderingCost", "stockOutPenalty", 
                   "unitRevenue", "leadTime", "forecast", "inTransitStock", 
                   "returnsToGo", "actions"]:
            if key in td and td[key].dim() > 1 and td[key].shape[1] >= window_end:
                td[key] = td[key][:, window_start:window_end]
        
        # Actualizar timestep
        td["currentTimestep"] = torch.full((batch_size, 1), window_start, device=device)
        nextAction = orderQuantityData[:, window_end:window_end+1]
        
        # Forward pass
        td = model.forward(td, nextOrderQuantity=nextAction)
        
        # Estado final de la ventana
        print("\nESTADO FINAL DE LA VENTANA:")
        print(f"  Timestep: {window_start}")
        print(f"  Stock físico: {td['onHandLevel'][:, -1].mean().item():.2f}")
        print(f"  Stock en tránsito: {td['inTransitStock'][:, -1].mean().item():.2f}")
        print(f"  Cantidad ordenada: {td['orderQuantity'].mean().item():.2f}")
        print(f"  Acción real: {nextAction.mean().item():.2f}")
        print(f"  Acción predicha: {td['predictedAction'].mean().item():.2f}")
        
        # Detalles de algunos elementos del batch
        print("\nDETALLES DE ELEMENTOS DEL BATCH:")
        for i in range(min(3, batch_size)):
            print(f"\nElemento {i}:")
            print(f"  Stock físico: {td['onHandLevel'][i, -1].item():.2f}")
            print(f"  Stock en tránsito: {td['inTransitStock'][i, -1].mean().item():.2f}")
            print(f"  Cantidad ordenada: {td['orderQuantity'][i].item():.2f}")
            print(f"  Acción real: {nextAction[i].item():.2f}")
            print(f"  Acción predicha: {td['predictedAction'][i].item():.2f}")
            print(f"  Demanda actual: {td['forecast'][i, 0, 0].item():.2f}")
        
        # Pausa entre ventanas
        if window_end < trajectory_length - 1:
            input("\nPresiona Enter para continuar a la siguiente ventana...")

    # 6. Métricas finales
    if 'test_metrics' in td:
        print("\n" + "="*50)
        print("MÉTRICAS FINALES DE LA SIMULACIÓN")
        print("="*50)
        metrics = td['test_metrics']
        print("\nCostes e ingresos medios:")
        print(f"  - Coste almacenamiento: {np.mean(metrics['holding_costs']):.2f}")
        print(f"  - Coste pedido: {np.mean(metrics['ordering_costs']):.2f}")
        print(f"  - Coste rotura: {np.mean(metrics['stockout_costs']):.2f}")
        print(f"  - Ingresos ventas: {np.mean(metrics['sales_revenue']):.2f}")
        print(f"  - Coste total: {np.mean(metrics['total_costs']):.2f}")
        print("\nNiveles de stock medios:")
        print(f"  - Stock físico: {np.mean(metrics['on_hand_levels']):.2f}")
        print(f"  - Stock en tránsito: {np.mean(metrics['in_transit_levels']):.2f}")

    print("\n" + "="*50)
    print("FIN DE LA SIMULACIÓN")
    print("="*50)


#long contexto
#ventana de contexto indices de tiempo
#CAMBIAR RELU A LA QUE MULTIPLICA POR LA PENDIENTE HECHO
#BUCLE ACTUALIZACION DE RETURNS TO GO HECHO
#AUMENTAR EL BATCH SIZE HECHO 
