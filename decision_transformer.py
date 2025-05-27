import torch.nn as nn
import torch
import numpy as np
from tensordict import TensorDict
from generate_tajectories import RETURN_TO_GO_WINDOW, FORECAST_LENGTH, MIN_DEMAND_MEAN, MAX_DEMAND_MEAN, MIN_DEMAND_STD, MAX_DEMAND_STD, MAX_LEAD_TIME, TRAJECTORY_LENGTH
from transformers import DecisionTransformerGPT2Model
from decision_transformer_config import DecisionTransformerConfig


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
   
    def forward(self, td, nextOrderQuantity=None): 

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
        if not self.training:
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
            orderQuantity = self.outputProjection(output)  # [batchSize, 1]
            orderQuantity = self.softplus(orderQuantity)
            #orderQuantity = self.relu(orderQuantity)
            predictedAction = orderQuantity
        
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
            #predictedAction = self.relu(predictedAction)
            orderQuantity = nextOrderQuantity

        if nextOrderQuantity is None:
            orderQuantity = predictedAction # Asegura que la cantidad a ordenar sea no negativa, lo multiplico por 100 para ver los resultados
        else: 
            orderQuantity = nextOrderQuantity      

        td["orderQuantity"] = orderQuantity
        td["predictedAction"] = predictedAction
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
        
        # Actualizar onHandLevel con el stock en tránsito que llega
        td["onHandLevel"] = td["onHandLevel"] + td["inTransitStock"][batch_idx, 0].unsqueeze(-1)
        
        # Calcular stockout y income para el timestep actual
        # Obtener el índice del timestep actual para cada batch
        currentTimeStep = td["currentTimestep"].long()

        #current_demand = td["forecast"][..., currentTimeStep[0][0], 0]  # [batch_size]
        current_demand = td["forecast"][batch_idx, currentTimeStep.squeeze(-1), 0]
        print(f"Current demand shape: {current_demand.shape}")
        current_stock = td["onHandLevel"][batch_idx, currentTimeStep.squeeze(-1)]  # [batch_size]
        print(f"Current stock shape: {current_stock.shape}")
        # Calcular income y stockout para el timestep actual
        stockOutPenalty = (td["stockOutPenalty"][batch_idx, currentTimeStep.squeeze(-1)] * torch.max(torch.zeros_like(current_demand), 
                                                           current_demand - current_stock)).unsqueeze(-1)  # [batch_size, 1]
        print(f"Stock out penalty shape: {stockOutPenalty.shape}")
        income = (td["unitRevenue"][batch_idx, currentTimeStep.squeeze(-1)] * torch.min(current_demand, current_stock)).unsqueeze(-1)  # [batch_size, 1]
        print(f"Income shape: {income.shape}")

        td["onHandLevel"] = torch.clamp(torch.sub(td["onHandLevel"], td["forecast"][..., 0]), min=0) 

        # Calcular holding cost y ordering cost para el timestep actual
        holdingCost = (td["holdingCost"][batch_idx, currentTimeStep.squeeze(-1)] * current_stock).unsqueeze(-1)  # [batch_size, 1]
        print(f"Holding cost shape: {holdingCost.shape}")
        
        # Corregir el cálculo de orderingCost para que tenga forma [batch_size, 1]
        orderingCost = torch.where(
            orderQuantity.squeeze(-1) > 0,  # [batch_size]
            td["orderingCost"][batch_idx, currentTimeStep.squeeze(-1)],  # [batch_size]
            torch.zeros_like(td["orderingCost"][batch_idx, currentTimeStep.squeeze(-1)])  # [batch_size]
        ).unsqueeze(-1)  # [batch_size, 1]
        print(f"Ordering cost shape: {orderingCost.shape}")
        print("\nDespués de recibir stock en tránsito:")
        print(f"Stock físico shape: {td['onHandLevel'].shape}")
    
    
        currentTimeStep = td["currentTimestep"].long()
        print(f"Current timestep: {currentTimeStep}")
        
        # Calcular el beneficio actual (ahora será [batch_size, 1])
        benefitUpdate = (income - holdingCost - stockOutPenalty - orderingCost).float()
        print(f"BenefitUpdate shape: {benefitUpdate.shape}")  # Debería ser [batch_size, 1]
        
        
        # Calcular el beneficio medio hasta el momento actual
        mean_benefit = td["benefit"][..., :int(currentTimeStep[0][0])+1].mean(dim=-1, keepdim=True)  # [batch_size, 1]
        print(f"Mean benefit shape: {mean_benefit.shape}")
        
        returnToGo = td["returnsToGo"]  # [batch_size, seq_len]
        print(f"Returns to go shape antes de actualizar: {returnToGo.shape}")

        # Asegurar que las dimensiones coincidan
        mask = currentTimeStep >= RETURN_TO_GO_WINDOW
        print(f"\nDimensiones de los tensores:")
        print(f"returnToGo shape: {returnToGo.shape}")
        print(f"benefitUpdate shape: {benefitUpdate.shape}")
        print(f"td['benefit'][...,0] shape: {td['benefit'][...,0].shape}")
        print(f"mask shape: {mask.shape}")
        print(f"currentTimeStep shape: {currentTimeStep.shape}")

        # Actualizar returnsToGo usando el beneficio medio y el beneficio actual
        current_returns = returnToGo[batch_idx, currentTimeStep.squeeze(-1)].unsqueeze(-1)  # [batch_size, 1]
        current_benefit = td["benefit"][batch_idx, 0].unsqueeze(-1)  # [batch_size, 1]
        
        new_returns = torch.where(
            mask,
            (current_returns * RETURN_TO_GO_WINDOW - benefitUpdate + current_benefit) / RETURN_TO_GO_WINDOW,
            current_returns
        ).squeeze(-1)  # [batch_size]
        new_returns_to_go = td["returnsToGo"].clone()
        new_returns_to_go[batch_idx, currentTimeStep.squeeze(-1)] = new_returns
        td["returnsToGo"] = new_returns_to_go

        if mask.any():
            # Hacer roll del beneficio
            td["benefit"] = torch.roll(td["benefit"], shifts=-1, dims=-1)
            # Actualizar el último valor con el beneficio nuevo
            print(f"Beneficio actualizado: {benefitUpdate}")
            print(f"Ultimo beneficio: {td['benefit'][..., -1]}")
            td["benefit"][..., -1] = benefitUpdate.squeeze()
            print(f"Nuevo beneficio: {td['benefit'][..., -1]}")
        else:
            # Si aún no llegamos a RETURN_TO_GO_WINDOW, actualizar normalmente
            td["benefit"][...,currentTimeStep] = benefitUpdate

        print(f"Returns to go shape después de actualizar: {td['returnsToGo'].shape}")

        # Actualizar stock en tránsito
        # Desplazar el tensor una posición (representa el paso del tiempo)
        td["inTransitStock"] = torch.roll(td["inTransitStock"], shifts=-1, dims=-1)
        # La última posición se pone a cero ya que es nueva
        td["inTransitStock"][..., -1] = 0
        # Añadir la nueva orden en la última posición
        td["inTransitStock"][..., -1] = td["orderQuantity"].squeeze()        
        
        td["currentTimestep"] = td["currentTimestep"] + 1

        td["forecast"] = torch.roll(td["forecast"], shifts=-1, dims=-1)
        # Generar nuevo forecast con la forma correcta [batch_size, FORECAST_LENGTH]
        new_forecast = torch.normal(
            mean=demand_mean, 
            std=demand_std, 
            size=(td["forecast"].shape[0], 1)
        )
        td["forecast"][..., -1] = new_forecast

        print("\nEstado Final:")
        print(f"Nuevo timestep: {td['currentTimestep']}")
        print(f"Stock físico final: {td['onHandLevel']}")
    
        print("\n=== Fin Forward Pass ===\n")

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
    # 1. Crear datos de prueba
    print("\n=== Generando datos de prueba ===")
    
    # 2. Crear TensorDict inicial con la estructura correcta
    batch_size = 1  # Batch size = 1 como en generate_tajectories.py
    trajectory_length = TRAJECTORY_LENGTH  # Usar la constante definida
    
    # Crear el TensorDict con la estructura correcta
    print("\n=== Generando datos de prueba ===")
    
    batch_size = 1
    td = TensorDict({
        'batch_size': torch.tensor([batch_size]),
        'leadTime': torch.tensor([[15]]),
        
        # Estados iniciales
        'onHandLevel': torch.tensor([[50.0]]),  # Stock inicial
        'inTransitStock': torch.zeros(batch_size, 15-1),  # Stock en tránsito
        'demand': torch.tensor([[20.0]]),  # Demanda actual
        'forecast': torch.normal(mean=demand_mean, std=demand_std, size=(batch_size, FORECAST_LENGTH)),  # Previsión de demanda
        'holdingCost': torch.tensor([[5.0]]),  # Coste de almacenamiento
        'orderingCost': torch.tensor([[100.0]]),  # Coste de pedido
        'stockOutPenalty': torch.tensor([[50.0]]),  # Penalización por rotura
        'unitRevenue': torch.tensor([[20.0]]),  # Ingreso unitario
        'timesStep': torch.tensor([[0]]),  # Paso de tiempo inicial
        
        
        # Acciones y returns
        'actions': torch.zeros(batch_size, trajectory_length),  # Acciones iniciales
        'returnsToGo': torch.zeros(batch_size, trajectory_length)  # Returns to go iniciales
    }, batch_size=[batch_size])
    
    print("\n=== Valores iniciales ===")
    print(f"Batch size: {td['batch_size']}")
    print(f"Trajectory length: {trajectory_length}")
    # Estado inicial del sistema
    #'onHandLevel': torch.tensor([[100], [150]]),               # Stock inicial diferente para cada batch
    #'inTransitStock': torch.zeros(batch_size, 4),  # Sin pedidos en tránsito inicialmente
    #'forecast': torch.tensor([[20.0, 22.0, 18.0, 25.0, 21.0]] * batch_size),  # Previsión de demanda para 5 períodos
    #'returnsToGo':torch.tensor([[500]] * batch_size)
    #})
    
    print(f"inTransitStock shape: {td['inTransitStock'].shape}")
   
    
    print("\n=== Valores iniciales ===")
    print(f"Batch size: {td['batch_size']}")
    print(f"\nParámetros del sistema:")
    print(f"Holding Cost: {td['holdingCost']}")
    print(f"Ordering Cost: {td['orderingCost']}")
    print(f"Unit Revenue: {td['unitRevenue']}")
    print(f"Lead Time: {td['leadTime']}")
    print(f"returnsToGo: {td['returnsToGo']}")
    
    print("\n--- Estados iniciales ---")
    #for key, value in td['states'].items():
     #   print(f"{key}: {value}")
    
    print("\n--- Acciones y Returns ---")
    print(f"Actions shape: {td['actions'].shape}")
    print(f"Returns to go shape: {td['returnsToGo'].shape}")
    
    print("\n=== Inicializando modelo ===")
    config = DecisionTransformerConfig()
    model = DecisionTransformer(config)
    model.eval()
    tdNew = model.initModel(td)
    
    # 4. Ejecutar el forward pass
    print("\n=== Iniciando Forward Pass ===")
    for step in range(trajectory_length):
        print(f"\nPaso {step}")
        tdNew = model.forward(tdNew)
        
        # Opcional: hacer una pausa entre pasos
    print("\n=== Verificando inicialización ===")
    print(f"Batch size: {tdNew['batch_size']}")
    
    print("\n--- Tensores inicializados a cero ---")
    print(f"currentTimestep shape: {tdNew['currentTimestep'].shape}")
    print(f"orderQuantity shape: {tdNew['orderQuantity'].shape}")
    print(f"onHandLevel shape: {tdNew['onHandLevel'].shape}")
    print(f"inTransitStock shape: {tdNew['inTransitStock'].shape}")
    print(f"forecast shape: {tdNew['forecast'].shape}")
    
    print("\n--- Embeddings iniciales ---")
    print(f"statesEmbedding shape: {tdNew['statesEmbedding'].shape}")
    print(f"actionsEmbedding shape: {tdNew['actionsEmbedding'].shape}")
    print(f"returnsToGoEmbedding shape: {tdNew['returnsToGoEmbedding'].shape}")
    
    print("\n--- Valores constantes ---")
    print(f"holdingCost: {tdNew['holdingCost']}")
    print(f"orderingCost: {tdNew['orderingCost']}")
    print(f"stockOutPenalty: {tdNew['stockOutPenalty']}")
    print(f"unitRevenue: {tdNew['unitRevenue']}")
    print(f"leadTime: {tdNew['leadTime']}")
    
    print("\n=== Verificando valores ===")
    print("Todos los tensores deberían ser cero:")
    print(f"currentTimestep sum: {tdNew['currentTimestep'].sum()}")
    print(f"orderQuantity sum: {tdNew['orderQuantity'].sum()}")
    print(f"onHandLevel sum: {tdNew['onHandLevel'].sum()}")
    
    print("\nTodos los embeddings deberían estar vacíos (dim 1 = 0):")
    print(f"statesEmbedding size[1]: {tdNew['statesEmbedding'].size(1)}")
    print(f"actionsEmbedding size[1]: {tdNew['actionsEmbedding'].size(1)}")
    print(f"returnsToGoEmbedding size[1]: {tdNew['returnsToGoEmbedding'].size(1)}")

    print("\n=== Iniciando Forward Pass ===")
    for step in range(5):
        print(f"\nPaso {step}")
        tdNew = model.forward(tdNew)
        input("Presiona Enter para continuar al siguiente paso...")


#long contexto
#ventana de contexto indices de tiempo
#CAMBIAR RELU A LA QUE MULTIPLICA POR LA PENDIENTE HECHO
#BUCLE ACTUALIZACION DE RETURNS TO GO HECHO
#AUMENTAR EL BATCH SIZE HECHO 
