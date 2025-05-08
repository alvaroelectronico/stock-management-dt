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

        self.maxSeqLength = 15



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
        #CAMBIAR ESTA RELU A LA QUE MULTIPLICA POR LA PENDIENTE
        #PROBAR TAMBIEN CON SOFTPLUS NN.SOFTPLUS
    

        self.transformer = DecisionTransformerGPT2Model(decisionTransformerConfig)


    def initModel(self, td): 
        batchSize = td["leadTime"].size(0) #change this constant to the batch size of the data  
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
        print(f"Stock en tránsito: {td['inTransitStock']}")
        print(f"Forecast: {td['forecast']}")
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
        
        print(f"\nScalar data: {scalarData}")

        demandData = td["forecast"]    
        stockInTransitData = td["inTransitStock"].float().unsqueeze(-1)

        batch_size = td["forecast"].shape[0]
        # project the scalar data, the demand and the stock in transit data
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
        timeIndicesDemand = torch.arange(TRAJECTORY_LENGTH, device=td["forecast"].device).long() #no estoy segura de si es así
        timeIndicesStockInTransit = torch.arange(leadTime-1, device=td["inTransitStock"].device).long() #considerando que todos los elementos del batch tienen el mismo lead time
    
        timeDataProjectionDemand = self.projectTimeData(timeIndicesDemand) 
        timeDataProjectionStockInTransit = self.projectTimeData(timeIndicesStockInTransit) 

        # add the time data to the demand and the stock in transit data
        demandTimeEmbedding = demandDataProjection + timeDataProjectionDemand.unsqueeze(0) #[batch_size, FORECAST_LENGTH, embedding_dim] + [1, FORECAST_LENGTH, embedding_dim] = [batch_size, FORECAST_LENGTH, embedding_dim]
        StockInTransitTimeEmbedding = stockInTransitDataProjection + timeDataProjectionStockInTransit.unsqueeze(0) # [batch_size, lead_time-1, embedding_dim] + [1, lead_time-1, embedding_dim] = [batch_size, lead_time-1, embedding_dim]

        print(f"\nDemand time embedding shape: {demandTimeEmbedding.shape}")
        print(f"Stock in transit time embedding shape: {StockInTransitTimeEmbedding.shape}")

        # concatenate the demand and the stock in transit data
        demandStockTimeEmbedding = torch.cat([demandTimeEmbedding, StockInTransitTimeEmbedding], dim=1)

        print(f"\nDemand stock time embedding shape: {demandStockTimeEmbedding.shape}")

        # apply the MHA to the scalar data and the concatenated demand and stock in transit data
        mhaState, _ = self.mhaState(
            query=scalarDataProjection,
            key=demandStockTimeEmbedding,
            value=demandStockTimeEmbedding 
            )
        print("\nDespués de MHA:")
        print(f"States embedding shape: {td['statesEmbedding'].shape}")
        print(f"Actions embedding shape: {td['actionsEmbedding'].shape}")
        print(f"Returns-to-go embedding shape: {td['returnsToGoEmbedding'].shape}")
    

        td["statesEmbedding"] = self.addSequenceData(td, td["statesEmbedding"], mhaState)
        print(f"States embedding shape: {td['statesEmbedding'].shape}")
        returnsToGo=td["returnsToGo"].float()
        print(f"Returns to go shape: {returnsToGo.shape}")
        print(f"Returns to go reshape: {returnsToGo.reshape(-1,1).shape}")
        embeddingsReturnsToGo = self.embeddingReturnsToGo(returnsToGo.reshape(-1,1)).unsqueeze(1).permute(1,0,2)
        print(f"Embeddings returns to go shape: {embeddingsReturnsToGo.shape}")
        td["returnsToGoEmbedding"] = self.addSequenceData(td, td["returnsToGoEmbedding"],
                                                          embeddingsReturnsToGo)
        
        print(f"\nDimensiones en forward:")
        print(f"returnsToGo shape: {returnsToGo.shape}")
        print(f"embeddingsReturnsToGo shape: {embeddingsReturnsToGo.shape}")
        print(f"returnsToGoEmbedding shape: {td['returnsToGoEmbedding'].shape}")

        # stack the returns to go embedding, the states embedding and the actions embedding
        if not self.training:
            stackedInputs = (
                torch.stack((td["returnsToGoEmbedding"], td["statesEmbedding"],
                             torch.cat((td["actionsEmbedding"],
                                        torch.zeros(batchSize, td["statesEmbedding"].size(1) - td["actionsEmbedding"].size(1), self.embeddingDim, device=self.device)), dim=1)),
                            dim=1)
                .permute(0, 2, 1, 3)
                .reshape(batchSize, 3 * td["statesEmbedding"].size(1), self.embeddingDim)
            )
        
            #apply the transformer to the stacked inputs
            output = self.transformer(inputs_embeds=stackedInputs) 
            output = output["last_hidden_state"]
            output = output.reshape(batchSize, td["statesEmbedding"].size(1), 3, self.embeddingDim).permute(0, 2, 1, 3)
            output = output[:, 1, -1, :] #output = output[:, 2, -1, :]
            orderQuantity = self.outputProjection(output)  # [batchSize, 1]
            orderQuantity = self.softplus(orderQuantity)
            predictedAction = orderQuantity
        
        else:
            stackedInputs = (
                torch.stack((td["returnsToGoEmbedding"], td["statesEmbedding"],
                             torch.cat((td["actionsEmbedding"],
                                        torch.zeros(batchSize, td["statesEmbedding"].size(1) - td["actionsEmbedding"].size(1), self.embeddingDim, device=self.device)), dim=1)),
                            dim=1)
                .permute(0, 2, 1, 3)
                .reshape(batchSize, 3 * td["statesEmbedding"].size(1), self.embeddingDim)
            )
        
            #apply the transformer to the stacked inputs
            output = self.transformer(inputs_embeds=stackedInputs) 
            output = output["last_hidden_state"]
            output = output.reshape(batchSize, td["statesEmbedding"].size(1), 3, self.embeddingDim).permute(0, 2, 1, 3)
            output = output[:, 1, -1, :] #output = output[:, 2, -1, :]
            predictedAction = self.outputProjection(output)  # [batchSize, 1]
            predictedAction = self.softplus(predictedAction)
            orderQuantity = nextOrderQuantity

        if nextOrderQuantity is None:
            orderQuantity = predictedAction # Asegura que la cantidad a ordenar sea no negativa, lo multiplico por 100 para ver los resultados
        else: 
            orderQuantity = nextOrderQuantity      

        td["orderQuantity"] = orderQuantity
        print("\nDecisión de Orden:")
        print(f"Cantidad ordenada: {td['orderQuantity']}")
    

        # project the order quantity to an embedding and add it to the actions embedding
        actionEmbedding = self.embeddingAction(orderQuantity).unsqueeze(1)  # Necesitas añadir esta capa en __init__
        td["actionsEmbedding"] = self.addSequenceData(td, td["actionsEmbedding"], actionEmbedding)
        print(f"Actions embedding shape: {td['actionsEmbedding'].shape}")

        #Actualizar todos los datos
        
        print("\nActualización del Sistema:")
        print("Antes de actualizar:")
        print(f"Stock físico: {td['onHandLevel']}")
        print(f"Stock físico shape: {td['onHandLevel'].shape}")
        print(f"Forecast: {(td['forecast'][...,0:1]).shape}")
        print(f"Forecast: {td['forecast'][...,0:1]}")
    

        #td["onHandLevel"] = torch.add(td["onHandLevel"], td["inTransitStock"][...,0:1])
        td["onHandLevel"] = torch.add(td["onHandLevel"], td["inTransitStock"][...,0])
        print(f"Stock físico shape: {td['onHandLevel'].shape}")
        stockOutPenalty = (td["stockOutPenalty"] * torch.max(torch.zeros_like(td["forecast"][..., 0]), td["forecast"][..., 0] - td["onHandLevel"]))
        income = td["unitRevenue"] * torch.min(td["forecast"][..., 0], td["onHandLevel"])

        td["onHandLevel"] = torch.clamp(torch.sub(td["onHandLevel"], td["forecast"][..., 0]), min=0) 

        holdingCost = (td["holdingCost"] * td["onHandLevel"])
        orderingCost = torch.where(
            orderQuantity > 0,
            td["orderingCost"],  # Si hay pedido
            torch.zeros_like(td["orderingCost"])  # Si no hay pedido
            )

        print("\nDespués de recibir stock en tránsito:")
        print(f"Stock físico: {td['onHandLevel']}")
        print(f"Stock físico shape: {td['onHandLevel'].shape}")
        print("\nCostes y Beneficios:")
        print(f"Penalización por rotura: {stockOutPenalty}")
        print(f"Ingresos: {income}")
        print(f"Coste de almacenamiento: {holdingCost}")
        print(f"Coste de pedido: {orderingCost}")
        print(f"Return-to-go actual: {td['returnsToGo']}")
    
    
        currentTimeStep = td["currentTimestep"].long()
        print(f"Current timestep: {currentTimeStep}")
        # Calcular el beneficio actual
        benefitUpdate = (income - holdingCost - stockOutPenalty - orderingCost).float()
        # Asegurar que benefitUpdate tenga la forma [batch_size, 1]
        benefitUpdate = benefitUpdate[:,currentTimeStep]
        print(f"Beneficio actualizado: {benefitUpdate}")
        print(f"Beneficio: {td['benefit']}")
        print(f"Beneficio shape: {td['benefit'].shape}")
        print(f"currentTimestep shape: {currentTimeStep.shape}")

        returnToGo = td["returnsToGo"]

        mask = currentTimeStep >= RETURN_TO_GO_WINDOW
        td["returnsToGo"] = torch.where(
            mask,
            (returnToGo*RETURN_TO_GO_WINDOW - benefitUpdate + 
             td["benefit"][...,0])/RETURN_TO_GO_WINDOW,
            returnToGo
        )
        if mask.any():
            oldBenefit = td["benefit"][...,0]
            # Hacer roll del beneficio
            td["benefit"] = torch.roll(td["benefit"], shifts=-1, dims=-1)
            # Actualizar el último valor con el beneficio nuevo
            print(f"Beneficio actualizado: {benefitUpdate}")
            print(f"Ultimo beneficio: {td['benefit'][..., -1]}")
            td["benefit"][..., -1] = benefitUpdate.squeeze()
            print(f"Nuevo beneficio: {td['benefit'][..., -1]}")
            td["returnsToGo"] = torch.where(
                mask,
                (returnToGo*RETURN_TO_GO_WINDOW - benefitUpdate + 
                 oldBenefit)/RETURN_TO_GO_WINDOW,  # Usamos el penúltimo valor que acabamos de desplazar
                returnToGo)
        else:
            # Si aún no llegamos a RETURN_TO_GO_WINDOW, actualizar normalmente
            td["benefit"][...,currentTimeStep] = benefitUpdate
            #for b in range(td["benefit"].size(0)):
                #td["benefit"][b, currentTimeStep[b, 0]] = benefitUpdate[b, 0]


        print(f"Beneficio: {td['benefit']}")
        print(f"Return-to-go actual: {td['returnsToGo']}")
    


         # Actualizar stock en tránsito
        # Desplazar el tensor una posición (representa el paso del tiempo)
        td["inTransitStock"] = torch.roll(td["inTransitStock"], shifts=-1, dims=-1)
        # La última posición se pone a cero ya que es nueva
        td["inTransitStock"][..., -1] = 0
        # Añadir la nueva orden en la última posición
        td["inTransitStock"][..., -1] = td["orderQuantity"].squeeze()        
        
        td["currentTimestep"] = td["currentTimestep"] + 1

        td["forecast"] = torch.roll(td["forecast"], shifts=-1, dims=-1)
        td["forecast"][..., -1] = torch.normal(mean=demand_mean, std=demand_std, size=(td["forecast"].shape[0],))   #el size sirve para que se genere un valor por cada batch

        print("\nEstado Final:")
        print(f"Nuevo timestep: {td['currentTimestep']}")
        print(f"Stock físico final: {td['onHandLevel']}")
        print(f"Stock en tránsito actualizado: {td['inTransitStock']}")
        print(f"Nuevo forecast: {td['forecast']}")
    
        print("\n=== Fin Forward Pass ===\n")


        return td

    def addSequenceData(self, td, tensor, data):
        print(f"\nDebug addSequenceData:")
        print(f"Tensor original shape: {tensor.shape}")
        print(f"Data to add shape: {data.shape}")

        currentTimestep = td["currentTimestep"].long()
        
        # Asegurar que data tenga la forma correcta [batch_size, 1, embedding_dim]
        if data.dim() == 2:
            data = data.unsqueeze(1)
        
        # Asegurar que el tensor de tiempo tenga la forma correcta
        time_embedding = self.projectTimeData(currentTimestep)
        if time_embedding.dim() == 2:
            time_embedding = time_embedding.unsqueeze(1)
            
        # Sumar data y time_embedding
        new_data = data + time_embedding
        
        # Manejar el caso cuando el tensor está lleno
        if tensor.size(1) >= self.maxSeqLength:
            # Mantener solo los últimos maxSeqLength-1 elementos
            tensor = tensor[:, -self.maxSeqLength+1:, :]
            # Asegurar que new_data tenga la misma forma que tensor
            new_data = new_data[:, :1, :]  # Tomar solo el primer elemento
            result = torch.cat((tensor, new_data), dim=1)
        else:
            result = torch.cat((tensor, new_data), dim=1)
            
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
    
    print("\n--- Estados iniciales ---")
    #for key, value in td['states'].items():
     #   print(f"{key}: {value}")
    
    print("\n--- Acciones y Returns ---")
    print(f"Actions shape: {td['actions'].shape}")
    print(f"Returns to go shape: {td['returnsToGo'].shape}")
    
    # 3. Crear modelo y ejecutar initModel
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
        input("Presiona Enter para continuar al siguiente paso...")


#long contexto
#ventana de contexto indices de tiempo
#CAMBIAR RELU A LA QUE MULTIPLICA POR LA PENDIENTE HECHO
#BUCLE ACTUALIZACION DE RETURNS TO GO HECHO
#AUMENTAR EL BATCH SIZE HECHO 
