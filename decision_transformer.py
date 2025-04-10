import torch.nn as nn
import torch
import numpy as np
from tensordict import TensorDict
from generate_tajectories import RETURN_TO_GO_WINDOW, FORECAST_LENGHT, MIN_DEMAND_MEAN, MAX_DEMAND_MEAN, MIN_DEMAND_STD, MAX_DEMAND_STD, MAX_LEAD_TIME, TRAYECTORY_LENGHT
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
        self.projectDemandData = nn.Linear(FORECAST_LENGHT, self.embeddingDim) # FORECAST_LENGHT es el número de períodos de prevision de demanda
        #self.projectDemandData = nn.Linear(1, self.embeddingDim)
        
        #projection for the time data
        maxTimeLength = max(TRAYECTORY_LENGHT, MAX_LEAD_TIME) #hago esto para que el embedding de tiempo tenga suficiente capacidad para almacenar los valores de tiempo de todos los datos (demanda y stock en tránsito)
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
        self.relu = nn.ReLU()

        self.transformer = DecisionTransformerGPT2Model(decisionTransformerConfig)


    def initModel(self, td): 
        batchSize = td["leadTime"].size(0) #change this constant to the batch size of the data  
        print(f" este es Batch size: {batchSize}")
        print(f"Lead time: {td['leadTime'][0][0]}")
        leadTime = int(td["leadTime"][0][0])
        print(f"Lead time: {leadTime}")
        device = getTorchDevice()
        tdNew = td.clone() #clone the tensor dict to avoid modifying the original one

        #initialize some values of the new tensor dict
        tdNew["currentTimestep"] = torch.zeros((batchSize, 1), dtype=torch.long) #initially the current timestep is 0
        tdNew["orderQuantity"] = torch.zeros((batchSize, 1), dtype=torch.int64)
        tdNew["onHandLevel"] = td["onHandLevel"]
        tdNew["inTransitStock"] = torch.zeros(batchSize, leadTime-1, dtype=torch.int64) #initially the in transit stock is 0
        tdNew["forecast"] = td["forecast"]
        tdNew["orderingCost"] = td["orderingCost"]
        tdNew["stockOutPenalty"] = td["stockoutPenalty"]
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
        leadTime = td["leadTime"][0][0].item() #¿el lead time es el mismo para todos los batches?
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
        # demandDataProjection = demandDataProjection.view(batch_size, TRAYECTORY_LENGHT, FORECAST_LENGHT, -1)
        print(f"DEMAND DATA PROJECTION SHAPE: {demandDataProjection.shape}") # [1,3,5] batch_size = 1, TRAYECTORY_LENGHT = 3, FORECAST_LENGHT = 5
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
        timeIndicesDemand = torch.arange(TRAYECTORY_LENGHT, device=td["forecast"].device).long() #no estoy segura de si es así
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
            orderQuantity = self.relu(orderQuantity)*100
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
            predictedAction = self.relu(predictedAction)*100
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
        print(f"Stock en tránsito: {td['inTransitStock'][...,0:1]}")
        print(f"Stock en tránsito shape: {td['inTransitStock'][...,0:1].shape}")
        print(f"Forecast: {(td['forecast'][...,0:1]).shape}")
        print(f"Forecast: {td['forecast'][...,0:1]}")
    

        td["onHandLevel"] = torch.add(td["onHandLevel"], td["inTransitStock"][...,0:1])
        print(f"Stock físico shape: {td['onHandLevel'].shape}")
        stockOutPenalty = (td["stockOutPenalty"] * torch.max(torch.zeros_like(td["forecast"][..., 0]), td["forecast"][..., 0] - td["onHandLevel"]))
        income = td["unitRevenue"] * torch.min(td["forecast"][..., 0], td["onHandLevel"])

        td["onHandLevel"] = torch.clamp(torch.sub(td["onHandLevel"], td["forecast"][..., 0]), min=0) #no se si puede ser 0 mirAR BIEN COMO SE ACTUALIZA FORECAST

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
        benefitUpdate = (income - holdingCost - stockOutPenalty - orderingCost).float()
        print(f"Beneficio actualizado: {benefitUpdate}")
        print(f"Beneficio: {td['benefit']}")
        print(f"Beneficio shape: {td['benefit'].shape}")
        print(f"currentTimestep shape: {currentTimeStep.shape}")

        returnToGo = td["returnsToGo"]

        mask = currentTimeStep >= RETURN_TO_GO_WINDOW
        td["returnsToGo"] = torch.where(
            mask,
            (returnToGo*RETURN_TO_GO_WINDOW - benefitUpdate + 
             td["benefit"][...,0:1])/RETURN_TO_GO_WINDOW,
            returnToGo
        )
        if mask.any():
            oldBenefit = td["benefit"][...,0:1]
            # Hacer roll del beneficio
            td["benefit"] = torch.roll(td["benefit"], shifts=-1, dims=-1)
            # Actualizar el último valor con el beneficio nuevo
            td["benefit"][..., -1] = benefitUpdate.squeeze()
            td["returnsToGo"] = torch.where(
                mask,
                (returnToGo*RETURN_TO_GO_WINDOW - benefitUpdate + 
                 oldBenefit)/RETURN_TO_GO_WINDOW,  # Usamos el penúltimo valor que acabamos de desplazar
                returnToGo)
        else:
            # Si aún no llegamos a RETURN_TO_GO_WINDOW, actualizar normalmente
            for b in range(td["benefit"].size(0)):
                td["benefit"][b, currentTimeStep[b, 0]] = benefitUpdate[b, 0]


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
        
        if tensor.size(1) >= self.maxSeqLength:
            result= torch.cat((tensor[:, 1:, :], data + self.projectTimeData(currentTimestep)), dim=1)
        else:
            result= torch.cat((tensor, data + self.projectTimeData(currentTimestep)), dim=1)
        print(f"Result shape: {result.shape}")
        return result
    
    def setInitalReturnToGo(self, td, returnsToGo):
        td["returnsToGo"] = returnsToGo if returnsToGo is not None else torch.zeros(self.batchSize, device=self.device)



if __name__ == "__main__":
    # 1. Crear datos de prueba
    
    print("\n=== Generando datos de prueba ===")
    # Generar una trayectoria de ejemplo
    
    # 2. Crear TensorDict inicial
    batch_size = 2  # Probamos con batch_size = 2
    td = TensorDict({
        'batch_size': torch.tensor([batch_size]),
        
        # Costes y parámetros (valores arbitrarios para prueba)
        'holdingCost': torch.tensor([[5.0]] * batch_size),      # Coste almacenamiento = 5
        'orderingCost': torch.tensor([[100.0]] * batch_size),   # Coste pedido = 100
        'stockoutPenalty': torch.tensor([[50.0]] * batch_size), # Penalización rotura = 50
        'unitRevenue': torch.tensor([[20.0]] * batch_size),     # Ingreso unitario = 20
        'leadTime': torch.tensor([[5]] * batch_size),  
        'orderQuantity': torch.tensor([[5]] * batch_size),        # Lead time = 15 períodos
        
        # Estado inicial del sistema
        'onHandLevel': torch.tensor([[100], [150]]),               # Stock inicial diferente para cada batch
        'inTransitStock': torch.zeros(batch_size, 4),  # Sin pedidos en tránsito inicialmente
        'forecast': torch.tensor([[20.0, 22.0, 18.0, 25.0, 21.0]] * batch_size),  # Previsión de demanda para 5 períodos
        'returnsToGo':torch.tensor([[500]] * batch_size)
    })
    
    # Añadir algunos pedidos en tránsito para prueba
    print(f"inTransitStock shape: {td['inTransitStock'].shape}")
    td['inTransitStock'][0, 2] = 100   # Batch 0: pedido de 100 unidades que llega en t=5
    td['inTransitStock'][1, 3] = 150  # Batch 1: pedido de 150 unidades que llega en t=10
    
    print("\n=== Valores iniciales ===")
    print(f"Batch size: {td['batch_size']}")
    print(f"\nParámetros del sistema:")
    print(f"Holding Cost: {td['holdingCost']}")  
    print(f"Ordering Cost: {td['orderingCost']}")
    print(f"Stockout Penalty: {td['stockoutPenalty']}")
    print(f"Unit Revenue: {td['unitRevenue']}")
    print(f"Lead Time: {td['leadTime']}")
    print(f"returnsToGo: {td['returnsToGo']}")
    
    print(f"\nEstado inicial:")
    print(f"On Hand Level: {td['onHandLevel']}")
    print(f"Forecast: {td['forecast']}")
    
   
    
    # 3. Crear modelo y ejecutar initModel
    print("\n=== Inicializando modelo ===")
    config = DecisionTransformerConfig()
    model = DecisionTransformer(config)  # Ajusta los parámetros según tu configuración
    model.eval()
    tdNew = model.initModel(td)
    
    # 4. Imprimir resultados
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
    
        
    # 6. Verificar que los tensores están correctamente inicializados
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
    # 3. Ejecutar el forward
    for step in range(5):  # Por ejemplo, 5 pasos de tiempo
        print(f"\nPaso {step}")
        tdNew = model.forward(tdNew)  # Aquí empieza el forward
        
        # Opcional: hacer una pausa entre pasos para mejor visualización
        input("Presiona Enter para continuar al siguiente paso...")



