import torch.nn as nn
import torch
from tensordict import TensorDict
from generate_tajectories import generateInstanceData, generateTrajectory, FORECAST_LENGHT

def getTorchDevice():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
   


class DecisionTransformer(nn.Module):

    def __init__(self):
        super().__init__()
        self.embeddingDim = 128


         # Projections
        #projection for the scalar data
        self.projectScalarData= nn.Linear(7, self.embeddingDim) # 7 values: onHandLevel, inTransitStock, holdingCost, OrderCost, StockOutPenalty, UnitRevenue, leadTime
        #projection for the demand data
        self.projectDemandData = nn.Linear(FORECAST_LENGHT, self.embeddingDim) # FORECAST_LENGHT es el número de períodos de prevision de demanda
        self.projectTimeData = nn.Embedding(FORECAST_LENGHT, self.embeddingDim) 


     # MHA para el forward
        self.mhaState = nn.MultiheadAttention(
            embed_dim=self.embeddingDim,
            num_heads=4,
            batch_first=True
        )




    def initModel(self, td): 
        #Leemos algunos valores
        batchSize = 2       
        device = getTorchDevice()
        tdNew = td.clone()

        tdNew["currentTimestep"] = torch.zeros((batchSize, 1), dtype=torch.int64)
        tdNew["orderQuantity"] = torch.zeros((batchSize, 1))
        tdNew["onHandLevel"] = torch.zeros(batchSize, dtype=torch.int64)
        tdNew["inTransitStock"] = torch.zeros(batchSize, dtype=torch.int64)
        #tdNew["forecast"] = torch.zeros(batchSize, dtype=torch.int64)
        tdNew["demand"] = torch.zeros(batchSize, dtype=torch.int64)
        tdNew["initialProblemState"] = torch.zeros(batchSize, dtype=torch.int64)

    

        tdNew["statesEmbedding"] = torch.zeros((batchSize, 0, self.embeddingDim), dtype=torch.float, device=device)
        tdNew["actionsEmbedding"] = torch.zeros((batchSize, 0, self.embeddingDim), dtype=torch.float, device=device)
        tdNew["returnsToGoEmbedding"] = torch.zeros((batchSize, 0, self.embeddingDim), dtype=torch.float, device=device)

                                            
        self.createInitialState(tdNew)
        return tdNew
   
    def forward(self, td): 
        
        return td

   
    def createInitialState(self, td):

        dataScalar = td['dataScalar'] #incluye los valores inciales del td de onHandLevel, inTransitStock, holdingCost, orderingCost, stockOutPenalty, unitRevenue, leadTime
        dataDemand = td['forecast'] #incluye los valores inciales del td de forecast
        dataTimeStep = td['Timestep'] #incluye los valores inciales del td de currentTimestep  torch.arange(FORECAST_LENGHT)
        

        scalarDataProjection = self.projectScalarData(dataScalar)
        demandDataProjection = self.projectDemandData(dataDemand)
        timeDataProjection = self.projectTimeData(dataTimeStep)

        #demandTimeEmbedding = demandDataProjection + timeDataProjection
        #stateEmbedding = self.mhaState(
        #scalarDataProjection.unsqueeze(1),
        #demandTimeEmbedding,
        #demandTimeEmbedding
        #)[0].squeeze(1)


        #allData = torch.cat([scalarDataProjection, stateEmbedding], dim=-1)
        #encodedState = self.encoder(allData)

        #td["initialProblemState"] = encodedState
        #if not self.training:


    def addSequenceData(self, td, tensor, data):
        if tensor.size(1) == self.maxSeqLength:
            return torch.cat((tensor[:, 1:, :], data + self.embeddingTimestep(td["currentTimestep"])), dim=1)
        return torch.cat((tensor, data + self.embeddingTimestep(td["currentTimestep"])), dim=1)


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
        'stockOutPenalty': torch.tensor([[50.0]] * batch_size), # Penalización rotura = 50
        'unitRevenue': torch.tensor([[20.0]] * batch_size),     # Ingreso unitario = 20
        'leadTime': torch.tensor([[15]] * batch_size),          # Lead time = 15 períodos
        
        # Estado inicial del sistema
        'onHandLevel': torch.tensor([100, 150]),               # Stock inicial diferente para cada batch
        'inTransitStock': torch.zeros(batch_size, 15),  # Sin pedidos en tránsito inicialmente
        'forecast': torch.tensor([[20.0, 22.0, 18.0, 25.0, 21.0]] * batch_size)  # Previsión de demanda para 5 períodos
    })
    td["dataScalar"] = torch.cat((
    td["onHandLevel"].view(batch_size, -1),  # Aseguramos forma (batch, features)
    td["inTransitStock"].view(batch_size, -1),
    td["holdingCost"].view(batch_size, -1),
    td["orderingCost"].view(batch_size, -1),
    td["stockOutPenalty"].view(batch_size, -1),
    td["unitRevenue"].view(batch_size, -1),
    td["leadTime"].view(batch_size, -1)
), dim=-1)
    
    # Añadir algunos pedidos en tránsito para prueba
    td['inTransitStock'][0, 5] = 100   # Batch 0: pedido de 100 unidades que llega en t=5
    td['inTransitStock'][1, 10] = 150  # Batch 1: pedido de 150 unidades que llega en t=10
    
    print("\n=== Valores iniciales ===")
    print(f"Batch size: {td['batch_size']}")
    print(f"\nParámetros del sistema:")
    print(f"Holding Cost: {td['holdingCost']}")
    print(f"Ordering Cost: {td['orderingCost']}")
    print(f"Stockout Penalty: {td['stockOutPenalty']}")
    print(f"Unit Revenue: {td['unitRevenue']}")
    print(f"Lead Time: {td['leadTime']}")
    
    print(f"\nEstado inicial:")
    print(f"On Hand Level: {td['onHandLevel']}")
    print(f"Forecast: {td['forecast']}")
    
    print(f"\nPedidos en tránsito:")

    for b in range(batch_size):
        nonzero = td['inTransitStock'][b].nonzero()
        print(f"Batch {b}:")
        for idx in nonzero:
            print(f"  t={idx.item()}: {td['inTransitStock'][b, idx].item()} unidades")
    
    # 3. Crear modelo y ejecutar initModel
    print("\n=== Inicializando modelo ===")
    model = DecisionTransformer()  # Ajusta los parámetros según tu configuración
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
    print(f"demand shape: {tdNew['demand'].shape}")
    
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






