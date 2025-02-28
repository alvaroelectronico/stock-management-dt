import math
import torch

class StockManagementExperiment():

    def __init__(self, instance, solution):
        self.instance = instance
        self.solution = solution
        self.previousTransitStock = None #Stock en tránsito del período anterior
        self.previousOrderQuantity = None #Cantidad a ordenar del período anterior

    def get_objective(self) -> float:
        return self.solution["totalProfit"]

    def check_solution(self,td):
        for k in range(self.solution["batch_size"][0]):
            print(f"\n=== Verificando batch {k} ===")
            sumProfit = 0
            averageProfit = 0
            
            # Convertir tensores a listas para facilitar las verificaciones
            inTransitStock = td["inTransitStock"][k].cpu().tolist()
            forecast = td["forecast"][k].cpu().tolist()
            benefit = td["benefit"][k].cpu().tolist()
            
            print(f"\nValores iniciales:")
            print(f"Stock en tránsito: {inTransitStock}")
            print(f"Previsión demanda: {forecast}")
            print(f"Beneficios por período: {benefit}")
            
            # Extraer valores escalares
            onHandLevel = td["onHandLevel"][k].item()
            orderQuantity = td["orderQuantity"][k].item()
            leadTime = td["leadTime"][k].item()
            holdingCost = td["holdingCost"][k].item()
            orderingCost = td["orderingCost"][k].item()
            stockOutPenalty = td["stockOutPenalty"][k].item()
            unitRevenue = td["unitRevenue"][k].item()
            t = td["t"][k].item()
            # Variables para verificación
            periodProfit = 0
            currentStock = onHandLevel

            print(f"\nParámetros:")
            print(f"Stock físico: {onHandLevel}")
            print(f"Órdenes: {orderQuantity}")
            print(f"Lead Time: {leadTime}")
            print(f"Holding Cost: {holdingCost}")
            print(f"Ordering Cost: {orderingCost}")
            print(f"Stockout Penalty: {stockOutPenalty}")
            print(f"Unit Revenue: {unitRevenue}")
            
            # Verificar cada período
            print(f"\n--- Período {t} ---")

            #Verificar restricciones básicas: stock físico, en tránsito y la cantidad a ordenar no pueden ser negativos
            assert onHandLevel >= 0, "Stock físico negativo detectado"
            assert all(x >= 0 for x in inTransitStock), "Stock en tránsito negativo detectado"
            assert orderQuantity >= 0, "Orden negativa detectada"
            
            # 1. Verificar actualización de stock en tránsito

            if self.previousTransitStock is not None:
                assert inTransitStock[:-1] == self.previousTransitStock[1:],\
                f"Error en la actualización del stock en tránsito en t={t}"

            self.previousTransitStock = inTransitStock[t]

            if self.previousOrderQuantity is not None:
                assert inTransitStock[-1] == self.previousOrderQuantity, \
                f"La orden no se añadió correctamente al stock en tránsito en t={t}"


            print(f"Verificando actualización stock en tránsito:")
            print(f"Stock t: {inTransitStock[t]}")

            # 2. Verificar recepción de pedidos y actualización de stock
                
            if t >= leadTime and orderQuantity[t-leadTime] > 0:
                current_stock += orderQuantity[t-leadTime]
                print(f"Recepción de pedido: +{orderQuantity[t-leadTime]}")
                print(f"Stock tras recepción: {current_stock}")

            # 3. Verificar satisfacción de demanda y cálculo de beneficios
            demand = forecast[0]
            sales = min(currentStock, demand)
            stockout = max(0, demand - currentStock)
            currentStock = max(0, currentStock - demand)
                
            print(f"Demanda del período: {demand}")
            print(f"Ventas realizadas: {sales}")
            print(f"Stockout: {stockout}")
            print(f"Stock final: {currentStock}")

            # 4. Verificar costes y beneficios del período
            periodRevenue = sales * unitRevenue
            periodHolding = currentStock * holdingCost
            periodStockout = stockout * stockOutPenalty
            periodOrdering = orderingCost if orderQuantity > 0 else 0
                
            periodProfit = periodRevenue - periodHolding - periodStockout - periodOrdering
            sumProfit += periodProfit
            averageProfit = sumProfit/t

            print(f"\nCálculo de beneficios:")
            print(f"Ingresos: {periodRevenue}")
            print(f"Coste almacenamiento: {periodHolding}")
            print(f"Coste stockout: {periodStockout}")
            print(f"Coste pedido: {periodOrdering}")
            print(f"Beneficio período: {periodProfit}")
            print(f"Beneficio reportado: {benefit[t]}")

             

            # 6. Verificar beneficio total
            assert math.isclose(
                periodProfit, 
                benefit[t], 
                rel_tol=1e-4
            ), f"Error en el cálculo del beneficio total para batch {k}"

            # 7. Verificar return-to-go
            print("\n=== Verificando Return-to-Go ===")
            returnsToGo = td["returnsToGo"][k].cpu().tolist()
            window_size = len(returnsToGo)
            print(f"Window size: {window_size}")
            print(f"Returns-to-go actuales: {returnsToGo}")
                
            for t in range(window_size):
                if t > 0:
                    print(f"\nPeríodo {t}:")
                    print(f"Beneficio actual: {benefit[t]}")
                    print(f"Beneficio anterior: {benefit[t-1]}")
                    expectedRTG = self.calculate_return_to_go(
                        benefit, t, window_size, averageProfit)
                    assert math.isclose(
                        expectedRTG,
                        returnsToGo[t],   
                        rel_tol=1e-4
                    ), f"Error en el cálculo del return-to-go en t={t}"
                    print(f"Return-to-go calculado: {expectedRTG}")
                    print(f"Return-to-go reportado: {returnsToGo[t]}")

    def calculate_return_to_go(self, benefit, t, window_size, averageProfit):
        """Calcula el return-to-go esperado para un período específico"""
        if t <= window_size:
            return averageProfit
        
        old_benefit = benefit[t-window_size]  # beneficio del período anterior
        print(f"Beneficio anterior: {old_benefit}")
        new_benefit = benefit[t]    # beneficio del período actual
        print(f"Beneficio actual: {new_benefit}")
        result = (averageProfit * window_size - new_benefit + old_benefit) / window_size
        print(f"Resultado: {result}")
        return (averageProfit * window_size - new_benefit + old_benefit) / window_size
   

if __name__ == "__main__":
    print("\n=== Iniciando prueba del Experiment ===")
    
    # Crear datos de prueba
    batch_size = 2
    td = {
        "batch_size": torch.tensor([batch_size]),
        "leadTime": torch.tensor([[5]] * batch_size),
        "holdingCost": torch.tensor([[2.0]] * batch_size),
        "orderingCost": torch.tensor([[100.0]] * batch_size),
        "stockOutPenalty": torch.tensor([[50.0]] * batch_size),
        "unitRevenue": torch.tensor([[20.0]] * batch_size),
        "onHandLevel": torch.tensor([[100], [150]]),
        "inTransitStock": torch.zeros(batch_size, 4),
        "orderQuantity": torch.tensor([[50], [75]]),
        "forecast": torch.tensor([[20.0, 22.0, 18.0, 25.0, 21.0]] * batch_size),
        "benefit": torch.tensor([[100.0, 140.0, 120.0, 80.0, 90.0]] * batch_size),
        "returnsToGo": torch.tensor([[108.0, 140.0, 98.0]] * batch_size),
        "t":torch.tensor([[1],[2]])
    }

    # Añadir algunos pedidos en tránsito para prueba
    td["inTransitStock"][0, 2] = 100  # Batch 0: pedido de 100 unidades
    td["inTransitStock"][1, 3] = 150  # Batch 1: pedido de 150 unidades

    print("\n=== Datos de prueba creados ===")
    print(f"Batch size: {td['batch_size']}")
    print(f"Lead Time: {td['leadTime']}")
    print(f"Stock inicial: {td['onHandLevel']}")
    print(f"Stock en tránsito inicial: {td['inTransitStock']}")

    # Crear instancia del experimento
    experiment = StockManagementExperiment(None, td)

    # Ejecutar verificación
    print("\n=== Ejecutando verificación ===")
    try:
        experiment.check_solution(td)
        print("\n✅ Verificación completada con éxito")
    except AssertionError as e:
        print(f"\n❌ Error en la verificación: {e}")
    except Exception as e:
        import traceback
        print(f"\n❌ Error inesperado:")
        print(f"Tipo de error: {type(e).__name__}")
        print(f"Mensaje: {str(e)}")
        print("\nTraceback completo:")
        traceback.print_exc()