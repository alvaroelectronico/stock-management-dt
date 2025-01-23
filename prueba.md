# Gestión de Stocks

*January 17, 2025*

## 1. Descripción del problema

Satisfacer la demanda de un único producto contando con un proveedor.

- Suponemos que el tiempo avanza de forma discreta.
- Suponemos capacidad del almacén infinita
- Costes relevantes a considerar: costes de emisión, costes de almacenamiento y costes por rotura de stock.
- Consideramos demanda y lead time no deterministas

### 1.1 Parámetros y variables
- Nivel de inventario (Stock físico y Stock en tránsito)
- Tiempo de entrega o lead time (constante)
- Previsión de demanda
- Costes de almacenamiento
- Costes de emisión o pedido
- Penalización por rotura
- Ingresos unitarios por venta

### 1.2 Objetivo del problema
Para cada período: ¿Pedimos? ¿Cuánto pedimos?

## 2. Aplicación al Decision Transformer

### 2.1 Generación de trayectorias
- Modelos de optimización
- Políticas clásicas de gestión de Stocks

### 2.2 Estado, Acción y Return-to-go

| Estado | Acción | Return-to-go |
|--------|--------|--------------|
| Stock físico | Número (unidades pedidas) | reward (ingresos-costes) |
| Stock en tránsito | | |
| Previsión de demanda | | |
| Costes de almacenamiento y emisión | | |
| Ingresos unitarios por venta | | |
| Penalización por rotura | | |
| Lead time | | |

### 2.3 Estado

Concatenar todos los escalares (stock físico, stock en tránsito, costes de almacenamiento, de emisión y de rotura, ingresos y lead time) en un tensor con 7 valores y convertirlo a un único embedding.

`DimEmbeddingsEscalares = batch, 1, 128`

La previsión de demanda no es un escalar, contiene un valor de previsión de demanda para cada período del horizonte considerado (H), por lo tanto generamos H embeddings por cada vector de previsión de demanda. Sumamos los H embeddings de cada previsión de demanda y nos quedaríamos con H embeddings. Para tener en cuenta en que momento nos encontramos vamos a incluir el time step (es más importante cumplir con la demanda de los días más próximos para evitar una rotura de stock que con los días más alejados).

Para ello sumamos los embeddings de la demanda más los embeddings del timestep:

`DemandaPrevista + TimeStep = batch, H, 128 + batch, H, 128 = batch, H, 128`

Como al Decision Transformer solo le podemos pasar 1 embedding realizamos un MHA pasándole el embedding de las variables escalares y el de la demanda prevista (incluye el timestep)

### 2.4 Return-to-go

Representa los beneficios a los que podemos aspirar. Establecemos un horizonte para calcular la media de los beneficios.

Ejemplo: Si tengo una trayectoria de 100 días, le paso 30 días (lo que he hecho hasta ahora) para obtener lo que debería ir haciendo a partir de ese momento. Para calcular el return-to-go, multiplicamos el beneficio medio por 30, le restamos los beneficios obtenidos en la acción que nos devuelve el transformer y le sumamos los beneficios del primer período del horizonte.

### 2.5 Acción

Se corresponde con el número de unidades que pedimos, si tiene valor 0 quiere decir que no pedimos nada.

**Opción 1:** Utilizar la función de activación ReLU, por debajo de un valor te devuelve el valor 0 y por encima de dicho valor no está acotada.

**Opción 2:** Utilizar dos redes:
- La primera red nos devuelve un número entre 0 y 1 que se correspondería con la probabilidad de pedir (si es mayor que 0.5, entonces pedimos).
- Si pedimos lo pasamos por una segunda red que nos devolvería los parámetros de una distribución normal que representaría la cantidad pedida.

### 2.6 Cosas a considerar
- Normalización de los datos 