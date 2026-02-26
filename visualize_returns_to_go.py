import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path


def get_project_directory():
    """
    Obtiene el directorio raíz del proyecto.
    """
    return str(Path(__file__).resolve().parent)


def load_training_data(data_path):
    """
    Carga los datos de entrenamiento desde un archivo .pt.

    Args:
        data_path: Ruta al archivo de datos de entrenamiento.

    Returns:
        TensorDict con los datos de entrenamiento.
    """
    training_data = torch.load(data_path, weights_only=False)
    return training_data


def extract_returns_to_go(training_data):
    """
    Extrae los valores de returns to go de los datos de entrenamiento.

    Args:
        training_data: TensorDict con los datos de entrenamiento.

    Returns:
        Array de numpy con los returns to go.
    """
    returns_to_go = training_data["returnsToGo"]
    
    # Convertir a numpy y aplanar si es necesario.
    if isinstance(returns_to_go, torch.Tensor):
        returns_to_go = returns_to_go.cpu().numpy()
    
    # Aplanar si tiene más de una dimensión.
    if returns_to_go.ndim > 1:
        returns_to_go = returns_to_go.flatten()
    
    return returns_to_go


def visualize_returns_to_go(data_path):
    """
    Visualiza los returns to go de los datos de entrenamiento con múltiples gráficas interactivas.

    Args:
        data_path: Ruta al archivo de datos de entrenamiento.
    """
    # Cargar datos.
    print(f"Cargando datos desde: {data_path}")
    training_data = load_training_data(data_path)
    
    # Extraer returns to go.
    returns_to_go = extract_returns_to_go(training_data)
    
    # Calcular estadísticas.
    mean_rtg = returns_to_go.mean()
    median_rtg = np.median(returns_to_go)
    min_rtg = returns_to_go.min()
    max_rtg = returns_to_go.max()
    std_rtg = returns_to_go.std()
    var_rtg = returns_to_go.var()
    p25 = np.percentile(returns_to_go, 25)
    p75 = np.percentile(returns_to_go, 75)
    iqr = p75 - p25
    
    print(f"Número de trayectorias: {len(returns_to_go)}")
    print(f"Returns to go - Min: {min_rtg:.2f}, Max: {max_rtg:.2f}")
    print(f"Returns to go - Media: {mean_rtg:.2f}, Mediana: {median_rtg:.2f}")
    print(f"Returns to go - Std: {std_rtg:.2f}")
    
    # Crear subplots con plotly (1 fila, 2 columnas).
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Histograma de Returns to Go', 'Estadísticas de Resumen'),
        specs=[[{"type": "xy"}, {"type": "table"}]],
        horizontal_spacing=0.15
    )
    
    # 1. Histograma.
    fig.add_trace(
        go.Histogram(
            x=returns_to_go,
            nbinsx=50,
            name='Frecuencia',
            marker_color='lightblue',
            marker_line_color='black',
            marker_line_width=1,
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Añadir líneas verticales para media y mediana.
    fig.add_vline(
        x=mean_rtg, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Media: {mean_rtg:.2f}",
        row=1, col=1
    )
    fig.add_vline(
        x=median_rtg, 
        line_dash="dash", 
        line_color="green",
        annotation_text=f"Mediana: {median_rtg:.2f}",
        row=1, col=1
    )
    
    # 2. Tabla de estadísticas.
    stats_data = [
        ['Total de trayectorias', f'{len(returns_to_go):,}'],
        ['Mínimo', f'{min_rtg:.2f}'],
        ['Máximo', f'{max_rtg:.2f}'],
        ['Media', f'{mean_rtg:.2f}'],
        ['Mediana', f'{median_rtg:.2f}'],
        ['Desviación estándar', f'{std_rtg:.2f}'],
        ['Varianza', f'{var_rtg:.2f}'],
        ['Percentil 25', f'{p25:.2f}'],
        ['Percentil 75', f'{p75:.2f}'],
        ['Rango intercuartílico', f'{iqr:.2f}']
    ]
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=['Métrica', 'Valor'],
                fill_color='paleturquoise',
                align='left',
                font=dict(size=12, color='black')
            ),
            cells=dict(
                values=list(zip(*stats_data)),
                fill_color='lavender',
                align='left',
                font=dict(size=11)
            )
        ),
        row=1, col=2
    )
    
    # Actualizar layout.
    fig.update_xaxes(title_text="Returns to Go", row=1, col=1)
    fig.update_yaxes(title_text="Frecuencia", row=1, col=1)
    
    fig.update_layout(
        title_text=f'Análisis de Returns to Go - {Path(data_path).name}',
        title_x=0.5,
        height=500,
        showlegend=False
    )
    
    # Guardar la figura como HTML.
    output_path = Path(data_path).parent / f"returns_to_go_analysis_{Path(data_path).stem}.html"
    fig.write_html(str(output_path))
    print(f"\nGràfica guardada en: {output_path}")
    
    # Mostrar la gráfica.
    fig.show()


if __name__ == "__main__":
    # Ruta a los datos de entrenamiento (modificar según sea necesario).
    data_path = get_project_directory() + "/data/rl_solution_trajectories.pt"
    
    # Si quieres usar otra ruta, descomenta y modifica la siguiente línea:
    # data_path = "data/test_data.pt"
    
    try:
        visualize_returns_to_go(data_path)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {data_path}")
        print("Por favor, verifica que la ruta sea correcta.")
    except Exception as e:
        print(f"Error al procesar los datos: {e}")
        import traceback
        traceback.print_exc()

