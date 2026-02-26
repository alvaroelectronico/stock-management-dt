import torch
from tensordict import TensorDict
from typing import Dict, Optional, Tuple, List


def scale_on_hand_level(data: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Escala el nivel de stock en mano.
    
    Args:
        data: Tensor con onHandLevel
        mean: Media para normalización
        std: Desviación estándar para normalización
    
    Returns:
        Tensor escalado
    """
    return (data - mean) / std


def unscale_on_hand_level(data: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Desescala el nivel de stock en mano.
    
    Args:
        data: Tensor escalado con onHandLevel
        mean: Media usada en normalización
        std: Desviación estándar usada en normalización
    
    Returns:
        Tensor desescalado
    """
    return data * std + mean


def scale_holding_cost(data: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Escala el coste de mantener stock.
    
    Args:
        data: Tensor con holdingCost
        mean: Media para normalización
        std: Desviación estándar para normalización
    
    Returns:
        Tensor escalado
    """
    return (data - mean) / std


def unscale_holding_cost(data: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Desescala el coste de mantener stock.
    
    Args:
        data: Tensor escalado con holdingCost
        mean: Media usada en normalización
        std: Desviación estándar usada en normalización
    
    Returns:
        Tensor desescalado
    """
    return data * std + mean


def scale_ordering_cost(data: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Escala el coste de hacer pedido.
    
    Args:
        data: Tensor con orderingCost
        mean: Media para normalización
        std: Desviación estándar para normalización
    
    Returns:
        Tensor escalado
    """
    return (data - mean) / std


def unscale_ordering_cost(data: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Desescala el coste de hacer pedido.
    
    Args:
        data: Tensor escalado con orderingCost
        mean: Media usada en normalización
        std: Desviación estándar usada en normalización
    
    Returns:
        Tensor desescalado
    """
    return data * std + mean


def scale_stock_out_penalty(data: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Escala la penalización por falta de stock.
    
    Args:
        data: Tensor con stockOutPenalty
        mean: Media para normalización
        std: Desviación estándar para normalización
    
    Returns:
        Tensor escalado
    """
    return (data - mean) / std


def unscale_stock_out_penalty(data: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Desescala la penalización por falta de stock.
    
    Args:
        data: Tensor escalado con stockOutPenalty
        mean: Media usada en normalización
        std: Desviación estándar usada en normalización
    
    Returns:
        Tensor desescalado
    """
    return data * std + mean


def scale_unit_revenue(data: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Escala el ingreso por unidad.
    
    Args:
        data: Tensor con unitRevenue
        mean: Media para normalización
        std: Desviación estándar para normalización
    
    Returns:
        Tensor escalado
    """
    return (data - mean) / std


def unscale_unit_revenue(data: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Desescala el ingreso por unidad.
    
    Args:
        data: Tensor escalado con unitRevenue
        mean: Media usada en normalización
        std: Desviación estándar usada en normalización
    
    Returns:
        Tensor desescalado
    """
    return data * std + mean


def scale_lead_time(data: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Escala el tiempo de entrega.
    
    Args:
        data: Tensor con leadTime
        mean: Media para normalización
        std: Desviación estándar para normalización
    
    Returns:
        Tensor escalado
    """
    return (data - mean) / std


def unscale_lead_time(data: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Desescala el tiempo de entrega.
    
    Args:
        data: Tensor escalado con leadTime
        mean: Media usada en normalización
        std: Desviación estándar usada en normalización
    
    Returns:
        Tensor desescalado
    """
    return data * std + mean


def scale_forecast(data: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Escala el pronóstico de demanda.
    
    Args:
        data: Tensor con forecast [batch, time_steps, forecast_length]
        mean: Media para normalización
        std: Desviación estándar para normalización
    
    Returns:
        Tensor escalado
    """
    return (data - mean) / std


def unscale_forecast(data: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Desescala el pronóstico de demanda.
    
    Args:
        data: Tensor escalado con forecast
        mean: Media usada en normalización
        std: Desviación estándar usada en normalización
    
    Returns:
        Tensor desescalado
    """
    return data * std + mean


def scale_in_transit_stock(data: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Escala el stock en tránsito.
    
    Args:
        data: Tensor con inTransitStock [batch, max_lead_time]
        mean: Media para normalización
        std: Desviación estándar para normalización
    
    Returns:
        Tensor escalado
    """
    return (data - mean) / std


def unscale_in_transit_stock(data: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Desescala el stock en tránsito.
    
    Args:
        data: Tensor escalado con inTransitStock
        mean: Media usada en normalización
        std: Desviación estándar usada en normalización
    
    Returns:
        Tensor desescalado
    """
    return data * std + mean


def scale_demand(data: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Escala la demanda real.
    
    Args:
        data: Tensor con demand [batch, time_steps]
        mean: Media para normalización
        std: Desviación estándar para normalización
    
    Returns:
        Tensor escalado
    """
    return (data - mean) / std


def unscale_demand(data: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Desescala la demanda real.
    
    Args:
        data: Tensor escalado con demand
        mean: Media usada en normalización
        std: Desviación estándar usada en normalización
    
    Returns:
        Tensor desescalado
    """
    return data * std + mean


def scale_order_quantity(data: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """
    Escala la cantidad de pedido usando Min-Max.
    
    Args:
        data: Tensor con orderQuantity
        min_val: Valor mínimo para normalización
        max_val: Valor máximo para normalización
    
    Returns:
        Tensor escalado a [0, 1]
    """
    if max_val - min_val < 1e-8:
        return torch.zeros_like(data)
    return (data - min_val) / (max_val - min_val)


def unscale_order_quantity(data: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """
    Desescala la cantidad de pedido usando Min-Max.
    
    Args:
        data: Tensor escalado con orderQuantity [0, 1]
        min_val: Valor mínimo usado en normalización
        max_val: Valor máximo usado en normalización
    
    Returns:
        Tensor desescalado
    """
    if max_val - min_val < 1e-8:
        return torch.zeros_like(data)
    return data * (max_val - min_val) + min_val


def scale_returns_to_go(data: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Escala el retorno esperado.
    
    Args:
        data: Tensor con returnsToGo [batch, return_window]
        mean: Media para normalización
        std: Desviación estándar para normalización
    
    Returns:
        Tensor escalado
    """
    return data


def unscale_returns_to_go(data: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Desescala el retorno esperado.
    
    Args:
        data: Tensor escalado con returnsToGo
        mean: Media usada en normalización
        std: Desviación estándar usada en normalización
    
    Returns:
        Tensor desescalado
    """
    return data


def scale_benefit(data: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Escala el beneficio acumulado.
    
    Args:
        data: Tensor con benefit [batch, return_window]
        mean: Media para normalización
        std: Desviación estándar para normalización
    
    Returns:
        Tensor escalado
    """
    return (data - mean) / std


def unscale_benefit(data: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Desescala el beneficio acumulado.
    
    Args:
        data: Tensor escalado con benefit
        mean: Media usada en normalización
        std: Desviación estándar usada en normalización
    
    Returns:
        Tensor desescalado
    """
    return data * std + mean


def scale_tensordict(td: TensorDict, scaling_params: Dict[str, Tuple[float, float]]) -> TensorDict:
    """
    Escala todos los campos del TensorDict usando los parámetros proporcionados.
    
    Args:
        td: TensorDict sin escalar
        scaling_params: Diccionario con parámetros de escalado (mean, std) para cada campo
    
    Returns:
        TensorDict escalado con los mismos campos que el original
    """
    if hasattr(td, 'clone'):
        td_scaled = td.clone()
    else:
        td_scaled = {k: v.clone() for k, v in td.items()}
    
    if isinstance(td_scaled, dict):
        td_scaled = TensorDict(td_scaled, batch_size=td.batch_size if hasattr(td, 'batch_size') else None)
    
    scale_functions = {
        'onHandLevel': scale_on_hand_level,
        'holdingCost': scale_holding_cost,
        'orderingCost': scale_ordering_cost,
        'stockOutPenalty': scale_stock_out_penalty,
        'unitRevenue': scale_unit_revenue,
        'leadTime': scale_lead_time,
        'forecast': scale_forecast,
        'inTransitStock': scale_in_transit_stock,
        'demand': scale_demand,
        'orderQuantity': scale_order_quantity,
        'returnsToGo': scale_returns_to_go,
        'benefit': scale_benefit,
    }
    
    for field_name, scale_func in scale_functions.items():
        if field_name in td_scaled and field_name in scaling_params:
            mean, std = scaling_params[field_name]
            td_scaled[field_name] = scale_func(td_scaled[field_name], mean, std)
    
    return td_scaled


def unscale_tensordict(td: TensorDict, scaling_params: Dict[str, Tuple[float, float]]) -> TensorDict:
    """
    Desescala todos los campos del TensorDict usando los parámetros proporcionados.
    
    Args:
        td: TensorDict escalado
        scaling_params: Diccionario con parámetros de escalado (mean, std) para cada campo
    
    Returns:
        TensorDict desescalado con los mismos campos que el original
    """
    if hasattr(td, 'clone'):
        td_unscaled = td.clone()
    else:
        td_unscaled = {k: v.clone() for k, v in td.items()}
    
    if isinstance(td_unscaled, dict):
        td_unscaled = TensorDict(td_unscaled, batch_size=td.batch_size if hasattr(td, 'batch_size') else None)
    
    unscale_functions = {
        'onHandLevel': unscale_on_hand_level,
        'holdingCost': unscale_holding_cost,
        'orderingCost': unscale_ordering_cost,
        'stockOutPenalty': unscale_stock_out_penalty,
        'unitRevenue': unscale_unit_revenue,
        'leadTime': unscale_lead_time,
        'forecast': unscale_forecast,
        'inTransitStock': unscale_in_transit_stock,
        'demand': unscale_demand,
        'orderQuantity': unscale_order_quantity,
        'returnsToGo': unscale_returns_to_go,
        'benefit': unscale_benefit,
    }
    
    for field_name, unscale_func in unscale_functions.items():
        if field_name in td_unscaled and field_name in scaling_params:
            mean, std = scaling_params[field_name]
            td_unscaled[field_name] = unscale_func(td_unscaled[field_name], mean, std)
    
    return td_unscaled


def compute_scaling_params_from_training_data(data_paths: List[str]) -> Dict[str, Tuple[float, float]]:
    """
    Calcula los parámetros de escalado (mean, std) a partir de los datos de entrenamiento.
    
    Args:
        data_paths: Lista de rutas a los archivos de datos de entrenamiento (.pt)
    
    Returns:
        Diccionario con parámetros de escalado (mean, std) para cada campo
    """
    all_states = []
    all_actions = []
    all_returns_to_go = []
    
    for path in data_paths:
        element = torch.load(path, weights_only=False)
        all_states.append(element["states"])
        all_actions.append(element["actions"])
        all_returns_to_go.append(element["returnsToGo"])
    
    states = torch.cat(all_states, dim=0) if len(all_states) > 1 else all_states[0]
    actions = torch.cat(all_actions, dim=0) if len(all_actions) > 1 else all_actions[0]
    returns_to_go = torch.cat(all_returns_to_go, dim=0) if len(all_returns_to_go) > 1 else all_returns_to_go[0]
    
    scaling_params = {}
    
    if "onHandLevel" in states:
        data = states["onHandLevel"].float()
        mean = data.mean().item()
        std = data.std().item()
        if std < 1e-8:
            std = 1.0
        scaling_params["onHandLevel"] = (mean, std)
    
    if "holdingCost" in states:
        data = states["holdingCost"].float()
        mean = data.mean().item()
        std = data.std().item()
        if std < 1e-8:
            std = 1.0
        scaling_params["holdingCost"] = (mean, std)
    
    if "orderingCost" in states:
        data = states["orderingCost"].float()
        mean = data.mean().item()
        std = data.std().item()
        if std < 1e-8:
            std = 1.0
        scaling_params["orderingCost"] = (mean, std)
    
    if "stockOutPenalty" in states:
        data = states["stockOutPenalty"].float()
        mean = data.mean().item()
        std = data.std().item()
        if std < 1e-8:
            std = 1.0
        scaling_params["stockOutPenalty"] = (mean, std)
    
    if "unitRevenue" in states:
        data = states["unitRevenue"].float()
        mean = data.mean().item()
        std = data.std().item()
        if std < 1e-8:
            std = 1.0
        scaling_params["unitRevenue"] = (mean, std)
    
    if "leadTime" in states:
        data = states["leadTime"].float()
        mean = data.mean().item()
        std = data.std().item()
        if std < 1e-8:
            std = 1.0
        scaling_params["leadTime"] = (mean, std)
    
    if "forecast" in states:
        data = states["forecast"].float()
        mean = data.mean().item()
        std = data.std().item()
        if std < 1e-8:
            std = 1.0
        scaling_params["forecast"] = (mean, std)
    
    if "inTransitStock" in states:
        data = states["inTransitStock"].float()
        mean = data.mean().item()
        std = data.std().item()
        if std < 1e-8:
            std = 1.0
        scaling_params["inTransitStock"] = (mean, std)
    
    if "demand" in states:
        data = states["demand"].float()
        mean = data.mean().item()
        std = data.std().item()
        if std < 1e-8:
            std = 1.0
        scaling_params["demand"] = (mean, std)
    
    if actions is not None:
        data = actions.float()
        min_val = data.min().item()
        max_val = data.max().item()
        if max_val - min_val < 1e-8:
            max_val = min_val + 1.0
        scaling_params["orderQuantity"] = (min_val, max_val)
    
    if returns_to_go is not None:
        data = returns_to_go.float()
        mean = data.mean().item()
        std = data.std().item()
        if std < 1e-8:
            std = 1.0
        scaling_params["returnsToGo"] = (mean, std)
    
    return scaling_params

