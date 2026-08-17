"""
Утилиты для нормализации и денормализации данных.
"""

import numpy as np
from typing import Tuple, Optional


def minmax_normalize(data: np.ndarray, 
                     data_min: Optional[np.ndarray] = None,
                     data_max: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Min-Max нормализация к диапазону [0, 1].
    
    Args:
        data: Данные для нормализации
        data_min: Минимум (если уже вычислен)
        data_max: Максимум (если уже вычислен)
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            (нормализованные данные, минимум, максимум)
    """
    if data_min is None:
        data_min = data.min(axis=0)
    if data_max is None:
        data_max = data.max(axis=0)
    
    # Избегаем деления на ноль
    data_range = data_max - data_min
    data_range[data_range == 0] = 1.0
    
    normalized = (data - data_min) / data_range
    
    return normalized, data_min, data_max


def minmax_denormalize(data: np.ndarray, 
                       data_min: np.ndarray, 
                       data_max: np.ndarray) -> np.ndarray:
    """
    Обратное преобразование Min-Max нормализации.
    
    Args:
        data: Нормализованные данные
        data_min: Минимум от нормализации
        data_max: Максимум от нормализации
        
    Returns:
        np.ndarray: Денормализованные данные
    """
    data_range = data_max - data_min
    data_range[data_range == 0] = 1.0
    
    return data * data_range + data_min


def zscore_normalize(data: np.ndarray,
                     data_mean: Optional[np.ndarray] = None,
                     data_std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score нормализация (стандартизация).
    
    Args:
        data: Данные для нормализации
        data_mean: Среднее (если уже вычислено)
        data_std: Стандартное отклонение (если уже вычислено)
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            (нормализованные данные, среднее, стандартное отклонение)
    """
    if data_mean is None:
        data_mean = data.mean(axis=0)
    if data_std is None:
        data_std = data.std(axis=0)
    
    # Избегаем деления на ноль
    data_std[data_std == 0] = 1.0
    
    normalized = (data - data_mean) / data_std
    
    return normalized, data_mean, data_std


def zscore_denormalize(data: np.ndarray,
                       data_mean: np.ndarray,
                       data_std: np.ndarray) -> np.ndarray:
    """
    Обратное преобразование Z-score нормализации.
    
    Args:
        data: Нормализованные данные
        data_mean: Среднее от нормализации
        data_std: Стандартное отклонение от нормализации
        
    Returns:
        np.ndarray: Денормализованные данные
    """
    return data * data_std + data_mean
