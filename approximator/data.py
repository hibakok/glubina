"""
Модуль для работы с данными: пары вход-выход и датасеты.
"""

import numpy as np
from typing import List, Optional, Tuple


class DataPair:
    """Представляет одну пару вход-выход."""
    
    def __init__(self, inputs: List[float], outputs: List[float]):
        """
        Инициализация пары данных.
        
        Args:
            inputs: Список входных значений
            outputs: Список выходных значений
        """
        self.inputs = list(inputs)
        self.outputs = list(outputs)
    
    def __repr__(self) -> str:
        return f"DataPair(inputs={self.inputs}, outputs={self.outputs})"


class Dataset:
    """Набор пар вход-выход, загруженный из файла."""
    
    def __init__(self):
        """Инициализация пустого датасета."""
        self.pairs: List[DataPair] = []
        self.input_dim: int = 0
        self.output_dim: int = 0
        self._input_min: Optional[np.ndarray] = None
        self._input_max: Optional[np.ndarray] = None
        self._output_min: Optional[np.ndarray] = None
        self._output_max: Optional[np.ndarray] = None
        self._input_mean: Optional[np.ndarray] = None
        self._input_std: Optional[np.ndarray] = None
        self._output_mean: Optional[np.ndarray] = None
        self._output_std: Optional[np.ndarray] = None
        self._normalized: bool = False
        self._normalization_type: str = "minmax"  # или "zscore"
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'Dataset':
        """
        Загрузка датасета из файла.
        
        Args:
            filepath: Путь к файлу с данными
            
        Returns:
            Dataset: Загруженный датасет
            
        Raises:
            FileNotFoundError: Если файл не найден
            ValueError: Если формат файла неверный или размерности не совпадают
        """
        dataset = cls()
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл не найден: {filepath}")
        except Exception as e:
            raise ValueError(f"Ошибка чтения файла: {e}")
        
        pairs = []
        input_dim = None
        output_dim = None
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Пропускаем пустые строки и комментарии
            if not line or line.startswith('#'):
                continue
            
            # Разделяем по вертикальной черте
            if '|' not in line:
                raise ValueError(
                    f"Строка {line_num}: отсутствует разделитель '|'. "
                    f"Строка: {line}"
                )
            
            parts = line.split('|')
            if len(parts) != 2:
                raise ValueError(
                    f"Строка {line_num}: неверный формат (несколько разделителей '|'). "
                    f"Строка: {line}"
                )
            
            input_part, output_part = parts
            
            # Парсим входные значения
            try:
                inputs = [float(x.strip()) for x in input_part.split()]
            except ValueError:
                raise ValueError(
                    f"Строка {line_num}: неверный формат входных данных. "
                    f"Строка: {line}"
                )
            
            # Парсим выходные значения
            try:
                outputs = [float(x.strip()) for x in output_part.split()]
            except ValueError:
                raise ValueError(
                    f"Строка {line_num}: неверный формат выходных данных. "
                    f"Строка: {line}"
                )
            
            if len(inputs) == 0:
                raise ValueError(
                    f"Строка {line_num}: нет входных значений. "
                    f"Строка: {line}"
                )
            
            if len(outputs) == 0:
                raise ValueError(
                    f"Строка {line_num}: нет выходных значений. "
                    f"Строка: {line}"
                )
            
            # Проверяем консистентность размерностей
            if input_dim is None:
                input_dim = len(inputs)
            elif len(inputs) != input_dim:
                raise ValueError(
                    f"Строка {line_num}: несоответствие размерности входов "
                    f"(ожидается {input_dim}, получено {len(inputs)}). "
                    f"Строка: {line}"
                )
            
            if output_dim is None:
                output_dim = len(outputs)
            elif len(outputs) != output_dim:
                raise ValueError(
                    f"Строка {line_num}: несоответствие размерности выходов "
                    f"(ожидается {output_dim}, получено {len(outputs)}). "
                    f"Строка: {line}"
                )
            
            pairs.append(DataPair(inputs, outputs))
        
        if len(pairs) == 0:
            raise ValueError("Файл не содержит корректных данных")
        
        dataset.pairs = pairs
        dataset.input_dim = input_dim
        dataset.output_dim = output_dim
        
        return dataset
    
    def validate(self) -> bool:
        """
        Проверка консистентности размерностей.
        
        Returns:
            bool: True если все пары имеют одинаковую размерность
        """
        if len(self.pairs) == 0:
            return False
        
        for pair in self.pairs:
            if len(pair.inputs) != self.input_dim:
                return False
            if len(pair.outputs) != self.output_dim:
                return False
        
        return True
    
    def get_input_matrix(self) -> np.ndarray:
        """
        Возврат матрицы входов.
        
        Returns:
            np.ndarray: Матрица размером (n_samples, input_dim)
        """
        return np.array([pair.inputs for pair in self.pairs])
    
    def get_output_matrix(self) -> np.ndarray:
        """
        Возврат матрицы выходов.
        
        Returns:
            np.ndarray: Матрица размером (n_samples, output_dim)
        """
        return np.array([pair.outputs for pair in self.pairs])
    
    def normalize(self, method: str = "minmax") -> None:
        """
        Применение нормализации к данным.
        
        Args:
            method: Тип нормализации ("minmax" или "zscore")
        """
        if len(self.pairs) == 0:
            return
        
        X = self.get_input_matrix()
        y = self.get_output_matrix()
        
        self._normalization_type = method
        
        if method == "minmax":
            # Min-Max нормализация к диапазону [0, 1]
            self._input_min = X.min(axis=0)
            self._input_max = X.max(axis=0)
            self._output_min = y.min(axis=0)
            self._output_max = y.max(axis=0)
            
            # Избегаем деления на ноль
            input_range = self._input_max - self._input_min
            input_range[input_range == 0] = 1.0
            output_range = self._output_max - self._output_min
            output_range[output_range == 0] = 1.0
            
            X_norm = (X - self._input_min) / input_range
            y_norm = (y - self._output_min) / output_range
            
        elif method == "zscore":
            # Z-score нормализация (стандартизация)
            self._input_mean = X.mean(axis=0)
            self._input_std = X.std(axis=0)
            self._output_mean = y.mean(axis=0)
            self._output_std = y.std(axis=0)
            
            # Избегаем деления на ноль
            self._input_std[self._input_std == 0] = 1.0
            self._output_std[self._output_std == 0] = 1.0
            
            X_norm = (X - self._input_mean) / self._input_std
            y_norm = (y - self._output_mean) / self._output_std
        else:
            raise ValueError(f"Неизвестный метод нормализации: {method}")
        
        # Обновляем пары нормализованными данными
        for i, pair in enumerate(self.pairs):
            pair.inputs = X_norm[i].tolist()
            pair.outputs = y_norm[i].tolist()
        
        self._normalized = True
    
    def denormalize_outputs(self, outputs: np.ndarray) -> np.ndarray:
        """
        Обратное преобразование выходов.
        
        Args:
            outputs: Нормализованные выходные значения
            
        Returns:
            np.ndarray: Денормализованные значения
        """
        if not self._normalized:
            return outputs
        
        if self._normalization_type == "minmax":
            output_range = self._output_max - self._output_min
            output_range[output_range == 0] = 1.0
            return outputs * output_range + self._output_min
        elif self._normalization_type == "zscore":
            return outputs * self._output_std + self._output_mean
        
        return outputs
    
    def denormalize_inputs(self, inputs: np.ndarray) -> np.ndarray:
        """
        Обратное преобразование входов.
        
        Args:
            inputs: Нормализованные входные значения
            
        Returns:
            np.ndarray: Денормализованные значения
        """
        if not self._normalized:
            return inputs
        
        if self._normalization_type == "minmax":
            input_range = self._input_max - self._input_min
            input_range[input_range == 0] = 1.0
            return inputs * input_range + self._input_min
        elif self._normalization_type == "zscore":
            return inputs * self._input_std + self._input_mean
        
        return inputs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __repr__(self) -> str:
        return (f"Dataset(pairs={len(self.pairs)}, "
                f"input_dim={self.input_dim}, output_dim={self.output_dim})")
