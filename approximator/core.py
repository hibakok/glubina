"""
Главный класс универсального аппроксиматора.
Объединяет все компоненты системы.
"""

import numpy as np
import pickle
import time
from typing import List, Tuple, Optional, Dict, Any
from .data import Dataset
from .tree import ExpressionTree
from .population import Population


class UniversalApproximator:
    """Главный класс аппроксиматора."""
    
    def __init__(self):
        """Инициализация аппроксиматора."""
        self.dataset: Optional[Dataset] = None
        self.population: Optional[Population] = None
        self.best_individual: Optional[ExpressionTree] = None
        self.best_fitness: float = float('inf')
        self.normalizer_params: Dict[str, Any] = {}
        self.generation_history: List[Tuple[float, float]] = []
        self.training_time: float = 0.0
        self._trained: bool = False
        
        # Параметры эволюции (по умолчанию)
        self.pop_size: int = 50
        self.generations: int = 100
        self.mutation_rate: float = 0.1
        self.crossover_rate: float = 0.7
        self.elite_count: int = 2
        self.max_depth: int = 5
    
    def load_data(self, filepath: str) -> None:
        """
        Загрузка данных из файла.
        
        Args:
            filepath: Путь к файлу с данными
        """
        self.dataset = Dataset.load_from_file(filepath)
        self._trained = False
        self.generation_history = []
        self.best_individual = None
        self.best_fitness = float('inf')
    
    def set_params(self, pop_size: int = None, generations: int = None,
                   mutation_rate: float = None, crossover_rate: float = None,
                   elite_count: int = None, max_depth: int = None) -> None:
        """
        Настройка параметров эволюции.
        
        Args:
            pop_size: Размер популяции
            generations: Количество поколений
            mutation_rate: Вероятность мутации
            crossover_rate: Вероятность кроссовера
            elite_count: Количество элитных особей
            max_depth: Максимальная глубина дерева
        """
        if pop_size is not None:
            self.pop_size = pop_size
        if generations is not None:
            self.generations = generations
        if mutation_rate is not None:
            self.mutation_rate = mutation_rate
        if crossover_rate is not None:
            self.crossover_rate = crossover_rate
        if elite_count is not None:
            self.elite_count = elite_count
        if max_depth is not None:
            self.max_depth = max_depth
    
    def train(self, generations: int = None, pop_size: int = None,
              mutation_rate: float = None, crossover_rate: float = None,
              elite_count: int = None, max_depth: int = None,
              verbose: bool = True) -> None:
        """
        Обучение модели.
        
        Args:
            generations: Количество поколений (переопределяет параметр по умолчанию)
            pop_size: Размер популяции
            mutation_rate: Вероятность мутации
            crossover_rate: Вероятность кроссовера
            elite_count: Количество элитных особей
            max_depth: Максимальная глубина дерева
            verbose: Выводить ли прогресс
        """
        if self.dataset is None:
            raise ValueError("Сначала загрузите данные через load_data()")
        
        if len(self.dataset) == 0:
            raise ValueError("Датасет пуст")
        
        # Применяем параметры
        gen = generations if generations is not None else self.generations
        pop = pop_size if pop_size is not None else self.pop_size
        mut = mutation_rate if mutation_rate is not None else self.mutation_rate
        cross = crossover_rate if crossover_rate is not None else self.crossover_rate
        elite = elite_count if elite_count is not None else self.elite_count
        depth = max_depth if max_depth is not None else self.max_depth
        
        # Нормализуем данные
        self.dataset.normalize(method="minmax")
        
        # Инициализируем популяцию
        self.population = Population()
        self.population.initialize_random(
            size=pop,
            input_dim=self.dataset.input_dim,
            max_depth=depth
        )
        
        # Запускаем эволюцию
        start_time = time.time()
        
        if verbose:
            print(f"Начало обучения: {gen} поколений, популяция={pop}")
            print("-" * 60)
        
        self.generation_history = self.population.evolve(
            generations=gen,
            mutation_rate=mut,
            crossover_rate=cross,
            elite_count=elite,
            dataset=self.dataset
        )
        
        self.training_time = time.time() - start_time
        
        # Получаем лучшую особь
        self.best_individual, self.best_fitness = self.population.get_best()
        self._trained = True
        
        if verbose:
            print("-" * 60)
            print(f"Обучение завершено за {self.training_time:.2f} сек")
            print(f"Лучшая fitness (MSE): {self.best_fitness:.6f}")
            print(f"Лучшая формула: {self.best_individual.to_string()}")
    
    def predict(self, inputs: List[float]) -> List[float]:
        """
        Предсказание выхода для новых входных данных.
        
        Args:
            inputs: Входные данные (список значений)
            
        Returns:
            List[float]: Предсказанные выходные значения
            
        Raises:
            ValueError: Если модель не обучена или размерность не совпадает
        """
        if self.best_individual is None:
            raise ValueError("Модель не обучена. Сначала вызовите train()")
        
        if len(inputs) != self.dataset.input_dim:
            raise ValueError(
                f"Неверная размерность входа: ожидается {self.dataset.input_dim}, "
                f"получено {len(inputs)}"
            )
        
        # Нормализуем входы
        inputs_array = np.array(inputs).reshape(1, -1)
        
        if self.dataset._normalized:
            if self.dataset._normalization_type == "minmax":
                input_range = self.dataset._input_max - self.dataset._input_min
                input_range[input_range == 0] = 1.0
                inputs_norm = (inputs_array - self.dataset._input_min) / input_range
            elif self.dataset._normalization_type == "zscore":
                inputs_norm = (inputs_array - self.dataset._input_mean) / self.dataset._input_std
            else:
                inputs_norm = inputs_array
        else:
            inputs_norm = inputs_array
        
        # Вычисляем предсказание
        pred_norm = self.best_individual.evaluate(inputs_norm[0])
        
        # Денормализуем выход
        if self.dataset._normalized:
            if self.dataset._normalization_type == "minmax":
                output_range = self.dataset._output_max - self.dataset._output_min
                output_range[output_range == 0] = 1.0
                pred = pred_norm * output_range[0] + self.dataset._output_min[0]
            elif self.dataset._normalization_type == "zscore":
                pred = pred_norm * self.dataset._output_std[0] + self.dataset._output_mean[0]
            else:
                pred = pred_norm
        else:
            pred = pred_norm
        
        return [float(pred)]
    
    def get_best_formula(self) -> str:
        """
        Возврат лучшей найденной формулы в читаемом виде.
        
        Returns:
            str: Формула в виде строки
        """
        if self.best_individual is None:
            return "Модель не обучена"
        
        vars_str = ", ".join([f"x{i}" for i in range(self.dataset.input_dim)])
        return f"f({vars_str}) = {self.best_individual.to_string()}"
    
    def save_model(self, filepath: str) -> None:
        """
        Сохранение модели в файл.
        
        Args:
            filepath: Путь к файлу для сохранения
        """
        if self.best_individual is None:
            raise ValueError("Нет модели для сохранения. Сначала обучите модель")
        
        model_data = {
            'best_individual': self.best_individual,
            'best_fitness': self.best_fitness,
            'input_dim': self.dataset.input_dim if self.dataset else 0,
            'output_dim': self.dataset.output_dim if self.dataset else 0,
            'normalizer_params': {
                '_normalized': self.dataset._normalized if self.dataset else False,
                '_normalization_type': self.dataset._normalization_type if self.dataset else 'minmax',
                '_input_min': self.dataset._input_min.tolist() if self.dataset and self.dataset._input_min is not None else None,
                '_input_max': self.dataset._input_max.tolist() if self.dataset and self.dataset._input_max is not None else None,
                '_output_min': self.dataset._output_min.tolist() if self.dataset and self.dataset._output_min is not None else None,
                '_output_max': self.dataset._output_max.tolist() if self.dataset and self.dataset._output_max is not None else None,
                '_input_mean': self.dataset._input_mean.tolist() if self.dataset and self.dataset._input_mean is not None else None,
                '_input_std': self.dataset._input_std.tolist() if self.dataset and self.dataset._input_std is not None else None,
                '_output_mean': self.dataset._output_mean.tolist() if self.dataset and self.dataset._output_mean is not None else None,
                '_output_std': self.dataset._output_std.tolist() if self.dataset and self.dataset._output_std is not None else None,
            },
            'generation_history': self.generation_history,
            'training_time': self.training_time,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str) -> None:
        """
        Загрузка сохраненной модели.
        
        Args:
            filepath: Путь к файлу с моделью
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл модели не найден: {filepath}")
        except Exception as e:
            raise ValueError(f"Ошибка загрузки модели: {e}")
        
        self.best_individual = model_data['best_individual']
        self.best_fitness = model_data['best_fitness']
        self.generation_history = model_data.get('generation_history', [])
        self.training_time = model_data.get('training_time', 0.0)
        
        # Восстанавливаем датасет с параметрами нормализации
        self.dataset = Dataset()
        self.dataset.input_dim = model_data['input_dim']
        self.dataset.output_dim = model_data['output_dim']
        self.dataset._normalized = model_data['normalizer_params'].get('_normalized', False)
        self.dataset._normalization_type = model_data['normalizer_params'].get('_normalization_type', 'minmax')
        
        norm_params = model_data['normalizer_params']
        if norm_params.get('_input_min') is not None:
            self.dataset._input_min = np.array(norm_params['_input_min'])
        if norm_params.get('_input_max') is not None:
            self.dataset._input_max = np.array(norm_params['_input_max'])
        if norm_params.get('_output_min') is not None:
            self.dataset._output_min = np.array(norm_params['_output_min'])
        if norm_params.get('_output_max') is not None:
            self.dataset._output_max = np.array(norm_params['_output_max'])
        if norm_params.get('_input_mean') is not None:
            self.dataset._input_mean = np.array(norm_params['_input_mean'])
        if norm_params.get('_input_std') is not None:
            self.dataset._input_std = np.array(norm_params['_input_std'])
        if norm_params.get('_output_mean') is not None:
            self.dataset._output_mean = np.array(norm_params['_output_mean'])
        if norm_params.get('_output_std') is not None:
            self.dataset._output_std = np.array(norm_params['_output_std'])
        
        self._trained = True
    
    def get_training_stats(self) -> dict:
        """
        Получение статистики обучения.
        
        Returns:
            dict: Статистика обучения
        """
        if len(self.generation_history) == 0:
            return {}
        
        initial_fitness = self.generation_history[0][0]
        final_fitness = self.generation_history[-1][0]
        
        # Подсчет количества улучшений
        improvements = 0
        prev_best = float('inf')
        for best, _ in self.generation_history:
            if best < prev_best:
                improvements += 1
                prev_best = best
        
        return {
            'total_generations': len(self.generation_history),
            'training_time': self.training_time,
            'initial_fitness': initial_fitness,
            'final_fitness': final_fitness,
            'improvements': improvements,
            'best_fitness': self.best_fitness,
            'formula': self.get_best_formula(),
        }
    
    def __repr__(self) -> str:
        status = "обучена" if self._trained else "не обучена"
        return f"UniversalApproximator(status={status}, best_fitness={self.best_fitness:.6f})"
