"""
Модуль для работы с популяцией особей и эволюцией.
"""

import numpy as np
import random
from typing import List, Tuple, Optional
from .tree import ExpressionTree
from .data import Dataset


class Population:
    """Популяция особей (ExpressionTree) для эволюции."""
    
    def __init__(self):
        """Инициализация пустой популяции."""
        self.individuals: List[ExpressionTree] = []
        self.fitness_scores: List[float] = []
        self.input_dim: int = 1
        self.max_depth: int = 5
    
    def initialize_random(self, size: int, input_dim: int, max_depth: int = 5) -> None:
        """
        Случайная инициализация популяции.
        
        Args:
            size: Размер популяции
            input_dim: Размерность входа
            max_depth: Максимальная глубина деревьев
        """
        self.individuals = []
        self.fitness_scores = []
        self.input_dim = input_dim
        self.max_depth = max_depth
        
        for _ in range(size):
            tree = ExpressionTree.random(input_dim, max_depth)
            self.individuals.append(tree)
    
    def evaluate_fitness(self, dataset: Dataset) -> None:
        """
        Вычисление fitness для всех особей популяции.
        Fitness = MSE (mean squared error) между предсказаниями и реальными данными.
        
        Args:
            dataset: Датасет для вычисления fitness
        """
        self.fitness_scores = []
        X = dataset.get_input_matrix()
        y = dataset.get_output_matrix()
        
        for individual in self.individuals:
            predictions = []
            for i in range(len(X)):
                pred = individual.evaluate(X[i])
                predictions.append(pred)
            
            predictions = np.array(predictions)
            
            # Для многомерного выхода берем среднее по всем измерениям
            if len(y.shape) == 1:
                mse = np.mean((predictions - y) ** 2)
            else:
                # Если многомерный выход, усредняем
                mse = np.mean((predictions - y[:, 0]) ** 2)
            
            self.fitness_scores.append(float(mse))
    
    def select_tournament(self, k: int = 3) -> Tuple[ExpressionTree, float]:
        """
        Турнирная селекция одной особи.
        
        Args:
            k: Размер турнира (количество участников)
            
        Returns:
            Tuple[ExpressionTree, float]: Лучшая особь и её fitness
        """
        # Выбираем k случайных индексов
        indices = random.sample(range(len(self.individuals)), min(k, len(self.individuals)))
        
        # Находим лучшего среди выбранных
        best_idx = indices[0]
        best_fitness = self.fitness_scores[best_idx]
        
        for idx in indices[1:]:
            if self.fitness_scores[idx] < best_fitness:
                best_fitness = self.fitness_scores[idx]
                best_idx = idx
        
        return self.individuals[best_idx], best_fitness
    
    def evolve(self, generations: int, mutation_rate: float = 0.1, 
               crossover_rate: float = 0.7, elite_count: int = 2,
               dataset: Dataset = None) -> List[Tuple[float, float]]:
        """
        Основной цикл эволюции.
        
        Args:
            generations: Количество поколений
            mutation_rate: Вероятность мутации
            crossover_rate: Вероятность кроссовера
            elite_count: Количество элитных особей
            dataset: Датасет для вычисления fitness
            
        Returns:
            List[Tuple[float, float]]: История (best_fitness, avg_fitness) по поколениям
        """
        history = []
        
        for gen in range(generations):
            # Вычисляем fitness
            self.evaluate_fitness(dataset)
            
            # Сохраняем статистику
            best_fitness = min(self.fitness_scores)
            avg_fitness = np.mean(self.fitness_scores)
            history.append((best_fitness, avg_fitness))
            
            # Создаем новое поколение
            new_individuals = []
            
            # Элитизм: сохраняем лучших без изменений
            elite_indices = np.argsort(self.fitness_scores)[:elite_count]
            for idx in elite_indices:
                new_individuals.append(self.individuals[idx].copy())
            
            # Заполняем остальную часть популяции
            while len(new_individuals) < len(self.individuals):
                # Селекция родителей
                parent1, _ = self.select_tournament(k=3)
                parent2, _ = self.select_tournament(k=3)
                
                # Кроссовер
                if random.random() < crossover_rate:
                    child1, child2 = parent1.crossover(parent2)
                else:
                    child1 = parent1.copy()
                    child2 = parent2.copy()
                
                # Мутация
                child1.mutate(mutation_rate, self.max_depth)
                child2.mutate(mutation_rate, self.max_depth)
                
                # Добавляем потомков
                new_individuals.append(child1)
                if len(new_individuals) < len(self.individuals):
                    new_individuals.append(child2)
            
            # Обновляем популяцию
            self.individuals = new_individuals[:len(self.individuals)]
        
        # Финальная оценка fitness
        self.evaluate_fitness(dataset)
        
        return history
    
    def get_best(self) -> Tuple[ExpressionTree, float]:
        """
        Получение лучшей особи.
        
        Returns:
            Tuple[ExpressionTree, float]: Лучшая особь и её fitness
        """
        if len(self.individuals) == 0:
            return None, float('inf')
        
        best_idx = np.argmin(self.fitness_scores)
        return self.individuals[best_idx], self.fitness_scores[best_idx]
    
    def get_statistics(self) -> dict:
        """
        Получение статистики популяции.
        
        Returns:
            dict: Статистика (min, max, mean, std)
        """
        if len(self.fitness_scores) == 0:
            return {}
        
        return {
            'min': float(np.min(self.fitness_scores)),
            'max': float(np.max(self.fitness_scores)),
            'mean': float(np.mean(self.fitness_scores)),
            'std': float(np.std(self.fitness_scores)),
        }
    
    def __len__(self) -> int:
        return len(self.individuals)
    
    def __repr__(self) -> str:
        return f"Population(size={len(self.individuals)})"
