#!/usr/bin/env python3
"""
Эволюционирующий универсальный аппроксиматор
Использует генетические алгоритмы для поиска математических выражений,
аппроксимирующих заданные данные или функции.
"""

import random
import math
import operator
from typing import List, Callable, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time
import copy


# ==================== Структуры данных для многомерных случаев ====================

@dataclass
class DataPair:
    """Пара вход-выход для многомерных данных"""
    inputs: List[float]  # Входной вектор размерности N
    outputs: List[float]  # Выходной вектор размерности M


@dataclass
class ParsedData:
    """Результат парсинга файла с данными"""
    pairs: List[DataPair]  # Список пар вход-выход
    input_dim: int  # Размерность входа (N)
    output_dim: int  # Размерность выхода (M)
    total_lines: int  # Всего строк в файле
    parsed_lines: int  # Успешно распаршено
    skipped_lines: int  # Пропущено строк
    skip_reasons: Dict[str, int]  # Причины пропуска строк
    input_ranges: List[Tuple[float, float]]  # Диапазоны по каждому измерению входа
    output_ranges: List[Tuple[float, float]]  # Диапазоны по каждому измерению выхода
    error_message: Optional[str]  # Сообщение об ошибке, если есть


# ==================== Базовые компоненты ====================

@dataclass
class Node(ABC):
    """Базовый класс для узла дерева выражения"""
    
    @abstractmethod
    def evaluate(self, x: List[float]) -> float:
        """Вычислить значение узла. x - вектор входных значений"""
        pass
    
    @abstractmethod
    def evaluate_batch(self, X: List[List[float]]) -> List[float]:
        """Вычислить значение узла для набора точек.
        
        Args:
            X: Список входных векторов (каждый вектор размерности input_dim)
        
        Returns:
            Список результатов, по одному на каждую входную точку
        """
        pass
    
    @abstractmethod
    def depth(self) -> int:
        """Глубина узла"""
        pass
    
    @abstractmethod
    def copy(self) -> 'Node':
        """Создать копию узла"""
        pass
    
    @abstractmethod
    def to_string(self) -> str:
        """Строковое представление"""
        pass


class Constant(Node):
    """Константа"""
    
    def __init__(self, value: float):
        self.value = value
    
    def evaluate(self, x: List[float]) -> float:
        return self.value
    
    def evaluate_batch(self, X: List[List[float]]) -> List[float]:
        """Возвращает массив констант для всех точек."""
        return [self.value for _ in X]
    
    def depth(self) -> int:
        return 1
    
    def copy(self) -> 'Constant':
        return Constant(self.value)
    
    def to_string(self) -> str:
        return f"{self.value:.4f}"


class Variable(Node):
    """Переменная с индексом для многомерного входа (x[i])"""
    
    def __init__(self, index: int = 0):
        self.index = index
    
    def evaluate(self, x: List[float]) -> float:
        if 0 <= self.index < len(x):
            return x[self.index]
        # Если индекс выходит за границы, вернуть 0
        return 0.0
    
    def evaluate_batch(self, X: List[List[float]]) -> List[float]:
        """Возвращает массив значений переменной для всех точек."""
        result = []
        for x in X:
            if 0 <= self.index < len(x):
                val = x[self.index]
                # Защита от NaN и Inf
                if math.isnan(val) or math.isinf(val):
                    val = 0.0
            else:
                val = 0.0
            result.append(val)
        return result
    
    def depth(self) -> int:
        return 1
    
    def copy(self) -> 'Variable':
        return Variable(self.index)
    
    def to_string(self) -> str:
        return f"x[{self.index}]"


class BinaryOperator(Node):
    """Бинарный оператор"""
    
    def __init__(self, left: Node, right: Node, op: Callable[[float, float], float], symbol: str):
        self.left = left
        self.right = right
        self.op = op
        self.symbol = symbol
    
    def evaluate(self, x: List[float]) -> float:
        try:
            result = self.op(self.left.evaluate(x), self.right.evaluate(x))
            if math.isnan(result) or math.isinf(result):
                return 0.0
            return result
        except (ZeroDivisionError, ValueError, OverflowError):
            return 0.0
    
    def evaluate_batch(self, X: List[List[float]]) -> List[float]:
        """Вычисляет значение оператора для набора точек.
        
        Сначала вычисляются результаты левого и правого поддеревьев для всех точек,
        затем применяется операция поэлементно.
        """
        left_results = self.left.evaluate_batch(X)
        right_results = self.right.evaluate_batch(X)
        
        result = []
        for l, r in zip(left_results, right_results):
            try:
                val = self.op(l, r)
                if math.isnan(val) or math.isinf(val):
                    val = 0.0
            except (ZeroDivisionError, ValueError, OverflowError):
                val = 0.0
            result.append(val)
        return result
    
    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())
    
    def copy(self) -> 'BinaryOperator':
        return BinaryOperator(self.left.copy(), self.right.copy(), self.op, self.symbol)
    
    def to_string(self) -> str:
        return f"({self.left.to_string()} {self.symbol} {self.right.to_string()})"


class UnaryOperator(Node):
    """Унарный оператор"""
    
    def __init__(self, operand: Node, op: Callable[[float], float], symbol: str):
        self.operand = operand
        self.op = op
        self.symbol = symbol
    
    def evaluate(self, x: List[float]) -> float:
        try:
            result = self.op(self.operand.evaluate(x))
            if math.isnan(result) or math.isinf(result):
                return 0.0
            return result
        except (ValueError, ZeroDivisionError, OverflowError):
            return 0.0
    
    def evaluate_batch(self, X: List[List[float]]) -> List[float]:
        """Вычисляет значение оператора для набора точек.
        
        Сначала вычисляются результаты поддерева для всех точек,
        затем применяется операция поэлементно.
        """
        operand_results = self.operand.evaluate_batch(X)
        
        result = []
        for val in operand_results:
            try:
                res = self.op(val)
                if math.isnan(res) or math.isinf(res):
                    res = 0.0
            except (ValueError, ZeroDivisionError, OverflowError):
                res = 0.0
            result.append(res)
        return result
    
    def depth(self) -> int:
        return 1 + self.operand.depth()
    
    def copy(self) -> 'UnaryOperator':
        return UnaryOperator(self.operand.copy(), self.op, self.symbol)
    
    def to_string(self) -> str:
        return f"{self.symbol}({self.operand.to_string()})"


class ConditionalNode(Node):
    """Условный узел (If-Then-Else).
    
    Содержит три дочерних узла:
    - condition: условие (вычисляется как число, > 0 означает true)
    - then_branch: ветка "тогда" (выполняется если условие > 0)
    - else_branch: ветка "иначе" (выполняется если условие <= 0)
    """
    
    def __init__(self, condition: Node, then_branch: Node, else_branch: Node):
        self.condition = condition
        self.then_branch = then_branch
        self.else_branch = else_branch
    
    def evaluate(self, x: List[float]) -> float:
        try:
            cond_value = self.condition.evaluate(x)
            if cond_value > 0:
                return self.then_branch.evaluate(x)
            else:
                return self.else_branch.evaluate(x)
        except (ValueError, ZeroDivisionError, OverflowError):
            return 0.0
    
    def evaluate_batch(self, X: List[List[float]]) -> List[float]:
        """Вычисляет значение условного узла для набора точек.
        
        Для каждой точки вычисляется условие, и в зависимости от него
        выбирается результат из then_branch или else_branch.
        """
        cond_results = self.condition.evaluate_batch(X)
        then_results = self.then_branch.evaluate_batch(X)
        else_results = self.else_branch.evaluate_batch(X)
        
        result = []
        for cond_val, then_val, else_val in zip(cond_results, then_results, else_results):
            try:
                if math.isnan(cond_val) or math.isinf(cond_val):
                    cond_val = 0.0
                if cond_val > 0:
                    res = then_val
                else:
                    res = else_val
                # Защита от NaN/Inf в результате
                if math.isnan(res) or math.isinf(res):
                    res = 0.0
            except (ValueError, ZeroDivisionError, OverflowError):
                res = 0.0
            result.append(res)
        return result
    
    def depth(self) -> int:
        return 1 + max(
            self.condition.depth(),
            self.then_branch.depth(),
            self.else_branch.depth()
        )
    
    def copy(self) -> 'ConditionalNode':
        return ConditionalNode(
            self.condition.copy(),
            self.then_branch.copy(),
            self.else_branch.copy()
        )
    
    def to_string(self) -> str:
        return f"({self.condition.to_string()} > 0 ? {self.then_branch.to_string()} : {self.else_branch.to_string()})"


# ==================== Генератор популяции ====================

class ExpressionGenerator:
    """Генератор случайных выражений"""
    
    def __init__(self, max_depth: int = 5, input_dim: int = 1):
        self.max_depth = max_depth
        self.input_dim = input_dim  # Размерность входного вектора
        
        # Бинарные операторы: (функция, символ)
        self.binary_ops = [
            (operator.add, "+"),
            (operator.sub, "-"),
            (operator.mul, "*"),
            (self.safe_div, "/"),
            (self.safe_pow, "^"),
            # Новые бинарные операторы
            (self.safe_min, "min"),
            (self.safe_max, "max"),
            (self.safe_mod, "mod"),
            (self.safe_atan2, "atan2"),
            (self.safe_hypot, "hypot"),
            # Операторы сравнения (возвращают 1.0 или 0.0)
            (self.safe_gt, ">"),
            (self.safe_lt, "<"),
            (self.safe_ge, ">="),
            (self.safe_le, "<="),
            (self.safe_eq, "=="),
            (self.safe_ne, "!="),
        ]
        
        # Унарные операторы: (функция, символ)
        self.unary_ops = [
            (math.sin, "sin"),
            (math.cos, "cos"),
            (math.tan, "tan"),
            (math.exp, "exp"),
            (math.sqrt, "sqrt"),
            (math.log, "log"),
            (self.safe_neg, "neg"),
            # Новые унарные функции
            (abs, "abs"),
            (math.floor, "floor"),
            (math.ceil, "ceil"),
            (math.sinh, "sinh"),
            (math.cosh, "cosh"),
            (math.tanh, "tanh"),
            (self.safe_asin, "asin"),
            (self.safe_acos, "acos"),
            (math.atan, "atan"),
            (self.safe_log2, "log2"),
            (self.safe_log10, "log10"),
            (self.safe_sign, "sign"),
            (self.safe_sigmoid, "sigmoid"),
            (self.safe_softplus, "softplus"),
            # Защищённые версии
            (self.safe_log_protected, "log_p"),
            (self.safe_sqrt_protected, "sqrt_p"),
        ]
        
        self._node_count = 0
        self._max_nodes = 100
    
    def reset_counter(self):
        self._node_count = 0
    
    @staticmethod
    def safe_div(a: float, b: float) -> float:
        if abs(b) < 1e-10:
            return 0.0
        return a / b
    
    @staticmethod
    def safe_pow(a: float, b: float) -> float:
        try:
            if abs(a) > 1e10:
                a = math.copysign(1e10, a)
            if abs(b) > 10:
                b = math.copysign(10, b)
            result = math.pow(abs(a), b) if a >= 0 else 0.0
            if math.isnan(result) or math.isinf(result):
                return 0.0
            return result
        except (ValueError, OverflowError):
            return 0.0
    
    @staticmethod
    def safe_neg(a: float) -> float:
        return -a
    
    # === Новые унарные функции ===
    
    @staticmethod
    def safe_asin(a: float) -> float:
        try:
            if abs(a) > 1:
                a = math.copysign(1, a)
            return math.asin(a)
        except (ValueError, OverflowError):
            return 0.0
    
    @staticmethod
    def safe_acos(a: float) -> float:
        try:
            if abs(a) > 1:
                a = math.copysign(1, a)
            return math.acos(a)
        except (ValueError, OverflowError):
            return 0.0
    
    @staticmethod
    def safe_log2(a: float) -> float:
        try:
            if a <= 0:
                return 0.0
            return math.log2(a)
        except (ValueError, OverflowError):
            return 0.0
    
    @staticmethod
    def safe_log10(a: float) -> float:
        try:
            if a <= 0:
                return 0.0
            return math.log10(a)
        except (ValueError, OverflowError):
            return 0.0
    
    @staticmethod
    def safe_sign(a: float) -> float:
        if a > 0:
            return 1.0
        elif a < 0:
            return -1.0
        return 0.0
    
    @staticmethod
    def safe_sigmoid(a: float) -> float:
        try:
            if a > 500:
                return 1.0
            if a < -500:
                return 0.0
            return 1.0 / (1.0 + math.exp(-a))
        except (ValueError, OverflowError):
            return 0.0
    
    @staticmethod
    def safe_softplus(a: float) -> float:
        try:
            if a > 500:
                return a
            if a < -500:
                return 0.0
            return math.log(1.0 + math.exp(-a)) if a < 0 else math.log(1.0 + math.exp(a))
        except (ValueError, OverflowError):
            return 0.0
    
    @staticmethod
    def safe_log_protected(a: float) -> float:
        """Защищённый логарифм: log(abs(x) + 1e-10)"""
        try:
            return math.log(abs(a) + 1e-10)
        except (ValueError, OverflowError):
            return 0.0
    
    @staticmethod
    def safe_sqrt_protected(a: float) -> float:
        """Защищённый корень: sqrt(abs(x))"""
        try:
            return math.sqrt(abs(a))
        except (ValueError, OverflowError):
            return 0.0
    
    # === Новые бинарные функции ===
    
    @staticmethod
    def safe_min(a: float, b: float) -> float:
        return min(a, b)
    
    @staticmethod
    def safe_max(a: float, b: float) -> float:
        return max(a, b)
    
    @staticmethod
    def safe_mod(a: float, b: float) -> float:
        try:
            if abs(b) < 1e-10:
                return 0.0
            return a % b
        except (ValueError, ZeroDivisionError, OverflowError):
            return 0.0
    
    @staticmethod
    def safe_atan2(y: float, x: float) -> float:
        try:
            return math.atan2(y, x)
        except (ValueError, OverflowError):
            return 0.0
    
    @staticmethod
    def safe_hypot(a: float, b: float) -> float:
        try:
            return math.hypot(a, b)
        except (ValueError, OverflowError):
            return 0.0
    
    # === Операторы сравнения (возвращают 1.0 или 0.0) ===
    
    EPSILON = 1e-9
    
    @staticmethod
    def safe_gt(a: float, b: float) -> float:
        return 1.0 if a > b else 0.0
    
    @staticmethod
    def safe_lt(a: float, b: float) -> float:
        return 1.0 if a < b else 0.0
    
    @staticmethod
    def safe_ge(a: float, b: float) -> float:
        return 1.0 if a >= b else 0.0
    
    @staticmethod
    def safe_le(a: float, b: float) -> float:
        return 1.0 if a <= b else 0.0
    
    @staticmethod
    def safe_eq(a: float, b: float) -> float:
        return 1.0 if abs(a - b) < ExpressionGenerator.EPSILON else 0.0
    
    @staticmethod
    def safe_ne(a: float, b: float) -> float:
        return 1.0 if abs(a - b) >= ExpressionGenerator.EPSILON else 0.0
    
    def _create_random_variable(self) -> Variable:
        """Создать случайную переменную из доступных индексов"""
        if self.input_dim <= 0:
            return Variable(0)
        index = random.randint(0, self.input_dim - 1)
        return Variable(index)
    
    def generate_random(self, depth: int = 0) -> Node:
        """Сгенерировать случайное выражение"""
        self._node_count += 1
        if self._node_count > self._max_nodes:
            # Достигнут лимит узлов, вернуть простой узел
            if random.random() < 0.5:
                return Constant(random.uniform(-5, 5))
            else:
                return self._create_random_variable()
        
        if depth >= self.max_depth or (depth > 0 and random.random() < 0.4):
            # Листовой узел: константа или переменная
            if random.random() < 0.7:
                return Constant(random.uniform(-5, 5))
            else:
                return self._create_random_variable()
        
        # Внутренний узел
        choice = random.random()
        
        # Вероятность генерации условного узла ~12%
        if choice < 0.12 and depth < self.max_depth - 1:
            # Условный узел (If-Then-Else)
            condition = self.generate_random(depth + 1)
            then_branch = self.generate_random(depth + 1)
            else_branch = self.generate_random(depth + 1)
            return ConditionalNode(condition, then_branch, else_branch)
        elif choice < 0.75:  # Бинарный оператор (~63%)
            op_func, op_symbol = random.choice(self.binary_ops)
            left = self.generate_random(depth + 1)
            right = self.generate_random(depth + 1)
            return BinaryOperator(left, right, op_func, op_symbol)
        else:  # Унарный оператор (~25%)
            op_func, op_symbol = random.choice(self.unary_ops)
            operand = self.generate_random(depth + 1)
            return UnaryOperator(operand, op_func, op_symbol)


# ==================== Генетический алгоритм ====================

class Individual:
    """Особь в популяции - может представлять одно дерево или набор деревьев для многомерного выхода"""
    
    def __init__(self, expression):
        """
        expression может быть:
        - Node: для одномерного выхода (M=1)
        - List[Node]: для многомерного выхода (M>1)
        """
        if isinstance(expression, list):
            self.expressions = expression  # Список деревьев для многомерного выхода
        else:
            self.expressions = [expression]  # Обертываем в список для единообразия
        
        self.fitness = float('inf')
    
    @property
    def expression(self):
        """Для обратной совместимости - возвращает первое дерево (для M=1)"""
        return self.expressions[0] if self.expressions else None
    
    @property
    def output_dim(self):
        """Размерность выхода (количество деревьев)"""
        return len(self.expressions)
    
    def copy(self) -> 'Individual':
        new_individual = Individual([expr.copy() for expr in self.expressions])
        new_individual.fitness = self.fitness
        return new_individual


class GeneticAlgorithm:
    """Генетический алгоритм для эволюции выражений"""
    
    def __init__(self, 
                 population_size: int = 100,
                 mutation_rate: float = 0.3,
                 crossover_rate: float = 0.7,
                 elitism_count: int = 5,
                 max_generations: int = 1000,
                 target_fitness: float = 1e-6,
                 max_depth: int = 8,
                 input_dim: int = 1,
                 output_dim: int = 1,
                 auto_scale_params: bool = True):
        """
        Инициализация генетического алгоритма.
        
        Args:
            population_size: Базовый размер популяции (будет масштабирован при auto_scale_params=True)
            mutation_rate: Вероятность мутации
            crossover_rate: Вероятность кроссовера
            elitism_count: Количество элитных особей
            max_generations: Базовое максимальное количество поколений (будет масштабировано)
            target_fitness: Целевая приспособленность
            max_depth: Максимальная глубина дерева выражения
            input_dim: Размерность входного вектора
            output_dim: Размерность выходного вектора
            auto_scale_params: Если True, автоматически масштабировать параметры в зависимости от размерности
        """
        
        # Автоматическое масштабирование параметров в зависимости от размерности задачи
        if auto_scale_params:
            # Масштабирующий коэффициент на основе общей сложности задачи
            # Чем выше размерность входа и выхода, тем сложнее задача
            total_dim = input_dim + output_dim
            scale_factor = max(1.0, total_dim / 2.0)
            
            # Увеличиваем размер популяции пропорционально размерности
            # Минимум 100, максимум 500
            scaled_population = int(population_size * scale_factor)
            self.population_size = max(100, min(500, scaled_population))
            
            # Увеличиваем количество поколений пропорционально размерности
            # Минимум исходное значение, максимум 2000
            scaled_generations = int(max_generations * scale_factor)
            self.max_generations = max(max_generations, min(2000, scaled_generations))
            
            # Немного увеличиваем вероятность мутации для более сложных задач
            # чтобы поддерживать разнообразие
            self.mutation_rate = min(0.5, mutation_rate + 0.05 * (total_dim - 2))
        else:
            self.population_size = population_size
            self.max_generations = max_generations
            self.mutation_rate = mutation_rate
        
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.target_fitness = target_fitness
        self.input_dim = input_dim  # Размерность входного вектора
        self.output_dim = output_dim  # Размерность выходного вектора (M)
        self.generator = ExpressionGenerator(max_depth=max_depth, input_dim=input_dim)
        self.population: List[Individual] = []
        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []
        self.tournament_size = 7
        self.auto_scale_params = auto_scale_params
    
    def initialize_population(self):
        """Инициализировать начальную популяцию
        
        Для многомерного выхода (M>1) каждая особь содержит M деревьев выражений.
        При инициализации гарантируется, что разные деревья используют разные входные переменные,
        чтобы эволюция начиналась с разнообразного поиска.
        """
        self.population = []
        for ind_idx in range(self.population_size):
            trees = []
            # Гарантируем, что хотя бы часть начальных деревьев использует разные переменные
            # Циклически распределяем индексы переменных между деревьями
            for tree_idx in range(self.output_dim):
                self.generator.reset_counter()
                # Для первого дерева каждой особи используем циклический индекс переменной
                # Это гарантирует покрытие всех входных измерений в начальной популяции
                if self.input_dim > 1 and tree_idx < self.input_dim:
                    # Создаём дерево с гарантированным использованием конкретной переменной
                    expr = self._generate_tree_with_variable(tree_idx % self.input_dim)
                else:
                    expr = self.generator.generate_random()
                trees.append(expr)
            individual = Individual(trees if self.output_dim > 1 else trees[0])
            self.population.append(individual)
    
    def _generate_tree_with_variable(self, var_index: int) -> Node:
        """Сгенерировать случайное выражение, гарантирующее использование переменной x[var_index]
        
        Метод создаёт дерево, в котором хотя бы один лист - переменная с заданным индексом.
        """
        # Сначала генерируем случайное дерево
        self.generator.reset_counter()
        expr = self.generator.generate_random()
        
        # С вероятностью 50% заменяем случайный узел на нужную переменную
        if random.random() < 0.5:
            node_to_replace = self.get_random_node(expr)
            if node_to_replace and node_to_replace is not expr:
                parent = self.find_parent(expr, node_to_replace)
                if parent:
                    new_var = Variable(var_index)
                    if isinstance(parent, BinaryOperator):
                        if parent.left is node_to_replace:
                            parent.left = new_var
                        else:
                            parent.right = new_var
                    elif isinstance(parent, UnaryOperator):
                        parent.operand = new_var
            elif expr.depth() == 1:
                # Если дерево очень простое, просто возвращаем переменную
                expr = Variable(var_index)
        
        return expr
    
    def evaluate_fitness(self, individual: Individual, 
                        target_values: List[List[float]],
                        test_points: List[List[float]]) -> float:
        """Вычислить приспособленность особи.
        
        test_points - список входных векторов (каждый вектор размерности input_dim)
        target_values - предвычисленные значения целевой функции для всех test_points.
                        target_values[i] - вектор выходов размерности output_dim для i-й точки.
        
        Для многомерного выхода fitness считается как MSE (средний квадрат ошибки) по всем точкам
        и по всем компонентам выхода.
        Формула: MSE = sum((predicted[i] - actual[i])^2) / (num_points * num_outputs)
        
        Используется батчевая обработка: для каждого дерева вычисляется значение сразу для всех точек,
        что сокращает количество обходов дерева в num_points раз.
        """
        total_squared_error = 0.0
        num_points = len(test_points)
        num_outputs = self.output_dim
        
        if num_points == 0:
            return float('inf')
        
        # Вычисляем предсказания для всех деревьев и всех точек сразу (батчевая обработка)
        # predicted_batch[i] = список из num_points значений для i-го дерева
        predicted_batches = []
        for i in range(num_outputs):
            predicted_batch = individual.expressions[i].evaluate_batch(test_points)
            predicted_batches.append(predicted_batch)
        
        # Вычисляем ошибку для каждой точки, используя предвычисленные целевые значения
        for point_idx in range(num_points):
            actual = target_values[point_idx]  # Берём готовые значения из массива
            
            # Суммировать квадрат ошибки по всем компонентам выхода
            for i in range(num_outputs):
                predicted = predicted_batches[i][point_idx]
                diff = predicted - actual[i]
                # Ограничиваем максимальную ошибку чтобы избежать переполнения
                diff = max(-1e6, min(1e6, diff))
                total_squared_error += diff * diff
        
        # MSE = сумма квадратов ошибок / (кол-во точек * кол-во выходов)
        mse = total_squared_error / (num_points * num_outputs)
        individual.fitness = mse
        return mse
    
    def select_tournament(self, tournament_size: int = None) -> Individual:
        """Турнирная селекция"""
        if tournament_size is None:
            tournament_size = self.tournament_size
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return min(tournament, key=lambda ind: ind.fitness)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Кроссовер между двумя особями с поддержкой многомерного выхода.
        
        При кроссовере можно обмениваться поддеревьями как внутри одного выхода,
        так и между разными выходами (если размерность совпадает).
        """
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1_trees = [expr.copy() for expr in parent1.expressions]
        child2_trees = [expr.copy() for expr in parent2.expressions]
        
        # Выполнить несколько кроссоверов для разных деревьев
        num_crossovers = max(1, self.output_dim // 2)
        
        for _ in range(num_crossovers):
            # Случайно выбрать индекс дерева для кроссовера
            tree_idx = random.randint(0, self.output_dim - 1)
            
            # Найти случайные узлы для обмена в выбранных деревьях
            node1 = self.get_random_node(child1_trees[tree_idx])
            node2 = self.get_random_node(child2_trees[tree_idx])
            
            # Заменить узлы друг с другом
            if node1 and node2 and node1 is not child1_trees[tree_idx] and node2 is not child2_trees[tree_idx]:
                node1_copy = self.find_corresponding_node(child1_trees[tree_idx], node1)
                node2_copy = self.find_corresponding_node(child2_trees[tree_idx], node2)
                
                if node1_copy and node2_copy:
                    parent1_parent = self.find_parent(child1_trees[tree_idx], node1_copy)
                    parent2_parent = self.find_parent(child2_trees[tree_idx], node2_copy)
                    
                    if parent1_parent:
                        if isinstance(parent1_parent, BinaryOperator):
                            if parent1_parent.left is node1_copy:
                                parent1_parent.left = node2_copy.copy()
                            else:
                                parent1_parent.right = node2_copy.copy()
                        elif isinstance(parent1_parent, UnaryOperator):
                            parent1_parent.operand = node2_copy.copy()
                        elif isinstance(parent1_parent, ConditionalNode):
                            # Для условного узла заменяем одну из ветвей
                            if parent1_parent.condition is node1_copy:
                                parent1_parent.condition = node2_copy.copy()
                            elif parent1_parent.then_branch is node1_copy:
                                parent1_parent.then_branch = node2_copy.copy()
                            else:
                                parent1_parent.else_branch = node2_copy.copy()
                    
                    if parent2_parent:
                        if isinstance(parent2_parent, BinaryOperator):
                            if parent2_parent.left is node2_copy:
                                parent2_parent.left = node1_copy.copy()
                            else:
                                parent2_parent.right = node1_copy.copy()
                        elif isinstance(parent2_parent, UnaryOperator):
                            parent2_parent.operand = node1_copy.copy()
                        elif isinstance(parent2_parent, ConditionalNode):
                            # Для условного узла заменяем одну из ветвей
                            if parent2_parent.condition is node2_copy:
                                parent2_parent.condition = node1_copy.copy()
                            elif parent2_parent.then_branch is node2_copy:
                                parent2_parent.then_branch = node1_copy.copy()
                            else:
                                parent2_parent.else_branch = node1_copy.copy()
        
        # С вероятностью выполнить кроссовер между разными деревьями (между выходами)
        if self.output_dim > 1 and random.random() < 0.3:
            # Выбрать два разных дерева в каждой особи
            tree_idx1 = random.randint(0, self.output_dim - 1)
            tree_idx2 = random.randint(0, self.output_dim - 1)
            if tree_idx1 != tree_idx2:
                # Обменяться случайными поддеревьями между разными выходами
                node1 = self.get_random_node(child1_trees[tree_idx1])
                node2 = self.get_random_node(child1_trees[tree_idx2])
                
                if node1 and node2:
                    node1_copy = self.find_corresponding_node(child1_trees[tree_idx1], node1)
                    node2_copy = self.find_corresponding_node(child1_trees[tree_idx2], node2)
                    
                    if node1_copy and node2_copy:
                        parent1_parent = self.find_parent(child1_trees[tree_idx1], node1_copy)
                        parent2_parent = self.find_parent(child1_trees[tree_idx2], node2_copy)
                        
                        if parent1_parent and parent2_parent:
                            if isinstance(parent1_parent, BinaryOperator):
                                if parent1_parent.left is node1_copy:
                                    parent1_parent.left = node2_copy.copy()
                                else:
                                    parent1_parent.right = node2_copy.copy()
                            elif isinstance(parent1_parent, UnaryOperator):
                                parent1_parent.operand = node2_copy.copy()
                            elif isinstance(parent1_parent, ConditionalNode):
                                # Для условного узла заменяем одну из ветвей
                                if parent1_parent.condition is node1_copy:
                                    parent1_parent.condition = node2_copy.copy()
                                elif parent1_parent.then_branch is node1_copy:
                                    parent1_parent.then_branch = node2_copy.copy()
                                else:
                                    parent1_parent.else_branch = node2_copy.copy()
                            
                            if isinstance(parent2_parent, BinaryOperator):
                                if parent2_parent.left is node2_copy:
                                    parent2_parent.left = node1_copy.copy()
                                else:
                                    parent2_parent.right = node1_copy.copy()
                            elif isinstance(parent2_parent, UnaryOperator):
                                parent2_parent.operand = node1_copy.copy()
                            elif isinstance(parent2_parent, ConditionalNode):
                                # Для условного узла заменяем одну из ветвей
                                if parent2_parent.condition is node2_copy:
                                    parent2_parent.condition = node1_copy.copy()
                                elif parent2_parent.then_branch is node2_copy:
                                    parent2_parent.then_branch = node1_copy.copy()
                                else:
                                    parent2_parent.else_branch = node1_copy.copy()
        
        return Individual(child1_trees if self.output_dim > 1 else child1_trees[0]), \
               Individual(child2_trees if self.output_dim > 1 else child2_trees[0])
    
    def find_corresponding_node(self, tree: Node, target: Node) -> Optional[Node]:
        """Найти узел в дереве, соответствующий целевому узлу по структуре"""
        # Для простоты используем случайный узел того же типа
        nodes = self.collect_all_nodes(tree)
        for node in nodes:
            if type(node) == type(target):
                return node
        return nodes[0] if nodes else None
    
    def get_random_node(self, root: Node) -> Optional[Node]:
        """Получить случайный узел из дерева"""
        nodes = self.collect_all_nodes(root)
        if nodes:
            return random.choice(nodes)
        return None
    
    def collect_all_nodes(self, root: Node) -> List[Node]:
        """Собрать все узлы дерева"""
        nodes = [root]
        
        if isinstance(root, BinaryOperator):
            nodes.extend(self.collect_all_nodes(root.left))
            nodes.extend(self.collect_all_nodes(root.right))
        elif isinstance(root, UnaryOperator):
            nodes.extend(self.collect_all_nodes(root.operand))
        elif isinstance(root, ConditionalNode):
            nodes.extend(self.collect_all_nodes(root.condition))
            nodes.extend(self.collect_all_nodes(root.then_branch))
            nodes.extend(self.collect_all_nodes(root.else_branch))
        
        return nodes
    
    def get_constants_from_tree(self, root: Node) -> List[Constant]:
        """Собрать все константы из дерева в порядке обхода"""
        constants = []
        
        if isinstance(root, Constant):
            constants.append(root)
        elif isinstance(root, BinaryOperator):
            constants.extend(self.get_constants_from_tree(root.left))
            constants.extend(self.get_constants_from_tree(root.right))
        elif isinstance(root, UnaryOperator):
            constants.extend(self.get_constants_from_tree(root.operand))
        elif isinstance(root, ConditionalNode):
            constants.extend(self.get_constants_from_tree(root.condition))
            constants.extend(self.get_constants_from_tree(root.then_branch))
            constants.extend(self.get_constants_from_tree(root.else_branch))
        
        return constants
    
    def set_constants_in_tree(self, root: Node, constants: List[float], index: int = 0) -> int:
        """Записать значения констант обратно в дерево. Возвращает следующий индекс."""
        if isinstance(root, Constant):
            if index < len(constants):
                root.value = constants[index]
                return index + 1
            return index
        elif isinstance(root, BinaryOperator):
            index = self.set_constants_in_tree(root.left, constants, index)
            index = self.set_constants_in_tree(root.right, constants, index)
            return index
        elif isinstance(root, UnaryOperator):
            index = self.set_constants_in_tree(root.operand, constants, index)
            return index
        elif isinstance(root, ConditionalNode):
            index = self.set_constants_in_tree(root.condition, constants, index)
            index = self.set_constants_in_tree(root.then_branch, constants, index)
            index = self.set_constants_in_tree(root.else_branch, constants, index)
            return index
        return index
    
    def _compute_mse_for_constants(self, expression: Node, constants: List[float], 
                                   target_values: List[List[float]],
                                   test_points: List[List[float]]) -> float:
        """Вычислить MSE для заданных значений констант.
        
        target_values - предвычисленные значения целевой функции для всех test_points.
                        target_values[i][0] - значение целевой функции для i-й точки (для первого выхода).
        """
        # Записать константы в дерево
        self.set_constants_in_tree(expression, constants, 0)
        
        # Вычислить ошибку, используя предвычисленные целевые значения
        total_squared_error = 0.0
        num_points = len(test_points)
        
        if num_points == 0:
            return float('inf')
        
        for idx, x_vec in enumerate(test_points):
            try:
                predicted = expression.evaluate(x_vec)
                actual = target_values[idx][0]  # Берём готовое значение из массива (для первого выхода)
                diff = predicted - actual
                diff = max(-1e6, min(1e6, diff))
                total_squared_error += diff * diff
            except:
                return float('inf')
        
        return total_squared_error / num_points
    
    def optimize_constants_nelder_mead(self, individual: Individual,
                                       target_values: List[List[float]],
                                       test_points: List[List[float]],
                                       max_iterations: int = 100,
                                       tol: float = 1e-8) -> bool:
        """Оптимизация констант методом Нелдера-Мида (simplex method).
        
        target_values - предвычисленные значения целевой функции для всех test_points.
        Возвращает True если оптимизация была выполнена успешно.
        """
        # Работаем только с первым деревом (для M=1)
        expr = individual.expressions[0].copy()
        
        # Собрать начальные значения констант
        constants_nodes = self.get_constants_from_tree(expr)
        n_consts = len(constants_nodes)
        
        if n_consts == 0:
            return False  # Нет констант для оптимизации
        
        if n_consts > 20:
            return False  # Слишком много констант, метод будет медленным
        
        initial_constants = [c.value for c in constants_nodes]
        
        # Функция ошибки для оптимизации
        def objective(consts):
            return self._compute_mse_for_constants(expr, list(consts), target_values, test_points)
        
        # Инициализация симплекса
        simplex = [initial_constants[:]]
        for i in range(n_consts):
            point = initial_constants[:]
            # Шаг для каждой координаты
            step = max(0.1, abs(point[i]) * 0.1)
            point[i] += step
            simplex.append(point)
        
        # Параметры метода Нелдера-Мида
        alpha = 1.0  # отражение
        gamma = 2.0  # растяжение
        rho = 0.5    # сжатие
        sigma = 0.5  # уменьшение
        
        best_fitness = objective(initial_constants)
        
        for iteration in range(max_iterations):
            # Сортировка симплекса по значению функции
            simplex.sort(key=objective)
            
            # Проверка сходимости
            best_val = objective(simplex[0])
            worst_val = objective(simplex[-1])
            if abs(worst_val - best_val) < tol:
                break
            
            # Центроид без худшей точки
            centroid = [sum(simplex[i][j] for i in range(n_consts)) / n_consts 
                       for j in range(n_consts)]
            
            # Отражение
            xr = [centroid[j] + alpha * (centroid[j] - simplex[-1][j]) for j in range(n_consts)]
            xr_fitness = objective(xr)
            
            if objective(simplex[0]) <= xr_fitness < objective(simplex[-2]):
                simplex[-1] = xr
                continue
            
            # Растяжение
            if xr_fitness < objective(simplex[0]):
                xe = [centroid[j] + gamma * (xr[j] - centroid[j]) for j in range(n_consts)]
                xe_fitness = objective(xe)
                if xe_fitness < xr_fitness:
                    simplex[-1] = xe
                else:
                    simplex[-1] = xr
                continue
            
            # Сжатие
            xc = [centroid[j] + rho * (simplex[-1][j] - centroid[j]) for j in range(n_consts)]
            xc_fitness = objective(xc)
            
            if xc_fitness < worst_val:
                simplex[-1] = xc
                continue
            
            # Уменьшение
            for i in range(1, len(simplex)):
                simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) 
                             for j in range(n_consts)]
        
        # Получить лучшее решение
        simplex.sort(key=objective)
        best_constants = simplex[0]
        final_fitness = objective(best_constants)
        
        # Применить лучшие константы к особи
        if final_fitness < best_fitness:
            self.set_constants_in_tree(individual.expressions[0], best_constants, 0)
            return True
        
        return False
    
    def optimize_constants_coordinate_descent(self, individual: Individual,
                                              target_values: List[List[float]],
                                              test_points: List[List[float]],
                                              max_iterations: int = 100,
                                              tol: float = 1e-8) -> bool:
        """Оптимизация констант координатным спуском.
        
        target_values - предвычисленные значения целевой функции для всех test_points.
        Поочерёдно оптимизируем каждую константу, фиксируя остальные.
        Возвращает True если оптимизация была выполнена успешно.
        """
        expr = individual.expressions[0].copy()
        constants_nodes = self.get_constants_from_tree(expr)
        n_consts = len(constants_nodes)
        
        if n_consts == 0:
            return False
        
        if n_consts > 30:
            return False
        
        current_constants = [c.value for c in constants_nodes]
        best_fitness = self._compute_mse_for_constants(expr, current_constants, target_values, test_points)
        
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            for i in range(n_consts):
                # Поиск оптимального значения для i-й константы методом золотого сечения
                const_i = current_constants[i]
                
                # Определяем диапазон поиска
                search_range = max(10.0, abs(const_i) * 2.0)
                a = const_i - search_range
                b = const_i + search_range
                
                # Золотое сечение
                phi = (1 + math.sqrt(5)) / 2
                resphi = 2 - phi
                
                c = a + resphi * (b - a)
                d = b - resphi * (b - a)
                
                # Вычисляем fitness для c и d
                test_consts_c = current_constants[:]
                test_consts_c[i] = c
                fitness_c = self._compute_mse_for_constants(expr, test_consts_c, target_values, test_points)
                
                test_consts_d = current_constants[:]
                test_consts_d[i] = d
                fitness_d = self._compute_mse_for_constants(expr, test_consts_d, target_values, test_points)
                
                # Итерации золотого сечения
                for _ in range(20):
                    if fitness_c < fitness_d:
                        b = d
                        d = c
                        fitness_d = fitness_c
                        c = a + resphi * (b - a)
                        test_consts_c = current_constants[:]
                        test_consts_c[i] = c
                        fitness_c = self._compute_mse_for_constants(expr, test_consts_c, target_values, test_points)
                    else:
                        a = c
                        c = d
                        fitness_c = fitness_d
                        d = b - resphi * (b - a)
                        test_consts_d = current_constants[:]
                        test_consts_d[i] = d
                        fitness_d = self._compute_mse_for_constants(expr, test_consts_d, target_values, test_points)
                
                # Лучшее значение
                best_val = min(fitness_c, fitness_d)
                best_const = c if fitness_c < fitness_d else d
                
                if best_val < current_constants[i]:
                    current_constants[i] = best_const
                    improved = True
        
        # Проверка улучшения
        final_fitness = self._compute_mse_for_constants(expr, current_constants, target_values, test_points)
        
        if final_fitness < best_fitness - tol:
            self.set_constants_in_tree(individual.expressions[0], current_constants, 0)
            return True
        
        return False
    
    def optimize_constants(self, individual: Individual,
                          target_values: List[List[float]],
                          test_points: List[List[float]],
                          method: str = 'hybrid') -> bool:
        """Общая функция оптимизации констант.
        
        Args:
            individual: Особь для оптимизации
            target_values: Предвычисленные значения целевой функции для всех test_points
            test_points: Точки данных
            method: 'nelder', 'coordinate', или 'hybrid'
        
        Returns:
            True если оптимизация улучшила фитнес
        """
        if method == 'nelder':
            return self.optimize_constants_nelder_mead(individual, target_values, test_points)
        elif method == 'coordinate':
            return self.optimize_constants_coordinate_descent(individual, target_values, test_points)
        else:  # hybrid
            # Сначала Нелдер-Мид, потом координатный спуск
            result1 = self.optimize_constants_nelder_mead(individual, target_values, test_points, max_iterations=50)
            result2 = self.optimize_constants_coordinate_descent(individual, target_values, test_points, max_iterations=50)
            return result1 or result2
    
    def combine_expressions(self, expr1: Node, expr2: Node) -> Node:
        """Комбинировать два выражения"""
        if random.random() < 0.5:
            op_func, op_symbol = random.choice(self.generator.binary_ops)
            return BinaryOperator(expr1.copy(), expr2.copy(), op_func, op_symbol)
        else:
            op_func, op_symbol = random.choice(self.generator.unary_ops)
            return UnaryOperator(expr1.copy(), op_func, op_symbol)
    
    def mutate(self, individual: Individual) -> Individual:
        """Мутация особи с поддержкой многомерного выхода.
        
        При мутации случайно выбирается, какое из M деревьев мутировать.
        """
        if random.random() > self.mutation_rate:
            return individual
        
        new_trees = [expr.copy() for expr in individual.expressions]
        
        # Для многомерного выхода: выбрать случайное дерево для мутации
        # или мутировать несколько деревьев
        if self.output_dim > 1:
            num_mutations = random.randint(1, max(1, self.output_dim // 2))
            trees_to_mutate = random.sample(range(self.output_dim), min(num_mutations, self.output_dim))
        else:
            trees_to_mutate = [0]
        
        for tree_idx in trees_to_mutate:
            tree = new_trees[tree_idx]
            
            # Типы мутаций с разными вероятностями
            mutation_type = random.choices(
                ['replace_subtree', 'change_constant', 'add_operator', 'simplify'],
                weights=[0.4, 0.3, 0.2, 0.1]
            )[0]
            
            if mutation_type == 'replace_subtree':
                # Заменить случайное поддерево на новое случайное выражение
                nodes = self.collect_all_nodes(tree)
                if len(nodes) > 1:
                    node_to_replace = random.choice(nodes[1:])  # Не корень
                    parent = self.find_parent(tree, node_to_replace)
                    
                    if parent:
                        if isinstance(parent, BinaryOperator):
                            if parent.left is node_to_replace:
                                parent.left = self.generator.generate_random()
                            else:
                                parent.right = self.generator.generate_random()
                        elif isinstance(parent, UnaryOperator):
                            parent.operand = self.generator.generate_random()
                        elif isinstance(parent, ConditionalNode):
                            # Для условного узла заменяем одну из ветвей
                            branch_choice = random.random()
                            if branch_choice < 0.33 and parent.condition is node_to_replace:
                                parent.condition = self.generator.generate_random()
                            elif branch_choice < 0.66 and parent.then_branch is node_to_replace:
                                parent.then_branch = self.generator.generate_random()
                            else:
                                parent.else_branch = self.generator.generate_random()
            
            elif mutation_type == 'change_constant':
                # Изменить случайную константу (тонкая настройка)
                constants = [n for n in self.collect_all_nodes(tree) if isinstance(n, Constant)]
                if constants:
                    const = random.choice(constants)
                    # Гауссовская мутация с уменьшающимся шагом
                    const.value += random.gauss(0, 0.5)
            
            elif mutation_type == 'add_operator':
                # Добавить оператор вокруг случайного узла
                nodes = self.collect_all_nodes(tree)
                if nodes and len(nodes) < 50:  # Ограничить размер
                    node = random.choice(nodes)
                    op_func, op_symbol = random.choice(self.generator.unary_ops)
                    new_unary = UnaryOperator(node.copy(), op_func, op_symbol)
                    # Нужно заменить node в родителе на new_unary
                    parent = self.find_parent(tree, node)
                    if parent:
                        if isinstance(parent, BinaryOperator):
                            if parent.left is node:
                                parent.left = new_unary
                            else:
                                parent.right = new_unary
                        elif isinstance(parent, UnaryOperator):
                            parent.operand = new_unary
                        elif isinstance(parent, ConditionalNode):
                            # Для условного узла заменяем одну из ветвей
                            if parent.condition is node:
                                parent.condition = new_unary
                            elif parent.then_branch is node:
                                parent.then_branch = new_unary
                            else:
                                parent.else_branch = new_unary
                    else:
                        new_trees[tree_idx] = new_unary
            
            elif mutation_type == 'simplify':
                # Упростить выражение, удалив сложные части
                nodes = self.collect_all_nodes(tree)
                if len(nodes) > 20:
                    # Удалить случайную ветку, заменив её на константу или переменную
                    deep_nodes = [n for n in nodes if n.depth() > 3]
                    if deep_nodes:
                        node_to_simplify = random.choice(deep_nodes)
                        parent = self.find_parent(tree, node_to_simplify)
                        if parent:
                            replacement = Variable() if random.random() < 0.5 else Constant(random.uniform(-5, 5))
                            if isinstance(parent, BinaryOperator):
                                if parent.left is node_to_simplify:
                                    parent.left = replacement
                                else:
                                    parent.right = replacement
                            elif isinstance(parent, UnaryOperator):
                                parent.operand = replacement
                            elif isinstance(parent, ConditionalNode):
                                # Для условного узла упрощаем одну из ветвей
                                if parent.condition is node_to_simplify:
                                    parent.condition = replacement
                                elif parent.then_branch is node_to_simplify:
                                    parent.then_branch = replacement
                                else:
                                    parent.else_branch = replacement
        
        # Проверка на слишком большое выражение
        for tree in new_trees:
            if tree.depth() > 15:
                # Вернуться к оригиналу или упростить
                return individual
        
        return Individual(new_trees if self.output_dim > 1 else new_trees[0])
    
    def find_parent(self, root: Node, target: Node) -> Optional[Node]:
        """Найти родителя узла"""
        if root is target:
            return None
        
        if isinstance(root, BinaryOperator):
            if root.left is target or root.right is target:
                return root
            left_parent = self.find_parent(root.left, target)
            if left_parent:
                return left_parent
            return self.find_parent(root.right, target)
        
        elif isinstance(root, UnaryOperator):
            if root.operand is target:
                return root
            return self.find_parent(root.operand, target)
        
        elif isinstance(root, ConditionalNode):
            # Для условного узла проверяем все три ветви
            if root.condition is target:
                return root
            cond_parent = self.find_parent(root.condition, target)
            if cond_parent:
                return cond_parent
            
            if root.then_branch is target:
                return root
            then_parent = self.find_parent(root.then_branch, target)
            if then_parent:
                return then_parent
            
            if root.else_branch is target:
                return root
            else_parent = self.find_parent(root.else_branch, target)
            return else_parent
        
        return None
    
    def evolve(self, target_function: Callable[[List[float]], List[float]],
               test_points: List[List[float]],
               verbose: bool = True) -> Tuple[Optional[Individual], dict]:
        """Запустить эволюцию.
        
        target_function - функция, принимающая вектор и возвращающая вектор выходов размерности output_dim
        test_points - список входных векторов (каждый вектор размерности input_dim)
        """
        
        try:
            # ПРЕДВЫЧИСЛЕНИЕ ЦЕЛЕВЫХ ЗНАЧЕНИЙ
            # Один раз вычисляем целевую функцию для всех тестовых точек
            # и сохраняем результаты в массив для последующего использования
            target_values = []
            for point in test_points:
                result = target_function(point)
                target_values.append(result)
            
            self.initialize_population()
            
            # Оценить начальную популяцию, используя предвычисленные значения
            for individual in self.population:
                self.evaluate_fitness(individual, target_values, test_points)
            
            best_ever = min(self.population, key=lambda ind: ind.fitness)
            
            if verbose:
                print(f"Начальная лучшая приспособленность: {best_ever.fitness:.6f}")
            
            start_time = time.time()
            generation = 0
            no_improvement_count = 0
            max_no_improvement = 150  # Максимальное количество поколений без улучшения
            
            while generation < self.max_generations:
                # Сортировать по приспособленности
                self.population.sort(key=lambda ind: ind.fitness)
                
                best_current = self.population[0]
                avg_fitness = sum(ind.fitness for ind in self.population) / len(self.population)
                
                self.best_fitness_history.append(best_current.fitness)
                self.avg_fitness_history.append(avg_fitness)
                
                if best_current.fitness < best_ever.fitness - 1e-10:
                    best_ever = best_current.copy()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                # Проверка критерия остановки - целевая приспособленность
                if best_current.fitness < self.target_fitness:
                    elapsed = time.time() - start_time
                    if verbose:
                        # Очистить строку прогресса и вывести финальное сообщение
                        print("\r" + " " * 100 + "\r", end="")
                        print(f"\n🎯 ДОСТИГНУТА ЦЕЛЕВАЯ ПРИСПОСОБЛЕННОСТЬ на поколении {generation}!")
                        print(f"   Лучшая ошибка: {best_current.fitness:.6f}")
                        print(f"   Время: {elapsed:.1f}с")
                    break
                
                # Обновление прогресса в реальном времени (каждое поколение)
                if verbose:
                    elapsed = time.time() - start_time
                    progress_line = f"\rПоколение {generation}/{self.max_generations} | Лучшая ошибка: {best_current.fitness:.6f} | Средняя: {avg_fitness:.6f} | Время: {elapsed:.1f}с"
                    print(progress_line, end="", flush=True)
                    
                    # Раз в 50 поколений выводить сводку новой строкой
                    if generation > 0 and generation % 50 == 0:
                        print(f"\n📊 Сводка на поколении {generation}: лучшая={best_current.fitness:.6f}, средняя={avg_fitness:.6f}, время={elapsed:.1f}с")
                
                # Проверка на застой
                if no_improvement_count >= max_no_improvement:
                    if verbose:
                        print(f"\n⚠️ Застой обнаружен на поколении {generation}, перезапуск...")
                    # Частичный перезапуск популяции
                    self._partial_restart()
                    no_improvement_count = 0
                
                # Адаптивная оптимизация констант для лучших особей (раз в 5 поколений)
                if generation > 0 and generation % 5 == 0:
                    # Оптимизируем топ-20% популяции, используя предвычисленные значения
                    elite_count = max(1, self.population_size // 5)
                    for i in range(elite_count):
                        self.optimize_constants(self.population[i], target_values, test_points, method='hybrid')
                    # Переоценить фитнес после оптимизации
                    for i in range(elite_count):
                        self.evaluate_fitness(self.population[i], target_values, test_points)
                
                # Создание нового поколения
                new_population = []
                
                # Элитизм
                for i in range(self.elitism_count):
                    new_population.append(self.population[i].copy())
                
                # Заполнение остальной части популяции
                while len(new_population) < self.population_size:
                    parent1 = self.select_tournament()
                    parent2 = self.select_tournament()
                    
                    child1, child2 = self.crossover(parent1, parent2)
                    
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)
                    
                    new_population.append(child1)
                    if len(new_population) < self.population_size:
                        new_population.append(child2)
                
                # Оценить новую популяцию, используя предвычисленные значения
                for individual in new_population:
                    self.evaluate_fitness(individual, target_values, test_points)
                
                self.population = new_population
                generation += 1
            
            elapsed_time = time.time() - start_time
            
            # Если цикл завершился без достижения целевой приспособленности
            if best_ever.fitness >= self.target_fitness and verbose:
                print("\r" + " " * 100 + "\r", end="")
                print(f"\n🏁 Эволюция завершена после {generation} поколений")
                print(f"   Лучшая ошибка: {best_ever.fitness:.6f}")
                print(f"   Время: {elapsed_time:.1f}с")
            
            # Создать словарь с информацией об эволюции
            evolution_info = {
                'generations': generation,
                'elapsed_time': elapsed_time,
                'final_fitness': best_ever.fitness,
                'population_size': self.population_size,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'max_depth': self.generator.max_depth,
                'interrupted': False,
                'converged': best_ever.fitness < self.target_fitness
            }
            
            return best_ever, evolution_info
            
        except KeyboardInterrupt:
            # Обработка прерывания Ctrl+C во время эволюции
            elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
            
            if verbose:
                print("\n\n⚠️  ЭВОЛЮЦИЯ ПРЕРВАНА ПОЛЬЗОВАТЕЛЕМ (Ctrl+C)")
                if 'best_ever' in locals():
                    print(f"\n📊 ТЕКУЩИЙ ЛУЧШИЙ РЕЗУЛЬТАТ НА МОМЕНТ ПРЕРЫВАНИЯ:")
                    print(f"   Поколение: {generation}")
                    print(f"   Лучшая ошибка: {best_ever.fitness:.6f}")
                    print(f"   Выражение: {best_ever.expression.to_string()}")
                    print(f"   Прошло времени: {elapsed_time:.1f}с")
                else:
                    print("Эволюция прервана до получения первых результатов.")
            
            evolution_info = {
                'generations': generation if 'generation' in locals() else 0,
                'elapsed_time': elapsed_time,
                'final_fitness': best_ever.fitness if 'best_ever' in locals() else float('inf'),
                'population_size': self.population_size,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'max_depth': self.generator.max_depth,
                'interrupted': True,
                'converged': False
            }
            
            return best_ever if 'best_ever' in locals() else None, evolution_info
        
        except Exception as e:
            # Обработка любых других исключений
            if verbose:
                print(f"\n⚠️  ПРОИЗОШЛА ОШИБКА ВО ВРЕМЯ ЭВОЛЮЦИИ: {e}")
                print("Эволюция прервана. Возврат в меню.")
            
            evolution_info = {
                'generations': generation if 'generation' in locals() else 0,
                'elapsed_time': 0,
                'final_fitness': float('inf'),
                'population_size': self.population_size,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'max_depth': self.generator.max_depth,
                'interrupted': True,
                'converged': False,
                'error': str(e)
            }
            
            return None, evolution_info
    
    def _partial_restart(self):
        """Частичный перезапуск популяции для выхода из локального оптимума"""
        # Сохранить лучших особей
        elite_count = self.elitism_count * 2
        self.population.sort(key=lambda ind: ind.fitness)
        elite = [self.population[i].copy() for i in range(elite_count)]
        
        # Заменить худших новыми случайными особями
        new_count = self.population_size - elite_count
        for i in range(elite_count, self.population_size):
            self.generator.reset_counter()
            trees = []
            for _ in range(self.output_dim):
                expr = self.generator.generate_random()
                trees.append(expr)
            self.population[i] = Individual(trees if self.output_dim > 1 else trees[0])
        
        # Восстановить элиту
        for i in range(elite_count):
            self.population[i] = elite[i]


# ==================== Тестирование ====================

class ApproximatorTester:
    """Класс для тестирования аппроксиматора"""
    
    def __init__(self, input_dim: int = 1, output_dim: int = 1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Функции теперь принимают вектор [x] вместо скаляра x и возвращают список выходов
        self.test_functions = {
            "Линейная": lambda x: [2 * x[0] + 3],
            "Квадратичная": lambda x: [x[0] ** 2],
            "Синус": lambda x: [math.sin(x[0])],
            "Косинус": lambda x: [math.cos(x[0])],
            "Экспонента": lambda x: [math.exp(x[0] / 5)],
            "Комбинированная": lambda x: [math.sin(x[0]) + 0.5 * x[0] ** 2],
            "Сложная": lambda x: [math.sin(x[0]) * math.cos(x[0]) + 0.1 * x[0] ** 3],
        }
        
        # Многомерные тестовые функции (для output_dim > 1)
        self.multidim_test_functions = {
            "2D_линейная_квадрат": lambda x: [2 * x[0] + 3, x[0] ** 2],
            "2D_синус_косинус": lambda x: [math.sin(x[0]), math.cos(x[0])],
            "3D_вектор": lambda x: [x[0], x[0] ** 2, math.sin(x[0])],
        }
        
        # Многомерные функции с многомерным входом (input_dim > 1)
        self.multidim_input_functions = {
            "2D_вход_линейная": lambda x: [x[0] + x[1]],  # x[0] + x[1]
            "2D_вход_скалярное_произведение": lambda x: [x[0] * x[1]],  # x[0] * x[1]
            "3D_вход_сумма": lambda x: [x[0] + x[1] + x[2]],  # сумма всех компонент
            "4D_вход_комбо": lambda x: [x[0] * x[1] + math.sin(x[2]) + x[3]],  # комбинированная
        }
        
        # Словесные описания и формулы встроенных функций
        self.function_descriptions = {
            "Линейная": "f(x) = [2x + 3]",
            "Квадратичная": "f(x) = [x²]",
            "Синус": "f(x) = [sin(x)]",
            "Косинус": "f(x) = [cos(x)]",
            "Экспонента": "f(x) = [e^(x/5)]",
            "Комбинированная": "f(x) = [sin(x) + 0.5x²]",
            "Сложная": "f(x) = [sin(x)·cos(x) + 0.1x³]",
        }
        
        self.multidim_function_descriptions = {
            "2D_линейная_квадрат": "f(x) = [2x + 3, x²]",
            "2D_синус_косинус": "f(x) = [sin(x), cos(x)]",
            "3D_вектор": "f(x) = [x, x², sin(x)]",
        }
        
        self.multidim_input_descriptions = {
            "2D_вход_линейная": "f(x₁,x₂) = [x₁ + x₂]",
            "2D_вход_скалярное_произведение": "f(x₁,x₂) = [x₁·x₂]",
            "3D_вход_сумма": "f(x₁,x₂,x₃) = [x₁ + x₂ + x₃]",
            "4D_вход_комбо": "f(x₁,x₂,x₃,x₄) = [x₁·x₂ + sin(x₃) + x₄]",
        }
        
        # Базовые диапазоны для генерации тестовых точек
        self.base_ranges = {
            "default": (-3.0, 3.0, 0.1),  # (min, max, step)
            "wide": (-5.0, 5.0, 0.2),
            "narrow": (-1.0, 1.0, 0.05),
        }
        
        # Генерируем test_ranges для одномерного входа (обратная совместимость)
        self.test_ranges = {
            "default": [[i * 0.1] for i in range(-30, 31)],
            "wide": [[i * 0.2] for i in range(-50, 51)],
            "narrow": [[i * 0.05] for i in range(-20, 21)],
        }
    
    def generate_test_points(self, input_dim: int = 1, range_name: str = "default",
                            custom_ranges: List[Tuple[float, float]] = None) -> List[List[float]]:
        """Генерировать тестовые точки для заданной размерности входа.
        
        Args:
            input_dim: Размерность входного вектора
            range_name: Название предустановленного диапазона ("default", "wide", "narrow")
            custom_ranges: Список кортежей (min, max) для каждого измерения.
                          Если None, используется range_name.
        
        Returns:
            Список входных векторов размерности input_dim
        
        Для многомерного входа генерируются точки как декартово произведение диапазонов
        по каждому измерению (с ограничением общего количества точек).
        """
        if custom_ranges is not None:
            # Использовать пользовательские диапазоны
            ranges = custom_ranges
        else:
            # Использовать предустановленные диапазоны
            if range_name not in self.base_ranges:
                range_name = "default"
            min_val, max_val, step = self.base_ranges[range_name]
            ranges = [(min_val, max_val)] * input_dim
        
        # Генерируем точки
        if input_dim == 1:
            # Одномерный случай - просто последовательность
            min_val, max_val = ranges[0]
            points = []
            current = min_val
            while current <= max_val:
                points.append([current])
                current += step
            return points
        else:
            # Многомерный случай - декартово произведение с ограничением количества точек
            # Чтобы не генерировать слишком много точек, используем разумный лимит
            max_points = 500  # Максимальное количество точек
            
            # Сначала генерируем 1D точки для каждого измерения
            dims_points = []
            for (min_val, max_val) in ranges:
                dim_points = []
                # Адаптируем шаг чтобы уложиться в лимит
                estimated_per_dim = int(max_points ** (1.0 / input_dim))
                estimated_per_dim = max(5, min(estimated_per_dim, 20))  # от 5 до 20 точек на измерение
                
                step = (max_val - min_val) / (estimated_per_dim - 1) if estimated_per_dim > 1 else 0
                current = min_val
                while current <= max_val + step/2:  # +step/2 для включения последней точки
                    dim_points.append(current)
                    current += step
                dims_points.append(dim_points)
            
            # Генерируем декартово произведение
            from itertools import product
            points = []
            for combo in product(*dims_points):
                points.append(list(combo))
                if len(points) >= max_points:
                    break
            
            return points
    
    def run_test(self, func_name: str, func: Callable[[List[float]], List[float]],
                test_points: List[List[float]], ga: GeneticAlgorithm) -> Tuple[Individual, dict]:
        """Запустить тест для одной функции"""
        
        print(f"\n{'='*60}")
        print(f"Тестирование функции: {func_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        best_individual, evolution_info = ga.evolve(func, test_points, verbose=True)
        
        end_time = time.time()
        
        # Оценка качества на отдельных тестовых точках
        validation_points = [[i * 0.15] for i in range(-25, 26)]
        validation_errors = []
        
        for x_vec in validation_points:
            predicted = [best_individual.expressions[i].evaluate(x_vec) for i in range(ga.output_dim)]
            actual = func(x_vec)
            error = sum(abs(predicted[i] - actual[i]) for i in range(ga.output_dim))
            validation_errors.append(error)
        
        avg_validation_error = sum(validation_errors) / len(validation_errors)
        max_validation_error = max(validation_errors)
        
        # Формирование выражений для каждой компоненты
        expressions = {}
        for i in range(ga.output_dim):
            expressions[f'component_{i}'] = best_individual.expressions[i].to_string()
        
        results = {
            "function_name": func_name,
            "output_dim": ga.output_dim,
            "training_fitness": best_individual.fitness,
            "validation_avg_error": avg_validation_error,
            "validation_max_error": max_validation_error,
            "evolution_time": evolution_info['elapsed_time'],
            "expressions": expressions,
            "generations": evolution_info['generations'],
            "evolution_info": evolution_info
        }
        
        print(f"\nРезультаты:")
        if ga.output_dim == 1:
            print(f"  Выражение: {results['expressions']['component_0']}")
        else:
            print(f"  Найдено выражений: {ga.output_dim}")
            for i in range(ga.output_dim):
                print(f"  Компонента {i}: {results['expressions'][f'component_{i}']}")
        
        print(f"  Ошибка на обучении (MSE): {results['training_fitness']:.6f}")
        print(f"  Средняя ошибка на валидации: {results['validation_avg_error']:.6f}")
        print(f"  Максимальная ошибка на валидации: {results['validation_max_error']:.6f}")
        print(f"  Время эволюции: {results['evolution_time']:.2f} сек")
        print(f"  Поколений: {results['generations']}")
        
        # Пример предсказаний
        print(f"\nПримеры предсказаний:")
        sample_points = [[-2.0], [-1.0], [0.0], [1.0], [2.0]]
        for x_vec in sample_points:
            predicted = [best_individual.expressions[i].evaluate(x_vec) for i in range(ga.output_dim)]
            actual = func(x_vec)
            errors = [abs(predicted[i] - actual[i]) for i in range(ga.output_dim)]
            
            if ga.output_dim == 1:
                print(f"  x={x_vec[0]:4.1f}: предсказано={predicted[0]:8.4f}, фактически={actual[0]:8.4f}, ошибка={errors[0]:.4f}")
            else:
                pred_str = ", ".join([f"{p:.4f}" for p in predicted])
                act_str = ", ".join([f"{a:.4f}" for a in actual])
                err_str = ", ".join([f"{e:.4f}" for e in errors])
                print(f"  x={x_vec[0]:4.1f}: предсказано=[{pred_str}], фактически=[{act_str}], ошибки=[{err_str}]")
        
        return best_individual, results
    
    def run_all_tests(self, output_dim: int = 1, input_dim: int = None) -> List[dict]:
        """Запустить все тесты
        
        Args:
            output_dim: Размерность выхода
            input_dim: Размерность входа (если None, используется self.input_dim)
        """
        
        if input_dim is None:
            input_dim = self.input_dim
        
        print("\n" + "="*60)
        print("ЭВОЛЮЦИОНИРУЮЩИЙ УНИВЕРСАЛЬНЫЙ АППРОКСИМАТОР")
        print(f"Размерность входа: {input_dim}, Размерность выхода: {output_dim}")
        print("="*60)
        print("\nЗапуск полного тестирования...\n")
        
        all_results = []
        
        # Выбрать набор функций в зависимости от размерности
        if input_dim > 1:
            # Для многомерного входа использовать функции с многомерным входом
            test_functions = self.multidim_input_functions
            func_descriptions = self.multidim_input_descriptions
        elif output_dim == 1:
            # Одномерный вход и выход
            test_functions = self.test_functions
            func_descriptions = self.function_descriptions
        else:
            # Многомерный выход, но одномерный вход
            test_functions = self.multidim_test_functions
            func_descriptions = self.multidim_function_descriptions
        
        # Сгенерировать тестовые точки для заданной размерности входа
        test_points = self.generate_test_points(input_dim=input_dim, range_name="default")
        
        for func_name, func in test_functions.items():
            # Создать новый ГА для каждого теста с оптимизированными параметрами
            # auto_scale_params=True автоматически масштабирует параметры под размерность
            ga = GeneticAlgorithm(
                population_size=150,
                mutation_rate=0.4,
                crossover_rate=0.8,
                elitism_count=10,
                max_generations=600,
                target_fitness=0.05,
                max_depth=8,
                input_dim=input_dim,
                output_dim=output_dim,
                auto_scale_params=True  # Автоматически масштабировать параметры
            )
            
            _, results = self.run_test(func_name, func, test_points, ga)
            all_results.append(results)
        
        # Итоговая статистика
        print(f"\n{'='*60}")
        print("ИТОГОВАЯ СТАТИСТИКА")
        print(f"{'='*60}")
        
        avg_fitness = sum(r['training_fitness'] for r in all_results) / len(all_results)
        avg_val_error = sum(r['validation_avg_error'] for r in all_results) / len(all_results)
        total_time = sum(r['evolution_time'] for r in all_results)
        
        print(f"\nВсего тестов: {len(all_results)}")
        print(f"Средняя ошибка обучения (MSE): {avg_fitness:.6f}")
        print(f"Средняя ошибка валидации: {avg_val_error:.6f}")
        print(f"Общее время тестирования: {total_time:.2f} сек")
        
        print(f"\nДетальные результаты по функциям:")
        for r in all_results:
            status = "✓" if r['validation_avg_error'] < 0.5 else "⚠"
            print(f"  {status} {r['function_name']:25s}: ошибка={r['validation_avg_error']:.4f}, MSE={r['training_fitness']:.6f}")
        
        return all_results


# ==================== Парсер данных ====================

class DataParser:
    """Парсер файлов с данными в формате: вход1 вход2 ... | выход1 выход2 ..."""
    
    @staticmethod
    def parse_file(filename: str) -> ParsedData:
        """
        Загрузить данные из файла.
        
        Формат файла:
        - Каждая строка: входное_число_1 ... входное_число_N | выходное_число_1 ... выходное_число_M
        - Комментарии начинаются с #
        - Пустые строки пропускаются
        - Разделитель | обязателен
        
        Возвращает ParsedData с результатами парсинга или ошибкой.
        """
        pairs: List[DataPair] = []
        input_dim: int = -1
        output_dim: int = -1
        total_lines: int = 0
        parsed_lines: int = 0
        skipped_lines: int = 0
        skip_reasons: Dict[str, int] = {}
        
        # Для вычисления диапазонов
        input_mins: List[float] = []
        input_maxs: List[float] = []
        output_mins: List[float] = []
        output_maxs: List[float] = []
        
        error_message: Optional[str] = None
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    total_lines += 1
                    line = line.strip()
                    
                    # Пропустить пустые строки
                    if not line:
                        skipped_lines += 1
                        skip_reasons['empty'] = skip_reasons.get('empty', 0) + 1
                        continue
                    
                    # Пропустить комментарии
                    if line.startswith('#'):
                        skipped_lines += 1
                        skip_reasons['comment'] = skip_reasons.get('comment', 0) + 1
                        continue
                    
                    # Проверить наличие разделителя |
                    if '|' not in line:
                        skipped_lines += 1
                        skip_reasons['no_separator'] = skip_reasons.get('no_separator', 0) + 1
                        continue
                    
                    # Разделить по |
                    parts = line.split('|')
                    
                    # Должно быть ровно две части (вход и выход)
                    if len(parts) != 2:
                        skipped_lines += 1
                        skip_reasons['multiple_separators'] = skip_reasons.get('multiple_separators', 0) + 1
                        continue
                    
                    input_part, output_part = parts
                    
                    # Распарсить входные значения
                    try:
                        inputs = [float(x.strip()) for x in input_part.split()]
                    except ValueError:
                        skipped_lines += 1
                        skip_reasons['invalid_input'] = skip_reasons.get('invalid_input', 0) + 1
                        continue
                    
                    # Распарсить выходные значения
                    try:
                        outputs = [float(y.strip()) for y in output_part.split()]
                    except ValueError:
                        skipped_lines += 1
                        skip_reasons['invalid_output'] = skip_reasons.get('invalid_output', 0) + 1
                        continue
                    
                    # Проверка: должны быть хотя бы одно входное и одно выходное значение
                    if len(inputs) == 0 or len(outputs) == 0:
                        skipped_lines += 1
                        skip_reasons['empty_vector'] = skip_reasons.get('empty_vector', 0) + 1
                        continue
                    
                    # Определить размерность по первой корректной строке
                    if input_dim == -1:
                        input_dim = len(inputs)
                        output_dim = len(outputs)
                        # Инициализировать диапазоны
                        input_mins = list(inputs)
                        input_maxs = list(inputs)
                        output_mins = list(outputs)
                        output_maxs = list(outputs)
                    else:
                        # Проверить соответствие размерности
                        if len(inputs) != input_dim:
                            skipped_lines += 1
                            skip_reasons['input_dim_mismatch'] = skip_reasons.get('input_dim_mismatch', 0) + 1
                            continue
                        
                        if len(outputs) != output_dim:
                            skipped_lines += 1
                            skip_reasons['output_dim_mismatch'] = skip_reasons.get('output_dim_mismatch', 0) + 1
                            continue
                        
                        # Обновить диапазоны
                        for i, val in enumerate(inputs):
                            input_mins[i] = min(input_mins[i], val)
                            input_maxs[i] = max(input_maxs[i], val)
                        
                        for i, val in enumerate(outputs):
                            output_mins[i] = min(output_mins[i], val)
                            output_maxs[i] = max(output_maxs[i], val)
                    
                    # Добавить пару
                    pairs.append(DataPair(inputs=inputs, outputs=outputs))
                    parsed_lines += 1
            
            # Проверка: файл пустой или недостаточно данных
            if parsed_lines == 0:
                if total_lines == 0:
                    error_message = "Файл пустой"
                elif total_lines == skipped_lines:
                    reasons_str = ', '.join([f"{k}: {v}" for k, v in skip_reasons.items()])
                    error_message = f"Все строки пропущены ({reasons_str})"
                else:
                    error_message = "Нет корректных строк с данными"
                
                return ParsedData(
                    pairs=pairs,
                    input_dim=input_dim if input_dim >= 0 else 0,
                    output_dim=output_dim if output_dim >= 0 else 0,
                    total_lines=total_lines,
                    parsed_lines=parsed_lines,
                    skipped_lines=skipped_lines,
                    skip_reasons=skip_reasons,
                    input_ranges=[],
                    output_ranges=[],
                    error_message=error_message
                )
            
            # Проверка: минимум 3 пары
            if parsed_lines < 3:
                error_message = f"Недостаточно данных: найдено только {parsed_lines} пар (требуется минимум 3)"
                return ParsedData(
                    pairs=pairs,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    total_lines=total_lines,
                    parsed_lines=parsed_lines,
                    skipped_lines=skipped_lines,
                    skip_reasons=skip_reasons,
                    input_ranges=list(zip(input_mins, input_maxs)),
                    output_ranges=list(zip(output_mins, output_maxs)),
                    error_message=error_message
                )
            
            # Сформировать диапазоны
            input_ranges = list(zip(input_mins, input_maxs))
            output_ranges = list(zip(output_mins, output_maxs))
            
            return ParsedData(
                pairs=pairs,
                input_dim=input_dim,
                output_dim=output_dim,
                total_lines=total_lines,
                parsed_lines=parsed_lines,
                skipped_lines=skipped_lines,
                skip_reasons=skip_reasons,
                input_ranges=input_ranges,
                output_ranges=output_ranges,
                error_message=None
            )
            
        except FileNotFoundError:
            return ParsedData(
                pairs=[],
                input_dim=0,
                output_dim=0,
                total_lines=0,
                parsed_lines=0,
                skipped_lines=0,
                skip_reasons={},
                input_ranges=[],
                output_ranges=[],
                error_message=f"Файл '{filename}' не найден"
            )
        except PermissionError:
            return ParsedData(
                pairs=[],
                input_dim=0,
                output_dim=0,
                total_lines=0,
                parsed_lines=0,
                skipped_lines=0,
                skip_reasons={},
                input_ranges=[],
                output_ranges=[],
                error_message=f"Нет доступа к файлу '{filename}'"
            )
        except UnicodeDecodeError:
            return ParsedData(
                pairs=[],
                input_dim=0,
                output_dim=0,
                total_lines=0,
                parsed_lines=0,
                skipped_lines=0,
                skip_reasons={},
                input_ranges=[],
                output_ranges=[],
                error_message=f"Неверная кодировка файла '{filename}' (ожидалась UTF-8)"
            )
        except Exception as e:
            return ParsedData(
                pairs=[],
                input_dim=0,
                output_dim=0,
                total_lines=0,
                parsed_lines=0,
                skipped_lines=0,
                skip_reasons={},
                input_ranges=[],
                output_ranges=[],
                error_message=f"Ошибка при чтении файла: {e}"
            )
    
    @staticmethod
    def print_summary(data: ParsedData) -> None:
        """Вывести сводку по загруженным данным"""
        print("\n" + "=" * 60)
        print("СВОДКА ПО ЗАГРУЖЕННЫМ ДАННЫМ")
        print("=" * 60)
        
        if data.error_message:
            print(f"❌ ОШИБКА: {data.error_message}")
            print("=" * 60)
            return
        
        print(f"✅ Успешно загружено пар: {data.parsed_lines}")
        print(f"Размерность входа (N): {data.input_dim}")
        print(f"Размерность выхода (M): {data.output_dim}")
        
        # Диапазоны входов
        if data.input_ranges:
            print("\nДиапазоны входных переменных:")
            for i, (min_val, max_val) in enumerate(data.input_ranges):
                print(f"  X[{i}]: [{min_val:.6f}, {max_val:.6f}]")
        
        # Диапазоны выходов
        if data.output_ranges:
            print("\nДиапазоны выходных переменных:")
            for i, (min_val, max_val) in enumerate(data.output_ranges):
                print(f"  Y[{i}]: [{min_val:.6f}, {max_val:.6f}]")
        
        # Пропущенные строки
        if data.skipped_lines > 0:
            print(f"\n⚠️  Пропущено строк: {data.skipped_lines}")
            if data.skip_reasons:
                print("Причины пропуска:")
                reason_map = {
                    'empty': 'пустые строки',
                    'comment': 'комментарии (#)',
                    'no_separator': 'отсутствует разделитель |',
                    'multiple_separators': 'несколько разделителей |',
                    'invalid_input': 'некорректные входные значения',
                    'invalid_output': 'некорректные выходные значения',
                    'empty_vector': 'пустой вектор входа или выхода',
                    'input_dim_mismatch': 'несоответствие размерности входа',
                    'output_dim_mismatch': 'несоответствие размерности выхода'
                }
                for reason, count in data.skip_reasons.items():
                    reason_text = reason_map.get(reason, reason)
                    print(f"  - {reason_text}: {count}")
        
        print("=" * 60)


# ==================== Основная программа ====================

class EvolutionaryApproximatorApp:
    """Основное приложение с интерактивным меню"""
    
    def __init__(self):
        self.ga: Optional[GeneticAlgorithm] = None
        self.best_individual: Optional[Individual] = None
        self.last_results: Optional[dict] = None
        self.tester = ApproximatorTester()
        
        # Параметры генетического алгоритма по умолчанию
        self.default_ga_params = {
            'population_size': 150,
            'mutation_rate': 0.4,
            'crossover_rate': 0.8,
            'elitism_count': 10,
            'max_generations': 600,
            'target_fitness': 0.05,
            'max_depth': 8
        }
    
    def ask_evolution_params(self) -> None:
        """Спросить пользователя о параметрах эволюции
        
        Если пользователь нажимает Enter или вводит Y — используются параметры по умолчанию.
        Если вводит N — последовательно спрашиваем 3 ключевых параметра.
        """
        print("\n" + "=" * 50)
        print("НАСТРОЙКИ ЭВОЛЮЦИИ")
        print("=" * 50)
        print(f"Параметры по умолчанию:")
        print(f"  Размер популяции: {self.default_ga_params['population_size']}")
        print(f"  Максимум поколений: {self.default_ga_params['max_generations']}")
        print(f"  Максимальная глубина выражения: {self.default_ga_params['max_depth']}")
        print()
        
        try:
            choice = input("Использовать настройки по умолчанию? (Y/n): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n")
            return
        
        # Если пользователь нажал Enter или ввел Y — используем настройки по умолчанию
        if choice == '' or choice == 'y' or choice == 'yes' or choice == 'д' or choice == 'да':
            print("\nИспользуются настройки по умолчанию.")
            return
        
        # Если пользователь ввел N — спрашиваем параметры
        if choice == 'n' or choice == 'no' or choice == 'н' or choice == 'нет':
            print("\nВведите параметры (нажмите Enter для использования значения по умолчанию):")
            
            # Размер популяции
            try:
                pop_input = input(f"Размер популяции [{self.default_ga_params['population_size']}]: ").strip()
                if pop_input:
                    pop_size = int(pop_input)
                    if pop_size > 0:
                        self.default_ga_params['population_size'] = pop_size
            except ValueError:
                print(f"  Неверное значение, используется {self.default_ga_params['population_size']}")
            except (KeyboardInterrupt, EOFError):
                print("\n")
            
            # Максимум поколений
            try:
                gen_input = input(f"Максимум поколений [{self.default_ga_params['max_generations']}]: ").strip()
                if gen_input:
                    max_gen = int(gen_input)
                    if max_gen > 0:
                        self.default_ga_params['max_generations'] = max_gen
            except ValueError:
                print(f"  Неверное значение, используется {self.default_ga_params['max_generations']}")
            except (KeyboardInterrupt, EOFError):
                print("\n")
            
            # Максимальная глубина выражения
            try:
                depth_input = input(f"Максимальная глубина выражения [{self.default_ga_params['max_depth']}]: ").strip()
                if depth_input:
                    max_d = int(depth_input)
                    if max_d > 0 and max_d <= 15:
                        self.default_ga_params['max_depth'] = max_d
                    elif max_d > 15:
                        print(f"  Глубина ограничена 15, используется 15")
                        self.default_ga_params['max_depth'] = 15
            except ValueError:
                print(f"  Неверное значение, используется {self.default_ga_params['max_depth']}")
            except (KeyboardInterrupt, EOFError):
                print("\n")
            
            print(f"\nПараметры установлены:")
            print(f"  Размер популяции: {self.default_ga_params['population_size']}")
            print(f"  Максимум поколений: {self.default_ga_params['max_generations']}")
            print(f"  Максимальная глубина: {self.default_ga_params['max_depth']}")
        else:
            print("\nИспользуются настройки по умолчанию.")
    
    def display_menu(self) -> None:
        """Отобразить главное меню"""
        print("\n" + "=" * 50)
        print("       ЭВОЛЮЦИОНИРУЮЩИЙ АППРОКСИМАТОР")
        print("=" * 50)
        print("1. Загрузить данные из файла и запустить эволюцию")
        print("2. Запустить эволюцию на встроенной функции")
        print("3. Посмотреть результаты последнего запуска")
        print("4. Сохранить найденное выражение в файл")
        print("5. Выход")
        print("=" * 50)
    
    def get_user_choice(self) -> Optional[int]:
        """Получить выбор пользователя"""
        try:
            choice = input("\nВыберите пункт меню (1-5): ").strip()
            return int(choice)
        except ValueError:
            return None
    
    def load_data_from_file(self) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        """Загрузить данные из файла с использованием нового парсера.
        
        Новый формат: вход1 вход2 ... | выход1 выход2 ...
        
        Возвращает кортеж (x_values, y_values) или (None, None) при ошибке.
        Для многомерных случаев возвращает только первые измерения.
        """
        filename = input("Введите имя файла с данными: ").strip()
        
        if not filename:
            print("Ошибка: имя файла не может быть пустым.")
            return None, None
        
        # Использовать новый парсер
        parsed_data = DataParser.parse_file(filename)
        
        # Вывести сводку
        DataParser.print_summary(parsed_data)
        
        # Проверить на ошибки
        if parsed_data.error_message:
            return None, None
        
        # Для обратной совместимости вернуть только первые измерения
        # (старый код ожидает одномерные x и y)
        x_values = [pair.inputs[0] for pair in parsed_data.pairs]
        y_values = [pair.outputs[0] for pair in parsed_data.pairs]
        
        return x_values, y_values
    
    def create_target_function_from_data(self, x_values: List[float], y_values: List[float]) -> Callable[[float], float]:
        """Создать целевую функцию из данных для аппроксимации"""
        # Используем интерполяцию для создания функции
        def target_func(x: float) -> float:
            # Найти ближайшую точку
            min_dist = float('inf')
            closest_y = y_values[0] if y_values else 0.0
            
            for xv, yv in zip(x_values, y_values):
                dist = abs(x - xv)
                if dist < min_dist:
                    min_dist = dist
                    closest_y = yv
            
            # Простая линейная интерполяция между двумя ближайшими точками
            if len(x_values) >= 2:
                sorted_points = sorted(zip(x_values, y_values), key=lambda p: p[0])
                
                # Найти соседние точки
                for i in range(len(sorted_points) - 1):
                    x1, y1 = sorted_points[i]
                    x2, y2 = sorted_points[i + 1]
                    
                    if x1 <= x <= x2:
                        # Линейная интерполяция
                        if abs(x2 - x1) > 1e-10:
                            t = (x - x1) / (x2 - x1)
                            return y1 + t * (y2 - y1)
                
                # Если x вне диапазона, использовать ближайшую границу
                if x < sorted_points[0][0]:
                    return sorted_points[0][1]
                else:
                    return sorted_points[-1][1]
            
            return closest_y
        
        return target_func
    
    def run_evolution_on_data(self, x_values: List[float], y_values: List[float]) -> None:
        """Запустить эволюцию на загруженных данных"""
        print("\n" + "=" * 50)
        print("ЗАПУСК ЭВОЛЮЦИИ НА ЗАГРУЖЕННЫХ ДАННЫХ")
        print("=" * 50)
        
        # Спросить пользователя о параметрах эволюции
        self.ask_evolution_params()
        
        try:
            # Создать целевую функцию из данных
            target_func = self.create_target_function_from_data(x_values, y_values)
            
            # Использовать x_values как тестовые точки
            test_points = x_values
            
            # Инициализировать ГА
            self.ga = GeneticAlgorithm(**self.default_ga_params)
            
            print(f"\nПараметры эволюции:")
            print(f"  Размер популяции: {self.default_ga_params['population_size']}")
            print(f"  Максимум поколений: {self.default_ga_params['max_generations']}")
            print(f"  Точек данных: {len(test_points)}")
            print()
            
            # Запустить эволюцию
            self.best_individual, self.evolution_info = self.ga.evolve(target_func, test_points, verbose=True)
            
            # Проверка на прерывание
            if self.evolution_info.get('interrupted', False):
                print("\nЭволюция была прервана. Возврат в меню.")
                return
            
            # Сохранить результаты с полной информацией
            if self.best_individual:
                # Вычислить ошибки на валидационных точках (отдельный набор)
                validation_points = [x_values[0] + (x_values[-1] - x_values[0]) * i / 10 for i in range(11)]
                validation_errors = []
                for x in validation_points:
                    predicted = self.best_individual.expression.evaluate(x)
                    actual = target_func(x)
                    error = abs(predicted - actual)
                    validation_errors.append(error)
                
                avg_val_error = sum(validation_errors) / len(validation_errors) if validation_errors else 0
                max_val_error = max(validation_errors) if validation_errors else 0
                
                self.last_results = {
                    'type': 'data',
                    'expression': self.best_individual.expression.to_string(),
                    'fitness': self.best_individual.fitness,
                    'validation_avg_error': avg_val_error,
                    'validation_max_error': max_val_error,
                    'data_points': len(x_values),
                    'generations': self.evolution_info['generations'],
                    'elapsed_time': self.evolution_info['elapsed_time'],
                    'evolution_info': self.evolution_info,
                    'x_values': x_values,
                    'y_values': y_values,
                    'target_func': target_func
                }
                
                print("\n" + "-" * 50)
                print("РЕЗУЛЬТАТ:")
                print(f"  Выражение: {self.last_results['expression']}")
                print(f"  Ошибка на обучении (MSE): {self.last_results['fitness']:.6f}")
                print(f"  Средняя ошибка на валидации: {avg_val_error:.6f}")
                print(f"  Поколений: {self.last_results['generations']}")
                print(f"  Время эволюции: {self.evolution_info['elapsed_time']:.2f} сек")
                
                # Примеры предсказаний на данных
                print("\nПримеры предсказаний на данных:")
                sample_indices = [0, len(x_values)//2, len(x_values)-1]
                for i in sample_indices:
                    if i < len(x_values):
                        x = x_values[i]
                        actual = y_values[i]
                        predicted = self.best_individual.expression.evaluate(x)
                        error = abs(predicted - actual)
                        print(f"  x={x:.4f}: предсказано={predicted:.4f}, фактически={actual:.4f}, ошибка={error:.4f}")
            else:
                print("\n⚠️  Эволюция не смогла найти подходящее выражение.")
        
        except Exception as e:
            print(f"\n⚠️  ПРОИЗОШЛА ОШИБКА ПРИ ЗАПУСКЕ ЭВОЛЮЦИИ НА ДАННЫХ: {e}")
            print("Возврат в меню...")
    
    def select_builtin_function(self) -> Optional[Tuple[str, Callable[[float], float]]]:
        """Предложить пользователю выбрать встроенную функцию"""
        print("\nДоступные встроенные функции:")
        print("-" * 50)
        
        func_list = list(self.tester.test_functions.items())
        desc_list = list(self.tester.function_descriptions.items())
        
        for i, ((name, _), (_, formula)) in enumerate(zip(func_list, desc_list), 1):
            print(f"  {i}. {name}: {formula}")
        
        print("-" * 50)
        print("  0. Вернуться в главное меню")
        print("-" * 50)
        
        try:
            choice = input(f"Выберите функцию (0-{len(func_list)}): ").strip()
            choice_idx = int(choice)
            
            # Пункт "Вернуться в главное меню"
            if choice_idx == 0:
                return None
            
            # Корректировка индекса (пользователь видит 1-based, но у нас есть пункт 0)
            choice_idx -= 1
            
            if 0 <= choice_idx < len(func_list):
                return func_list[choice_idx]
            else:
                print("Ошибка: неверный номер функции.")
                return None
        except ValueError:
            print("Ошибка: введите число.")
            return None
    
    def run_evolution_on_builtin(self) -> None:
        """Запустить эволюцию на встроенной функции"""
        func_result = self.select_builtin_function()
        
        # Если пользователь выбрал "Вернуться в главное меню" или произошла ошибка
        if func_result is None:
            print("\nВозврат в главное меню...")
            return
        
        func_name, target_func = func_result
        
        print("\n" + "=" * 50)
        print(f"ЗАПУСК ЭВОЛЮЦИИ НА ФУНКЦИИ: {func_name}")
        print("=" * 50)
        
        # Спросить пользователя о параметрах эволюции
        self.ask_evolution_params()
        
        try:
            # Инициализировать ГА
            self.ga = GeneticAlgorithm(**self.default_ga_params)
            
            # Использовать тестовые точки по умолчанию
            test_points = self.tester.test_ranges["default"]
            
            print(f"\nПараметры эволюции:")
            print(f"  Функция: {func_name}")
            print(f"  Размер популяции: {self.default_ga_params['population_size']}")
            print(f"  Максимум поколений: {self.default_ga_params['max_generations']}")
            print(f"  Тестовых точек: {len(test_points)}")
            print()
            
            # Запустить эволюцию
            self.best_individual, self.evolution_info = self.ga.evolve(target_func, test_points, verbose=True)
            
            # Проверка на прерывание
            if self.evolution_info.get('interrupted', False):
                print("\nЭволюция была прервана. Возврат в меню.")
                return
            
            # Сохранить результаты с полной информацией
            if self.best_individual:
                # Оценка на валидационных точках
                validation_points = [i * 0.15 for i in range(-25, 26)]
                validation_errors = []
                
                for x in validation_points:
                    predicted = self.best_individual.expression.evaluate(x)
                    actual = target_func(x)
                    error = abs(predicted - actual)
                    validation_errors.append(error)
                
                avg_val_error = sum(validation_errors) / len(validation_errors)
                max_val_error = max(validation_errors)
                
                self.last_results = {
                    'type': 'builtin',
                    'function_name': func_name,
                    'expression': self.best_individual.expression.to_string(),
                    'fitness': self.best_individual.fitness,
                    'validation_avg_error': avg_val_error,
                    'validation_max_error': max_val_error,
                    'generations': self.evolution_info['generations'],
                    'elapsed_time': self.evolution_info['elapsed_time'],
                    'evolution_info': self.evolution_info,
                    'target_func': target_func
                }
                
                print("\n" + "-" * 50)
                print("РЕЗУЛЬТАТ:")
                print(f"  Выражение: {self.last_results['expression']}")
                print(f"  Ошибка на обучении (MSE): {self.last_results['fitness']:.6f}")
                print(f"  Средняя ошибка на валидации: {avg_val_error:.6f}")
                print(f"  Поколений: {self.last_results['generations']}")
                print(f"  Время эволюции: {self.evolution_info['elapsed_time']:.2f} сек")
                
                # Примеры предсказаний
                print("\nПримеры предсказаний:")
                sample_points = [-2.0, -1.0, 0.0, 1.0, 2.0]
                for x in sample_points:
                    predicted = self.best_individual.expression.evaluate(x)
                    actual = target_func(x)
                    error = abs(predicted - actual)
                    print(f"  x={x:4.1f}: предсказано={predicted:8.4f}, фактически={actual:8.4f}, ошибка={error:.4f}")
            else:
                print("\n⚠️  Эволюция не смогла найти подходящее выражение.")
        
        except Exception as e:
            print(f"\n⚠️  ПРОИЗОШЛА ОШИБКА ПРИ ЗАПУСКЕ ЭВОЛЮЦИИ НА ВСТРОЕННОЙ ФУНКЦИИ: {e}")
            print("Возврат в меню...")
    
    def view_last_results(self) -> None:
        """Показать результаты последнего запуска"""
        print("\n" + "=" * 50)
        print("РЕЗУЛЬТАТЫ ПОСЛЕДНЕГО ЗАПУСКА")
        print("=" * 50)
        
        if self.last_results is None:
            print("Нет результатов последнего запуска.")
            print("Сначала выполните запуск эволюции (пункт 1 или 2).")
            return
        
        if self.last_results.get('type') == 'data':
            print("Тип: Аппроксимация данных из файла")
            print(f"Количество точек данных: {self.last_results.get('data_points', 'N/A')}")
        elif self.last_results.get('type') == 'builtin':
            print("Тип: Встроенная функция")
            print(f"Функция: {self.last_results.get('function_name', 'N/A')}")
        
        print(f"\nНайденное выражение:")
        print(f"  {self.last_results.get('expression', 'N/A')}")
        
        print(f"\nМетрики качества:")
        print(f"  Ошибка на обучении (MSE): {self.last_results.get('fitness', 'N/A')}")
        
        if 'validation_avg_error' in self.last_results:
            print(f"  Средняя ошибка на валидации: {self.last_results['validation_avg_error']:.6f}")
            print(f"  Максимальная ошибка на валидации: {self.last_results['validation_max_error']:.6f}")
        
        print(f"\nСтатистика эволюции:")
        print(f"  Поколений: {self.last_results.get('generations', 'N/A')}")
        print(f"  Время эволюции: {self.last_results.get('elapsed_time', 'N/A'):.2f} сек" if self.last_results.get('elapsed_time') else "  Время эволюции: N/A")
        
        # Таблица сравнения для 7 точек
        print(f"\nТаблица сравнения предсказаний:")
        print("-" * 70)
        print(f"{'X':>10} | {'Фактическое':>14} | {'Предсказанное':>14} | {'Ошибка':>10}")
        print("-" * 70)
        
        target_func = self.last_results.get('target_func')
        x_values = self.last_results.get('x_values')
        y_values = self.last_results.get('y_values')
        
        if target_func and self.last_results.get('type') == 'builtin':
            # Для встроенных функций используем равномерные точки
            sample_points = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
            for x in sample_points:
                actual = target_func(x)
                predicted = self.best_individual.expression.evaluate(x)
                error = abs(predicted - actual)
                print(f"{x:>10.4f} | {actual:>14.6f} | {predicted:>14.6f} | {error:>10.6f}")
        elif x_values and y_values and self.last_results.get('type') == 'data':
            # Для данных выбираем 7 равномерных точек
            n = len(x_values)
            if n > 0:
                step = max(1, n // 7)
                indices = [min(i * step, n - 1) for i in range(7)]
                for i in indices:
                    x = x_values[i]
                    actual = y_values[i]
                    predicted = self.best_individual.expression.evaluate(x)
                    error = abs(predicted - actual)
                    print(f"{x:>10.4f} | {actual:>14.6f} | {predicted:>14.6f} | {error:>10.6f}")
        else:
            # Резервный вариант
            sample_points = [-2.0, -1.0, 0.0, 1.0, 2.0]
            for x in sample_points:
                predicted = self.best_individual.expression.evaluate(x)
                actual = target_func(x) if target_func else 0
                error = abs(predicted - actual)
                print(f"{x:>10.4f} | {actual:>14.6f} | {predicted:>14.6f} | {error:>10.6f}")
        
        print("-" * 70)
    
    def save_expression_to_file(self) -> None:
        """Сохранить найденное выражение в файл"""
        print("\n" + "=" * 50)
        print("СОХРАНЕНИЕ ВЫРАЖЕНИЯ В ФАЙЛ")
        print("=" * 50)
        
        if self.best_individual is None:
            print("Нет выражения для сохранения.")
            print("Сначала выполните запуск эволюции (пункт 1 или 2).")
            return
        
        # Запрос имени файла с дефолтным значением
        filename = input("Введите имя файла для сохранения [result.txt]: ").strip()
        
        if not filename:
            filename = "result.txt"
        
        expression = self.best_individual.expression.to_string()
        fitness = self.best_individual.fitness
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("РЕЗУЛЬТАТ ЭВОЛЮЦИОННОГО АППРОКСИМАТОРА\n")
                f.write("=" * 60 + "\n\n")
                
                # Найденное выражение
                f.write("НАЙДЕННОЕ МАТЕМАТИЧЕСКОЕ ВЫРАЖЕНИЕ:\n")
                f.write(f"  {expression}\n\n")
                
                # Параметры эволюции
                f.write("ПАРАМЕТРЫ ЭВОЛЮЦИИ:\n")
                if self.ga:
                    f.write(f"  Размер популяции: {self.ga.population_size}\n")
                    f.write(f"  Вероятность мутации: {self.ga.mutation_rate}\n")
                    f.write(f"  Вероятность кроссовера: {self.ga.crossover_rate}\n")
                    f.write(f"  Максимум поколений: {self.ga.max_generations}\n")
                    f.write(f"  Максимальная глубина дерева: {self.ga.generator.max_depth}\n")
                f.write("\n")
                
                # Итоговые ошибки
                f.write("ИТОГОВЫЕ ОШИБКИ:\n")
                f.write(f"  Ошибка на обучающих данных (MSE): {fitness:.6f}\n")
                if 'validation_avg_error' in self.last_results:
                    f.write(f"  Средняя ошибка на валидации: {self.last_results['validation_avg_error']:.6f}\n")
                    f.write(f"  Максимальная ошибка на валидации: {self.last_results['validation_max_error']:.6f}\n")
                f.write("\n")
                
                # Статистика эволюции
                f.write("СТАТИСТИКА ЭВОЛЮЦИИ:\n")
                f.write(f"  Поколений: {self.last_results.get('generations', 'N/A')}\n")
                if self.last_results.get('elapsed_time'):
                    f.write(f"  Время эволюции: {self.last_results['elapsed_time']:.2f} сек\n")
                f.write("\n")
                
                # Тип задачи
                if self.last_results.get('type') == 'data':
                    f.write("ТИП ЗАДАЧИ: Аппроксимация данных из файла\n")
                    f.write(f"  Количество точек данных: {self.last_results.get('data_points', 'N/A')}\n")
                elif self.last_results.get('type') == 'builtin':
                    f.write("ТИП ЗАДАЧИ: Встроенная функция\n")
                    f.write(f"  Функция: {self.last_results.get('function_name', 'N/A')}\n")
                f.write("\n")
                
                # Таблица предсказаний
                f.write("ТАБЛИЦА СРАВНЕНИЯ ПРЕДСКАЗАНИЙ:\n")
                f.write("-" * 70 + "\n")
                f.write(f"{'X':>12} | {'Фактическое':>14} | {'Предсказанное':>14} | {'Ошибка':>12}\n")
                f.write("-" * 70 + "\n")
                
                target_func = self.last_results.get('target_func')
                x_values = self.last_results.get('x_values')
                y_values = self.last_results.get('y_values')
                
                if target_func and self.last_results.get('type') == 'builtin':
                    # Для встроенных функций используем равномерные точки
                    sample_points = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
                    for x in sample_points:
                        actual = target_func(x)
                        predicted = self.best_individual.expression.evaluate(x)
                        error = abs(predicted - actual)
                        f.write(f"{x:>12.4f} | {actual:>14.6f} | {predicted:>14.6f} | {error:>12.6f}\n")
                elif x_values and y_values and self.last_results.get('type') == 'data':
                    # Для данных выбираем 7 равномерных точек
                    n = len(x_values)
                    if n > 0:
                        step = max(1, n // 7)
                        indices = [min(i * step, n - 1) for i in range(7)]
                        for i in indices:
                            x = x_values[i]
                            actual = y_values[i]
                            predicted = self.best_individual.expression.evaluate(x)
                            error = abs(predicted - actual)
                            f.write(f"{x:>12.4f} | {actual:>14.6f} | {predicted:>14.6f} | {error:>12.6f}\n")
                
                f.write("-" * 70 + "\n")
                f.write("\n" + "=" * 60 + "\n")
                f.write("Конец отчета\n")
                f.write("=" * 60 + "\n")
            
            print(f"Результат успешно сохранен в файл '{filename}'.")
            
        except PermissionError:
            print(f"Ошибка: нет прав на запись в файл '{filename}'.")
        except Exception as e:
            print(f"Ошибка при сохранении файла: {e}")
    
    def run(self) -> None:
        """Запустить главный цикл приложения"""
        print("\n" + "=" * 50)
        print("   ДОБРО ПОЖАЛОВАТЬ В ЭВОЛЮЦИОННЫЙ АППРОКСИМАТОР!")
        print("=" * 50)
        
        try:
            while True:
                try:
                    self.display_menu()
                    choice = self.get_user_choice()
                    
                    if choice == 1:
                        x_values, y_values = self.load_data_from_file()
                        if x_values and y_values:
                            self.run_evolution_on_data(x_values, y_values)
                    
                    elif choice == 2:
                        self.run_evolution_on_builtin()
                    
                    elif choice == 3:
                        self.view_last_results()
                    
                    elif choice == 4:
                        self.save_expression_to_file()
                    
                    elif choice == 5:
                        print("\n" + "=" * 50)
                        print("СПАСИБО ЗА ИСПОЛЬЗОВАНИЕ ПРОГРАММЫ!")
                        print("До свидания!")
                        print("=" * 50 + "\n")
                        break
                    
                    else:
                        print("\nНекорректный ввод. Пожалуйста, выберите пункт от 1 до 5.")
                
                except KeyboardInterrupt:
                    # Обработка Ctrl+C в меню
                    print("\n\n⚠️  Получено прерывание (Ctrl+C)")
                    print("Вы действительно хотите выйти? (y/n): ", end="")
                    try:
                        confirm = input().strip().lower()
                        if confirm in ('y', 'yes', 'д', 'да'):
                            print("\n" + "=" * 50)
                            print("СПАСИБО ЗА ИСПОЛЬЗОВАНИЕ ПРОГРАММЫ!")
                            print("До свидания!")
                            print("=" * 50 + "\n")
                            break
                        else:
                            print("Возврат в меню...")
                    except (KeyboardInterrupt, EOFError):
                        print("\n\nВыход из программы...")
                        print("=" * 50)
                        print("До свидания!")
                        print("=" * 50 + "\n")
                        break
        except Exception as e:
            print(f"\n⚠️  ПРОИЗОШЛА НЕОЖИДАННАЯ ОШИБКА: {e}")
            print("Программа вынуждена завершить работу. Приносим извинения.")


def main():
    """Основная функция"""
    app = EvolutionaryApproximatorApp()
    app.run()


if __name__ == "__main__":
    main()
