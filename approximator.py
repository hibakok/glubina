#!/usr/bin/env python3
"""
Эволюционный аппроксиматор функций — ядро системы (версия 1)

Система эволюционирует программы (деревья выражений с условиями и циклами),
чтобы аппроксимировать произвольную вычислимую функцию.
"""

# =============================================================================
# CUSTOM TARGETS
# =============================================================================
# Вставьте сюда свою целевую функцию для использования с --target custom
# Пример:
#
# def custom_target(x0, x1=None):
#     """Ваша целевая функция."""
#     return x0 * x0 + x1 * x1 if x1 is not None else x0 * x0
#
# CUSTOM_BOUNDS = [(-5.0, 5.0)]  # границы для x0
# # Для двух переменных: CUSTOM_BOUNDS = [(-5.0, 5.0), (-5.0, 5.0)]
# =============================================================================

import argparse
import copy
import math
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


# =============================================================================
# Узлы дерева выражений
# =============================================================================

@dataclass
class ConstNode:
    """Константа (вещественное число)."""
    value: float
    
    def arity(self) -> int:
        return 0
    
    def to_string(self) -> str:
        return f"{self.value:.6g}"
    
    def to_python(self) -> str:
        return f"{self.value}"
    
    def hash_key(self) -> tuple:
        return ('const', self.value)


@dataclass
class VarNode:
    """Переменная (x0, x1, ..., acc, i)."""
    name: str
    
    def arity(self) -> int:
        return 0
    
    def to_string(self) -> str:
        return self.name
    
    def to_python(self) -> str:
        return self.name
    
    def hash_key(self) -> tuple:
        return ('var', self.name)


@dataclass
class UnaryNode:
    """Унарный узел: sin, cos, tanh, exp, log, sqrt, abs, neg."""
    op: str
    child: Any  # тип узла
    
    def arity(self) -> int:
        return 1
    
    def to_string(self) -> str:
        return f"{self.op}({self.child.to_string()})"
    
    def to_python(self) -> str:
        return f"{self.op}({self.child.to_python()})"
    
    def hash_key(self) -> tuple:
        return ('unary', self.op, self.child.hash_key())


@dataclass
class BinaryNode:
    """Бинарный узел: +, -, *, /, min, max, less."""
    op: str
    left: Any
    right: Any
    
    def arity(self) -> int:
        return 2
    
    def to_string(self) -> str:
        op_map = {
            'add': '+', 'sub': '-', 'mul': '*', 'div': '/',
            'min': 'min', 'max': 'max', 'less': 'less'
        }
        if self.op in ('add', 'sub', 'mul', 'div'):
            return f"({self.left.to_string()} {op_map[self.op]} {self.right.to_string()})"
        else:
            return f"{self.op}({self.left.to_string()}, {self.right.to_string()})"
    
    def to_python(self) -> str:
        if self.op == 'add':
            return f"({self.left.to_python()} + {self.right.to_python()})"
        elif self.op == 'sub':
            return f"({self.left.to_python()} - {self.right.to_python()})"
        elif self.op == 'mul':
            return f"({self.left.to_python()} * {self.right.to_python()})"
        elif self.op == 'div':
            return f"_safe_div({self.left.to_python()}, {self.right.to_python()})"
        elif self.op == 'min':
            return f"min({self.left.to_python()}, {self.right.to_python()})"
        elif self.op == 'max':
            return f"max({self.left.to_python()}, {self.right.to_python()})"
        elif self.op == 'less':
            return f"(1.0 if ({self.left.to_python()}) < ({self.right.to_python()}) else 0.0)"
        return ""
    
    def hash_key(self) -> tuple:
        return ('binary', self.op, self.left.hash_key(), self.right.hash_key())


@dataclass
class TernaryNode:
    """Тернарный узел: if(c, a, b) — возвращает a если c > 0, иначе b."""
    cond: Any
    then_branch: Any
    else_branch: Any
    
    def arity(self) -> int:
        return 3
    
    def to_string(self) -> str:
        return f"if({self.cond.to_string()}, {self.then_branch.to_string()}, {self.else_branch.to_string()})"
    
    def to_python(self) -> str:
        return f"(({self.then_branch.to_python()}) if ({self.cond.to_python()}) > 0 else ({self.else_branch.to_python()}))"
    
    def hash_key(self) -> tuple:
        return ('ternary', self.cond.hash_key(), self.then_branch.hash_key(), self.else_branch.hash_key())


@dataclass
class LoopNode:
    """
    Узел цикла: loop(n, init, body).
    n — количество итераций (приводится к int, зажимается 0..1000).
    init — начальное значение accum.
    body — тело цикла, где доступны acc и i.
    """
    n_expr: Any
    init_expr: Any
    body_expr: Any
    
    def arity(self) -> int:
        return 3
    
    def to_string(self) -> str:
        return f"loop({self.n_expr.to_string()}, {self.init_expr.to_string()}, {self.body_expr.to_string()})"
    
    def to_python(self) -> str:
        return f"_loop({self.n_expr.to_python()}, {self.init_expr.to_python()}, lambda acc, i: {self.body_expr.to_python()})"
    
    def hash_key(self) -> tuple:
        return ('loop', self.n_expr.hash_key(), self.init_expr.hash_key(), self.body_expr.hash_key())


NodeType = Union[ConstNode, VarNode, UnaryNode, BinaryNode, TernaryNode, LoopNode]


# =============================================================================
# Защищённые математические операции
# =============================================================================

def _safe_div(a: float, b: float) -> float:
    """Защищённое деление: при |b| < 1e-10 вернуть 1.0."""
    if abs(b) < 1e-10:
        return 1.0
    return a / b


def _safe_log(x: float) -> float:
    """Защищённый логарифм: log(|x| + 1e-10)."""
    return math.log(abs(x) + 1e-10)


def _safe_sqrt(x: float) -> float:
    """Защищённый корень: sqrt(|x|)."""
    return math.sqrt(abs(x))


def _safe_exp(x: float) -> float:
    """Защищённый экспонент: аргумент зажимается в [-700, 700]."""
    x_clamped = max(-700.0, min(700.0, x))
    return math.exp(x_clamped)


UNARY_OPS = {
    'sin': math.sin,
    'cos': math.cos,
    'tanh': math.tanh,
    'exp': _safe_exp,
    'log': _safe_log,
    'sqrt': _safe_sqrt,
    'abs': abs,
    'neg': lambda x: -x,
}

BINARY_OPS = {
    'add': lambda a, b: a + b,
    'sub': lambda a, b: a - b,
    'mul': lambda a, b: a * b,
    'div': _safe_div,
    'min': min,
    'max': max,
    'less': lambda a, b: 1.0 if a < b else 0.0,
}


# =============================================================================
# Исполнитель с бюджетом узлов
# =============================================================================

class ExecutionBudgetExceeded(Exception):
    """Исключение при превышении бюджета узлов."""
    pass


class InvalidProgram(Exception):
    """Исключение для невалидной программы (NaN, Inf, и т.п.)."""
    pass


class Executor:
    """
    Исполнитель деревьев выражений с бюджетом узлов и защищёнными операциями.
    """
    
    def __init__(self, budget: int = 20000):
        self.budget = budget
        self.nodes_executed = 0
        self.variables: Dict[str, float] = {}
    
    def reset(self, variables: Dict[str, float]) -> None:
        """Сбросить счётчик и установить переменные окружения."""
        self.nodes_executed = 0
        self.variables = variables
    
    def _count_node(self) -> None:
        """Учесть выполнение узла, проверить бюджет."""
        self.nodes_executed += 1
        if self.nodes_executed > self.budget:
            raise ExecutionBudgetExceeded(f"Превышен бюджет узлов: {self.budget}")
    
    def evaluate(self, node: NodeType) -> float:
        """Вычислить значение узла рекурсивно."""
        self._count_node()
        
        if isinstance(node, ConstNode):
            return node.value
        
        elif isinstance(node, VarNode):
            if node.name not in self.variables:
                raise InvalidProgram(f"Переменная {node.name} не определена")
            return self.variables[node.name]
        
        elif isinstance(node, UnaryNode):
            child_val = self.evaluate(node.child)
            if math.isnan(child_val) or math.isinf(child_val):
                raise InvalidProgram("NaN/Inf в унарном узле")
            result = UNARY_OPS[node.op](child_val)
            if math.isnan(result) or math.isinf(result):
                raise InvalidProgram("NaN/Inf после унарной операции")
            return result
        
        elif isinstance(node, BinaryNode):
            left_val = self.evaluate(node.left)
            right_val = self.evaluate(node.right)
            if math.isnan(left_val) or math.isinf(left_val) or math.isnan(right_val) or math.isinf(right_val):
                raise InvalidProgram("NaN/Inf в бинарном узле")
            result = BINARY_OPS[node.op](left_val, right_val)
            if math.isnan(result) or math.isinf(result):
                raise InvalidProgram("NaN/Inf после бинарной операции")
            return result
        
        elif isinstance(node, TernaryNode):
            cond_val = self.evaluate(node.cond)
            if math.isnan(cond_val) or math.isinf(cond_val):
                raise InvalidProgram("NaN/Inf в условии if")
            if cond_val > 0:
                return self.evaluate(node.then_branch)
            else:
                return self.evaluate(node.else_branch)
        
        elif isinstance(node, LoopNode):
            # Считаем узел loop
            self._count_node()
            
            n_val = self.evaluate(node.n_expr)
            if math.isnan(n_val) or math.isinf(n_val):
                raise InvalidProgram("NaN/Inf в счётчике цикла")
            n_int = int(n_val)
            n_int = max(0, min(1000, n_int))
            
            acc_val = self.evaluate(node.init_expr)
            if math.isnan(acc_val) or math.isinf(acc_val):
                raise InvalidProgram("NaN/Inf в начальном значении цикла")
            
            for i in range(n_int):
                # Считаем узел тела цикла на каждой итерации
                self._count_node()
                old_acc = self.variables.get('acc')
                old_i = self.variables.get('i')
                self.variables['acc'] = acc_val
                self.variables['i'] = float(i)
                try:
                    acc_val = self.evaluate(node.body_expr)
                finally:
                    if old_acc is not None:
                        self.variables['acc'] = old_acc
                    elif 'acc' in self.variables:
                        del self.variables['acc']
                    if old_i is not None:
                        self.variables['i'] = old_i
                    elif 'i' in self.variables:
                        del self.variables['i']
                
                if math.isnan(acc_val) or math.isinf(acc_val):
                    raise InvalidProgram("NaN/Inf в теле цикла")
            
            return acc_val
        
        else:
            raise InvalidProgram(f"Неизвестный тип узла: {type(node)}")


# =============================================================================
# Генерация случайных деревьев
# =============================================================================

class TreeGenerator:
    """Генератор случайных деревьев выражений."""
    
    def __init__(self, rng: random.Random, max_depth: int = 12, max_nodes: int = 200):
        self.rng = rng
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.unary_ops = ['sin', 'cos', 'tanh', 'exp', 'log', 'sqrt', 'abs', 'neg']
        self.binary_ops = ['add', 'sub', 'mul', 'div', 'min', 'max', 'less']
    
    def generate(self, depth: int, var_names: List[str], method: str = 'half') -> NodeType:
        """
        Сгенерировать случайное дерево.
        method: 'grow' — только терминалы на максимальной глубине,
                'full' — только нетерминалы до максимальной глубины,
                'half' — случайно между grow и full.
        """
        if depth >= self.max_depth:
            return self._random_terminal(var_names)
        
        if method == 'half':
            method = self.rng.choice(['grow', 'full'])
        
        if method == 'full' and depth < self.max_depth - 1:
            return self._random_nonterminal(depth, var_names)
        elif method == 'grow':
            if depth == 0:
                return self._random_terminal(var_names)
            choice = self.rng.random()
            if choice < 0.3:
                return self._random_terminal(var_names)
            else:
                return self._random_nonterminal(depth, var_names)
        else:
            return self._random_nonterminal(depth, var_names)
    
    def _random_terminal(self, var_names: List[str]) -> NodeType:
        """Случайный терминал: константа или переменная."""
        if self.rng.random() < 0.5 and var_names:
            name = self.rng.choice(var_names)
            return VarNode(name=name)
        else:
            value = self.rng.uniform(-5.0, 5.0)
            return ConstNode(value=value)
    
    def _random_nonterminal(self, depth: int, var_names: List[str]) -> NodeType:
        """Случайный нетерминальный узел."""
        choice = self.rng.random()
        
        if choice < 0.15:
            # Унарный оператор
            op = self.rng.choice(self.unary_ops)
            child = self.generate(depth + 1, var_names, 'half')
            return UnaryNode(op=op, child=child)
        
        elif choice < 0.85:
            # Бинарный оператор
            op = self.rng.choice(self.binary_ops)
            left = self.generate(depth + 1, var_names, 'half')
            right = self.generate(depth + 1, var_names, 'half')
            return BinaryNode(op=op, left=left, right=right)
        
        elif choice < 0.95:
            # Тернарный if
            cond = self.generate(depth + 1, var_names, 'half')
            then_b = self.generate(depth + 1, var_names, 'half')
            else_b = self.generate(depth + 1, var_names, 'half')
            return TernaryNode(cond=cond, then_branch=then_b, else_branch=else_b)
        
        else:
            # Цикл loop
            n_expr = self.generate(depth + 1, var_names, 'half')
            init_expr = self.generate(depth + 1, var_names, 'half')
            # В теле цикла добавляем acc и i к переменным
            body_vars = var_names + ['acc', 'i']
            body_expr = self.generate(depth + 1, body_vars, 'half')
            return LoopNode(n_expr=n_expr, init_expr=init_expr, body_expr=body_expr)


# =============================================================================
# Подсчёт узлов и глубины
# =============================================================================

def count_nodes(node: NodeType) -> int:
    """Подсчитать количество узлов в дереве."""
    if isinstance(node, (ConstNode, VarNode)):
        return 1
    elif isinstance(node, UnaryNode):
        return 1 + count_nodes(node.child)
    elif isinstance(node, BinaryNode):
        return 1 + count_nodes(node.left) + count_nodes(node.right)
    elif isinstance(node, TernaryNode):
        return 1 + count_nodes(node.cond) + count_nodes(node.then_branch) + count_nodes(node.else_branch)
    elif isinstance(node, LoopNode):
        return 1 + count_nodes(node.n_expr) + count_nodes(node.init_expr) + count_nodes(node.body_expr)
    return 0


def get_depth(node: NodeType) -> int:
    """Получить глубину дерева."""
    if isinstance(node, (ConstNode, VarNode)):
        return 1
    elif isinstance(node, UnaryNode):
        return 1 + get_depth(node.child)
    elif isinstance(node, BinaryNode):
        return 1 + max(get_depth(node.left), get_depth(node.right))
    elif isinstance(node, TernaryNode):
        return 1 + max(get_depth(node.cond), get_depth(node.then_branch), get_depth(node.else_branch))
    elif isinstance(node, LoopNode):
        return 1 + max(get_depth(node.n_expr), get_depth(node.init_expr), get_depth(node.body_expr))
    return 0


# =============================================================================
# Кэш оценок
# =============================================================================

class EvaluationCache:
    """Кэш оценок программ на наборе данных."""
    
    def __init__(self, max_size: int = 100000):
        self.cache: Dict[tuple, List[float]] = {}
        self.max_size = max_size
    
    def get(self, hash_key: tuple) -> Optional[List[float]]:
        """Получить значение из кэша."""
        return self.cache.get(hash_key)
    
    def put(self, hash_key: tuple, values: List[float]) -> None:
        """Положить значение в кэш."""
        if len(self.cache) >= self.max_size:
            # Простая очистка половины кэша при переполнении
            keys_to_remove = list(self.cache.keys())[:self.max_size // 2]
            for k in keys_to_remove:
                del self.cache[k]
        self.cache[hash_key] = values
    
    def clear(self) -> None:
        """Очистить кэш."""
        self.cache.clear()


# =============================================================================
# Эволюционный движок
# =============================================================================

class Individual:
    """Особь популяции."""
    
    def __init__(self, tree: NodeType):
        self.tree = tree
        self.fitness: float = float('inf')
        self.raw_mse: float = float('inf')
        self.valid: bool = False
        self.cached_outputs: Optional[List[float]] = None
    
    def copy(self) -> 'Individual':
        """Глубокая копия особи."""
        new_ind = Individual(copy.deepcopy(self.tree))
        new_ind.fitness = self.fitness
        new_ind.raw_mse = self.raw_mse
        new_ind.valid = self.valid
        if self.cached_outputs is not None:
            new_ind.cached_outputs = list(self.cached_outputs)  # Копия списка значений
        return new_ind


class EvolutionEngine:
    """
    Эволюционный движок для аппроксимации функций.
    """
    
    def __init__(
        self,
        target_func: Callable,
        bounds: List[Tuple[float, float]],
        rng: random.Random,
        pop_size: int = 300,
        generations: int = 100,
        max_depth: int = 12,
        max_nodes: int = 200,
        parsimony: float = 1e-4,
        tournament_size: int = 3,
        elite_size: int = 2,
        crossover_prob: float = 0.85,
        mutation_prob: float = 0.15,
        train_points: int = 30,
        val_points: int = 100,
        budget: int = 20000,
    ):
        self.target_func = target_func
        self.bounds = bounds
        self.rng = rng
        self.pop_size = pop_size
        self.generations = generations
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.parsimony = parsimony
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.train_points = train_points
        self.val_points = val_points
        self.budget = budget
        
        self.executor = Executor(budget=budget)
        self.generator = TreeGenerator(rng, max_depth, max_nodes)
        self.cache = EvaluationCache(100000)
        
        self.train_data: List[Tuple[List[float], float]] = []
        self.val_data: List[Tuple[List[float], float]] = []
        self.var_names: List[str] = []
        self.target_variance: float = 1.0
        
        self.best_individual: Optional[Individual] = None
        self.best_fitness_history: List[float] = []
    
    def generate_data(self) -> None:
        """Сгенерировать обучающую и контрольную выборки."""
        n_vars = len(self.bounds)
        self.var_names = [f"x{i}" for i in range(n_vars)]
        
        def sample_point() -> Tuple[List[float], float]:
            point = [self.rng.uniform(b[0], b[1]) for b in self.bounds]
            target_val = self.target_func(*point)
            return point, target_val
        
        self.train_data = [sample_point() for _ in range(self.train_points)]
        self.val_data = [sample_point() for _ in range(self.val_points)]
        
        target_vals = [t[1] for t in self.train_data]
        mean_val = sum(target_vals) / len(target_vals)
        self.target_variance = sum((v - mean_val) ** 2 for v in target_vals) / len(target_vals)
    
    def _evaluate_on_data(self, tree: NodeType, data: List[Tuple[List[float], float]]) -> Tuple[List[float], bool]:
        """
        Вычислить outputs программы на наборе данных.
        Возвращает (outputs, valid).
        """
        hash_key = tree.hash_key()
        cached = self.cache.get(hash_key)
        if cached is not None:
            return cached, True
        
        outputs = []
        try:
            for point, _ in data:
                env = dict(zip(self.var_names, point))
                self.executor.reset(env)
                val = self.executor.evaluate(tree)
                outputs.append(val)
        except (ExecutionBudgetExceeded, InvalidProgram, ZeroDivisionError, ValueError, OverflowError):
            return [], False
        
        for v in outputs:
            if math.isnan(v) or math.isinf(v):
                return [], False
        
        self.cache.put(hash_key, outputs)
        return outputs, True
    
    def compute_fitness(self, individual: Individual) -> float:
        """Вычислить приспособленность особи."""
        outputs, valid = self._evaluate_on_data(individual.tree, self.train_data)
        
        if not valid:
            individual.valid = False
            individual.fitness = 1e9
            individual.raw_mse = 1e9
            return 1e9
        
        individual.valid = True
        individual.cached_outputs = outputs
        
        mse = 0.0
        for o, t in zip(outputs, self.train_data):
            diff = o - t[1]
            # Защита от переполнения
            if math.isinf(diff) or math.isnan(diff):
                individual.valid = False
                individual.fitness = 1e9
                individual.raw_mse = 1e9
                return 1e9
            mse += diff * diff
            # Ранний выход при слишком большой ошибке
            if mse > 1e18:
                individual.valid = False
                individual.fitness = 1e9
                individual.raw_mse = 1e9
                return 1e9
        mse /= len(self.train_data)
        individual.raw_mse = mse
        
        normalized_mse = mse / (self.target_variance + 1e-12)
        nodes = count_nodes(individual.tree)
        fitness = normalized_mse + self.parsimony * nodes
        
        if fitness > 1e9:
            fitness = 1e9
        
        individual.fitness = fitness
        return fitness
    
    def tournament_select(self, population: List[Individual]) -> Individual:
        """Турнирная селекция."""
        candidates = self.rng.sample(population, min(self.tournament_size, len(population)))
        return min(candidates, key=lambda ind: ind.fitness)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """
        Кроссовер: обмен случайными поддеревьями.
        Возвращает нового потомка.
        """
        tree1 = copy.deepcopy(parent1.tree)
        tree2 = copy.deepcopy(parent2.tree)
        
        def collect_nodes(node: NodeType, nodes: List[Tuple[Any, str, Any]]) -> None:
            """Собрать все узлы с путями для замены."""
            if isinstance(node, ConstNode):
                nodes.append((node, 'const', None))
            elif isinstance(node, VarNode):
                nodes.append((node, 'var', None))
            elif isinstance(node, UnaryNode):
                nodes.append((node, 'unary', None))
                collect_nodes(node.child, nodes)
            elif isinstance(node, BinaryNode):
                nodes.append((node, 'binary', None))
                collect_nodes(node.left, nodes)
                collect_nodes(node.right, nodes)
            elif isinstance(node, TernaryNode):
                nodes.append((node, 'ternary', None))
                collect_nodes(node.cond, nodes)
                collect_nodes(node.then_branch, nodes)
                collect_nodes(node.else_branch, nodes)
            elif isinstance(node, LoopNode):
                nodes.append((node, 'loop', None))
                collect_nodes(node.n_expr, nodes)
                collect_nodes(node.init_expr, nodes)
                collect_nodes(node.body_expr, nodes)
        
        nodes1: List[Tuple[Any, str, Any]] = []
        nodes2: List[Tuple[Any, str, Any]] = []
        collect_nodes(tree1, nodes1)
        collect_nodes(tree2, nodes2)
        
        if not nodes1 or not nodes2:
            return Individual(copy.deepcopy(parent1.tree))
        
        idx1 = self.rng.randint(0, len(nodes1) - 1)
        idx2 = self.rng.randint(0, len(nodes2) - 1)
        
        node1, type1, _ = nodes1[idx1]
        node2, type2, _ = nodes2[idx2]
        
        def get_parent_and_attr(node_to_find: NodeType, root: NodeType) -> Tuple[Optional[Any], Optional[str]]:
            """Найти родителя и атрибут узла."""
            if root is node_to_find:
                return None, None
            
            if isinstance(root, UnaryNode):
                if root.child is node_to_find:
                    return root, 'child'
                return get_parent_and_attr(node_to_find, root.child)
            elif isinstance(root, BinaryNode):
                if root.left is node_to_find:
                    return root, 'left'
                if root.right is node_to_find:
                    return root, 'right'
                res = get_parent_and_attr(node_to_find, root.left)
                if res[0] is not None:
                    return res
                return get_parent_and_attr(node_to_find, root.right)
            elif isinstance(root, TernaryNode):
                if root.cond is node_to_find:
                    return root, 'cond'
                if root.then_branch is node_to_find:
                    return root, 'then_branch'
                if root.else_branch is node_to_find:
                    return root, 'else_branch'
                for attr in ['cond', 'then_branch', 'else_branch']:
                    res = get_parent_and_attr(node_to_find, getattr(root, attr))
                    if res[0] is not None:
                        return res
                return None, None
            elif isinstance(root, LoopNode):
                if root.n_expr is node_to_find:
                    return root, 'n_expr'
                if root.init_expr is node_to_find:
                    return root, 'init_expr'
                if root.body_expr is node_to_find:
                    return root, 'body_expr'
                for attr in ['n_expr', 'init_expr', 'body_expr']:
                    res = get_parent_and_attr(node_to_find, getattr(root, attr))
                    if res[0] is not None:
                        return res
                return None, None
            return None, None
        
        parent1_node, attr1 = get_parent_and_attr(node1, tree1)
        new_subtree = copy.deepcopy(node2)
        
        if parent1_node is None:
            tree1 = new_subtree
        else:
            setattr(parent1_node, attr1, new_subtree)
        
        if get_depth(tree1) > self.max_depth or count_nodes(tree1) > self.max_nodes:
            return Individual(copy.deepcopy(parent1.tree))
        
        return Individual(tree1)
    
    def mutate(self, parent: Individual) -> Individual:
        """
        Мутация: один из трёх типов с равной вероятностью.
        1. Замена случайного поддерева новым случайным.
        2. Точечная замена узла на операцию той же арности.
        3. Возмущение случайной константы.
        """
        tree = copy.deepcopy(parent.tree)
        
        def collect_all_nodes(node: NodeType, nodes: List[Any], var_context: List[str]) -> None:
            """Собрать все узлы."""
            nodes.append((node, list(var_context)))
            if isinstance(node, UnaryNode):
                collect_all_nodes(node.child, nodes, var_context)
            elif isinstance(node, BinaryNode):
                collect_all_nodes(node.left, nodes, var_context)
                collect_all_nodes(node.right, nodes, var_context)
            elif isinstance(node, TernaryNode):
                collect_all_nodes(node.cond, nodes, var_context)
                collect_all_nodes(node.then_branch, nodes, var_context)
                collect_all_nodes(node.else_branch, nodes, var_context)
            elif isinstance(node, LoopNode):
                collect_all_nodes(node.n_expr, nodes, var_context)
                collect_all_nodes(node.init_expr, nodes, var_context)
                body_vars = var_context + ['acc', 'i']
                collect_all_nodes(node.body_expr, nodes, body_vars)
        
        all_nodes: List[Tuple[Any, List[str]]] = []
        collect_all_nodes(tree, all_nodes, self.var_names)
        
        if not all_nodes:
            return Individual(copy.deepcopy(parent.tree))
        
        mutation_type = self.rng.choice(['replace', 'swap_op', 'perturb'])
        
        if mutation_type == 'replace':
            idx = self.rng.randint(0, len(all_nodes) - 1)
            node_to_replace, var_ctx = all_nodes[idx]
            parent_node, attr = self._find_parent(tree, node_to_replace)
            new_subtree = self.generator.generate(0, var_ctx, 'half')
            
            if parent_node is None:
                tree = new_subtree
            else:
                setattr(parent_node, attr, new_subtree)
        
        elif mutation_type == 'swap_op':
            suitable_nodes = []
            for node, _ in all_nodes:
                if isinstance(node, UnaryNode) or isinstance(node, BinaryNode):
                    suitable_nodes.append((node, 'unary' if isinstance(node, UnaryNode) else 'binary'))
            
            if suitable_nodes:
                idx = self.rng.randint(0, len(suitable_nodes) - 1)
                node, node_type = suitable_nodes[idx]
                
                if node_type == 'unary':
                    new_op = self.rng.choice(self.generator.unary_ops)
                    node.op = new_op
                else:
                    new_op = self.rng.choice(self.generator.binary_ops)
                    node.op = new_op
        
        elif mutation_type == 'perturb':
            const_nodes = [(n, p, a) for n, p, a in 
                          [(nd, *self._find_parent(tree, nd)) for nd, _ in all_nodes]
                          if isinstance(n, ConstNode)]
            if const_nodes:
                idx = self.rng.randint(0, len(const_nodes) - 1)
                const_node, parent_node, attr = const_nodes[idx]
                perturbation = self.rng.gauss(0, 0.5)
                const_node.value += perturbation
        
        if get_depth(tree) > self.max_depth or count_nodes(tree) > self.max_nodes:
            return Individual(copy.deepcopy(parent.tree))
        
        return Individual(tree)
    
    def _find_parent(self, root: NodeType, target: NodeType) -> Tuple[Optional[Any], Optional[str]]:
        """Найти родителя и атрибут целевого узла."""
        if root is target:
            return None, None
        
        if isinstance(root, UnaryNode):
            if root.child is target:
                return root, 'child'
            return self._find_parent(root.child, target)
        elif isinstance(root, BinaryNode):
            if root.left is target:
                return root, 'left'
            if root.right is target:
                return root, 'right'
            res = self._find_parent(root.left, target)
            if res[0] is not None:
                return res
            return self._find_parent(root.right, target)
        elif isinstance(root, TernaryNode):
            if root.cond is target:
                return root, 'cond'
            if root.then_branch is target:
                return root, 'then_branch'
            if root.else_branch is target:
                return root, 'else_branch'
            for attr in ['cond', 'then_branch', 'else_branch']:
                res = self._find_parent(getattr(root, attr), target)
                if res[0] is not None:
                    return res
            return None, None
        elif isinstance(root, LoopNode):
            if root.n_expr is target:
                return root, 'n_expr'
            if root.init_expr is target:
                return root, 'init_expr'
            if root.body_expr is target:
                return root, 'body_expr'
            for attr in ['n_expr', 'init_expr', 'body_expr']:
                res = self._find_parent(getattr(root, attr), target)
                if res[0] is not None:
                    return res
            return None, None
        return None, None
    
    def initialize_population(self) -> List[Individual]:
        """Инициализация популяции методом ramped half-and-half."""
        population = []
        depths = list(range(2, 7))
        
        for i in range(self.pop_size):
            depth = depths[i % len(depths)]
            method = 'full' if i % 2 == 0 else 'grow'
            tree = self.generator.generate(0, self.var_names, method)
            
            attempts = 0
            while (get_depth(tree) > self.max_depth or count_nodes(tree) > self.max_nodes) and attempts < 10:
                tree = self.generator.generate(0, self.var_names, method)
                attempts += 1
            
            population.append(Individual(tree))
        
        return population
    
    def evolve(self, verbose: bool = True) -> Individual:
        """Запустить эволюцию."""
        self.generate_data()
        population = self.initialize_population()
        
        for ind in population:
            self.compute_fitness(ind)
        
        population.sort(key=lambda ind: ind.fitness)
        self.best_individual = population[0].copy()
        best_raw_fitness = self.best_individual.raw_mse / (self.target_variance + 1e-12)
        self.best_fitness_history.append(best_raw_fitness)
        
        start_time = time.time()
        stagnation_counter = 0
        prev_best_fitness = best_raw_fitness
        
        for gen in range(self.generations):
            new_population: List[Individual] = []
            
            elite_count = min(self.elite_size, len(population))
            for i in range(elite_count):
                new_population.append(population[i].copy())
            
            while len(new_population) < self.pop_size:
                if self.rng.random() < self.crossover_prob:
                    p1 = self.tournament_select(population)
                    p2 = self.tournament_select(population)
                    
                    child = None
                    for _ in range(5):
                        child = self.crossover(p1, p2)
                        if get_depth(child.tree) <= self.max_depth and count_nodes(child.tree) <= self.max_nodes:
                            break
                    
                    if child is None or get_depth(child.tree) > self.max_depth or count_nodes(child.tree) > self.max_nodes:
                        child = p1.copy()
                    
                    new_population.append(child)
                else:
                    p = self.tournament_select(population)
                    
                    child = None
                    for _ in range(5):
                        child = self.mutate(p)
                        if get_depth(child.tree) <= self.max_depth and count_nodes(child.tree) <= self.max_nodes:
                            break
                    
                    if child is None or get_depth(child.tree) > self.max_depth or count_nodes(child.tree) > self.max_nodes:
                        child = p.copy()
                    
                    new_population.append(child)
            
            for ind in new_population[elite_count:]:
                self.compute_fitness(ind)
            
            new_population.sort(key=lambda ind: ind.fitness)
            population = new_population
            
            current_best = population[0]
            current_raw_fitness = current_best.raw_mse / (self.target_variance + 1e-12)
            
            if current_raw_fitness < prev_best_fitness:
                self.best_individual = current_best.copy()
                best_raw_fitness = current_raw_fitness
                self.best_fitness_history.append(best_raw_fitness)
                stagnation_counter = 0
                prev_best_fitness = current_raw_fitness
                
                if verbose:
                    elapsed = time.time() - start_time
                    print(f"  [Рекорд] Поколение {gen}: fitness={best_raw_fitness:.6e}, решение: {current_best.tree.to_string()}")
            else:
                stagnation_counter += 1
            
            if stagnation_counter >= 30:
                worst_count = max(1, int(0.2 * self.pop_size))
                for i in range(self.pop_size - worst_count, self.pop_size):
                    depth = self.rng.randint(2, 6)
                    tree = self.generator.generate(0, self.var_names, 'half')
                    population[i] = Individual(tree)
                    self.compute_fitness(population[i])
                stagnation_counter = 0
            
            if verbose:
                elapsed = time.time() - start_time
                avg_fitness = sum(ind.fitness for ind in population) / len(population)
                best_size = count_nodes(population[0].tree)
                print(f"Поколение {gen}: лучший={best_raw_fitness:.6e}, средний={avg_fitness:.6e}, размер={best_size}, время={elapsed:.1f}с")
            
            if best_raw_fitness < 1e-6:
                if verbose:
                    print(f"\n*** УСПЕХ! Достигнут порог точности 1e-6 ***\n")
                break
        
        self.best_individual = population[0].copy()
        return self.best_individual
    
    def evaluate_on_validation(self, individual: Individual) -> float:
        """Оценить нормированную MSE на контрольной выборке."""
        outputs, valid = self._evaluate_on_data(individual.tree, self.val_data)
        
        if not valid:
            return 1e9
        
        mse = sum((o - t[1]) ** 2 for o, t in zip(outputs, self.val_data)) / len(self.val_data)
        return mse / (self.target_variance + 1e-12)


# =============================================================================
# Встроенные целевые функции
# =============================================================================

def quadratic(x0: float) -> float:
    """f(x) = x^2 + 3x + 2"""
    return x0 * x0 + 3 * x0 + 2


def trig_mix(x0: float) -> float:
    """f(x) = sin(x) + 0.5*cos(2x)"""
    return math.sin(x0) + 0.5 * math.cos(2 * x0)


def piecewise(x0: float) -> float:
    """f(x) = x^2 при x < 0, иначе sin(x)"""
    if x0 < 0:
        return x0 * x0
    else:
        return math.sin(x0)


def two_vars(x0: float, x1: float) -> float:
    """f(x0, x1) = x0*x1 + sin(x0) - x1^2"""
    return x0 * x1 + math.sin(x0) - x1 * x1


def logistic30(x0: float) -> float:
    """Логистическое отображение, 30 итераций."""
    t = x0
    for _ in range(30):
        t = 3.7 * t * (1 - t)
    return t


TARGETS = {
    'quadratic': (quadratic, [(-3.0, 3.0)]),
    'trig_mix': (trig_mix, [(-5.0, 5.0)]),
    'piecewise': (piecewise, [(-3.0, 3.0)]),
    'two_vars': (two_vars, [(-3.0, 3.0), (-3.0, 3.0)]),
    'logistic30': (logistic30, [(0.01, 0.99)]),
}


# =============================================================================
# Трансляция в Python-код
# =============================================================================

PYTHON_HELPERS = '''
import math

def _safe_div(a, b):
    """Защищённое деление."""
    if abs(b) < 1e-10:
        return 1.0
    return a / b

def _safe_log(x):
    """Защищённый логарифм."""
    return math.log(abs(x) + 1e-10)

def _safe_sqrt(x):
    """Защищённый корень."""
    return math.sqrt(abs(x))

def _safe_exp(x):
    """Защищённый экспонент."""
    return math.exp(max(-700.0, min(700.0, x)))

def _loop(n, init, body):
    """
    Выполнить цикл n раз.
    n приводится к int и зажимается в 0..1000.
    body — функция(lambda acc, i: ...).
    """
    n = int(n)
    n = max(0, min(1000, n))
    acc = init
    for i in range(n):
        acc = body(acc, i)
    return acc

'''


def generate_python_code(tree: NodeType) -> str:
    """Сгенерировать готовый к использованию Python-код."""
    expr = tree.to_python()
    return PYTHON_HELPERS + f"\ndef solution(x0, x1=None, x2=None, x3=None, x4=None):\n    return {expr}\n"


# =============================================================================
# Self-test
# =============================================================================

def run_selftest() -> bool:
    """Запустить самотестирование."""
    all_passed = True
    
    print("=" * 60)
    print("SELFTEST")
    print("=" * 60)
    
    # Тест 1: Детерминизм
    print("\n1. Детерминизм...", end=" ")
    rng1 = random.Random(42)
    engine1 = EvolutionEngine(
        target_func=quadratic,
        bounds=[(-3.0, 3.0)],
        rng=rng1,
        pop_size=50,
        generations=10,
        train_points=10,
        val_points=20,
    )
    engine1.generate_data()
    pop1 = engine1.initialize_population()
    for ind in pop1:
        engine1.compute_fitness(ind)
    best1_fitness = min(ind.fitness for ind in pop1)
    best1_tree = min(pop1, key=lambda ind: ind.fitness).tree.to_string()
    
    rng2 = random.Random(42)
    engine2 = EvolutionEngine(
        target_func=quadratic,
        bounds=[(-3.0, 3.0)],
        rng=rng2,
        pop_size=50,
        generations=10,
        train_points=10,
        val_points=20,
    )
    engine2.generate_data()
    pop2 = engine2.initialize_population()
    for ind in pop2:
        engine2.compute_fitness(ind)
    best2_fitness = min(ind.fitness for ind in pop2)
    best2_tree = min(pop2, key=lambda ind: ind.fitness).tree.to_string()
    
    if best1_fitness == best2_fitness and best1_tree == best2_tree:
        print("PASS")
    else:
        print("FAIL")
        print(f"  best1_fitness={best1_fitness}, best2_fitness={best2_fitness}")
        print(f"  best1_tree={best1_tree}")
        print(f"  best2_tree={best2_tree}")
        all_passed = False
    
    # Тест 2: Бюджет
    print("2. Бюджет цикла...", end=" ")
    # Цикл с большим телом, чтобы превысить бюджет
    # loop(1e9, 0, acc+i+1) — тело из 5 узлов, 1000 итераций * 5 = 5000 + overhead
    huge_loop = LoopNode(
        n_expr=ConstNode(1e9),
        init_expr=ConstNode(0),
        body_expr=BinaryNode(op='add', left=VarNode(name='acc'), 
                            right=BinaryNode(op='add', left=VarNode(name='i'), right=ConstNode(1)))
    )
    executor = Executor(budget=2000)  # Меньший бюджет
    executor.reset({'acc': 0.0, 'i': 0.0})
    start = time.time()
    try:
        executor.evaluate(huge_loop)
        print("FAIL (не выбросил исключение)")
        all_passed = False
    except ExecutionBudgetExceeded:
        elapsed = time.time() - start
        if elapsed < 1.0:
            print("PASS")
        else:
            print(f"FAIL (слишком долго: {elapsed}s)")
            all_passed = False
    except Exception as e:
        print(f"FAIL (неверное исключение: {e})")
        all_passed = False
    
    # Тест 3: Корректность цикла
    print("3. Корректность цикла...", end=" ")
    loop_5_0_acc_plus_i = LoopNode(
        n_expr=ConstNode(5),
        init_expr=ConstNode(0),
        body_expr=BinaryNode(op='add', left=VarNode(name='acc'), right=VarNode(name='i'))
    )
    executor.reset({'acc': 0.0, 'i': 0.0})
    try:
        result = executor.evaluate(loop_5_0_acc_plus_i)
        expected = 0 + 1 + 2 + 3 + 4  # = 10
        if abs(result - expected) < 1e-9:
            print("PASS")
        else:
            print(f"FAIL (ожидалось {expected}, получено {result})")
            all_passed = False
    except Exception as e:
        print(f"FAIL (исключение: {e})")
        all_passed = False
    
    # Тест 4: Защищённое деление
    print("4. Защищённое деление...", end=" ")
    div_by_zero = BinaryNode(op='div', left=ConstNode(5.0), right=ConstNode(0.0))
    executor.reset({})
    try:
        result = executor.evaluate(div_by_zero)
        if result == 1.0:
            print("PASS")
        else:
            print(f"FAIL (ожидалось 1.0, получено {result})")
            all_passed = False
    except Exception as e:
        print(f"FAIL (исключение: {e})")
        all_passed = False
    
    # Тест 5: Кроссовер и мутация не изменяют родителей
    print("5. Кроссовер/мутация...", end=" ")
    rng = random.Random(123)
    engine = EvolutionEngine(
        target_func=quadratic,
        bounds=[(-3.0, 3.0)],
        rng=rng,
        pop_size=10,
        generations=1,
    )
    engine.generate_data()
    
    tree1 = BinaryNode(op='add', left=ConstNode(1.0), right=ConstNode(2.0))
    tree2 = BinaryNode(op='mul', left=ConstNode(3.0), right=ConstNode(4.0))
    ind1 = Individual(tree1)
    ind2 = Individual(tree2)
    engine.compute_fitness(ind1)
    engine.compute_fitness(ind2)
    
    original_tree1_str = tree1.to_string()
    original_tree2_str = tree2.to_string()
    
    child = engine.crossover(ind1, ind2)
    
    if tree1.to_string() != original_tree1_str or tree2.to_string() != original_tree2_str:
        print("FAIL (родители изменены после кроссовера)")
        all_passed = False
    elif get_depth(child.tree) > engine.max_depth or count_nodes(child.tree) > engine.max_nodes:
        print("FAIL (потомок превышает лимиты)")
        all_passed = False
    else:
        mutant = engine.mutate(ind1)
        if tree1.to_string() != original_tree1_str:
            print("FAIL (родитель изменён после мутации)")
            all_passed = False
        elif get_depth(mutant.tree) > engine.max_depth or count_nodes(mutant.tree) > engine.max_nodes:
            print("FAIL (мутант превышает лимиты)")
            all_passed = False
        else:
            print("PASS")
    
    # Тест 6: Интеграционная проверка на quadratic
    print("6. Интеграция (quadratic)...", end=" ")
    rng = random.Random(0)
    engine = EvolutionEngine(
        target_func=quadratic,
        bounds=[(-3.0, 3.0)],
        rng=rng,
        pop_size=300,
        generations=150,
        max_depth=12,
        parsimony=1e-4,
        train_points=30,
        val_points=100,
    )
    best = engine.evolve(verbose=False)
    final_mse = best.raw_mse / (engine.target_variance + 1e-12)
    
    if final_mse < 1e-3:
        print("PASS")
    else:
        print(f"FAIL (MSE={final_mse:.6e}, ожидалось < 1e-3)")
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
    else:
        print("НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОЙДЕНЫ")
    print("=" * 60)
    
    return all_passed


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Эволюционный аппроксиматор функций')
    parser.add_argument('--target', type=str, default='quadratic',
                       help='Целевая функция: quadratic, trig_mix, piecewise, two_vars, logistic30, custom')
    parser.add_argument('--pop', type=int, default=300, help='Размер популяции')
    parser.add_argument('--gens', type=int, default=100, help='Число поколений')
    parser.add_argument('--seed', type=int, default=0, help='Seed для ГПСЧ')
    parser.add_argument('--depth', type=int, default=12, help='Максимальная глубина дерева')
    parser.add_argument('--max-nodes', type=int, default=200, help='Максимум узлов в дереве')
    parser.add_argument('--parsimony', type=float, default=1e-4, help='Коэффициент простоты')
    parser.add_argument('--train-points', type=int, default=30, help='Точек обучения')
    parser.add_argument('--val-points', type=int, default=100, help='Точек валидации')
    parser.add_argument('--budget', type=int, default=20000, help='Бюджет узлов на вычисление')
    parser.add_argument('--selftest', action='store_true', help='Запустить самотестирование')
    
    args = parser.parse_args()
    
    if args.selftest:
        success = run_selftest()
        sys.exit(0 if success else 1)
    
    if args.target == 'custom':
        try:
            target_func = custom_target
            bounds = CUSTOM_BOUNDS
        except NameError:
            print("ERROR: custom_target и CUSTOM_BOUNDS не определены!")
            print("Откройте файл и добавьте определение функции и границ в блок CUSTOM TARGETS.")
            sys.exit(1)
    elif args.target in TARGETS:
        target_func, bounds = TARGETS[args.target]
    else:
        print(f"ERROR: Неизвестная цель '{args.target}'")
        print(f"Доступные: {list(TARGETS.keys())}, custom")
        sys.exit(1)
    
    print("=" * 60)
    print("ЭВОЛЮЦИОННЫЙ АППРОКСИМАТОР ФУНКЦИЙ")
    print("=" * 60)
    print(f"Цель: {args.target}")
    print(f"Seed: {args.seed}")
    print(f"Популяция: {args.pop}, Поколений: {args.gens}")
    print(f"Макс. глубина: {args.depth}, Макс. узлов: {args.max_nodes}")
    print(f"Точки: обучение={args.train_points}, валидация={args.val_points}")
    print("=" * 60)
    
    rng = random.Random(args.seed)
    
    engine = EvolutionEngine(
        target_func=target_func,
        bounds=bounds,
        rng=rng,
        pop_size=args.pop,
        generations=args.gens,
        max_depth=args.depth,
        max_nodes=args.max_nodes,
        parsimony=args.parsimony,
        train_points=args.train_points,
        val_points=args.val_points,
        budget=args.budget,
    )
    
    start_time = time.time()
    best = engine.evolve(verbose=True)
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТ")
    print("=" * 60)
    
    train_mse = best.raw_mse / (engine.target_variance + 1e-12)
    val_mse = engine.evaluate_on_validation(best)
    
    print(f"Лучшее решение (инфикс):")
    print(f"  {best.tree.to_string()}")
    print()
    print(f"Метрики:")
    print(f"  Нормированная MSE (обучение): {train_mse:.6e}")
    print(f"  Нормированная MSE (валидация): {val_mse:.6e}")
    print(f"  Число узлов: {count_nodes(best.tree)}")
    print(f"  Успех (< 1e-6): {'ДА' if train_mse < 1e-6 else 'НЕТ'}")
    print(f"  Время: {total_time:.1f}с")
    
    if abs(train_mse - val_mse) / (train_mse + 1e-12) > 0.5:
        print(f"  ВНИМАНИЕ: Разрыв между обучением и валидацией — возможное переобучение")
    
    print()
    print("Python-код решения:")
    print("-" * 40)
    print(generate_python_code(best.tree))
    print("-" * 40)
    
    print("\nГотово.")


if __name__ == '__main__':
    main()
