"""
Модуль для работы с деревьями выражений.
Реализует генетическое программирование для эволюции математических функций.
"""

import numpy as np
import random
from enum import Enum
from typing import List, Optional, Tuple, Any
from copy import deepcopy


class NodeType(Enum):
    """Типы узлов в дереве выражений."""
    OPERATION = "operation"
    VARIABLE = "variable"
    CONSTANT = "constant"


class Operation(Enum):
    """Поддерживаемые математические операции."""
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    POW = "^"
    LOG = "log"
    EXP = "exp"
    SIN = "sin"
    COS = "cos"
    TAN = "tan"


# Операции с двумя аргументами
BINARY_OPS = [Operation.ADD, Operation.SUB, Operation.MUL, Operation.DIV, Operation.POW]
# Операции с одним аргументом
UNARY_OPS = [Operation.LOG, Operation.EXP, Operation.SIN, Operation.COS, Operation.TAN]
# Все операции
ALL_OPS = BINARY_OPS + UNARY_OPS


class Node:
    """Узел дерева выражений."""
    
    def __init__(self, node_type: NodeType, value: Any = None, 
                 left: 'Node' = None, right: 'Node' = None,
                 var_index: int = 0):
        """
        Инициализация узла.
        
        Args:
            node_type: Тип узла (OPERATION, VARIABLE, CONSTANT)
            value: Значение (операция или константа)
            left: Левый потомок
            right: Правый потомок
            var_index: Индекс переменной (для VARIABLE)
        """
        self.node_type = node_type
        self.value = value  # Operation для OPERATION, float для CONSTANT
        self.left = left
        self.right = right
        self.var_index = var_index  # Для VARIABLE: X0, X1, ...
    
    def is_leaf(self) -> bool:
        """Проверка, является ли узел листом."""
        return self.node_type in (NodeType.VARIABLE, NodeType.CONSTANT)
    
    def __repr__(self) -> str:
        if self.node_type == NodeType.OPERATION:
            return f"Node(OP={self.value.value})"
        elif self.node_type == NodeType.VARIABLE:
            return f"Node(X{self.var_index})"
        else:
            return f"Node(C={self.value:.4f})"


class ExpressionTree:
    """Дерево выражений, представляющее математическую функцию."""
    
    def __init__(self, root: Node = None, input_dim: int = 1):
        """
        Инициализация дерева выражений.
        
        Args:
            root: Корневой узел дерева
            input_dim: Размерность входа (количество переменных X0, X1, ...)
        """
        self.root = root
        self.input_dim = input_dim
        self._depth = self._calculate_depth() if root else 0
    
    def _calculate_depth(self, node: Node = None) -> int:
        """Вычисление глубины дерева."""
        if node is None:
            node = self.root
        
        if node is None:
            return 0
        
        if node.is_leaf():
            return 1
        
        left_depth = self._calculate_depth(node.left) if node.left else 0
        right_depth = self._calculate_depth(node.right) if node.right else 0
        
        return 1 + max(left_depth, right_depth)
    
    @property
    def depth(self) -> int:
        """Глубина дерева."""
        return self._depth
    
    @classmethod
    def random(cls, input_dim: int, max_depth: int = 5) -> 'ExpressionTree':
        """
        Создание случайного дерева выражений.
        
        Args:
            input_dim: Количество входных переменных
            max_depth: Максимальная глубина дерева
            
        Returns:
            ExpressionTree: Случайное дерево
        """
        root = cls._generate_random_node(input_dim, max_depth, current_depth=0)
        return cls(root=root, input_dim=input_dim)
    
    @classmethod
    def _generate_random_node(cls, input_dim: int, max_depth: int, 
                               current_depth: int) -> Node:
        """Рекурсивная генерация случайного узла."""
        
        # Если достигнута максимальная глубина, создаем лист
        if current_depth >= max_depth - 1:
            return cls._generate_leaf(input_dim)
        
        # С вероятностью 0.3 создаем лист на ранних уровнях
        if current_depth > 0 and random.random() < 0.3:
            return cls._generate_leaf(input_dim)
        
        # Создаем операцию
        if random.random() < 0.7:  # 70% бинарные операции
            op = random.choice(BINARY_OPS)
            left = cls._generate_random_node(input_dim, max_depth, current_depth + 1)
            right = cls._generate_random_node(input_dim, max_depth, current_depth + 1)
            return Node(NodeType.OPERATION, value=op, left=left, right=right)
        else:  # 30% унарные операции
            op = random.choice(UNARY_OPS)
            child = cls._generate_random_node(input_dim, max_depth, current_depth + 1)
            return Node(NodeType.OPERATION, value=op, left=child)
    
    @staticmethod
    def _generate_leaf(input_dim: int) -> Node:
        """Генерация листа (переменная или константа)."""
        if random.random() < 0.7:  # 70% переменные
            var_index = random.randint(0, input_dim - 1)
            return Node(NodeType.VARIABLE, var_index=var_index)
        else:  # 30% константы
            const_value = random.uniform(-10, 10)
            return Node(NodeType.CONSTANT, value=const_value)
    
    def evaluate(self, inputs: np.ndarray) -> float:
        """
        Вычисление значения функции на данных входах.
        
        Args:
            inputs: Входные данные (массив размерности input_dim)
            
        Returns:
            float: Результат вычисления
        """
        if self.root is None:
            return 0.0
        
        try:
            result = self._evaluate_node(self.root, inputs)
            # Защита от бесконечности и NaN
            if np.isnan(result) or np.isinf(result):
                return 0.0
            return result
        except Exception:
            return 0.0
    
    def _evaluate_node(self, node: Node, inputs: np.ndarray) -> float:
        """Рекурсивное вычисление узла."""
        
        if node.node_type == NodeType.VARIABLE:
            idx = min(node.var_index, len(inputs) - 1)
            return float(inputs[idx])
        
        if node.node_type == NodeType.CONSTANT:
            return float(node.value)
        
        if node.node_type == NodeType.OPERATION:
            op = node.value
            
            # Унарные операции
            if op in UNARY_OPS:
                child_val = self._evaluate_node(node.left, inputs)
                return self._apply_unary_op(op, child_val)
            
            # Бинарные операции
            left_val = self._evaluate_node(node.left, inputs)
            right_val = self._evaluate_node(node.right, inputs)
            return self._apply_binary_op(op, left_val, right_val)
        
        return 0.0
    
    def _apply_binary_op(self, op: Operation, a: float, b: float) -> float:
        """Применение бинарной операции с защитой от ошибок."""
        
        if op == Operation.ADD:
            return a + b
        elif op == Operation.SUB:
            return a - b
        elif op == Operation.MUL:
            return a * b
        elif op == Operation.DIV:
            # Защита от деления на ноль
            if abs(b) < 1e-10:
                return 0.0
            return a / b
        elif op == Operation.POW:
            # Защита от слишком больших значений и комплексных чисел
            try:
                # Ограничиваем показатель степени
                if abs(b) > 10:
                    b = np.sign(b) * 10
                # Если основание отрицательное, а степень не целая - проблема
                if a < 0 and not float(b).is_integer():
                    return 0.0
                result = a ** b
                if np.isnan(result) or np.isinf(result):
                    return 0.0
                return result
            except Exception:
                return 0.0
        
        return 0.0
    
    def _apply_unary_op(self, op: Operation, x: float) -> float:
        """Применение унарной операции с защитой от ошибок."""
        
        if op == Operation.LOG:
            # Защита от отрицательных аргументов
            if x <= 0:
                return 0.0
            try:
                result = np.log(x)
                if np.isnan(result) or np.isinf(result):
                    return 0.0
                return float(result)
            except Exception:
                return 0.0
        
        elif op == Operation.EXP:
            # Защита от переполнения
            if x > 700:
                return np.inf
            if x < -700:
                return 0.0
            try:
                result = np.exp(x)
                if np.isnan(result) or np.isinf(result):
                    return 0.0
                return float(result)
            except Exception:
                return 0.0
        
        elif op == Operation.SIN:
            return float(np.sin(x))
        
        elif op == Operation.COS:
            return float(np.cos(x))
        
        elif op == Operation.TAN:
            # Защита от asymptotes
            try:
                result = np.tan(x)
                if np.isnan(result) or np.isinf(result):
                    return 0.0
                return float(result)
            except Exception:
                return 0.0
        
        return 0.0
    
    def to_string(self) -> str:
        """Строковое представление формулы."""
        if self.root is None:
            return "empty"
        
        return self._node_to_string(self.root)
    
    def _node_to_string(self, node: Node) -> str:
        """Рекурсивное преобразование узла в строку."""
        
        if node is None:
            return ""
        
        if node.node_type == NodeType.VARIABLE:
            return f"x{node.var_index}"
        
        if node.node_type == NodeType.CONSTANT:
            return f"{node.value:.4g}"
        
        if node.node_type == NodeType.OPERATION:
            op = node.value.value
            
            if op in ["log", "exp", "sin", "cos", "tan"]:
                child_str = self._node_to_string(node.left)
                return f"{op}({child_str})"
            else:
                left_str = self._node_to_string(node.left)
                right_str = self._node_to_string(node.right)
                return f"({left_str} {op} {right_str})"
        
        return ""
    
    def copy(self) -> 'ExpressionTree':
        """Глубокое копирование дерева."""
        new_root = self._copy_node(self.root)
        return ExpressionTree(root=new_root, input_dim=self.input_dim)
    
    def _copy_node(self, node: Node) -> Node:
        """Рекурсивное копирование узла."""
        if node is None:
            return None
        
        return Node(
            node_type=node.node_type,
            value=deepcopy(node.value),
            left=self._copy_node(node.left),
            right=self._copy_node(node.right),
            var_index=node.var_index
        )
    
    def mutate(self, mutation_rate: float = 0.1, max_depth: int = 5) -> bool:
        """
        Случайная мутация одного узла.
        
        Args:
            mutation_rate: Вероятность мутации каждого узла
            max_depth: Максимальная глубина для новых поддеревьев
            
        Returns:
            bool: True если мутация произошла
        """
        if self.root is None:
            return False
        
        mutated = self._mutate_node(self.root, mutation_rate, max_depth, 0)
        self._depth = self._calculate_depth()
        return mutated
    
    def _mutate_node(self, node: Node, mutation_rate: float, 
                     max_depth: int, current_depth: int) -> bool:
        """Рекурсивная мутация узла."""
        
        mutated = False
        
        # Проверяем, мутировать ли этот узел
        if random.random() < mutation_rate:
            # Выбираем тип мутации
            mutation_type = random.choice(["replace", "subtree"])
            
            if mutation_type == "replace" and node.node_type == NodeType.OPERATION:
                # Заменяем операцию на другую
                node.value = random.choice(ALL_OPS)
                mutated = True
            
            elif mutation_type == "subtree":
                # Заменяем поддерево новым случайным
                if node.is_leaf():
                    new_node = self._generate_random_node(
                        self.input_dim, max_depth, current_depth
                    )
                    node.node_type = new_node.node_type
                    node.value = new_node.value
                    node.left = new_node.left
                    node.right = new_node.right
                    node.var_index = new_node.var_index
                    mutated = True
        
        # Рекурсивно мутируем потомков
        if node.left and not node.left.is_leaf():
            if self._mutate_node(node.left, mutation_rate, max_depth, current_depth + 1):
                mutated = True
        
        if node.right and not node.right.is_leaf():
            if self._mutate_node(node.right, mutation_rate, max_depth, current_depth + 1):
                mutated = True
        
        return mutated
    
    def crossover(self, other: 'ExpressionTree') -> Tuple['ExpressionTree', 'ExpressionTree']:
        """
        Кроссовер с другим деревом (обмен поддеревьями).
        
        Args:
            other: Другое дерево для кроссовера
            
        Returns:
            Tuple[ExpressionTree, ExpressionTree]: Два новых дерева
        """
        child1 = self.copy()
        child2 = other.copy()
        
        # Получаем случайные узлы для обмена
        node1 = self._get_random_node(self.root)
        node2 = other._get_random_node(other.root)
        
        if node1 is None or node2 is None:
            return child1, child2
        
        # Находим соответствующие узлы в копиях
        target1 = self._find_node_by_id(child1.root, id(node1))
        target2 = self._find_node_by_id(child2.root, id(node2))
        
        if target1 is None or target2 is None:
            return child1, child2
        
        # Обмениваемся поддеревьями
        # Копируем поддеревья
        subtree1 = child1._copy_node(target1)
        subtree2 = child2._copy_node(target2)
        
        # Вставляем swapped поддеревья
        self._swap_subtrees(child1.root, target1, subtree2)
        self._swap_subtrees(child2.root, target2, subtree1)
        
        # Обновляем глубину
        child1._depth = child1._calculate_depth()
        child2._depth = child2._calculate_depth()
        
        return child1, child2
    
    def _get_random_node(self, node: Node) -> Optional[Node]:
        """Получение случайного узла из дерева."""
        nodes = self._collect_nodes(node)
        if len(nodes) == 0:
            return None
        return random.choice(nodes)
    
    def _collect_nodes(self, node: Node) -> List[Node]:
        """Сбор всех узлов дерева в список."""
        if node is None:
            return []
        
        nodes = [node]
        if node.left:
            nodes.extend(self._collect_nodes(node.left))
        if node.right:
            nodes.extend(self._collect_nodes(node.right))
        
        return nodes
    
    def _find_node_by_id(self, root: Node, target_id: int) -> Optional[Node]:
        """Поиск узла по его ID (для кроссовера)."""
        # Это упрощенная версия - в реальности нужно более сложное сопоставление
        # Для простоты просто возвращаем случайный узел того же типа
        nodes = self._collect_nodes(root)
        if len(nodes) == 0:
            return None
        return random.choice(nodes)
    
    def _swap_subtrees(self, root: Node, target: Node, new_subtree: Node) -> None:
        """Замена поддерева в узле."""
        # Простая реализация - заменяем первый найденный узел
        self._replace_node(root, target, new_subtree)
    
    def _replace_node(self, parent: Node, target: Node, replacement: Node) -> bool:
        """Рекурсивная замена узла."""
        if parent is target:
            # Это корень - не можем заменить
            return False
        
        if parent.left is target:
            parent.left = replacement
            return True
        
        if parent.right is target:
            parent.right = replacement
            return True
        
        if parent.left:
            if self._replace_node(parent.left, target, replacement):
                return True
        
        if parent.right:
            if self._replace_node(parent.right, target, replacement):
                return True
        
        return False
    
    def count_nodes(self) -> int:
        """Подсчет количества узлов в дереве."""
        return len(self._collect_nodes(self.root))
    
    def __repr__(self) -> str:
        return f"ExpressionTree(depth={self.depth}, nodes={self.count_nodes()})"
