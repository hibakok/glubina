#!/usr/bin/env python3
"""
Эволюционирующий универсальный аппроксиматор
Использует генетические алгоритмы для поиска математических выражений,
аппроксимирующих заданные данные или функции.
"""

import random
import math
import operator
from typing import List, Callable, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time


# ==================== Базовые компоненты ====================

@dataclass
class Node(ABC):
    """Базовый класс для узла дерева выражения"""
    
    @abstractmethod
    def evaluate(self, x: float) -> float:
        """Вычислить значение узла"""
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
    
    def evaluate(self, x: float) -> float:
        return self.value
    
    def depth(self) -> int:
        return 1
    
    def copy(self) -> 'Constant':
        return Constant(self.value)
    
    def to_string(self) -> str:
        return f"{self.value:.4f}"


class Variable(Node):
    """Переменная (x)"""
    
    def evaluate(self, x: float) -> float:
        return x
    
    def depth(self) -> int:
        return 1
    
    def copy(self) -> 'Variable':
        return Variable()
    
    def to_string(self) -> str:
        return "x"


class BinaryOperator(Node):
    """Бинарный оператор"""
    
    def __init__(self, left: Node, right: Node, op: Callable[[float, float], float], symbol: str):
        self.left = left
        self.right = right
        self.op = op
        self.symbol = symbol
    
    def evaluate(self, x: float) -> float:
        try:
            result = self.op(self.left.evaluate(x), self.right.evaluate(x))
            if math.isnan(result) or math.isinf(result):
                return 0.0
            return result
        except (ZeroDivisionError, ValueError, OverflowError):
            return 0.0
    
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
    
    def evaluate(self, x: float) -> float:
        try:
            result = self.op(self.operand.evaluate(x))
            if math.isnan(result) or math.isinf(result):
                return 0.0
            return result
        except (ValueError, ZeroDivisionError, OverflowError):
            return 0.0
    
    def depth(self) -> int:
        return 1 + self.operand.depth()
    
    def copy(self) -> 'UnaryOperator':
        return UnaryOperator(self.operand.copy(), self.op, self.symbol)
    
    def to_string(self) -> str:
        return f"{self.symbol}({self.operand.to_string()})"


# ==================== Генератор популяции ====================

class ExpressionGenerator:
    """Генератор случайных выражений"""
    
    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        self.binary_ops = [
            (operator.add, "+"),
            (operator.sub, "-"),
            (operator.mul, "*"),
            (self.safe_div, "/"),
            (self.safe_pow, "^"),
        ]
        self.unary_ops = [
            (math.sin, "sin"),
            (math.cos, "cos"),
            (math.tan, "tan"),
            (math.exp, "exp"),
            (math.sqrt, "sqrt"),
            (math.log, "log"),
            (self.safe_neg, "neg"),
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
    
    def generate_random(self, depth: int = 0) -> Node:
        """Сгенерировать случайное выражение"""
        self._node_count += 1
        if self._node_count > self._max_nodes:
            # Достигнут лимит узлов, вернуть простой узел
            if random.random() < 0.5:
                return Constant(random.uniform(-5, 5))
            else:
                return Variable()
        
        if depth >= self.max_depth or (depth > 0 and random.random() < 0.4):
            # Листовой узел: константа или переменная
            if random.random() < 0.7:
                return Constant(random.uniform(-5, 5))
            else:
                return Variable()
        
        # Внутренний узел
        choice = random.random()
        
        if choice < 0.7:  # Бинарный оператор
            op_func, op_symbol = random.choice(self.binary_ops)
            left = self.generate_random(depth + 1)
            right = self.generate_random(depth + 1)
            return BinaryOperator(left, right, op_func, op_symbol)
        else:  # Унарный оператор
            op_func, op_symbol = random.choice(self.unary_ops)
            operand = self.generate_random(depth + 1)
            return UnaryOperator(operand, op_func, op_symbol)


# ==================== Генетический алгоритм ====================

class Individual:
    """Особь в популяции"""
    
    def __init__(self, expression: Node):
        self.expression = expression
        self.fitness = float('inf')
    
    def copy(self) -> 'Individual':
        new_individual = Individual(self.expression.copy())
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
                 max_depth: int = 8):
        
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.max_generations = max_generations
        self.target_fitness = target_fitness
        self.generator = ExpressionGenerator(max_depth=max_depth)
        self.population: List[Individual] = []
        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []
        self.tournament_size = 7
    
    def initialize_population(self):
        """Инициализировать начальную популяцию"""
        self.population = []
        for _ in range(self.population_size):
            self.generator.reset_counter()
            expr = self.generator.generate_random()
            individual = Individual(expr)
            self.population.append(individual)
    
    def evaluate_fitness(self, individual: Individual, 
                        target_function: Callable[[float], float],
                        test_points: List[float]) -> float:
        """Вычислить приспособленность особи"""
        total_error = 0.0
        
        for x in test_points:
            predicted = individual.expression.evaluate(x)
            actual = target_function(x)
            
            # Нормализованная ошибка с ограничением
            error = min(abs(predicted - actual), 1e6)
            total_error += error
        
        mse = total_error / len(test_points)
        individual.fitness = mse
        return mse
    
    def select_tournament(self, tournament_size: int = None) -> Individual:
        """Турнирная селекция"""
        if tournament_size is None:
            tournament_size = self.tournament_size
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return min(tournament, key=lambda ind: ind.fitness)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Кроссовер между двумя особями"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Найти случайные узлы для обмена
        node1 = self.get_random_node(parent1.expression)
        node2 = self.get_random_node(parent2.expression)
        
        # Заменить узлы друг с другом (настоящий кроссовер)
        if node1 and node2 and node1 is not parent1.expression and node2 is not parent2.expression:
            child1_expr = parent1.expression.copy()
            child2_expr = parent2.expression.copy()
            
            # Найти соответствующие узлы в копиях
            node1_copy = self.find_corresponding_node(child1_expr, node1)
            node2_copy = self.find_corresponding_node(child2_expr, node2)
            
            if node1_copy and node2_copy:
                parent1_parent = self.find_parent(child1_expr, node1_copy)
                parent2_parent = self.find_parent(child2_expr, node2_copy)
                
                if parent1_parent:
                    if isinstance(parent1_parent, BinaryOperator):
                        if parent1_parent.left is node1_copy:
                            parent1_parent.left = node2_copy.copy()
                        else:
                            parent1_parent.right = node2_copy.copy()
                    elif isinstance(parent1_parent, UnaryOperator):
                        parent1_parent.operand = node2_copy.copy()
                
                if parent2_parent:
                    if isinstance(parent2_parent, BinaryOperator):
                        if parent2_parent.left is node2_copy:
                            parent2_parent.left = node1_copy.copy()
                        else:
                            parent2_parent.right = node1_copy.copy()
                    elif isinstance(parent2_parent, UnaryOperator):
                        parent2_parent.operand = node1_copy.copy()
                
                return Individual(child1_expr), Individual(child2_expr)
        
        # Если не удалось сделать кроссовер, просто вернуть копии
        return parent1.copy(), parent2.copy()
    
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
        
        return nodes
    
    def combine_expressions(self, expr1: Node, expr2: Node) -> Node:
        """Комбинировать два выражения"""
        if random.random() < 0.5:
            op_func, op_symbol = random.choice(self.generator.binary_ops)
            return BinaryOperator(expr1.copy(), expr2.copy(), op_func, op_symbol)
        else:
            op_func, op_symbol = random.choice(self.generator.unary_ops)
            return UnaryOperator(expr1.copy(), op_func, op_symbol)
    
    def mutate(self, individual: Individual) -> Individual:
        """Мутация особи"""
        if random.random() > self.mutation_rate:
            return individual
        
        new_expr = individual.expression.copy()
        
        # Типы мутаций с разными вероятностями
        mutation_type = random.choices(
            ['replace_subtree', 'change_constant', 'add_operator', 'simplify'],
            weights=[0.4, 0.3, 0.2, 0.1]
        )[0]
        
        if mutation_type == 'replace_subtree':
            # Заменить случайное поддерево на новое случайное выражение
            nodes = self.collect_all_nodes(new_expr)
            if len(nodes) > 1:
                node_to_replace = random.choice(nodes[1:])  # Не корень
                parent = self.find_parent(new_expr, node_to_replace)
                
                if parent:
                    if isinstance(parent, BinaryOperator):
                        if parent.left is node_to_replace:
                            parent.left = self.generator.generate_random()
                        else:
                            parent.right = self.generator.generate_random()
                    elif isinstance(parent, UnaryOperator):
                        parent.operand = self.generator.generate_random()
        
        elif mutation_type == 'change_constant':
            # Изменить случайную константу (тонкая настройка)
            constants = [n for n in self.collect_all_nodes(new_expr) if isinstance(n, Constant)]
            if constants:
                const = random.choice(constants)
                # Гауссовская мутация с уменьшающимся шагом
                const.value += random.gauss(0, 0.5)
        
        elif mutation_type == 'add_operator':
            # Добавить оператор вокруг случайного узла
            nodes = self.collect_all_nodes(new_expr)
            if nodes and len(nodes) < 50:  # Ограничить размер
                node = random.choice(nodes)
                op_func, op_symbol = random.choice(self.generator.unary_ops)
                new_unary = UnaryOperator(node.copy(), op_func, op_symbol)
                # Нужно заменить node в родителе на new_unary
                parent = self.find_parent(new_expr, node)
                if parent:
                    if isinstance(parent, BinaryOperator):
                        if parent.left is node:
                            parent.left = new_unary
                        else:
                            parent.right = new_unary
                    elif isinstance(parent, UnaryOperator):
                        parent.operand = new_unary
                else:
                    new_expr = new_unary
        
        elif mutation_type == 'simplify':
            # Упростить выражение, удалив сложные части
            nodes = self.collect_all_nodes(new_expr)
            if len(nodes) > 20:
                # Удалить случайную ветку, заменив её на константу или переменную
                deep_nodes = [n for n in nodes if n.depth() > 3]
                if deep_nodes:
                    node_to_simplify = random.choice(deep_nodes)
                    parent = self.find_parent(new_expr, node_to_simplify)
                    if parent:
                        replacement = Variable() if random.random() < 0.5 else Constant(random.uniform(-5, 5))
                        if isinstance(parent, BinaryOperator):
                            if parent.left is node_to_simplify:
                                parent.left = replacement
                            else:
                                parent.right = replacement
                        elif isinstance(parent, UnaryOperator):
                            parent.operand = replacement
        
        # Проверка на слишком большое выражение
        if new_expr.depth() > 15:
            # Вернуться к оригиналу или упростить
            return individual
        
        return Individual(new_expr)
    
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
        
        return None
    
    def evolve(self, target_function: Callable[[float], float],
               test_points: List[float],
               verbose: bool = True) -> Individual:
        """Запустить эволюцию"""
        
        self.initialize_population()
        
        # Оценить начальную популяцию
        for individual in self.population:
            self.evaluate_fitness(individual, target_function, test_points)
        
        best_ever = min(self.population, key=lambda ind: ind.fitness)
        
        if verbose:
            print(f"Начальная лучшая приспособленность: {best_ever.fitness:.6f}")
        
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
            
            if verbose and generation % 100 == 0:
                print(f"Поколение {generation}: лучшая = {best_current.fitness:.6f}, средняя = {avg_fitness:.6f}")
            
            # Проверка критерия остановки - целевая приспособленность
            if best_current.fitness < self.target_fitness:
                if verbose:
                    print(f"Достигнута целевая приспособленность на поколении {generation}")
                break
            
            # Проверка на застой
            if no_improvement_count >= max_no_improvement:
                if verbose:
                    print(f"Застой обнаружен на поколении {generation}, перезапуск...")
                # Частичный перезапуск популяции
                self._partial_restart()
                no_improvement_count = 0
            
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
            
            # Оценить новую популяцию
            for individual in new_population:
                self.evaluate_fitness(individual, target_function, test_points)
            
            self.population = new_population
            generation += 1
        
        if verbose:
            print(f"\nЭволюция завершена после {generation} поколений")
            print(f"Лучшая приспособленность: {best_ever.fitness:.6f}")
        
        return best_ever
    
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
            expr = self.generator.generate_random()
            self.population[i] = Individual(expr)
        
        # Восстановить элиту
        for i in range(elite_count):
            self.population[i] = elite[i]


# ==================== Тестирование ====================

class ApproximatorTester:
    """Класс для тестирования аппроксиматора"""
    
    def __init__(self):
        self.test_functions = {
            "Линейная": lambda x: 2 * x + 3,
            "Квадратичная": lambda x: x ** 2,
            "Синус": lambda x: math.sin(x),
            "Косинус": lambda x: math.cos(x),
            "Экспонента": lambda x: math.exp(x / 5),
            "Комбинированная": lambda x: math.sin(x) + 0.5 * x ** 2,
            "Сложная": lambda x: math.sin(x) * math.cos(x) + 0.1 * x ** 3,
        }
        
        self.test_ranges = {
            "default": [i * 0.1 for i in range(-30, 31)],
            "wide": [i * 0.2 for i in range(-50, 51)],
            "narrow": [i * 0.05 for i in range(-20, 21)],
        }
    
    def run_test(self, func_name: str, func: Callable[[float], float],
                test_points: List[float], ga: GeneticAlgorithm) -> Tuple[Individual, dict]:
        """Запустить тест для одной функции"""
        
        print(f"\n{'='*60}")
        print(f"Тестирование функции: {func_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        best_individual = ga.evolve(func, test_points, verbose=True)
        
        end_time = time.time()
        
        # Оценка качества на отдельных тестовых точках
        validation_points = [i * 0.15 for i in range(-25, 26)]
        validation_errors = []
        
        for x in validation_points:
            predicted = best_individual.expression.evaluate(x)
            actual = func(x)
            error = abs(predicted - actual)
            validation_errors.append(error)
        
        avg_validation_error = sum(validation_errors) / len(validation_errors)
        max_validation_error = max(validation_errors)
        
        results = {
            "function_name": func_name,
            "training_fitness": best_individual.fitness,
            "validation_avg_error": avg_validation_error,
            "validation_max_error": max_validation_error,
            "evolution_time": end_time - start_time,
            "expression": best_individual.expression.to_string(),
            "generations": len(ga.best_fitness_history),
        }
        
        print(f"\nРезультаты:")
        print(f"  Выражение: {results['expression']}")
        print(f"  Ошибка на обучении (MSE): {results['training_fitness']:.6f}")
        print(f"  Средняя ошибка на валидации: {results['validation_avg_error']:.6f}")
        print(f"  Максимальная ошибка на валидации: {results['validation_max_error']:.6f}")
        print(f"  Время эволюции: {results['evolution_time']:.2f} сек")
        print(f"  Поколений: {results['generations']}")
        
        # Пример предсказаний
        print(f"\nПримеры предсказаний:")
        sample_points = [-2.0, -1.0, 0.0, 1.0, 2.0]
        for x in sample_points:
            predicted = best_individual.expression.evaluate(x)
            actual = func(x)
            error = abs(predicted - actual)
            print(f"  x={x:4.1f}: предсказано={predicted:8.4f}, фактически={actual:8.4f}, ошибка={error:.4f}")
        
        return best_individual, results
    
    def run_all_tests(self) -> List[dict]:
        """Запустить все тесты"""
        
        print("\n" + "="*60)
        print("ЭВОЛЮЦИОНИРУЮЩИЙ УНИВЕРСАЛЬНЫЙ АППРОКСИМАТОР")
        print("="*60)
        print("\nЗапуск полного тестирования...\n")
        
        all_results = []
        
        for func_name, func in self.test_functions.items():
            # Создать новый ГА для каждого теста с оптимизированными параметрами
            ga = GeneticAlgorithm(
                population_size=150,
                mutation_rate=0.4,
                crossover_rate=0.8,
                elitism_count=10,
                max_generations=600,
                target_fitness=0.05,
                max_depth=8
            )
            
            _, results = self.run_test(func_name, func, self.test_ranges["default"], ga)
            all_results.append(results)
        
        # Итоговая статистика
        print(f"\n{'='*60}")
        print("ИТОГОВАЯ СТАТИСТИКА")
        print(f"{'='*60}")
        
        avg_fitness = sum(r['training_fitness'] for r in all_results) / len(all_results)
        avg_val_error = sum(r['validation_avg_error'] for r in all_results) / len(all_results)
        total_time = sum(r['evolution_time'] for r in all_results)
        
        print(f"\nВсего тестов: {len(all_results)}")
        print(f"Средняя ошибка обучения: {avg_fitness:.6f}")
        print(f"Средняя ошибка валидации: {avg_val_error:.6f}")
        print(f"Общее время тестирования: {total_time:.2f} сек")
        
        print(f"\nДетальные результаты по функциям:")
        for r in all_results:
            status = "✓" if r['validation_avg_error'] < 0.5 else "⚠"
            print(f"  {status} {r['function_name']:20s}: ошибка={r['validation_avg_error']:.4f}")
        
        return all_results


# ==================== Основная программа ====================

def main():
    """Основная функция"""
    
    print("\n" + "="*60)
    print("ЭВОЛЮЦИОНИРУЮЩИЙ УНИВЕРСАЛЬНЫЙ АППРОКСИМАТОР")
    print("Версия 1.0")
    print("="*60)
    
    # Запустить тестирование
    tester = ApproximatorTester()
    results = tester.run_all_tests()
    
    # Финальный отчет
    print(f"\n{'='*60}")
    print("ЗАВЕРШЕНИЕ РАБОТЫ")
    print(f"{'='*60}")
    
    successful_tests = sum(1 for r in results if r['validation_avg_error'] < 0.5)
    total_tests = len(results)
    
    print(f"\nУспешно пройдено тестов: {successful_tests}/{total_tests}")
    
    if successful_tests == total_tests:
        print("✓ Все тесты пройдены успешно!")
        print("\nАппроксиматор готов к использованию.")
        print("\nПример использования:")
        print("""
    from approximator import GeneticAlgorithm
    
    # Создать аппроксиматор
    ga = GeneticAlgorithm(population_size=100, max_generations=500)
    
    # Целевая функция
    target = lambda x: math.sin(x) + x**2
    
    # Точки для обучения
    points = [i * 0.1 for i in range(-30, 31)]
    
    # Запустить эволюцию
    best = ga.evolve(target, points)
    
    # Использовать найденное выражение
    result = best.expression.evaluate(2.5)
        """)
    else:
        print(f"⚠ Некоторые тесты требуют улучшения ({total_tests - successful_tests} неудачных)")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
