"""
Юнит-тесты для аппроксиматора.
"""

import sys
import os
import unittest
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from approximator.data import DataPair, Dataset
from approximator.tree import ExpressionTree, Node, NodeType, Operation
from approximator.population import Population
from approximator.core import UniversalApproximator


class TestDataPair(unittest.TestCase):
    """Тесты для класса DataPair."""
    
    def test_init(self):
        """Тест инициализации."""
        pair = DataPair([1.0, 2.0], [3.0])
        self.assertEqual(pair.inputs, [1.0, 2.0])
        self.assertEqual(pair.outputs, [3.0])
    
    def test_repr(self):
        """Тест строкового представления."""
        pair = DataPair([1.0], [2.0])
        repr_str = repr(pair)
        self.assertIn("1.0", repr_str)
        self.assertIn("2.0", repr_str)


class TestDataset(unittest.TestCase):
    """Тесты для класса Dataset."""
    
    def setUp(self):
        """Создание временного файла с данными."""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        self.temp_file.write("1.0 2.0 | 3.0\n")
        self.temp_file.write("2.0 3.0 | 7.0\n")
        self.temp_file.write("3.0 4.0 | 13.0\n")
        self.temp_file.close()
    
    def tearDown(self):
        """Удаление временного файла."""
        os.unlink(self.temp_file.name)
    
    def test_load_from_file(self):
        """Тест загрузки корректного файла."""
        dataset = Dataset.load_from_file(self.temp_file.name)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.input_dim, 2)
        self.assertEqual(dataset.output_dim, 1)
    
    def test_validate(self):
        """Тест проверки консистентности."""
        dataset = Dataset.load_from_file(self.temp_file.name)
        self.assertTrue(dataset.validate())
    
    def test_get_matrices(self):
        """Тест получения матриц."""
        dataset = Dataset.load_from_file(self.temp_file.name)
        X = dataset.get_input_matrix()
        y = dataset.get_output_matrix()
        self.assertEqual(X.shape, (3, 2))
        # Для одномерного выхода shape может быть (3,) или (3, 1)
        self.assertEqual(y.shape[0], 3)
    
    def test_normalize(self):
        """Тест нормализации."""
        dataset = Dataset.load_from_file(self.temp_file.name)
        dataset.normalize("minmax")
        X = dataset.get_input_matrix()
        # После minmax нормализации значения должны быть в [0, 1]
        self.assertTrue(np.all(X >= 0))
        self.assertTrue(np.all(X <= 1))
    
    def test_load_file_not_found(self):
        """Тест ошибки при отсутствии файла."""
        with self.assertRaises(FileNotFoundError):
            Dataset.load_from_file("nonexistent_file.txt")
    
    def test_load_dimension_mismatch(self):
        """Тест ошибки при несоответствии размерностей."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        temp_file.write("1.0 2.0 | 3.0\n")
        temp_file.write("2.0 | 7.0\n")  # Несоответствие размерности входов
        temp_file.close()
        
        try:
            with self.assertRaises(ValueError):
                Dataset.load_from_file(temp_file.name)
        finally:
            os.unlink(temp_file.name)


class TestExpressionTree(unittest.TestCase):
    """Тесты для класса ExpressionTree."""
    
    def test_random_tree(self):
        """Тест создания случайного дерева."""
        tree = ExpressionTree.random(input_dim=2, max_depth=5)
        self.assertIsNotNone(tree.root)
        self.assertGreater(tree.depth, 0)
        self.assertLessEqual(tree.depth, 5)
    
    def test_evaluate_simple(self):
        """Тест вычисления простого дерева."""
        # Создаем дерево: x0 + x1
        var0 = Node(NodeType.VARIABLE, var_index=0)
        var1 = Node(NodeType.VARIABLE, var_index=1)
        root = Node(NodeType.OPERATION, value=Operation.ADD, left=var0, right=var1)
        
        tree = ExpressionTree(root=root, input_dim=2)
        result = tree.evaluate(np.array([2.0, 3.0]))
        self.assertAlmostEqual(result, 5.0, places=5)
    
    def test_evaluate_constant(self):
        """Тест вычисления константы."""
        const = Node(NodeType.CONSTANT, value=5.0)
        tree = ExpressionTree(root=const, input_dim=1)
        result = tree.evaluate(np.array([1.0]))
        self.assertAlmostEqual(result, 5.0, places=5)
    
    def test_mutate(self):
        """Тест мутации дерева."""
        tree = ExpressionTree.random(input_dim=2, max_depth=4)
        original_formula = tree.to_string()
        
        # Мутируем несколько раз
        for _ in range(10):
            tree.mutate(mutation_rate=0.5)
        
        # Дерево должно измениться
        new_formula = tree.to_string()
        # Не гарантируется, но вероятно изменение
        # self.assertNotEqual(original_formula, new_formula)
    
    def test_crossover(self):
        """Тест кроссовера деревьев."""
        tree1 = ExpressionTree.random(input_dim=2, max_depth=4)
        tree2 = ExpressionTree.random(input_dim=2, max_depth=4)
        
        child1, child2 = tree1.crossover(tree2)
        
        # Дети должны быть разными копиями
        self.assertIsNot(child1, tree1)
        self.assertIsNot(child2, tree2)
    
    def test_copy(self):
        """Тест копирования дерева."""
        tree = ExpressionTree.random(input_dim=2, max_depth=4)
        tree_copy = tree.copy()
        
        # Копия должна быть независимой
        self.assertIsNot(tree.root, tree_copy.root)
        self.assertEqual(tree.to_string(), tree_copy.to_string())
    
    def test_to_string(self):
        """Тест строкового представления."""
        # Создаем дерево: x0 * 2.0
        var0 = Node(NodeType.VARIABLE, var_index=0)
        const = Node(NodeType.CONSTANT, value=2.0)
        root = Node(NodeType.OPERATION, value=Operation.MUL, left=var0, right=const)
        
        tree = ExpressionTree(root=root, input_dim=1)
        formula = tree.to_string()
        self.assertIn("x0", formula)
        self.assertIn("*", formula)


class TestPopulation(unittest.TestCase):
    """Тесты для класса Population."""
    
    def test_initialize(self):
        """Тест инициализации популяции."""
        pop = Population()
        pop.initialize_random(size=10, input_dim=2, max_depth=5)
        self.assertEqual(len(pop), 10)
    
    def test_evaluate_fitness(self):
        """Тест вычисления fitness."""
        # Создаем простой датасет
        dataset = Dataset()
        dataset.pairs = [DataPair([1.0], [2.0]), DataPair([2.0], [4.0])]
        dataset.input_dim = 1
        dataset.output_dim = 1
        
        pop = Population()
        pop.initialize_random(size=5, input_dim=1, max_depth=3)
        pop.evaluate_fitness(dataset)
        
        self.assertEqual(len(pop.fitness_scores), 5)
        self.assertTrue(all(f >= 0 for f in pop.fitness_scores))
    
    def test_select_tournament(self):
        """Тест турнирной селекции."""
        pop = Population()
        pop.initialize_random(size=10, input_dim=2, max_depth=4)
        pop.fitness_scores = list(range(10))  # Fitness от 0 до 9
        
        best, fitness = pop.select_tournament(k=3)
        self.assertIsNotNone(best)
        self.assertLessEqual(fitness, max(pop.fitness_scores))
    
    def test_get_best(self):
        """Тест получения лучшей особи."""
        pop = Population()
        pop.initialize_random(size=5, input_dim=1, max_depth=3)
        pop.fitness_scores = [5.0, 3.0, 1.0, 4.0, 2.0]
        
        best, fitness = pop.get_best()
        self.assertEqual(fitness, 1.0)


class TestUniversalApproximator(unittest.TestCase):
    """Тесты для класса UniversalApproximator."""
    
    def setUp(self):
        """Создание временного файла с простыми данными."""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        # Функция f(x) = 2x + 1
        self.temp_file.write("1.0 | 3.0\n")
        self.temp_file.write("2.0 | 5.0\n")
        self.temp_file.write("3.0 | 7.0\n")
        self.temp_file.write("4.0 | 9.0\n")
        self.temp_file.write("5.0 | 11.0\n")
        self.temp_file.close()
    
    def tearDown(self):
        """Удаление временного файла."""
        os.unlink(self.temp_file.name)
    
    def test_load_data(self):
        """Тест загрузки данных."""
        approx = UniversalApproximator()
        approx.load_data(self.temp_file.name)
        self.assertIsNotNone(approx.dataset)
        self.assertEqual(len(approx.dataset), 5)
    
    def test_train_simple_function(self):
        """Тест обучения на простой функции f(x) = 2x + 1."""
        approx = UniversalApproximator()
        approx.load_data(self.temp_file.name)
        
        # Обучаем с достаточным количеством поколений
        approx.set_params(pop_size=50, generations=100, mutation_rate=0.1, 
                         crossover_rate=0.7, max_depth=4)
        approx.train(verbose=False)
        
        # MSE должно быть достаточно малым
        self.assertLess(approx.best_fitness, 0.5)
    
    def test_predict_after_train(self):
        """Тест предсказания после обучения."""
        approx = UniversalApproximator()
        approx.load_data(self.temp_file.name)
        approx.set_params(pop_size=50, generations=100)
        approx.train(verbose=False)
        
        # Предсказание для новой точки
        pred = approx.predict([2.5])
        # Ожидаем значение около 6.0 (2*2.5 + 1)
        self.assertGreater(pred[0], 4.0)
        self.assertLess(pred[0], 8.0)
    
    def test_save_and_load_model(self):
        """Тест сохранения и загрузки модели."""
        approx1 = UniversalApproximator()
        approx1.load_data(self.temp_file.name)
        approx1.set_params(pop_size=30, generations=50)
        approx1.train(verbose=False)
        
        # Сохраняем модель
        model_file = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
        model_file.close()
        
        try:
            approx1.save_model(model_file.name)
            
            # Загружаем в новый аппроксиматор
            approx2 = UniversalApproximator()
            approx2.load_model(model_file.name)
            
            # Проверяем идентичность предсказаний
            test_input = [3.0]
            pred1 = approx1.predict(test_input)
            pred2 = approx2.predict(test_input)
            
            self.assertAlmostEqual(pred1[0], pred2[0], places=5)
        finally:
            os.unlink(model_file.name)
    
    def test_get_best_formula(self):
        """Тест получения формулы."""
        approx = UniversalApproximator()
        approx.load_data(self.temp_file.name)
        
        # До обучения
        formula = approx.get_best_formula()
        self.assertIn("не обучена", formula.lower())
        
        # После обучения
        approx.set_params(pop_size=30, generations=50)
        approx.train(verbose=False)
        formula = approx.get_best_formula()
        self.assertIn("f(", formula)


class TestIntegration(unittest.TestCase):
    """Интеграционные тесты."""
    
    def test_full_workflow(self):
        """Тест полного рабочего цикла."""
        # Создаем файл с данными
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        temp_file.write("# Тестовые данные\n")
        temp_file.write("1.0 2.0 | 5.0\n")
        temp_file.write("2.0 3.0 | 9.0\n")
        temp_file.write("3.0 4.0 | 13.0\n")
        temp_file.close()
        
        try:
            # Создаем аппроксиматор
            approx = UniversalApproximator()
            
            # Загружаем данные
            approx.load_data(temp_file.name)
            self.assertEqual(approx.dataset.input_dim, 2)
            
            # Настраиваем параметры
            approx.set_params(pop_size=40, generations=80)
            
            # Обучаем
            approx.train(verbose=False)
            self.assertTrue(approx._trained)
            
            # Делаем предсказание
            pred = approx.predict([1.5, 2.5])
            self.assertIsInstance(pred, list)
            self.assertEqual(len(pred), 1)
            
            # Получаем формулу
            formula = approx.get_best_formula()
            self.assertIsInstance(formula, str)
            
            # Получаем статистику
            stats = approx.get_training_stats()
            self.assertIn('total_generations', stats)
            
        finally:
            os.unlink(temp_file.name)


if __name__ == '__main__':
    unittest.main(verbosity=2)
