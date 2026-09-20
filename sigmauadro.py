#!/usr/bin/env python3
"""
Sigmauadro - Универсальный эволюционный аппроксиматор
Основан на чистых мутациях без кроссовера
"""

import random
import math
import copy
import sys
import os
from decimal import Decimal, getcontext

# Максимальная точность вычислений
getcontext().prec = 50

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

class Config:
    """Настройки эволюции из config.txt"""
    DEFAULTS = {
        'offspring_per_parent': 4,
        'mutation_count': 3,
        'population_size': 20,
        'max_cpu_percent': 100,
    }
    
    def __init__(self):
        for k, v in self.DEFAULTS.items():
            setattr(self, k, v)
    
    def load(self, filename="config.txt"):
        """Загрузить настройки из файла"""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, val = line.split('=', 1)
                        key, val = key.strip(), val.strip()
                        if hasattr(self, key):
                            try:
                                setattr(self, key, int(val))
                            except ValueError:
                                pass
        return self
    
    def save(self, filename="config.txt"):
        """Сохранить настройки в файл"""
        with open(filename, 'w') as f:
            f.write("# Настройки Sigmauadro\n")
            for key in self.DEFAULTS:
                val = getattr(self, key)
                if isinstance(val, (int, float)):
                    f.write(f"{key} = {val}\n")
        return self

# ============================================================================
# МИНИМАЛЬНЫЕ КИРПИЧИКИ (БАЗОВЫЕ ОПЕРАЦИИ)
# ============================================================================

class Primitive:
    """Базовая операция - кирпичик для построения функций"""
    def __init__(self, name, func, arity, code, is_numeric=False):
        self.name = name      # Имя операции
        self.func = func      # Функция выполнения
        self.arity = arity    # Количество аргументов
        self.code = code      # Код для отображения
        self.is_numeric = is_numeric  # Числовой ли это примитив
    
    def execute(self, args):
        """Выполнить операцию над аргументами"""
        try:
            return self.func(*args)
        except (ZeroDivisionError, ValueError, OverflowError):
            return Decimal('0')

# Базовые примитивы
PRIMITIVES = [
    Primitive("add", lambda a, b: a + b, 2, "+"),
    Primitive("sub", lambda a, b: a - b, 2, "-"),
    Primitive("mul", lambda a, b: a * b, 2, "*"),
    Primitive("div", lambda a, b: a / b if b != 0 else Decimal('0'), 2, "/"),
    Primitive("neg", lambda a: -a, 1, "-"),
    Primitive("abs", lambda a: abs(a), 1, "abs"),
    Primitive("sqrt", lambda a: a.sqrt() if a >= 0 else Decimal('0'), 1, "sqrt"),
    Primitive("sin", lambda a: Decimal(str(math.sin(float(a)))), 1, "sin"),
    Primitive("cos", lambda a: Decimal(str(math.cos(float(a)))), 1, "cos"),
    Primitive("exp", lambda a: Decimal(str(math.exp(float(a)))) if a < 700 else Decimal('0'), 1, "exp"),
    Primitive("log", lambda a: Decimal(str(math.log(float(a)))) if a > 0 else Decimal('0'), 1, "log"),
    Primitive("pow", lambda a, b: Decimal(str(math.pow(float(a), float(b)))) if a > 0 else Decimal('0'), 2, "^"),
]

# Константы добавляются динамически
CONSTANT_PRIMITIVES_START_IDX = len(PRIMITIVES)
NUM_CONSTANT_PRIMITIVES = 20

def init_constants():
    """Инициализировать пул констант"""
    while len(PRIMITIVES) > CONSTANT_PRIMITIVES_START_IDX:
        PRIMITIVES.pop()
    for i in range(NUM_CONSTANT_PRIMITIVES):
        val = Decimal(str(random.uniform(-10, 10)))
        PRIMITIVES.append(Primitive(f"const_{i}", lambda v=val: v, 0, str(val), is_numeric=True))

# ============================================================================
# ОСОБЬ (ГЕНЕТИЧЕСКОЕ ПРЕДСТАВЛЕНИЕ)
# ============================================================================

class Individual:
    """Особь представляющая функцию в виде дерева операций"""
    
    def __init__(self, genome=None, input_size=1):
        self.genome = genome or []  # Список узлов дерева
        self.input_size = input_size
        self.error = None
        self.complexity = 0
        self._calc_complexity()
    
    def _calc_complexity(self):
        """Вычислить сложность особи (количество операций)"""
        self.complexity = len(self.genome)
    
    def execute(self, inputs):
        """Выполнить особь на входных данных"""
        if not self.genome:
            # Пустая особь возвращает 0
            return [Decimal('0')] * max(1, self.input_size)
        
        try:
            # Стек для вычислений
            stack = list(inputs)
            
            for node in self.genome:
                prim_idx, arg_count = node
                if prim_idx >= len(PRIMITIVES):
                    continue
                
                prim = PRIMITIVES[prim_idx]
                
                # Получаем аргументы
                args = []
                for _ in range(arg_count):
                    if stack:
                        args.append(stack.pop())
                    else:
                        args.append(Decimal('0'))
                
                # Выполняем операцию
                result = prim.execute(args)
                stack.append(result)
            
            # Результат - верхние значения стека
            if len(stack) < self.input_size:
                return stack + [Decimal('0')] * (self.input_size - len(stack))
            return stack[-self.input_size:] if self.input_size > 0 else stack
            
        except Exception:
            return [Decimal('0')] * max(1, self.input_size)
    
    def evaluate(self, data_pairs):
        """Вычислить ошибку на всех парах данных"""
        if not data_pairs:
            self.error = Decimal('inf')
            return self.error
        
        total_error = Decimal('0')
        count = 0
        
        for inputs, expected in data_pairs:
            outputs = self.execute([Decimal(x) for x in inputs])
            
            for i, exp_val in enumerate(expected):
                exp_val = Decimal(exp_val)
                out_val = outputs[i] if i < len(outputs) else Decimal('0')
                total_error += abs(out_val - exp_val)
                count += 1
        
        self.error = total_error / count if count > 0 else Decimal('inf')
        return self.error
    
    def mutate(self, mutation_count):
        """Применить мутации к особи"""
        new_genome = copy.deepcopy(self.genome)
        
        for _ in range(mutation_count):
            if not new_genome:
                # Добавить новую операцию
                prim_idx = random.randint(0, len(PRIMITIVES) - 1)
                prim = PRIMITIVES[prim_idx]
                new_genome.append((prim_idx, prim.arity))
            else:
                mut_type = random.choice(['replace', 'add', 'remove', 'tweak_const'])
                
                if mut_type == 'replace' and new_genome:
                    # Заменить узел
                    idx = random.randint(0, len(new_genome) - 1)
                    prim_idx = random.randint(0, len(PRIMITIVES) - 1)
                    prim = PRIMITIVES[prim_idx]
                    new_genome[idx] = (prim_idx, prim.arity)
                
                elif mut_type == 'add':
                    # Добавить узел
                    prim_idx = random.randint(0, len(PRIMITIVES) - 1)
                    prim = PRIMITIVES[prim_idx]
                    pos = random.randint(0, len(new_genome))
                    new_genome.insert(pos, (prim_idx, prim.arity))
                
                elif mut_type == 'remove' and len(new_genome) > 1:
                    # Удалить узел
                    idx = random.randint(0, len(new_genome) - 1)
                    new_genome.pop(idx)
                
                elif mut_type == 'tweak_const':
                    # Тонкая подстройка константы (числовая мутация)
                    const_indices = [i for i, (pidx, _) in enumerate(new_genome) 
                                    if pidx >= CONSTANT_PRIMITIVES_START_IDX]
                    if const_indices:
                        idx = random.choice(const_indices)
                        prim_idx, arity = new_genome[idx]
                        # Сдвинуть значение константы на малую величину
                        current_val = PRIMITIVES[prim_idx].func()
                        # Мутация на 1-15 знак после запятой
                        tweak = Decimal(str(random.uniform(-1e-15, 1e-15)))
                        new_val = current_val + tweak
                        # Обновить примитив новым значением
                        PRIMITIVES[prim_idx] = Primitive(f"const_{prim_idx - CONSTANT_PRIMITIVES_START_IDX}", 
                                                         lambda v=new_val: v, 0, str(new_val), is_numeric=True)
        
        offspring = Individual(new_genome, self.input_size)
        return offspring
    
    def to_readable(self):
        """Преобразовать геном в читаемое представление"""
        if not self.genome:
            return "f(x) = 0"
        
        # Построить выражение из дерева
        stack = []
        for node in self.genome:
            prim_idx, arg_count = node
            if prim_idx >= len(PRIMITIVES):
                continue
            
            prim = PRIMITIVES[prim_idx]
            
            args = []
            for _ in range(arg_count):
                if stack:
                    args.append(stack.pop())
                else:
                    args.append("x")
            
            if arg_count == 0:
                expr = prim.code
            elif arg_count == 1:
                expr = f"{prim.code}({args[0]})"
            else:
                expr = f"({args[1]} {prim.code} {args[0]})"
            
            stack.append(expr)
        
        if stack:
            return f"f(x) = {stack[-1]}"
        return "f(x) = 0"
    
    def save(self, filename):
        """Сохранить особь в читаемый файл"""
        with open(filename, 'w') as f:
            f.write(f"# Sigmauadro - Лучшая особь\n# {self.to_readable()}\n\n")
            f.write(f"input_size: {self.input_size}\nerror: {self.error}\ncomplexity: {self.complexity}\n\n")
            f.write("# Геном (индекс_примитива, количество_аргументов)\n")
            for node in self.genome:
                f.write(f"{node[0]} {node[1]}\n")
    
    @classmethod
    def load(cls, filename):
        """Загрузить особь из файла"""
        genome = []
        input_size = 1
        
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('input_size:'):
                    input_size = int(line.split(':')[1].strip())
                elif line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            prim_idx = int(parts[0])
                            arg_count = int(parts[1])
                            genome.append((prim_idx, arg_count))
                        except ValueError:
                            pass
        
        individual = cls(genome, input_size)
        return individual

# ============================================================================
# ЭВОЛЮЦИОННЫЙ ДВИЖОК
# ============================================================================

class EvolutionEngine:
    """Движок эволюции с внутренней и внешней популяцией"""
    
    def __init__(self, config):
        self.config = config
        self.inner_population = []  # Прародители
        self.best_individual = None
        self.data_pairs = []
        self.generations_run = 0
        self.generations_without_improvement = 0
        self.best_error_history = []
    
    def initialize(self, input_size):
        """Инициализировать внутреннюю популяцию с нуля"""
        # Инициализировать пул констант при старте
        init_constants()
        
        self.inner_population = []
        for _ in range(self.config.population_size):
            # Начать с пустой особи (абсолютный ноль)
            individual = Individual([], input_size)
            individual._calc_complexity()
            individual.evaluate(self.data_pairs)
            self.inner_population.append(individual)
        
        if self.inner_population:
            self.best_individual = min(self.inner_population, key=lambda x: x.error if x.error is not None else Decimal('inf'))
        self.generations_run = 0
        self.generations_without_improvement = 0
        self.best_error_history = []
    
    def load_data(self, filename):
        """Загрузить пары входных-выходных данных"""
        self.data_pairs = []
        
        if not os.path.exists(filename):
            print(f"Файл {filename} не найден")
            return False
        
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or '|' not in line:
                    continue
                
                try:
                    parts = line.split('|')
                    input_part = parts[0].strip().split()
                    output_part = parts[1].strip().split()
                    
                    inputs = [Decimal(x) for x in input_part]
                    outputs = [Decimal(x) for x in output_part]
                    
                    self.data_pairs.append((inputs, outputs))
                except (ValueError, IndexError):
                    continue
        
        if self.data_pairs:
            # Обновить размерность входа
            input_size = len(self.data_pairs[0][0])
            for ind in self.inner_population:
                ind.input_size = input_size
            if self.best_individual:
                self.best_individual.input_size = input_size
            # Инициализировать популяцию если пуста
            if not self.inner_population:
                self.initialize(input_size)
        
        return len(self.data_pairs) > 0
    
    def evolve_generation(self):
        """Провести одно поколение эволюции"""
        if not self.inner_population or not self.data_pairs:
            return
        
        # Каждым прародителем порождается потомство
        for i, parent in enumerate(self.inner_population):
            for _ in range(self.config.offspring_per_parent):
                # Порождение потомка через мутации
                offspring = parent.mutate(self.config.mutation_count)
                
                # Оценка приспособленности
                offspring.evaluate(self.data_pairs)
                
                # Проверка: может ли потомок заменить прародителя
                if offspring.error < parent.error:
                    # Потомок лучше - заменяем прародителя
                    self.inner_population[i] = offspring
                    parent = offspring  # Обновить ссылку для следующих мутаций
        
        # Найти лучшую особь в популяции
        current_best = min(self.inner_population, key=lambda x: x.error if x.error is not None else Decimal('inf'))
        
        # Сравнение с глобальным лучшим
        prev_best_error = self.best_individual.error if self.best_individual else Decimal('inf')
        
        if current_best.error < prev_best_error:
            self.best_individual = copy.deepcopy(current_best)
            self.generations_without_improvement = 0
        elif current_best.error == prev_best_error:
            if current_best.complexity < self.best_individual.complexity:
                self.best_individual = copy.deepcopy(current_best)
                self.generations_without_improvement = 0  # Упрощение тоже улучшение
        
        self.generations_run += 1
        self.generations_without_improvement += 1
        self.best_error_history.append(self.best_individual.error if self.best_individual else Decimal('inf'))
    
    def run_evolution(self, generations, display=True):
        """Запустить эволюцию на заданное количество поколений"""
        for gen in range(generations):
            self.evolve_generation()
            
            if display:
                # Отобразить прогресс
                best_err = self.best_individual.error if self.best_individual else Decimal('inf')
                complexity = self.best_individual.complexity if self.best_individual else 0
                
                # Очистить строку и вывести прогресс
                sys.stdout.write(f"\rПоколение {gen + 1}/{generations} | Ошибка: {best_err:.10f} | Без улучшений: {self.generations_without_improvement} | Сложность: {complexity}")
                sys.stdout.flush()
        
        if display:
            print()  # Новая строка после завершения
    
    def test_best(self):
        """Тестировать лучшую особь с пользовательским вводом"""
        if not self.best_individual:
            print("Нет обученной особи")
            return
        
        print(f"\nТестирование лучшей особи: {self.best_individual.to_readable()}")
        print("Введите входные данные (или 'выйти' для выхода)")
        
        while True:
            try:
                user_input = input("\nВход: ").strip()
                if user_input.lower() in ('выйти', 'exit', 'quit', 'q'):
                    break
                
                inputs = [Decimal(x) for x in user_input.split()]
                outputs = self.best_individual.execute(inputs)
                
                print(f"Выход: {' '.join(str(o) for o in outputs)}")
                
            except (ValueError, EOFError):
                print("Некорректный ввод")
    
    def save_best(self, filename="best_individual.txt"):
        """Сохранить лучшую особь в файл"""
        if self.best_individual:
            self.best_individual.save(filename)
            print(f"Лучшая особь сохранена в {filename}")
        else:
            print("Нет особи для сохранения")
    
    def save_population(self, filename="population.txt"):
        """Сохранить всю популяцию в файл"""
        with open(filename, 'w') as f:
            f.write(f"# Популяция Sigmauadro\n")
            f.write(f"# Особей: {len(self.inner_population)}\n\n")
            for i, ind in enumerate(self.inner_population):
                f.write(f"\n## Особь {i+1}\n")
                f.write(f"input_size: {ind.input_size}\n")
                f.write(f"error: {ind.error}\n")
                f.write(f"complexity: {ind.complexity}\n")
                for node in ind.genome:
                    f.write(f"{node[0]} {node[1]}\n")
        print(f"Популяция сохранена в {filename}")
    
    def load_population(self, filename):
        """Загрузить популяцию из файла"""
        if not os.path.exists(filename):
            print(f"Файл {filename} не найден")
            return False
        
        population = []
        current_genome = []
        current_input_size = 1
        
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('input_size:'):
                    current_input_size = int(line.split(':')[1].strip())
                elif line and not line.startswith('#') and not line.startswith('##'):
                    if ':' not in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                prim_idx = int(parts[0])
                                arg_count = int(parts[1])
                                current_genome.append((prim_idx, arg_count))
                            except ValueError:
                                pass
                    elif line.startswith('error:') or line.startswith('complexity:'):
                        # Конец особи - сохранить если есть геном
                        if current_genome:
                            ind = Individual(current_genome, current_input_size)
                            ind.evaluate(self.data_pairs)  # Вычислить ошибку
                            population.append(ind)
                            current_genome = []
        
        # Добавить последнюю особь
        if current_genome:
            ind = Individual(current_genome, current_input_size)
            ind.evaluate(self.data_pairs)
            population.append(ind)
        
        if population:
            self.inner_population = population
            self.best_individual = min(population, key=lambda x: x.error if x.error is not None else Decimal('inf'))
            print(f"Популяция загружена: {len(population)} особей")
            return True
        
        return False

# ============================================================================
# КОНСОЛЬНЫЙ ИНТЕРФЕЙС
# ============================================================================

def main_menu():
    """Главное меню программы"""
    config = Config().load()
    engine = EvolutionEngine(config)
    
    print("=" * 60)
    print("           SIGMAUADRO - Универсальный Аппроксиматор")
    print("=" * 60)
    print("\nЭволюционный алгоритм на основе чистых мутаций")
    print("Способен аппроксимировать любую вычислимую функцию\n")
    
    # Загрузить данные по умолчанию
    if not engine.load_data("data.txt"):
        print("Файл data.txt не найден. Загрузите данные через пункт 6.")
    
    while True:
        print("\n" + "=" * 60)
        print("ГЛАВНОЕ МЕНЮ")
        print("=" * 60)
        print("1. Начать эволюцию")
        print("2. Тестировать текущую лучшую особь")
        print("3. Сохранить лучшую особь в файл")
        print("4. Сохранить популяцию в файл")
        print("5. Загрузить популяцию из файла")
        print("6. Загрузить пары входных-выходных данных")
        print("7. Показать текущую лучшую особь")
        print("8. Настройки")
        print("9. Выход")
        
        choice = input("\nВыберите пункт: ").strip()
        
        if choice == '1':
            try:
                gens = int(input("Сколько поколений эволюции прогонять? ").strip())
                if gens > 0:
                    print("\nЗапуск эволюции...")
                    engine.run_evolution(gens)
                    print(f"\nЭволюция завершена. Пройдено поколений: {engine.generations_run}")
                    if engine.best_individual:
                        print(f"Лучшая ошибка: {engine.best_individual.error}")
                        print(f"Лучшая особь: {engine.best_individual.to_readable()}")
                else:
                    print("Количество поколений должно быть положительным")
            except ValueError:
                print("Некорректное число")
        
        elif choice == '2':
            engine.test_best()
        
        elif choice == '3':
            filename = input("Имя файла (лучшая_особь.txt): ").strip()
            if not filename:
                filename = "best_individual.txt"
            engine.save_best(filename)
        
        elif choice == '4':
            filename = input("Имя файла (популяция.txt): ").strip()
            if not filename:
                filename = "population.txt"
            engine.save_population(filename)
        
        elif choice == '5':
            filename = input("Имя файла для загрузки: ").strip()
            if not filename:
                print("Имя файла не указано")
            else:
                engine.load_population(filename)
        
        elif choice == '6':
            filename = input("Имя файла с данными (data.txt): ").strip()
            if not filename:
                filename = "data.txt"
            if engine.load_data(filename):
                print(f"Загружено пар данных: {len(engine.data_pairs)}")
            else:
                print("Не удалось загрузить данные")
        
        elif choice == '7':
            if engine.best_individual:
                print(f"\nЛучшая особь: {engine.best_individual.to_readable()}")
                print(f"Ошибка: {engine.best_individual.error}")
                print(f"Сложность: {engine.best_individual.complexity}")
                print(f"Поколений пройдено: {engine.generations_run}")
                print(f"Поколений без улучшений: {engine.generations_without_improvement}")
            else:
                print("Нет обученной особи")
        
        elif choice == '8':
            print("\nТекущие настройки:")
            print(f"  Потомков на прародителя: {config.offspring_per_parent}")
            print(f"  Мутаций на потомка: {config.mutation_count}")
            print(f"  Размер популяции: {config.population_size}")
            
            change = input("\nИзменить настройки? (y/n): ").strip().lower()
            if change == 'y':
                try:
                    val = int(input("Потомков на прародителя: ").strip())
                    if val > 0:
                        config.offspring_per_parent = val
                    val = int(input("Мутаций на потомка: ").strip())
                    if val > 0:
                        config.mutation_count = val
                    val = int(input("Размер популяции: ").strip())
                    if val > 0:
                        config.population_size = val
                        engine.initialize(len(engine.data_pairs[0][0]) if engine.data_pairs else 1)
                    config.save()
                    print("Настройки сохранены")
                except ValueError:
                    print("Некорректное значение")
        
        elif choice == '9':
            print("\nДо свидания!")
            break
        
        else:
            print("Неверный выбор")

if __name__ == "__main__":
    main_menu()
