#!/usr/bin/env python3
"""
Sigmauadro - Универсальный эволюционный аппроксиматор
Основан на чистых мутациях без кроссовера
Эволюция собирает универсальное решение из минимальных кирпичиков
"""

import random
import math
import copy
import sys
import os
from decimal import Decimal, getcontext

getcontext().prec = 50

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

class Config:
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
        with open(filename, 'w') as f:
            f.write("# Настройки Sigmauadro\n")
            for key in self.DEFAULTS:
                val = getattr(self, key)
                if isinstance(val, (int, float)):
                    f.write(f"{key} = {val}\n")
        return self

# ============================================================================
# ТИПЫ УЗЛОВ (ENUM)
# ============================================================================

NODE_CONST = 0      # Константа со значением
NODE_PRIM = 1       # Базовый примитив
NODE_INPUT = 2      # Входная переменная

# ============================================================================
# МИНИМАЛЬНЫЕ КИРПИЧИКИ (БАЗОВЫЕ ОПЕРАЦИИ)
# ============================================================================

class Primitive:
    """Базовая операция - кирпичик для построения функций"""
    def __init__(self, name, func, arity, code):
        self.name = name
        self.func = func
        self.arity = arity
        self.code = code
    
    def execute(self, args):
        try:
            return self.func(*args)
        except (ZeroDivisionError, ValueError, OverflowError):
            return Decimal('0')

# Базовые примитивы (неизменяемые)
BASE_PRIMITIVES = [
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

BASE_PRIMITIVE_COUNT = len(BASE_PRIMITIVES)

def get_primitive(idx):
    """Получить базовый примитив по индексу"""
    if 0 <= idx < BASE_PRIMITIVE_COUNT:
        return BASE_PRIMITIVES[idx]
    return None

# ============================================================================
# ОСОБЬ (ГЕНЕТИЧЕСКОЕ ПРЕДСТАВЛЕНИЕ)
# ============================================================================

class Individual:
    """Особь представляющая функцию в виде дерева операций
    
    Геном состоит из узлов трех типов:
    - NODE_CONST: (NODE_CONST, значение) - константа
    - NODE_PRIM: (NODE_PRIM, индекс_примитива) - базовая операция
    - NODE_INPUT: (NODE_INPUT, индекс_входа) - входная переменная
    """
    
    def __init__(self, genome=None, input_size=1, output_size=1):
        self.genome = genome or []
        self.input_size = input_size
        self.output_size = output_size
        self.error = None
        self._complexity = len(self.genome) if genome else 0
    
    @property
    def complexity(self):
        """Кэшированная сложность особи"""
        return self._complexity
    
    def _update_complexity(self):
        """Обновить сложность после изменения генома"""
        self._complexity = len(self.genome)
    
    def execute(self, inputs):
        """Выполнить особь на входных данных используя стек
        
        Возвращает список выходов размером output_size
        """
        if not self.genome:
            return [Decimal('0')] * self.output_size
        
        try:
            stack = list(inputs)
            
            for node in self.genome:
                node_type = node[0]
                
                if node_type == NODE_CONST:
                    stack.append(node[1])
                elif node_type == NODE_INPUT:
                    idx = node[1]
                    stack.append(inputs[idx] if 0 <= idx < len(inputs) else Decimal('0'))
                elif node_type == NODE_PRIM:
                    prim_idx = node[1]
                    prim = get_primitive(prim_idx)
                    if prim is None:
                        continue
                    args = []
                    for _ in range(prim.arity):
                        args.append(stack.pop() if stack else Decimal('0'))
                    stack.append(prim.execute(args))
            
            # Вернуть последние output_size значений
            result = stack[-self.output_size:] if len(stack) >= self.output_size else stack
            # Дополнить нулями если нужно
            while len(result) < self.output_size:
                result.append(Decimal('0'))
            return result
            
        except Exception:
            return [Decimal('0')] * self.output_size
    
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
        """Применить мутации к особи - каждая мутация минимальна"""
        new_genome = copy.deepcopy(self.genome)
        
        for _ in range(mutation_count):
            if not new_genome:
                node_type = random.choice([NODE_PRIM, NODE_INPUT, NODE_CONST])
                if node_type == NODE_PRIM:
                    new_genome.append((NODE_PRIM, random.randint(0, BASE_PRIMITIVE_COUNT - 1)))
                elif node_type == NODE_INPUT:
                    new_genome.append((NODE_INPUT, random.randint(0, max(0, self.input_size - 1))))
                else:
                    val = Decimal(str(random.uniform(-100, 100)))
                    new_genome.append((NODE_CONST, val))
            else:
                # Увеличить вероятность add для роста сложности
                weights = [0.35, 0.35, 0.15, 0.15]  # replace, add, remove, tweak_const
                mut_type = random.choices(['replace', 'add', 'remove', 'tweak_const'], weights=weights)[0]
                
                if mut_type == 'replace':
                    idx = random.randint(0, len(new_genome) - 1)
                    node_type = random.choice([NODE_PRIM, NODE_INPUT, NODE_CONST])
                    if node_type == NODE_PRIM:
                        new_genome[idx] = (NODE_PRIM, random.randint(0, BASE_PRIMITIVE_COUNT - 1))
                    elif node_type == NODE_INPUT:
                        new_genome[idx] = (NODE_INPUT, random.randint(0, max(0, self.input_size - 1)))
                    else:
                        val = Decimal(str(random.uniform(-100, 100)))
                        new_genome[idx] = (NODE_CONST, val)
                
                elif mut_type == 'add':
                    pos = random.randint(0, len(new_genome))
                    node_type = random.choice([NODE_PRIM, NODE_INPUT, NODE_CONST])
                    if node_type == NODE_PRIM:
                        new_genome.insert(pos, (NODE_PRIM, random.randint(0, BASE_PRIMITIVE_COUNT - 1)))
                    elif node_type == NODE_INPUT:
                        new_genome.insert(pos, (NODE_INPUT, random.randint(0, max(0, self.input_size - 1))))
                    else:
                        val = Decimal(str(random.uniform(-100, 100)))
                        new_genome.insert(pos, (NODE_CONST, val))
                
                elif mut_type == 'remove' and len(new_genome) > 1:
                    idx = random.randint(0, len(new_genome) - 1)
                    new_genome.pop(idx)
                
                elif mut_type == 'tweak_const':
                    const_indices = [i for i, n in enumerate(new_genome) if n[0] == NODE_CONST]
                    if const_indices:
                        idx = random.choice(const_indices)
                        current_val = new_genome[idx][1]
                        # Адаптивный размер твика - больше для больших значений
                        scale = max(abs(current_val), Decimal('1')) * Decimal('1e-10')
                        tweak = Decimal(str(random.uniform(-float(scale), float(scale))))
                        new_val = current_val + tweak
                        new_genome[idx] = (NODE_CONST, new_val)
        
        ind = Individual(new_genome, self.input_size, self.output_size)
        ind._update_complexity()
        return ind
    
    def to_readable(self):
        """Преобразовать геном в читаемое выражение с поддержкой multi-input/output"""
        if not self.genome:
            return "f(" + ", ".join(f"x{i}" for i in range(self.input_size)) + ") = 0"
        
        try:
            stack = []
            for node in self.genome:
                node_type = node[0]
                
                if node_type == NODE_CONST:
                    val = node[1]
                    # Форматировать константу красиво
                    if val == int(val):
                        stack.append(str(int(val)))
                    else:
                        stack.append(str(val))
                elif node_type == NODE_INPUT:
                    idx = node[1]
                    if self.input_size == 1:
                        stack.append("x")
                    else:
                        stack.append(f"x{idx}")
                elif node_type == NODE_PRIM:
                    prim = get_primitive(node[1])
                    if prim is None:
                        continue
                    args = []
                    for _ in range(prim.arity):
                        if stack:
                            args.append(stack.pop())
                        else:
                            args.append("x")
                    
                    if prim.arity == 0:
                        expr = prim.code
                    elif prim.arity == 1:
                        expr = f"{prim.code}({args[0]})"
                    else:
                        expr = f"({args[1]} {prim.code} {args[0]})"
                    stack.append(expr)
            
            if stack:
                if self.input_size == 1:
                    return f"f(x) = {stack[-1]}"
                else:
                    inputs = ", ".join(f"x{i}" for i in range(self.input_size))
                    return f"f({inputs}) = {stack[-1]}"
        except Exception:
            pass
        
        return "f(x) = 0"
    
    def describe(self):
        """Подробное описание что делает особь (для ИИ и человека)"""
        lines = []
        lines.append("=" * 60)
        lines.append("ОПИСАНИЕ ОСОБИ")
        lines.append("=" * 60)
        
        if not self.genome:
            lines.append("Пустая функция - всегда возвращает 0")
            return "\n".join(lines)
        
        lines.append(f"Входов: {self.input_size}")
        lines.append(f"Выходов: {self.output_size}")
        lines.append(f"Сложность (кол-во операций): {self.complexity}")
        lines.append(f"Ошибка на обучающих данных: {self.error}")
        lines.append("")
        
        # Построить структуру функции симулируя выполнение
        lines.append("СТРУКТУРА ФУНКЦИИ:")
        lines.append("-" * 40)
        
        try:
            stack = []
            step = 1
            input_counter = 0
            for node in self.genome:
                node_type = node[0]
                
                if node_type == NODE_CONST:
                    val = node[1]
                    val_str = str(int(val)) if val == int(val) else str(val)
                    stack.append(val_str)
                    lines.append(f"  [{step}] Константа: {val_str}")
                elif node_type == NODE_INPUT:
                    idx = node[1]
                    sym = "x" if self.input_size == 1 else f"x{idx}"
                    stack.append(sym)
                    lines.append(f"  [{step}] Вход: {sym}")
                elif node_type == NODE_PRIM:
                    prim = get_primitive(node[1])
                    if prim is None:
                        continue
                    args = []
                    for _ in range(prim.arity):
                        if stack:
                            args.append(stack.pop())
                        else:
                            # Если стек пуст, использовать заглушку
                            args.append(f"a{len(args)}")
                    
                    if prim.arity == 1:
                        op_name = {"neg": "-", "abs": "abs", "sqrt": "√", "sin": "sin", "cos": "cos", "exp": "exp", "log": "ln"}.get(prim.name, prim.code)
                        result = f"{op_name}({args[0]})"
                        lines.append(f"  [{step}] {op_name}({args[0]})")
                    elif prim.arity == 2:
                        result = f"({args[1]} {prim.code} {args[0]})"
                        lines.append(f"  [{step}] {args[1]} {prim.code} {args[0]}")
                    else:
                        result = prim.code
                        lines.append(f"  [{step}] {prim.code}")
                    stack.append(result)
                step += 1
            
            if stack:
                lines.append("")
                final_expr = stack[-1]
                if self.input_size == 1:
                    lines.append(f"ФОРМУЛА: f(x) = {final_expr}")
                else:
                    inputs = ", ".join(f"x{i}" for i in range(self.input_size))
                    lines.append(f"ФОРМУЛА: f({inputs}) = {final_expr}")
        except Exception as e:
            lines.append(f"Ошибка: {e}")
        
        lines.append("")
        lines.append("ГЕНОМ (машинное представление):")
        lines.append("-" * 40)
        type_names = {NODE_CONST: "CONST", NODE_PRIM: "PRIM", NODE_INPUT: "INPUT"}
        for i, node in enumerate(self.genome):
            node_type, val = node
            tname = type_names.get(node_type, "???")
            if node_type == NODE_PRIM:
                prim = get_primitive(val)
                vdesc = f"{prim.name} ({prim.code})" if prim else val
            elif node_type == NODE_INPUT:
                vdesc = f"x{val}"
            else:
                vdesc = val
            lines.append(f"  #{i}: {tname} = {vdesc}")
        
        return "\n".join(lines)
    
    def save(self, filename):
        """Сохранить особь в читаемый файл (UTF-8 для поддержки символов)"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("# Sigmauadro - Лучшая особь\n")
                f.write(f"# {self.to_readable()}\n\n")
                f.write(f"input_size: {self.input_size}\n")
                f.write(f"output_size: {self.output_size}\n")
                err_str = str(self.error) if self.error is not None else "inf"
                f.write(f"error: {err_str}\n")
                f.write(f"complexity: {self.complexity}\n\n")
                
                f.write("=" * 60 + "\n")
                f.write("ПОДРОБНОЕ ОПИСАНИЕ\n")
                f.write("=" * 60 + "\n")
                f.write(self.describe())
                f.write("\n\n")
                
                f.write("# ГЕНОМ (для загрузки программой)\n")
                f.write("# Формат: тип_узла значение\n")
                f.write("# Типы узлов: 0=константа, 1=примитив, 2=вход\n")
                f.write("# Примитивы: 0=add(+), 1=sub(-), 2=mul(*), 3=(/), 4=neg(-), 5=abs, 6=sqrt, 7=sin, 8=cos, 9=exp, 10=log, 11=pow(^)\n")
                for node in self.genome:
                    f.write(f"{node[0]} {node[1]}\n")
        except Exception as e:
            print(f"Ошибка сохранения: {e}")

    
    @classmethod
    def load(cls, filename):
        """Загрузить особь из файла"""
        genome = []
        input_size = 1
        output_size = 1
        
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('input_size:'):
                    input_size = int(line.split(':')[1].strip())
                elif line.startswith('output_size:'):
                    output_size = int(line.split(':')[1].strip())
                elif line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            node_type = int(parts[0])
                            val = parts[1]
                            if node_type == NODE_CONST:
                                val = Decimal(val)
                            else:
                                val = int(val)
                            genome.append((node_type, val))
                        except ValueError:
                            pass
        
        individual = cls(genome, input_size, output_size)
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
    
    def initialize(self, input_size, output_size=1):
        """Инициализировать внутреннюю популяцию с нуля"""
        self.inner_population = []
        for _ in range(self.config.population_size):
            individual = Individual([], input_size, output_size)
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
            # Определить размерности входа и выхода
            input_size = len(self.data_pairs[0][0])
            output_size = len(self.data_pairs[0][1])
            # Обновить особи
            for ind in self.inner_population:
                ind.input_size = input_size
                ind.output_size = output_size
            if self.best_individual:
                self.best_individual.input_size = input_size
                self.best_individual.output_size = output_size
            # Инициализировать популяцию если пуста
            if not self.inner_population:
                self.initialize(input_size, output_size)
        
        return len(self.data_pairs) > 0
    
    def evolve_generation(self, simplify_mode=False):
        """Провести одно поколение эволюции
        
        simplify_mode: если True, эволюция направлена на упрощение особи
                       Приоритет на снижение сложности, ошибка вторична
                       
        Логика режима упрощения (СИТУАЦИЯ 2 из ТЗ):
        - Если особь проще - она более приспособлена (даже если ошибка хуже)
        - Если особь сложнее - она НЕ приспособлена (даже если ошибка лучше)
        - При равной сложности - сравниваем по ошибке
        """
        if not self.inner_population or not self.data_pairs:
            return
        
        for i, parent in enumerate(self.inner_population):
            for _ in range(self.config.offspring_per_parent):
                offspring = parent.mutate(self.config.mutation_count)
                offspring.evaluate(self.data_pairs)
                
                replace_parent = False
                
                if simplify_mode:
                    # РЕЖИМ УПРОЩЕНИЯ: главный критерий - СЛОЖНОСТЬ
                    if offspring.complexity < parent.complexity:
                        # Потомок проще - принимаем (ошибка может быть любой)
                        replace_parent = True
                    elif offspring.complexity == parent.complexity:
                        # Сложность равна - сравниваем по ошибке
                        if offspring.error < parent.error:
                            replace_parent = True
                    # Если потомок сложнее - НЕ принимаем (даже если ошибка лучше)
                else:
                    # Обычный режим: приоритет на ошибку, затем сложность
                    if offspring.error < parent.error:
                        replace_parent = True
                    elif offspring.error == parent.error and offspring.complexity < parent.complexity:
                        replace_parent = True
                
                if replace_parent:
                    self.inner_population[i] = offspring
                    parent = offspring
        
        current_best = min(self.inner_population, key=lambda x: x.error if x.error is not None else Decimal('inf'))
        prev_best_error = self.best_individual.error if self.best_individual else Decimal('inf')
        prev_best_complexity = self.best_individual.complexity if self.best_individual else float('inf')
        
        if simplify_mode:
            # В режиме упрощения: приоритет на сложность при равной ошибке
            if current_best.error < prev_best_error:
                self.best_individual = copy.deepcopy(current_best)
                self.generations_without_improvement = 0
            elif current_best.error == prev_best_error and current_best.complexity < prev_best_complexity:
                self.best_individual = copy.deepcopy(current_best)
                self.generations_without_improvement = 0
            # ВАЖНО: в режиме упрощения счетчик без улучшений НЕ сбрасывается если сложность уменьшилась
            # но ошибка осталась той же - это считается улучшением
        else:
            # Обычный режим: приоритет на ошибку, затем сложность
            if current_best.error < prev_best_error:
                self.best_individual = copy.deepcopy(current_best)
                self.generations_without_improvement = 0
            elif current_best.error == prev_best_error and current_best.complexity < prev_best_complexity:
                self.best_individual = copy.deepcopy(current_best)
                self.generations_without_improvement = 0
        
        self.generations_run += 1
        # Сбрасываем счетчик только если действительно было улучшение
        if simplify_mode:
            # В режиме упрощения считаем улучшением и снижение сложности при равной ошибке
            if current_best.error < prev_best_error or (current_best.error == prev_best_error and current_best.complexity < prev_best_complexity):
                self.generations_without_improvement = 0
            else:
                self.generations_without_improvement += 1
        else:
            if current_best.error < prev_best_error or (current_best.error == prev_best_error and current_best.complexity < prev_best_complexity):
                self.generations_without_improvement = 0
            else:
                self.generations_without_improvement += 1
        
        self.best_error_history.append(self.best_individual.error if self.best_individual else Decimal('inf'))
    
    def run_evolution(self, generations, display=True, simplify_mode=False):
        """Запустить эволюцию на заданное количество поколений
        
        simplify_mode: если True, запустить режим эволюции на упрощение особи
        """
        for gen in range(generations):
            self.evolve_generation(simplify_mode=simplify_mode)
            
            if display:
                # Отобразить прогресс в столбик с полной точностью ошибки
                best_err = self.best_individual.error if self.best_individual else Decimal('inf')
                complexity = self.best_individual.complexity if self.best_individual else 0
                
                mode_prefix = "[УПРОЩЕНИЕ] " if simplify_mode else ""
                sys.stdout.write(f"\r{mode_prefix}Поколение {gen + 1}/{generations}\n")
                sys.stdout.write(f"{' ' * len(mode_prefix)}Ошибка: {best_err}\n")
                sys.stdout.write(f"{' ' * len(mode_prefix)}Без улучшений: {self.generations_without_improvement}\n")
                sys.stdout.write(f"{' ' * len(mode_prefix)}Сложность: {complexity}")
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
        try:
            with open(filename, 'w') as f:
                f.write(f"# Популяция Sigmauadro\n")
                f.write(f"# Особей: {len(self.inner_population)}\n\n")
                for i, ind in enumerate(self.inner_population):
                    f.write(f"\n## Особь {i+1}\n")
                    f.write(f"input_size: {ind.input_size}\n")
                    f.write(f"output_size: {ind.output_size}\n")
                    err_str = str(ind.error) if ind.error is not None else "inf"
                    f.write(f"error: {err_str}\n")
                    f.write(f"complexity: {ind.complexity}\n")
                    for node in ind.genome:
                        f.write(f"{node[0]} {node[1]}\n")
            print(f"Популяция сохранена в {filename}")
        except Exception as e:
            print(f"Ошибка сохранения популяции: {e}")
    
    def load_population(self, filename):
        """Загрузить популяцию из файла"""
        if not os.path.exists(filename):
            print(f"Файл {filename} не найден")
            return False
        
        population = []
        current_genome = []
        current_input_size = 1
        current_output_size = 1
        
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('input_size:'):
                    current_input_size = int(line.split(':')[1].strip())
                elif line.startswith('output_size:'):
                    current_output_size = int(line.split(':')[1].strip())
                elif line and not line.startswith('#') and not line.startswith('##'):
                    if ':' not in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                node_type = int(parts[0])
                                val = parts[1]
                                if node_type == NODE_CONST:
                                    val = Decimal(val)
                                else:
                                    val = int(val)
                                current_genome.append((node_type, val))
                            except ValueError:
                                pass
                    elif line.startswith('error:') or line.startswith('complexity:'):
                        if current_genome:
                            ind = Individual(current_genome, current_input_size, current_output_size)
                            ind.evaluate(self.data_pairs)
                            population.append(ind)
                            current_genome = []
        
        if current_genome:
            ind = Individual(current_genome, current_input_size, current_output_size)
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
                # Спросить тип эволюции
                print("\nВыберите режим эволюции:")
                print("1. Обычная эволюция")
                print("2. Эволюция на упрощение особи")
                
                # Гарантированная валидация выбора режима
                simplify_mode = False
                while True:
                    mode_choice = input("Ваш выбор (1 или 2): ").strip()
                    if mode_choice == '1':
                        simplify_mode = False
                        break
                    elif mode_choice == '2':
                        simplify_mode = True
                        break
                    print("Неверный выбор. Введите 1 или 2.")
                
                # Запрос количества поколений
                gens_input = input("Сколько поколений эволюции прогонять? ").strip()
                try:
                    gens = int(gens_input)
                except ValueError:
                    print("Некорректное число поколений")
                    continue
                
                if gens > 0:
                    print("\nЗапуск эволюции...")
                    engine.run_evolution(gens, simplify_mode=simplify_mode)
                    print(f"\nЭволюция завершена. Пройдено поколений: {engine.generations_run}")
                    if engine.best_individual:
                        print(f"Лучшая ошибка: {engine.best_individual.error}")
                        print(f"Лучшая особь: {engine.best_individual.to_readable()}")
                else:
                    print("Количество поколений должно быть положительным")
            except EOFError:
                print("\nПрервано пользователем")
            except KeyboardInterrupt:
                print("\nПрервано пользователем")
        
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
                        # Переинициализировать популяцию с новыми параметрами
                        input_size = len(engine.data_pairs[0][0]) if engine.data_pairs else 1
                        output_size = len(engine.data_pairs[0][1]) if engine.data_pairs else 1
                        engine.initialize(input_size, output_size)
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
