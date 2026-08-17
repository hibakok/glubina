"""
Демонстрационный скрипт для аппроксиматора.
Создает тестовые данные, обучает модель и сравнивает с исходной функцией.
"""

import sys
import os
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from approximator import UniversalApproximator


def create_test_data(filepath, func_str="x^2 + y"):
    """Создание тестовых данных для известной функции."""
    
    if func_str == "x^2 + y":
        # Функция f(x, y) = x^2 + y
        data = [
            (1.0, 2.0, 3.0),
            (2.0, 3.0, 7.0),
            (3.0, 4.0, 13.0),
            (0.5, 0.5, 0.75),
            (-1.0, 2.0, 3.0),
            (1.5, 1.5, 3.75),
            (2.5, 2.0, 8.25),
            (0.0, 1.0, 1.0),
            (2.0, 1.0, 5.0),
            (3.0, 2.0, 11.0),
        ]
    elif func_str == "2x + 1":
        # Функция f(x) = 2x + 1
        data = [(x, 0, 2*x + 1) for x in np.linspace(-5, 5, 11)]
    elif func_str == "sin(x)":
        # Функция f(x) = sin(x)
        data = [(x, 0, float(np.sin(x))) for x in np.linspace(-np.pi, np.pi, 20)]
    else:
        raise ValueError(f"Неизвестная функция: {func_str}")
    
    with open(filepath, 'w') as f:
        f.write(f"# Тестовые данные для функции: {func_str}\n")
        for row in data:
            if row[1] == 0 and func_str != "x^2 + y":
                f.write(f"{row[0]} | {row[2]}\n")
            else:
                f.write(f"{row[0]} {row[1]} | {row[2]}\n")
    
    return data


def evaluate_function(func_str, *args):
    """Вычисление значения исходной функции."""
    if func_str == "x^2 + y":
        return args[0]**2 + args[1]
    elif func_str == "2x + 1":
        return 2*args[0] + 1
    elif func_str == "sin(x)":
        return float(np.sin(args[0]))
    else:
        return None


def main():
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ ЭВОЛЮЦИОНИРУЮЩЕГО УНИВЕРСАЛЬНОГО АППРОКСИМАТОРА")
    print("=" * 70)
    
    # Создаем временный файл с данными
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    temp_file.close()
    
    try:
        # Выбор функции для тестирования
        print("\nВыберите функцию для аппроксимации:")
        print("1. f(x, y) = x^2 + y")
        print("2. f(x) = 2x + 1")
        print("3. f(x) = sin(x)")
        
        choice = input("\nВаш выбор (1-3): ").strip()
        
        if choice == "1":
            func_str = "x^2 + y"
            input_dim = 2
        elif choice == "2":
            func_str = "2x + 1"
            input_dim = 1
        elif choice == "3":
            func_str = "sin(x)"
            input_dim = 1
        else:
            func_str = "x^2 + y"
            input_dim = 2
        
        print(f"\n[1] Создание тестовых данных для функции: {func_str}")
        data = create_test_data(temp_file.name, func_str)
        print(f"    Создано {len(data)} пар данных")
        
        print(f"\n[2] Загрузка данных в аппроксиматор")
        approx = UniversalApproximator()
        approx.load_data(temp_file.name)
        print(f"    Загружено {len(approx.dataset)} пар")
        print(f"    Размерность входа: {approx.dataset.input_dim}")
        print(f"    Размерность выхода: {approx.dataset.output_dim}")
        
        print(f"\n[3] Настройка параметров эволюции")
        pop_size = 80
        generations = 150
        mutation_rate = 0.12
        crossover_rate = 0.75
        max_depth = 6
        
        approx.set_params(
            pop_size=pop_size,
            generations=generations,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            max_depth=max_depth
        )
        print(f"    Размер популяции: {pop_size}")
        print(f"    Количество поколений: {generations}")
        print(f"    Вероятность мутации: {mutation_rate}")
        print(f"    Вероятность кроссовера: {crossover_rate}")
        print(f"    Максимальная глубина: {max_depth}")
        
        print(f"\n[4] Обучение модели...")
        approx.train(verbose=True)
        
        print(f"\n[5] Сравнение с исходной функцией")
        print("-" * 70)
        print(f"{'Вход':<20} {'Ожидалось':<15} {'Предсказано':<15} {'Ошибка':<15}")
        print("-" * 70)
        
        total_error = 0
        test_points = []
        
        if func_str == "x^2 + y":
            test_points = [(1.0, 2.0), (2.0, 3.0), (0.5, 0.5), (2.5, 1.5)]
        elif func_str == "2x + 1":
            test_points = [(x,) for x in [1.0, 2.5, 4.0, -1.5]]
        elif func_str == "sin(x)":
            test_points = [(x,) for x in [0.0, np.pi/4, np.pi/2, np.pi]]
        
        for point in test_points:
            expected = evaluate_function(func_str, *point)
            predicted = approx.predict(list(point))[0]
            error = abs(predicted - expected)
            total_error += error
            
            point_str = str(point) if len(point) > 1 else f"({point[0]},)"
            print(f"{point_str:<20} {expected:<15.6f} {predicted:<15.6f} {error:<15.6f}")
        
        avg_error = total_error / len(test_points)
        print("-" * 70)
        print(f"Средняя ошибка: {avg_error:.6f}")
        
        print(f"\n[6] Найденная формула:")
        print(f"    {approx.get_best_formula()}")
        
        print(f"\n[7] Статистика обучения:")
        stats = approx.get_training_stats()
        print(f"    Время обучения: {stats['training_time']:.2f} сек")
        print(f"    Поколений: {stats['total_generations']}")
        print(f"    Начальная fitness: {stats['initial_fitness']:.6f}")
        print(f"    Конечная fitness: {stats['final_fitness']:.6f}")
        print(f"    Улучшений: {stats['improvements']}")
        
        print("\n" + "=" * 70)
        print("ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
        print("=" * 70)
        
    finally:
        # Удаляем временный файл
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


if __name__ == "__main__":
    main()
