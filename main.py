"""
Главная точка входа. Консольное меню управления аппроксиматором.
"""

import sys
import os

# Добавляем корневую директорию в path для импорта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from approximator import UniversalApproximator
from utils.visualization import print_training_history, print_ascii_chart, print_statistics


def print_menu():
    """Вывод главного меню."""
    print("\n" + "=" * 50)
    print("ЭВОЛЮЦИОНИРУЮЩИЙ УНИВЕРСАЛЬНЫЙ АППРОКСИМАТОР")
    print("=" * 50)
    print("1. Загрузить данные из файла")
    print("2. Настроить параметры эволюции")
    print("3. Запустить обучение")
    print("4. Просмотреть лучшую найденную формулу")
    print("5. Сделать предсказание для новых данных")
    print("6. Сохранить модель")
    print("7. Загрузить сохраненную модель")
    print("8. Показать статистику обучения")
    print("9. Выход")
    print("=" * 50)


def load_data(approximator: UniversalApproximator):
    """Пункт 1: Загрузка данных."""
    filepath = input("Введите путь к файлу с данными: ").strip()
    
    if not filepath:
        print("Ошибка: путь не указан")
        return
    
    try:
        approximator.load_data(filepath)
        print(f"\nДанные успешно загружены!")
        print(f"Количество пар: {len(approximator.dataset)}")
        print(f"Размерность входов: {approximator.dataset.input_dim}")
        print(f"Размерность выходов: {approximator.dataset.output_dim}")
    except FileNotFoundError as e:
        print(f"\nОшибка: {e}")
    except ValueError as e:
        print(f"\nОшибка формата данных: {e}")
    except Exception as e:
        print(f"\nНеизвестная ошибка: {e}")


def configure_params(approximator: UniversalApproximator):
    """Пункт 2: Настройка параметров эволюции."""
    print("\nНастройка параметров эволюции (Enter для значения по умолчанию):")
    
    try:
        pop_size = input(f"Размер популяции (по умолчанию {approximator.pop_size}): ").strip()
        if pop_size:
            approximator.pop_size = int(pop_size)
        
        generations = input(f"Количество поколений (по умолчанию {approximator.generations}): ").strip()
        if generations:
            approximator.generations = int(generations)
        
        mutation_rate = input(f"Вероятность мутации (по умолчанию {approximator.mutation_rate}): ").strip()
        if mutation_rate:
            approximator.mutation_rate = float(mutation_rate)
        
        crossover_rate = input(f"Вероятность кроссовера (по умолчанию {approximator.crossover_rate}): ").strip()
        if crossover_rate:
            approximator.crossover_rate = float(crossover_rate)
        
        elite_count = input(f"Количество элитных особей (по умолчанию {approximator.elite_count}): ").strip()
        if elite_count:
            approximator.elite_count = int(elite_count)
        
        max_depth = input(f"Максимальная глубина дерева (по умолчанию {approximator.max_depth}): ").strip()
        if max_depth:
            approximator.max_depth = int(max_depth)
        
        print("\nПараметры обновлены:")
        print(f"  Размер популяции: {approximator.pop_size}")
        print(f"  Количество поколений: {approximator.generations}")
        print(f"  Вероятность мутации: {approximator.mutation_rate}")
        print(f"  Вероятность кроссовера: {approximator.crossover_rate}")
        print(f"  Количество элитных особей: {approximator.elite_count}")
        print(f"  Максимальная глубина: {approximator.max_depth}")
        
    except ValueError as e:
        print(f"\nОшибка: неверный формат числа - {e}")


def train_model(approximator: UniversalApproximator):
    """Пункт 3: Запуск обучения."""
    if approximator.dataset is None:
        print("\nОшибка: сначала загрузите данные (пункт 1)")
        return
    
    print("\nЗапуск обучения...")
    
    try:
        approximator.train(verbose=True)
    except Exception as e:
        print(f"\nОшибка при обучении: {e}")


def view_formula(approximator: UniversalApproximator):
    """Пункт 4: Просмотр формулы."""
    if approximator.best_individual is None:
        print("\nМодель не обучена. Сначала запустите обучение (пункт 3)")
        return
    
    print("\n" + "=" * 50)
    print("ЛУЧШАЯ НАЙДЕННАЯ ФОРМУЛА")
    print("=" * 50)
    print(approximator.get_best_formula())
    print("=" * 50)


def make_prediction(approximator: UniversalApproximator):
    """Пункт 5: Предсказание."""
    if approximator.best_individual is None:
        print("\nМодель не обучена. Сначала запустите обучение (пункт 3)")
        return
    
    print(f"\nВведите {approximator.dataset.input_dim} значений через пробел:")
    inputs_str = input("> ").strip()
    
    try:
        inputs = [float(x) for x in inputs_str.split()]
        
        if len(inputs) != approximator.dataset.input_dim:
            print(f"Ошибка: ожидается {approximator.dataset.input_dim} значений, получено {len(inputs)}")
            return
        
        prediction = approximator.predict(inputs)
        print(f"\nПредсказание: {prediction[0]:.6f}")
        
    except ValueError as e:
        print(f"Ошибка: неверный формат ввода - {e}")
    except Exception as e:
        print(f"Ошибка при предсказании: {e}")


def save_model(approximator: UniversalApproximator):
    """Пункт 6: Сохранение модели."""
    if approximator.best_individual is None:
        print("\nНет модели для сохранения. Сначала обучите модель")
        return
    
    filepath = input("Введите имя файла для сохранения: ").strip()
    
    if not filepath:
        print("Ошибка: имя файла не указано")
        return
    
    # Добавляем расширение если нет
    if not filepath.endswith('.pkl'):
        filepath += '.pkl'
    
    # Сохраняем в папку models
    if not os.path.isabs(filepath):
        filepath = os.path.join('models', filepath)
    
    try:
        approximator.save_model(filepath)
        print(f"\nМодель успешно сохранена в {filepath}")
    except Exception as e:
        print(f"\nОшибка при сохранении: {e}")


def load_model(approximator: UniversalApproximator):
    """Пункт 7: Загрузка модели."""
    filepath = input("Введите имя файла модели: ").strip()
    
    if not filepath:
        print("Ошибка: имя файла не указано")
        return
    
    # Добавляем расширение если нет
    if not filepath.endswith('.pkl'):
        filepath += '.pkl'
    
    # Проверяем в папке models
    if not os.path.isabs(filepath):
        models_path = os.path.join('models', filepath)
        if os.path.exists(models_path):
            filepath = models_path
    
    try:
        approximator.load_model(filepath)
        print(f"\nМодель успешно загружена из {filepath}")
        print(f"Размерность входов: {approximator.dataset.input_dim}")
        print(f"Размерность выходов: {approximator.dataset.output_dim}")
        print(f"Лучшая fitness: {approximator.best_fitness:.6f}")
    except FileNotFoundError as e:
        print(f"\nОшибка: файл не найден - {e}")
    except Exception as e:
        print(f"\nОшибка при загрузке: {e}")


def show_statistics(approximator: UniversalApproximator):
    """Пункт 8: Статистика обучения."""
    if len(approximator.generation_history) == 0:
        print("\nНет статистики. Сначала запустите обучение (пункт 3)")
        return
    
    stats = approximator.get_training_stats()
    print_statistics(stats)
    
    # Показываем график
    choice = input("\nПоказать ASCII-график? (y/n): ").strip().lower()
    if choice == 'y':
        print_ascii_chart(approximator.generation_history)
    
    # Показываем историю
    choice = input("Показать историю по поколениям? (y/n): ").strip().lower()
    if choice == 'y':
        print_training_history(approximator.generation_history, interval=10)


def main():
    """Главная функция."""
    approximator = UniversalApproximator()
    
    print("\nДобро пожаловать в Эволюционирующий Универсальный Аппроксиматор!")
    
    while True:
        print_menu()
        
        choice = input("Выберите пункт меню (1-9): ").strip()
        
        if choice == '1':
            load_data(approximator)
        elif choice == '2':
            configure_params(approximator)
        elif choice == '3':
            train_model(approximator)
        elif choice == '4':
            view_formula(approximator)
        elif choice == '5':
            make_prediction(approximator)
        elif choice == '6':
            save_model(approximator)
        elif choice == '7':
            load_model(approximator)
        elif choice == '8':
            show_statistics(approximator)
        elif choice == '9':
            print("\nДо свидания!")
            break
        else:
            print("\nНеверный выбор. Пожалуйста, выберите от 1 до 9.")


if __name__ == "__main__":
    main()
