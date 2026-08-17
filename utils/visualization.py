"""
Утилиты для визуализации статистики обучения.
"""

from typing import List, Tuple


def print_training_history(history: List[Tuple[float, float]], 
                           interval: int = 10) -> None:
    """
    Вывод истории обучения в текстовом виде.
    
    Args:
        history: Список кортежей (best_fitness, avg_fitness) по поколениям
        interval: Интервал вывода (какие поколения показывать)
    """
    print("\n" + "=" * 70)
    print("ИСТОРИЯ ОБУЧЕНИЯ")
    print("=" * 70)
    print(f"{'Поколение':<12} {'Лучшая fitness':<18} {'Средняя fitness':<18}")
    print("-" * 70)
    
    for i, (best, avg) in enumerate(history):
        if i % interval == 0 or i == len(history) - 1:
            print(f"{i:<12} {best:<18.6f} {avg:<18.6f}")
    
    print("=" * 70)


def print_ascii_chart(history: List[Tuple[float, float]], 
                      width: int = 60, height: int = 15) -> None:
    """
    Создание ASCII-графика изменения fitness по поколениям.
    
    Args:
        history: Список кортежей (best_fitness, avg_fitness)
        width: Ширина графика в символах
        height: Высота графика в строках
    """
    if len(history) == 0:
        print("Нет данных для отображения")
        return
    
    # Извлекаем значения fitness
    best_values = [h[0] for h in history]
    avg_values = [h[1] for h in history]
    
    # Находим диапазон значений
    all_values = best_values + avg_values
    min_val = min(all_values)
    max_val = max(all_values)
    
    # Добавляем небольшой отступ
    value_range = max_val - min_val
    if value_range == 0:
        value_range = 1
        min_val -= 0.5
        max_val += 0.5
    
    # Создаем сетку графика
    chart = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Рисуем линии для best и avg
    for gen in range(len(history)):
        # Нормализуем позицию по x
        x = int((gen / max(1, len(history) - 1)) * (width - 1))
        
        # Нормализуем позицию best по y
        best_y_norm = (best_values[gen] - min_val) / value_range
        best_y = int((1 - best_y_norm) * (height - 1))
        best_y = max(0, min(height - 1, best_y))
        
        # Нормализуем позицию avg по y
        avg_y_norm = (avg_values[gen] - min_val) / value_range
        avg_y = int((1 - avg_y_norm) * (height - 1))
        avg_y = max(0, min(height - 1, avg_y))
        
        # Рисуем точки
        if 0 <= best_y < height and 0 <= x < width:
            chart[best_y][x] = '*'
        if 0 <= avg_y < height and 0 <= x < width:
            if chart[avg_y][x] == '*':
                chart[avg_y][x] = 'X'  # Пересечение
            else:
                chart[avg_y][x] = 'o'
    
    # Выводим график
    print("\n" + "=" * (width + 15))
    print("ГРАФИК FITNESS ПО ПОКОЛЕНИЯМ")
    print("=" * (width + 15))
    print(f"max: {max_val:.4f}")
    print("-" * (width + 2))
    
    for row in chart:
        print("|" + "".join(row) + "|")
    
    print("-" * (width + 2))
    print(f"min: {min_val:.4f}")
    print()
    print("Legend: * = лучшая fitness, o = средняя fitness, X = пересечение")
    print("=" * (width + 15))


def print_statistics(stats: dict) -> None:
    """
    Вывод финальной статистики обучения.
    
    Args:
        stats: Словарь со статистикой
    """
    print("\n" + "=" * 60)
    print("СТАТИСТИКА ОБУЧЕНИЯ")
    print("=" * 60)
    
    print(f"Всего поколений:     {stats.get('total_generations', 'N/A')}")
    print(f"Время обучения:      {stats.get('training_time', 0):.2f} сек")
    print(f"Начальная fitness:   {stats.get('initial_fitness', 'N/A'):.6f}" if stats.get('initial_fitness') else "Начальная fitness: N/A")
    print(f"Конечная fitness:    {stats.get('final_fitness', 'N/A'):.6f}" if stats.get('final_fitness') else "Конечная fitness: N/A")
    print(f"Количество улучшений: {stats.get('improvements', 'N/A')}")
    print(f"Лучшая формула:      {stats.get('formula', 'N/A')}")
    
    print("=" * 60)
