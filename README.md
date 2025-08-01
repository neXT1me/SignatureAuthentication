# SignatureAuthentication

## Обзор

SignatureAuthentication — это проект на Python, разработанный для аутентификации подписей с использованием современных алгоритмов машинного обучения. Проект включает реализацию классификации на основе Dynamic Time Warping (DTW), нечеткой классификации с оптимизацией признаков и весов с помощью гравитационного поиска (GSA), а также генерацию правил для классификаторов. Он предназначен для анализа данных подписи и построения надежных моделей бинарной и многоклассовой классификации.

## Возможности

- Анализ данных подписи с использованием DTW.
- Нечеткая классификация с оптимизацией признаков и весов.
- Отбор информативных признаков с помощью бинарного GSA.
- Оптимизация параметров классификатора с помощью непрерывного GSA.
- Генерация правил на основе экстремальных значений и горной кластеризации.
- Тестирование моделей с различными наборами данных.

## Установка

1. **Клонирование репозитория**:
   ```bash
   git clone https://github.com/neXT1me/SignatureAuthentication.git
   cd SignatureAuthentication
   ```

2. **Установка зависимостей**:
   Установите необходимые библиотеки из файла `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Подготовка данных**:
   Убедитесь, что файл `sign_data.csv` или `Данные подписи_Фурье.csv` находится в директории `datasets/` или укажите правильный путь в коде.

## Использование

### Пример запуска тестов
1. Перейдите в корневую директорию проекта.
2. Выполните основной скрипт для запуска эксперимента:
   ```bash
   python tests.py
   ```
   Это запустит эксперимент с многоклассовой классификацией, результат будет сохранен в файл `results/multiclass_gsa_depending_input_data.json`.

### Основные модули
- `DTW_analysis.py`: Анализ данных подписи с использованием DTW и классификация.
- `classifier.py`: Реализация нечеткого классификатора.
- `optimizer.py`: Оптимизация признаков и весов с помощью GSA.
- `rule_generation_algorithms.py`: Генерация правил на основе экстремальных значений и горной кластеризации.
- `tests.py`: Тестовые сценарии и эксперименты.

## Структура проекта

- `DTW/`: Модули, связанные с DTW-анализом.
- `Fuzzy_classifier/`: Реализация нечеткой классификации.
- `results/`: Содержит результаты экспериментов.
- `DTW_analysis.py`: Анализ данных с DTW.
- `classifier.py`: Нечеткий классификатор.
- `optimizer.py`: Оптимизация с GSA.
- `rule_generation_algorithms.py`: Алгоритмы генерации правил.
- `tests.py`: Тестовые скрипты.
- `.gitignore`: Файл исключений для Git.
- `README.md`: Этот файл.
- `requirements.txt`: Зависимости проекта.
