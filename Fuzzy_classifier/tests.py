from rule_generation_algorithms import extreme_feature_values
from optimizer import GSA_binary, GSA_continuous
from classifier import fuzzy_classifier

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np
import json

def get_data_sign(func='minmax') -> tuple[np.array, np.array]:
    '''Функция загрузки и нормировки данных'''
    df = pd.read_csv('../datasets/Данные подписи_Фурье.csv', delimiter=';')
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    if func == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    return scaler.fit_transform(X), y

def select_data(
    X: np.ndarray,
    y: np.ndarray,
    target_classes: list[int] = None,
    samples_per_class: int = None,
    feature_indices: list[int] = None
):
    '''
    Создание выборки с определенными таргетами, признаками и количеством экземпляров данных на каждый класс.

    :param X: np.ndarray, матрица входных данных формы (n_samples, n_features), содержащая признаки.
    :param y: np.ndarray, вектор меток классов формы (n_samples,), содержащий целочисленные метки.
    :param target_classes: list[int], список целевых классов для отбора (по умолчанию None, использует все уникальные классы).
    :param samples_per_class: int, желаемое количество экземпляров на каждый класс (по умолчанию None, использует все доступные).
    :param feature_indices: list[int], список индексов признаков для выбора (по умолчанию None, использует все признаки).
    :return: tuple[np.ndarray, np.ndarray], кортеж из (selected_X, selected_y), где selected_X — матрица отобранных данных формы
             (n_selected_samples, n_selected_features), а selected_y — вектор меток формы (n_selected_samples,).
    '''
    if target_classes is None:
        target_classes = np.unique(y).tolist()
    if feature_indices is None:
        feature_indices = list(range(X.shape[1]))

    selected_X = []
    selected_y = []

    for cls in target_classes:
        indices = np.where(y == cls)[0]

        # Пропуск, если недостаточно данных
        if samples_per_class is not None and len(indices) < samples_per_class:
            print(f"[!] Класс {cls} пропущен: найдено {len(indices)} экземпляров, требуется {samples_per_class}")
            continue

        if samples_per_class is not None:
            indices = np.random.choice(indices, samples_per_class, replace=False, )

        X_class = X[indices][:, feature_indices]
        y_class = y[indices]

        selected_X.append(X_class)
        selected_y.append(y_class)

    if not selected_X:
        raise ValueError("Ни один класс не прошёл условия отбора.")

    return np.vstack(selected_X), np.hstack(selected_y)

def convert_labels(y):
    '''Преобразование таргетов в последовательный вид: [1,6,12,18, ...] -> [0,1,2,3, ...]'''
    # Получаем уникальные метки и сортируем их
    unique_labels = np.unique(y)
    print(f"Исходные уникальные метки: {unique_labels}")

    # Создаем отображение: исходная метка -> новая метка
    original_to_new = {label: idx for idx, label in enumerate(unique_labels)}
    # Создаем обратное отображение: новая метка -> исходная метка
    new_to_original = {idx: label for idx, label in enumerate(unique_labels)}

    # Преобразуем массив меток
    y_converted = np.array([original_to_new[label] for label in y], dtype=int)

    print(f"Отображение (исходная -> новая): {original_to_new}")
    print(f"Обратное отображение (новая -> исходная): {new_to_original}")

    return y_converted, original_to_new, new_to_original


def create_fuzzy_class(X_train, X_test, y_train, y_test, list_classes, GSA_b=False, GSA_c=False) -> list:
    '''
    Создание модели нечеткого классификатора, проведение обучения и тестирования.

    :param X_train: np.ndarray, матрица обучающих данных формы (n_samples_train, n_features)
    :param X_test: np.ndarray, матрица тестовых данных формы (n_samples_test, n_features)
    :param y_train: np.ndarray, вектор меток классов для обучающих данных формы (n_samples_train,)
    :param y_test: np.ndarray, вектор меток классов для тестовых данных формы (n_samples_test,)
    :param list_classes: list, список уникальных меток классов для матрицы ошибок
    :param GSA_b: bool, флаг включения оптимизации бинарного вектора признаков (по умолчанию False)
    :param GSA_c: bool, флаг включения оптимизации весов (по умолчанию False)
    :return: list, матрица ошибок (confusion matrix) в виде списка списков
    '''
    n_features = X_train.shape[1]
    n_classes = len(set(y_train))
    # Инициализация вектора отбора признаков
    S_binar = np.ones(n_features)

    # Создание весов классификатора
    theta, target = extreme_feature_values(X_train, y_train, n_classes, n_features)
    if GSA_b:
        # Отбор информативных признаков
        optimize_b = GSA_binary(P=50, T=100, G0=10, alpha=10, epsilon=0.01, transfer_function='V2')
        S_binar = optimize_b.optimize(theta, target, X_train, y_train, n_classes, n_features)
    if GSA_c:
        # Оптимизация весов
        optimize_c = GSA_continuous(P=50, T=100, G0=10, alpha=10, epsilon=0.01)
        theta = optimize_c.optimize(theta, target, X_train, y_train, S_binar, n_classes, n_features)
    # a
    y_pred = fuzzy_classifier(X_test, theta, target, S_binar, n_classes, n_features)
    cm = confusion_matrix(y_test, y_pred, labels=list_classes).astype(int).tolist()
    return cm

def exp_multi_class_GSAb_GSAc() -> list:
    '''
    Эксперимент для рассмотрения поведения модели на различном количестве подписей пользователя [1:20]
    :return list: словарь с матрицами ошибок
    '''
    list_classes_many = [1, 8, 9, 10, 11, 13, 14, 15, 17, 18, 20, 22]

    X_org, y_org = get_data_sign()

    X, y = select_data(X_org, y_org, samples_per_class=100, target_classes=list_classes_many)
    y_new, _, _ = convert_labels(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_new, test_size=0.8, stratify=y)

    result = []

    # Цикл по количеству экземпляров на класс (от 1 до 20)
    unique_classes = np.unique(y_train)
    for n_samples_per_class in range(1, 21):
        # Формируем подмножество обучающей выборки для каждого класса
        X_train_subset = np.zeros((0, X_train.shape[1]))
        y_train_subset = np.array([])

        for class_label in unique_classes:
            # Индексы образцов текущего класса
            class_indices = np.where(y_train == class_label)[0]
            # Берем первые n_samples_per_class образцов (или все, если их меньше)
            selected_indices = class_indices[:min(n_samples_per_class, len(class_indices))]
            X_train_subset = np.vstack((X_train_subset, X_train[selected_indices]))
            y_train_subset = np.append(y_train_subset, y_train[selected_indices])

        # Создание нечеткого классификатора и проведения тестирования
        cm = create_fuzzy_class(
            X_train=X_train_subset,
            X_test=X_test,
            y_train=y_train_subset,
            y_test=y_test,
            list_classes=list(unique_classes),
            GSA_b=True,
            GSA_c=True)

        result.append(cm)

    return result

if __name__ == '__main__':
    results = exp_multi_class_GSAb_GSAc()

    filename = f"results/multiclass_gsa_depending_input_data.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)