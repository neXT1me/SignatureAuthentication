import numpy as np

def extreme_feature_values(X, y, m, n):
    """
    Генерирует начальный вектор theta и массив compute_cluster_labels для нечеткого классификатора по методу EC.

    Parameters:
    X (ndarray): Данные (p, n), где p - количество образцов, n - количество признаков.
    y (ndarray): Метки классов (p,), предполагается, что метки начинаются с 0.
    m (int): Количество классов.
    n (int): Количество признаков.

    :Returns:
    theta (ndarray): Вектор параметров theta, содержащий центры и отклонения для всех термов.
    compute_cluster_labels (ndarray): Массив меток классов для каждого правила (размер m), начинающийся с 0.
    """
    theta = []
    compute_cluster_labels = []

    for j in range(m):
        class_samples = X[y == j]

        compute_cluster_labels.append(j)

        if len(class_samples) == 0:
            for k in range(n):
                theta.extend([0.0, 0.1])
            continue

        for k in range(n):
            minclass_jk = np.min(class_samples[:, k])
            maxclass_jk = np.max(class_samples[:, k])

            s_jk = (minclass_jk + maxclass_jk) / 2
            sigma_jk = (maxclass_jk - minclass_jk) / 4
            if sigma_jk == 0:
                sigma_jk = 0.1

            theta.extend([s_jk, sigma_jk])

    theta = np.array(theta)
    compute_cluster_labels = np.array(compute_cluster_labels)

    return theta, compute_cluster_labels

def mountain_clustering(X, ra, epsilon=0.3, epsilon_bar=0.05, rb_factor=1.5):
    """
    Горная кластеризация

    X: обучающая выборка (n_samples, n_features)
    ra: радиус кластера
    epsilon: верхний порог
    epsilon_bar: нижний порог
    rb_factor: множитель для rb (rb = rb_factor * ra)

    :returns np.array: список с центрами
    """
    n_samples = X.shape[0]
    rb = rb_factor * ra

    # Шаг 1: Вычисление потенциалов
    potentials = np.zeros(n_samples)
    alpha = 4 / ra**2
    for i in range(n_samples):
        for j in range(n_samples):
            if i != j:
                dist = np.linalg.norm(X[i] - X[j])
                potentials[i] += np.exp(-alpha * dist**2)

    centers = []
    P_c1 = None

    while True:
        # Шаг 2: Находим точку с максимальным потенциалом
        new_idx = np.argmax(potentials)
        P_new = potentials[new_idx]

        if not centers:  # Если это первый центр
            centers.append(X[new_idx])
            P_c1 = P_new
        else:
            # Проверяем условия
            if P_new > epsilon * P_c1:
                centers.append(X[new_idx])
            elif P_new <= epsilon_bar * P_c1:
                break
            else:
                d_min = min(np.linalg.norm(X[new_idx] - center) for center in centers)
                if (d_min / ra + P_new / P_c1) >= 1:
                    centers.append(X[new_idx])
                else:
                    potentials[new_idx] = 0
                    continue

        # Шаг 3: Пересчет потенциалов
        new_center = X[new_idx]
        for i in range(n_samples):
            dist = np.linalg.norm(X[i] - new_center)
            potentials[i] -= P_new * np.exp(-4 * dist ** 2 / rb ** 2)

    return np.array(centers)

def assign_to_clusters(X_train, centers):
    """
    Определяет множество S_i для каждого кластера.
    X_train: обучающая выборка (n_samples, n_features)
    centers: центры кластеров (R, n_features)

    :Returns:
    Возвращает список S_i, где S_i — индексы точек, принадлежащих кластеру i.
    """
    R = centers.shape[0]
    S = [[] for _ in range(R)]

    for p in range(X_train.shape[0]):
        # Находим ближайший центр
        distances = np.linalg.norm(centers - X_train[p], axis=1)
        i = np.argmin(distances)
        S[i].append(p)
    return S

