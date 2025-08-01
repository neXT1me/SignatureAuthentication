import numpy as np
from tqdm import tqdm

from classifier import fuzzy_classifier

from sklearn.metrics import accuracy_score

class GSA_binary:
    """Класс для бинарного отбора признаков с помощью Binary Gravitational Search Algorithm (GSAb)."""

    def __init__(self, P=20, T=100, G0=100, alpha=20, epsilon=1e-10, transfer_function='S1', ElitistCheck=0):
        """
        Гравитационный поиск для отбора признаков

        Parameters:
        P (int): Количество частиц (размер популяции).
        T (int): Максимальное количество итераций.
        G0 (float): Начальная гравитационная константа.
        alpha (float): Параметр затухания гравитационной константы.
        epsilon (float): Малое значение для избежания деления на ноль.
        transfer_function (str): Трансферная функция для обновления позиций ('S1', 'S2', 'V1', 'V2').
        """
        self.P = P
        self.T = T
        self.G0 = G0
        self.alpha = alpha
        self.epsilon = epsilon
        self.transfer_function = transfer_function
        self.ElitistCheck = ElitistCheck
        if self.transfer_function not in ['S1', 'S2', 'V1', 'V2']:
            raise ValueError("transfer_function must be one of 'S1', 'S2', 'V1', 'V2'")

    def transfer_S1(self, v):
        return 1 / (1 + np.exp(-v))

    def transfer_S2(self, v):
        return 1 / (1 + np.exp(-self.alpha * v))

    def transfer_V1(self, v):
        return np.abs(2 / np.pi * np.arctan(np.pi / 2 * v))

    def transfer_V2(self, v):
        return self.transfer_V1(v)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def hamming_distance(self, s1, s2):
        return np.sum(s1 != s2)

    def optimize(self, theta, t_target, X_train, y_train, n_classes, n_features):
        """
        Выполняет бинарный отбор признаков, максимизируя GM.

        Parameters:
        theta: Параметры нечеткого классификатора (оптимизированные через GSAc).
        compute_cluster_labels: Правила для нечеткого классификатора (предварительно сгенерированы).
        s: Вектор масштабирования (предварительно задан).
        X_train, y_train: Обучающие данные.
        X_test, y_test: Тестовые данные.
        n_classes_binary (int): Количество классов (2 для бинарной классификации).
        n_features (int): Количество признаков.

        Returns:
        S_best: Лучший бинарный вектор признаков.
        best_fitness: Лучшее значение целевой функции (GM) на обучающих данных.
        test_fitness: Значение GM на тестовых данных для S_best.
        """
        # Инициализация популяции S
        S = np.random.randint(0, 2, size=(self.P, n_features))
        V = np.zeros((self.P, n_features))  # Начальные скорости

        # Проверка
        for i in range(self.P):
            if not np.any(S[i]):
                random_feature = np.random.randint(0, n_features)
                S[i, random_feature] = 1

        S_best = None
        best_fitness = -np.inf

        # Основной цикл GSA
        for t in tqdm(range(1, self.T+1), position=0, leave=True, desc='Optimize GSA'):
            fitness = np.zeros(self.P)
            for i in range(self.P):
                y_pred = fuzzy_classifier(X_train, theta, t_target, S[i], n_classes, n_features)

                ac = accuracy_score(y_train, y_pred)
                fitness[i] = ac

                if ac > best_fitness:
                    best_fitness = ac
                    S_best = S[i].copy()

            # Шаг 1: Находим best(t) и worst(t)
            best_t = np.max(fitness)
            worst_t = np.min(fitness[fitness > -np.inf]) if np.any(fitness > -np.inf) else 0

            # Шаг 2: Обновляем G(t)
            G_t = self.G0 * np.exp(-self.alpha * t / self.T)

            # Шаг 3: Вычисляем массы M_i(t)
            m = np.zeros(self.P)
            for i in range(self.P):
                if fitness[i] > -np.inf:
                    m[i] = (1 - fitness[i] - worst_t) / (best_t - worst_t + self.epsilon)
                else:
                    m[i] = 0
            M = m / (np.sum(m) + self.epsilon)

            # Шаг 4: Выбираем Kbest (либо линейно, либо все учавствуют)
            if not self.ElitistCheck:
                K = self.P
            else:
                K = 2 + (1 - t/self.T)*(100-2)
                K = round(self.P*K/100)

            best_indices = np.argsort(fitness)[::-1][:K]

            # Шаг 5: Вычисляем ускорения a_i(t) и обновляем скорости V_i(t+1)
            a = np.zeros((self.P, n_features))
            for i in range(self.P):
                for d in range(n_features):
                    F_i_d = 0
                    for j in best_indices:
                        if i == j:
                            continue
                        R_ij = self.hamming_distance(S[i], S[j]) + self.epsilon
                        F_i_d += np.random.rand() * G_t * (M[j] / R_ij) * (S[j][d] - S[i][d])
                    a[i, d] = F_i_d / (M[i] + self.epsilon)

                # Обновляем скорость V_i(t+1)
                V[i] = np.random.rand(n_features) * V[i] + a[i]

            # Шаг 6: Обновляем позиции S_i(t+1) с использованием выбранной трансферной функции
            for i in range(self.P):
                for d in range(n_features):
                    if self.transfer_function == 'S1':
                        prob = self.transfer_S1(V[i, d])
                        S[i, d] = 0 if np.random.rand() < prob else 1
                    elif self.transfer_function == 'S2':
                        prob = self.transfer_S2(V[i, d])
                        S[i, d] = 0 if np.random.rand() < prob else 1
                    elif self.transfer_function == 'V1':
                        prob = self.transfer_V1(V[i, d])
                        S[i, d] = 0 if np.random.rand() < prob else 1
                    elif self.transfer_function == 'V2':
                        prob = self.transfer_V2(V[i, d])
                        p = 1 if np.random.rand() < prob else 0
                        S[i, d] = S[i, d] ^ p  # Логическое XOR

        return S_best


class GSA_continuous:
    """Гравитационный поиск для оптимизации весов нечеткого классификатора"""

    def __init__(self, P=20, T=100, G0=100, alpha=20, epsilon=1e-10, ElitistCheck=True):
        """
        Инициализация оптимизатора GSA для непрерывной оптимизации весов.

        Parameters:
        P (int): Количество частиц (размер популяции).
        T (int): Максимальное количество итераций.
        G0 (float): Начальная гравитационная константа.
        alpha (float): Параметр затухания гравитационной константы.
        epsilon (float): Малое значение для избежания деления на ноль.
        ElitistCheck (bool): Включение элитарного подхода (True/False).
        """
        self.P = P
        self.T = T
        self.G0 = G0
        self.alpha = alpha
        self.epsilon = epsilon
        self.ElitistCheck = ElitistCheck

    def compute_fitness(self, theta, X_train, y_train, t_target, S_binary, n_classes, n_features):
        """Вычисление точности нечеткого классификатора для текущих весов."""
        y_pred = fuzzy_classifier(X_train, theta, t_target, S_binary, n_classes, n_features)
        return accuracy_score(y_train, y_pred)

    def optimize(self, theta_init, t_target, X_train, y_train, S_binary, n_classes, n_features):
        """
        Оптимизация весов нечеткого классификатора с помощью GSA.

        Parameters:
        theta_init (np.array): Начальные веса (размер n_features).
        t_target: Целевые значения для классификатора.
        X_train, y_train: Обучающие данные.
        S_binary: Вектор масштабирования (фиксированный).
        n_classes (int): Количество классов.
        n_features (int): Количество признаков.

        Returns:
        Theta_best (np.array): Оптимизированные веса.
        best_fitness (float): Лучшее значение точности.
        """
        # Инициализация популяции Theta (P частиц, каждая — вектор весов n_features)
        len_theta = len(theta_init)

        Theta = np.zeros((self.P, len_theta))
        for i in range(self.P):
            Theta[i, :] = theta_init + np.random.normal(0, 0.1, len_theta)

        V = np.zeros((self.P, len_theta))  # Скорости частиц
        best_fitness = -np.inf
        Theta_best = None

        # Основной цикл GSA
        for t in tqdm(range(1, self.T + 1), desc='Optimize GSA'):
            # Вычисляем значения целевой функции (точность) для каждой частицы
            fitness = np.array([self.compute_fitness(Theta[i], X_train, y_train, t_target, S_binary, n_classes, n_features)
                              for i in range(self.P)])

            # Обновление лучшего решения
            best_idx = np.argmax(fitness)
            if fitness[best_idx] > best_fitness:
                best_fitness = fitness[best_idx]
                Theta_best = Theta[best_idx].copy()

            # Шаг 1: Находим best(t) и worst(t)
            best_t = np.max(fitness)
            worst_t = np.min(fitness[fitness > -np.inf]) if np.any(fitness > -np.inf) else 0

            # Шаг 2: Обновляем гравитационную константу G(t)
            G_t = self.G0 * np.exp(-self.alpha * t / self.T)

            # Шаг 3: Вычисляем массы M_i(t)
            m = np.zeros(self.P)
            for i in range(self.P):
                if fitness[i] > -np.inf:
                    m[i] = (fitness[i] - worst_t) / (best_t - worst_t + self.epsilon)
                else:
                    m[i] = 0
            M = m / (np.sum(m) + self.epsilon)

            # Шаг 4: Определяем Kbest (элитарный подход)
            if not self.ElitistCheck:
                K = self.P
            else:
                K = max(2, int((1 - t / self.T) * (self.P - 2) + 2))

            # Сортируем по убыванию fitness и выбираем K лучших
            indices = np.argsort(fitness)[::-1][:K]

            # Шаг 5: Вычисляем силы и ускорения
            a = np.zeros((self.P, len_theta))
            for i in range(self.P):
                for d in range(n_features):
                    F_i_d = 0
                    for j in indices:
                        if i == j:
                            continue
                        R_ij = np.linalg.norm(Theta[i] - Theta[j]) + self.epsilon
                        F_i_d += np.random.rand() * G_t * (M[j] / R_ij) * (Theta[j, d] - Theta[i, d])
                    a[i, d] = F_i_d / (M[i] + self.epsilon)

            # Шаг 6: Обновляем скорости и позиции
            for i in range(self.P):
                # Обновление скорости (1-я формула)
                V[i] = np.random.rand(len_theta) * V[i] + a[i]
                # Обновление позиции (2-я формула)
                Theta[i] = Theta[i] + V[i]

        return Theta_best, best_fitness
