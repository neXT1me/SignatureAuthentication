import numpy as np

def fuzzy_classifier(X, theta, center_labels, S, n_classes, n_features) -> np.array:
    '''
    Формирования нечеткого классификатора с весами и проведение предсказания
    :param X: Входные данные для предсказания
    :param theta: Веса нечеткого классификатора
    :param center_labels: Таргеты весов
    :param S: Бинарный вектор для включения отобранных признаков
    :param n_classes: Количество классов
    :param n_features: Количество признаков
    :return: y_pred
    '''
    R = len(theta) // (2 * n_features)
    rule_params = []
    for i in range(R):
        rule = {}
        for j in range(n_features):
            idx = (i * n_features + j) * 2
            s_ij = theta[idx]
            sigma_ij = theta[idx + 1]
            # Создание правил нечеткого классификатора по весам theta
            rule[j] = lambda x, s=s_ij, sig=sigma_ij: np.exp(-((x - s) ** 2) / (sig ** 2)) if sig > 0 else 1.0
        rule_params.append(rule)

    # Проведение предсказания
    predictions = []
    for sample in X:
        beta = np.zeros(n_classes)
        for i in range(R):
            prod = 1.0
            for k in range(len(sample)):
                # Включение определенных признаков
                if S[k] == 1:
                    prod *= rule_params[i][k](sample[k])
            j = center_labels[i]
            beta[j] += prod
        predicted_class = np.argmax(beta)
        predictions.append(predicted_class)

    return np.array(predictions)

if __name__ == '__main__':
    a = np.array([1,0,1,0,0])
    b = np.array([1,1,1,1,1])
    print(np.sum(a, b))