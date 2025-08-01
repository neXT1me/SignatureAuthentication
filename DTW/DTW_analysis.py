import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tslearn.metrics import dtw_path, dtw

from tqdm import tqdm
import random

def get_data(data):
    # Сортировка данных по идентификатору подписи и временным меткам
    df_sorted = data.sort_values(by=['id_sign', 'pktID'])

    # Группировка данных по идентификатору подписи
    grouped = df_sorted.groupby('id_sign')

    # Определение параметров для извлечения (исключая служебные столбцы)
    parameters = df_sorted.columns.difference(['id_sign', 'id_sign.1', 'id_user', 'pktID']).tolist()

    # Создание итогового массива
    data_sign = [
        np.array([
            group[param].tolist()

            for param in parameters
        ])
        for _, group in grouped
    ]
    return data_sign


class DTW_classification():
    def __init__(self, train_true, test_fake, test_true=None):
        self.train_true = train_true
        self.test_fake = test_fake
        self.test_true = test_true

        self.value_sign = len(self.train_true)
        self.value_param = len(self.train_true[0])

        self.id_sign = list(range(self.value_sign))
        self.mass_params = [[data[i] for data in self.train_true] for i in range(self.value_param)]

        self.g_ref = []
        self._g_ref()

        self.summ_dtw_range = np.zeros(self.value_param)
        self._summ_dtw_range()

        self.psi = None
        self._psi()

        self.d_theshold = self.summ_dtw_range / self.value_sign+3 * self.psi

    def tuple_confusion(self) -> tuple[int]:
        tp = 0
        tn = 0

        for i in self.test_true:
            if self.predict(i):
                tp += 1

        for i in self.test_fake:
            if not self.predict(i):
                tn += 1

        return tp, tn, len(self.test_true)-tp, len(self.test_fake)-tn

    def predict(self, data) -> bool:
        result = np.zeros(self.value_param)
        for p in range(self.value_param):
            result[p] = dtw(data[p], self.g_ref[p])

        return (result <= self.d_theshold).all()

    def compute_dtw_barycenter(self, params):
        reference = params[0]
        aligned_params = [reference]

        for p in params[1:]:
            path, _ = dtw_path(reference, p)
            aligned_p = np.zeros(len(reference))
            count = np.zeros(len(reference))

            for i, j in path:
                aligned_p[i] += p[j]
                count[i] += 1

            aligned_p /= np.where(count == 0, 1, count)
            aligned_params.append(aligned_p)

        return np.mean(aligned_params, axis=0)

    def _summ_dtw_range(self):
        for sign in self.train_true:
            for param in range(self.value_param):
                self.summ_dtw_range[param] += dtw(self.g_ref[param], sign[param])

    def _g_ref(self):
        for mp in self.mass_params:
            self.g_ref.append(self.compute_dtw_barycenter(mp))

    def _psi(self):
        psi_2 = np.zeros(self.value_param)
        d_th = self.summ_dtw_range / self.value_sign

        for sign in self.train_true:
            for param in range(self.value_param):
                psi_2[param] += (dtw(self.g_ref[param],sign[param].reshape(-1,1)) - d_th[param])**2 / self.value_sign
        self.psi = np.sqrt(psi_2)

def experiment(data_train, data_fake_test, data_true_test):
    n = len(data_train)
    accuracy_param = []
    for i in tqdm(range(1, n + 1)):
        model = DTW_classification(data_train[:i], data_fake_test, data_true_test)
        accuracy_param.append(model.tuple_confusion())
    return accuracy_param

if __name__ == '__main__':
    file = 'sign_data.csv'
    df = pd.read_csv(file)


    # col = ['id_user', 'id_sign', 'pktID', 'X', 'Y', 'Z',
    #                       'pkNormalPressure',
    #                       'pkOrientationOrAltitude',
    #                       'pkOrientationOrAzimuth']
    # param = ['id_user', 'id_sign', 'pktID', 'X', 'Y', 'Z',
    #                       'pkNormalPressure',
    #                       'pkOrientationOrAltitude',
    #                       'pkOrientationOrAzimuth']

    list_id = [1, 2, 5, 6, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 22]

    # Данный тест проводится как проверка бинарной классификации, условия:
    # 1) Список используемых id [1, 2, 5, 6, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 22]
    # 2) Для каждого id используется 20 экземпялров данных
    # 3) количество тестовой выборки для каждого id (Хотелось бы рассмотреть когда количество тестовых данных одиноковое для каждых классов)
    for id in list_id:
        df_sign_true = df[df.id_user == id]
        df_sign_fake = df[df.id_user != id]



    data_sign_true = get_data(df_sign_true)
    data_sign_fake = get_data(df_sign_fake)
    # ---------------------- Тестирование при value_sign_true = 10 ----------------------------------
    list_acc = []
    n = 5

    for i in tqdm(range(n)):
        dt = random.sample(data_sign_true, k=10)
        result = experiment(data_train=dt,
                        data_fake_test=data_sign_fake[:2],
                        data_true_test=data_sign_true[:2])
        list_acc.append(result)
    # #
    # # with open('result.txt', 'w') as f:
    # #     f.write(str(list_acc))
    #
    # list_acc = []
    # n = 10
    #
    # for i in tqdm(range(n)):
    #     dt = random.sample(data_sign_true, k=20)
    #     dft = random.sample(data_sign_fake, k=50)
    #     dtt = random.sample(data_sign_true, k=50)
    #     result = experiment(data_train=dt,
    #                     data_fake_test=dft,
    #                     data_true_test=dtt)
    #     list_acc.append(result)
    #
    # with open('result_20_50_50_tft.txt', 'w') as f:
    #     f.write(str(list_acc))
    #
