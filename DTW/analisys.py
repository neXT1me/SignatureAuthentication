import json
import numpy as np
import matplotlib.pyplot as plt

def accuracy(data: list[list[int]]) -> list[float]:
    return [(tp + tn) / (tp + tn + fp + fn) for tp, tn, fn, fp in data]

def recall(data: list[list[int]]) -> list[float]:
    return [(tp) / (tp + fn) if tp + fn else 0 for tp, tn, fn, fp in data]

def fpr(data: list[list[int]]) -> list[float]:
    return [(fp) / (tn + fp) if tn + fp else 0 for tp, tn, fn, fp in data]

def far(data: list[list[int]]) -> list[float]:
    return [(fp) / (tn + fp) if tn + fp else 0 for tp, tn, fn, fp in data]

def frr(data: list[list[int]]) -> list[float]:
    return [(fn) / (tp + fn) if tp + fn else 0 for tp, tn, fn, fp in data]

def precision(data: list[list[int]]) -> list[float]:
    return [(tp) / (tp + fp) if tp + fp else 0 for tp, tn, fn, fp in data]

def show_plots(data_dict: dict, metric_func, metric_name: str):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    for id_key, experiments in data_dict.items():
        metric_values = []
        max_signatures = min(len(exp) for exp in experiments)
        for i in range(max_signatures):
            iter_metrics = []
            for exp in experiments:
                if i < len(exp):
                    iter_metrics.append(metric_func([exp[i]])[0])
            metric_values.append(np.mean(iter_metrics) if iter_metrics else 0)

        x_axis = list(range(1, max_signatures + 1))
        ax.plot(x_axis, metric_values, label=f'ID {id_key}', marker='o')

    ax.set_title(f'{metric_name} в зависимости от количества подписей')
    ax.set_xlabel('Количество подписей')
    ax.set_ylabel(metric_name)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{metric_name.lower().replace(" ", "_")}_per_id.png')
    plt.show()

def show_averaged_plots(data_dict: dict):
    # Подготовим данные для усреднения
    all_metrics = {'Accuracy': [], 'FAR': [], 'FRR': [], 'Precision': []}
    metric_funcs = {'Accuracy': accuracy, 'FAR': far, 'FRR': frr, 'Precision': precision}

    max_signatures = min(
        len(exp) for id_key, experiments in data_dict.items() for exp in experiments
    )

    signature_metrics_all = {metric: [[] for _ in range(max_signatures)] for metric in all_metrics}

    for i in range(max_signatures):
        signature_metrics = {'Accuracy': [], 'FAR': [], 'FRR': [], 'Precision': []}

        for id_key, experiments in data_dict.items():
            for exp in experiments:
                if i < len(exp):
                    for metric_name, metric_func in metric_funcs.items():
                        metric_value = metric_func([exp[i]])[0]
                        signature_metrics[metric_name].append(metric_value)
                        signature_metrics_all[metric_name][i].append(metric_value)

        for metric_name in all_metrics:
            if signature_metrics[metric_name]:
                all_metrics[metric_name].append(np.mean(signature_metrics[metric_name]))
            else:
                all_metrics[metric_name].append(0)

    for metric_name in signature_metrics_all:
        print(f"\nСтатистика для {metric_name} по количеству подписей:")
        for i in range(max_signatures):
            if signature_metrics_all[metric_name][i]:  # Проверяем, есть ли данные
                mean_value = np.mean(signature_metrics_all[metric_name][i])
                std_dev = np.std(signature_metrics_all[metric_name][i])
                print(f"  Подписей: {i + 1}, Среднее значение: {mean_value:.4f}, Среднее отклонение: {std_dev:.4f}")
            else:
                print(f"  Подписей: {i + 1}, Данных нет")

    for metric_name in all_metrics:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)

        x_axis = list(range(1, max_signatures + 1))
        ax.plot(x_axis, all_metrics[metric_name], label=f'Усредненный {metric_name}', marker='o')

        ax.set_title(f'{metric_name} (усредненный по всем ID) в зависимости от количества подписей')
        ax.set_xlabel('Количество подписей')
        ax.set_ylabel(metric_name)
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{metric_name.lower()}_averaged.png')
        plt.show()

if __name__ == '__main__':
    with open('results/result_20_50_50_tft.json', 'r', encoding='utf-8') as f:
        data_js = json.load(f)

    show_plots(data_js, accuracy, 'Accuracy')
    show_plots(data_js, far, 'False Acceptance Rate (FAR)')
    show_plots(data_js, frr, 'False Rejection Rate (FRR)')
    show_plots(data_js, precision, 'Precision')

    show_averaged_plots(data_js)