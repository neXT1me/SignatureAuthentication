import json
import numpy as np
import matplotlib.pyplot as plt
import os

# === НАСТРОЙКИ ===
FILE = 'multiclass_gsa_depending_input_data.json'  # замени на нужный файл
OUTPUT_DIR = 'plots_new_format'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Загрузка данных
with open(FILE, 'r') as f:
    raw_data = json.load(f)

# Преобразуем в массив (5, 19, 12, 12)
data = np.array(raw_data)  # (folds, signature_counts, users, users)
n_folds, n_sig_counts, n_users, _ = data.shape

signature_counts = np.arange(2, 21)  # от 2 до 20
accuracies = []
fars = []
frrs = []

# Для метрик по каждому пользователю
acc_per_id = {i: [] for i in range(n_users)}
far_per_id = {i: [] for i in range(n_users)}
frr_per_id = {i: [] for i in range(n_users)}

for sig_idx, n_signatures in enumerate(signature_counts):
    acc_fold = []
    far_fold = []
    frr_fold = []

    acc_fold_per_id = [[] for _ in range(n_users)]
    far_fold_per_id = [[] for _ in range(n_users)]
    frr_fold_per_id = [[] for _ in range(n_users)]

    for fold in range(n_folds):
        cm = np.array(data[fold, sig_idx])  # (12, 12)
        total = np.sum(cm)
        tp_total = np.trace(cm)
        acc_fold.append(tp_total / total)

        far_total = 0
        far_den = 0
        frr_total = 0
        frr_den = 0

        for i in range(n_users):
            row = cm[i]
            col = cm[:, i]

            tp = cm[i][i]
            fn = np.sum(row) - tp
            fp = np.sum(col) - tp
            tn = total - (tp + fn + fp)

            acc_val = tp / np.sum(row) if np.sum(row) > 0 else 0
            far_val = fp / (fp + tn) if (fp + tn) > 0 else 0
            frr_val = fn / (fn + tp) if (fn + tp) > 0 else 0

            acc_fold_per_id[i].append(acc_val)
            far_fold_per_id[i].append(far_val)
            frr_fold_per_id[i].append(frr_val)

            far_total += fp
            far_den += (fp + tn)
            frr_total += fn
            frr_den += (fn + tp)

        far_fold.append(far_total / far_den if far_den > 0 else 0)
        frr_fold.append(frr_total / frr_den if frr_den > 0 else 0)

    # Средние по фолдам
    accuracies.append(np.mean(acc_fold))
    fars.append(np.mean(far_fold))
    frrs.append(np.mean(frr_fold))

    # Средние по пользователям
    for i in range(n_users):
        acc_per_id[i].append(np.mean(acc_fold_per_id[i]))
        far_per_id[i].append(np.mean(far_fold_per_id[i]))
        frr_per_id[i].append(np.mean(frr_fold_per_id[i]))

    # Вывод в консоль
    acc_arr = [acc_per_id[i][-1] for i in range(n_users)]
    far_arr = [far_per_id[i][-1] for i in range(n_users)]
    frr_arr = [frr_per_id[i][-1] for i in range(n_users)]

    print(f'Подписей: {n_signatures:>2} | '
          f'Accuracy: {np.mean(acc_arr):.4f} ± {np.std(acc_arr):.4f} | '
          f'FAR: {np.mean(far_arr):.4f} ± {np.std(far_arr):.4f} | '
          f'FRR: {np.mean(frr_arr):.4f} ± {np.std(frr_arr):.4f}')

def plot_metric(y, label, ylabel, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(signature_counts, y, marker='o', label=label)
    plt.xlabel('Количество подписей')
    plt.ylabel(ylabel)
    plt.title(f'{label} в зависимости от количества подписей')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

plot_metric(accuracies, 'Accuracy', 'Accuracy', 'accuracy_total.png')
plot_metric(fars, 'Усреднённый FAR', 'FAR', 'far_total.png')
plot_metric(frrs, 'Усреднённый FRR', 'FRR', 'frr_total.png')


def plot_per_user(metric_dict, ylabel, filename_prefix):
    plt.figure(figsize=(10, 6))
    for uid, values in metric_dict.items():
        plt.plot(signature_counts, values, label=f'ID {uid + 1}', marker='o')
    plt.xlabel('Количество подписей')
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} по каждому пользователю')
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{filename_prefix}_per_id.png'))
    plt.close()

plot_per_user(acc_per_id, 'Accuracy', 'accuracy')
plot_per_user(far_per_id, 'FAR', 'far')
plot_per_user(frr_per_id, 'FRR', 'frr')
