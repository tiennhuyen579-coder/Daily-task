import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report

from main_1 import DCT, LGBM, XGB, X_test, y_test, class_names

plt.style.use("dark_background")
CYAN = '#00FFFF'
MAGENTA = '#FF00FF'
NEON_GREEN = '#39FF14'


def plot_main_1():

    models = {
        'Decision Tree': DCT(),
        'LGBM': LGBM(100, None),
        'XGBoost': XGB(100)
    }

    metrics = {'Accuracy': [], 'Macro F1': []}
    names = list(models.keys())

    for name, model in models.items():
        y_pred = model.predict(X_test)
        metrics['Accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['Macro F1'].append(f1_score(y_test, y_pred, average='macro'))

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, metrics['Accuracy'], width, label='Accuracy', color=CYAN, edgecolor='white')
    ax.bar(x + width / 2, metrics['Macro F1'], width, label='Macro F1', color=MAGENTA, edgecolor='white')

    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title('So sánh hiệu suất các mô hình (Main 1)', fontsize=14, color=NEON_GREEN)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.show()


def plot_main_2():
    n_estimators_list = list(range(50, 151, 10))
    latencies = []
    accuracies = []

    best_est = None
    best_acc = 0.0
    y_pred = None

    for n_est in n_estimators_list:
        model = LGBM(n_est, None)

        # Đo thời gian
        start_time = time.perf_counter()
        for i in range(1, 30):
            y_pred = model.predict(X_test)
        end_time = time.perf_counter()

        # Tính toán metric
        latency = ((end_time - start_time) * 1000) / len(X_test)
        acc = accuracy_score(y_test, y_pred)

        latencies.append(latency)
        accuracies.append(acc)

        if latency < 0.5 and acc > best_acc:
            best_acc = acc
            best_est = n_est

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Trục Y thứ 1 (Bên trái): Độ trễ
    ax1.set_xlabel('Số lượng cây (n_estimators)', fontsize=12)
    ax1.set_ylabel('Độ trễ (ms / mẫu)', fontsize=12, color=CYAN)
    line1 = ax1.plot(n_estimators_list, latencies, marker='o', linestyle='-', color=CYAN, linewidth=2, label='Latency')
    ax1.tick_params(axis='y', labelcolor=CYAN)
    ax1.axhline(y=0.5, color=MAGENTA, linestyle='--', linewidth=2, label='Threshold 0.5 ms')

    # Trục Y thứ 2 (Bên phải): Accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', fontsize=12, color=NEON_GREEN)
    line2 = ax2.plot(n_estimators_list, accuracies, marker='s', linestyle='-', color=NEON_GREEN, linewidth=2,
                     label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=NEON_GREEN)

    # Đánh dấu điểm Best Estimator
    if best_est:
        ax2.axvline(x=best_est, color='yellow', linestyle=':', linewidth=2)
        ax2.text(best_est, best_acc, f' Best: {best_est} cây\n Acc: {best_acc:.4f}',
                 color='yellow', fontsize=11, verticalalignment='bottom', horizontalalignment='right')

    # Gộp chung Legend của 2 trục cho gọn
    lines = line1 + [ax1.lines[-1]] + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', framealpha=0.3)

    ax1.set_title('Trade-off giữa Accuracy và Latency theo số lượng cây', fontsize=14, color='white')
    ax1.grid(color='#333333', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()


def plot_main_3():
    pr0 = None
    pr1 = 'balanced'

    model_pr0 = LGBM(110, pr0)
    model_pr1 = LGBM(110, pr1)


    y_pred0 = model_pr0.predict(X_test)
    y_pred1 = model_pr1.predict(X_test)


    # Lấy dữ liệu dạng dict để đưa vào Pandas DataFrame
    report0 = classification_report(y_test, y_pred0, target_names=class_names, output_dict=True)
    report1 = classification_report(y_test, y_pred1, target_names=class_names, output_dict=True)


    df0 = pd.DataFrame(report0).iloc[:-1, :len(class_names)].T
    df1 = pd.DataFrame(report1).iloc[:-1, :len(class_names)].T


    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # Heatmap 0: None
    sns.heatmap(df0, annot=True, cmap='cool', fmt='.2f', ax=axes[0], cbar=False, annot_kws={"size": 10})
    axes[0].set_title('LGBM với Class Weight = None (pr0)', fontsize=13, color='yellow')

    # Heatmap 1: Balanced
    sns.heatmap(df1, annot=True, cmap='cool', fmt='.2f', ax=axes[1], cbar=False, annot_kws={"size": 10})
    axes[1].set_title('LGBM với Class Weight = "Balanced" (pr1)', fontsize=13, color=CYAN)


    fig.suptitle('So sánh chi tiết các Metric giữa 3 phương pháp Weight (Main 3)', fontsize=16, color=NEON_GREEN)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Đang vẽ biểu đồ Main 1...")
    plot_main_1()
    print("Đang vẽ biểu đồ Main 2...")
    plot_main_2()
    print("Đang vẽ biểu đồ Main 3...")
    plot_main_3()