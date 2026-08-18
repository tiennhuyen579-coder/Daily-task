import time
import numpy as np
from main_1 import LGBM, X_test


def core2():
    best_est = None
    optimal_model = None
    total_time = 0
    latency = 0
    for n_est in range(50, 151, 10):
        model = LGBM(n_est, None)

        start_time = time.perf_counter()
        for i in range(1, 30):
            result_pred = model.predict(X_test)
        end_time = time.perf_counter()

        total_time = (end_time - start_time) * 1000
        latency = total_time / len(X_test)

        if latency < 0.5 :
            optimal_model = model
            best_est = n_est
    print(f'Best Estimators : {best_est}')
    print(f'Total time : {total_time}')
    print(f'Độ trễ : {latency}')

if __name__ == "__main__":
    core2()
