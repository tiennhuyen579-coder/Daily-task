from sklearn.metrics import accuracy_score, f1_score, classification_report
from data_loader_quickstart import load_clean_dataset
import sklearn.tree as tree
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split



X_raw, X_scaled, y, class_names, df = load_clean_dataset()

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def DCT():
    dct_md = tree.DecisionTreeClassifier(random_state = 42)
    dct_md.fit(X_train, y_train)
    return dct_md
def LGBM(n_est, none):
    # mặc định là 100 cây (n_est, Pre)
    lgb_md = lgb.LGBMClassifier(n_estimators = n_est, class_weight = none, random_state = 42)
    lgb_md.fit(X_train, y_train)
    return lgb_md
def XGB(n_est):
    # mặc định là 100 cây (n_est)
    xgb_md = xgb.XGBClassifier(n_estimators = n_est,random_state = 42)
    xgb_md.fit(X_train, y_train)
    return xgb_md

def core1():
    model_list = {'Decision tree' : DCT(),
                  'LGBM' : LGBM(100, None),
                  'XGBoost' : XGB(100)}
    for name, model in model_list.items():
        y_predict = model.predict(X_test)
        acc = accuracy_score(y_test, y_predict)
        f1 = classification_report(y_test, y_predict, target_names=class_names)

        print(f'{name}')
        print(f1)
        print('-' * 50)

if __name__ == '__main__':
    #core1()
    import pandas as pd

    print("Phân bố dữ liệu các class trong tập y:")
    print(pd.Series(y).value_counts())