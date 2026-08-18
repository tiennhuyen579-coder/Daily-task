from sklearn.metrics import classification_report
from main_1 import LGBM, X_test, y_test, class_names
pr1 = 'balanced'
pr2 = custom_weight = {0 : 5.0,
                     1 : 0.79,
                     2 : 0.80,
                     3 : 1.90,
                     4 : 3.80,
                     5 : 0.50,
                     6 : 5.11,}
def core3(pr):

    model_balanced = LGBM(110, pr)

    y_predict = model_balanced.predict(X_test)

    report = classification_report(y_test, y_predict, target_names=class_names)
    print(report)

if __name__ == '__main__':
    core3(pr1)
    core3(pr2)