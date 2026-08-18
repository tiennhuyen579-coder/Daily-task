from sklearn.metrics import classification_report
from main_1 import LGBM, X_test, y_test, class_names

pr0 = None
pr1 = 'balanced'

def core3(pr):

    model_balanced = LGBM(110, pr)

    y_predict = model_balanced.predict(X_test)

    report = classification_report(y_test, y_predict, target_names=class_names)
    print(report)

if __name__ == '__main__':
    core3(pr0)
    core3(pr1)