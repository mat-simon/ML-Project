import numpy as np
import scipy.io
import math
import util
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def main():
    dictionary_data = scipy.io.loadmat('MNISTmini.mat')

    train_fea1 = np.array(dictionary_data['train_fea1'])
    train_gnd1 = np.array(dictionary_data['train_gnd1'])
    test_fea1 = np.array(dictionary_data['test_fea1'])
    test_gnd1 = np.array(dictionary_data['test_gnd1'])
    # print("shape of test_gnd1:", test_gnd1.shape)

    # delete rows that are not 4 or 8 (our assigned digits to classify)
    train_fea1, train_gnd1 = util.reduce_data(train_fea1, train_gnd1, 4, 8)

    # test_fea1, test_gnd1 = util.reduce_data(test_fea1, test_gnd1, 4, 8)
    # num_features = math.log(train_gnd1.shape[0], 2)

    params = {'n_estimators': [x / 1 for x in range(2, 500)],
              'max_features': [x / 1 for x in range(2, 100)],
              'criterion': ['gini', 'entropy']
              }
    model = GridSearchCV(RandomForestClassifier(n_jobs=-1), params)
    model.fit(train_fea1, train_gnd1.ravel())
    print(f"Best estimator: {model.best_estimator_})")
    print(f"Accuracy = {model.best_score_}")

    # names = list(range(1, 100))
    # print("Features sorted by score:")
    # print(sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), names), reverse=True))

    # print("Best accuracy by training:")
    # best_accuracy = 0
    # best_accuracy_inx = 0
    # for i in range(1, 100):
    #     clf = RandomForestClassifier(n_estimators=1, max_features=i)
    #     clf.fit(train_fea1, train_gnd1.ravel())
    #     result = clf.score(test_fea1, test_gnd1.ravel())
    #     if best_accuracy < result:
    #         best_accuracy = result
    #         best_accuracy_inx = i
    #     # print(i, result)
    # print("Best accuracy:", best_accuracy_inx, best_accuracy)


if __name__ == '__main__':
    main()
