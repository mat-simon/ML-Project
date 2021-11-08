import numpy as np
import scipy.io
import math
import util
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def main():
    dictionary_data = scipy.io.loadmat('MNISTmini.mat')
    # print("dictionary keys:", dictionary_data.keys())

    train_fea1 = np.array(dictionary_data['train_fea1'])
    # print("shape of train_fea1 np array:", train_fea1.shape)

    train_gnd1 = np.array(dictionary_data['train_gnd1'])
    # print("shape of train_gnd1:", train_gnd1.shape)

    test_fea1 = np.array(dictionary_data['test_fea1'])
    # print("shape of test_fea1:", test_fea1.shape)

    test_gnd1 = np.array(dictionary_data['test_gnd1'])
    # print("shape of test_gnd1:", test_gnd1.shape)

    # delete rows that are not 4 or 8 (our assigned digits to classify)
    train_fea1, train_gnd1 = util.reduce_data(train_fea1, train_gnd1, 4, 8)
    test_fea1, test_gnd1 = util.reduce_data(test_fea1, test_gnd1, 4, 8)
    num_features = math.log(train_gnd1.shape[0], 2)

    clf = RandomForestClassifier(n_estimators=10, max_features=52)
    clf.fit(train_fea1, train_gnd1.ravel())

    result = clf.score(test_fea1, test_gnd1.ravel())
    print(result)

    names = list(range(1, 100))

    print("Features sorted by score:")
    print(sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), names), reverse=True))

    print("Best accuracy by training:")
    best_accuracy = 0
    best_accuracy_inx = 0
    for i in range(1, 100):
        clf = RandomForestClassifier(n_estimators=1, max_features=i)
        clf.fit(train_fea1, train_gnd1.ravel())
        result = clf.score(test_fea1, test_gnd1.ravel())
        if best_accuracy < result:
            best_accuracy = result
            best_accuracy_inx = i
        # print(i, result)

    print("Best accuracy:", best_accuracy_inx, best_accuracy)


if __name__ == '__main__':
    main()
