"""
Declare common infrastructure that may be needed for RF and LR
"""
import numpy as np
import scipy.io


def reduce_data(train, truth, x, y):
    # reduces full data to just data of digits x and y
    rows = truth.shape[0]
    bool_list = []
    for i in range(0, rows):
        if truth[i][0] == x or truth[i][0] == y:
            bool_list.append(True)
        else:
            bool_list.append(False)
    reduced_train = np.compress(bool_list, train, axis=0)
    reduced_truth = np.compress(bool_list, truth, axis=0)
    return reduced_train, reduced_truth


def get_data():
    print("Loading data...")
    try:
        dictionary_data = scipy.io.loadmat('MNISTmini.mat')
        train_fea1 = np.array(dictionary_data['train_fea1'])
        train_gnd1 = np.array(dictionary_data['train_gnd1'])
        test_fea1 = np.array(dictionary_data['test_fea1'])
        test_gnd1 = np.array(dictionary_data['test_gnd1'])
        print("success")
    except Exception as e:
        print(e)
    print("Reducing train data to first 1000 rows of 4s and 8s...")
    print("Reducing validation data to second 1000 rows of 4s and 8s...")
    try:
        train_fea1, train_gnd1 = reduce_data(train_fea1, train_gnd1, 4, 8)
        test_fea1, test_gnd1 = reduce_data(test_fea1, test_gnd1, 4, 8)
        # 4's: [0-999], 8's: [5842-6841]
        train_X = train_fea1[np.r_[0:1000, 5842:6842], :]
        train_Y = train_gnd1[np.r_[0:1000, 5842:6842], :]
        validation_X = train_fea1[np.r_[1000:2000, 6842:7842], :]
        validation_Y = train_gnd1[np.r_[1000:2000, 6842:7842], :]
        print("success\n")
    except Exception as e:
        print(e)
    return train_X, train_Y.ravel(), validation_X, validation_Y, test_fea1, test_gnd1
