import numpy as np
import scipy.io
from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler


def main():
    dictionary_data = scipy.io.loadmat('MNISTmini.mat')
    print("dictionary keys:", dictionary_data.keys())

    train_fea1 = np.array(dictionary_data['train_fea1'])
    print("shape of train_fea1 np array:", train_fea1.shape)

    train_gnd1 = np.array(dictionary_data['train_gnd1'])
    print("shape of train_gnd1:", train_gnd1.shape)

    test_fea1 = np.array(dictionary_data['test_fea1'])
    print("shape of test_fea1", test_fea1.shape)

    test_gnd1 = np.array(dictionary_data['test_gnd1'])
    print("shape of test_gnd1", test_gnd1.shape)

    # delete rows that are not 4 or 8 (our assigned digits to classify)
    train_fea1, train_gnd1 = reduce_data(train_fea1, train_gnd1, 4, 8)
    test_fea1, test_gnd1 = reduce_data(test_fea1, test_gnd1, 4, 8)

    print("shape of new train data:", train_fea1.shape)
    print("train.rows = ground_truth.rows? :", train_fea1.shape[0] == train_gnd1.shape[0])

    # data values from [0, 255] to [0, 1]
    train_fea1 = np.divide(train_fea1, 255)
    test_fea1 = np.divide(test_fea1, 255)


def reduce_data(train, truth, x, y):
    # reduces full data to just data of digits x and y
    rows = truth.shape[0]
    hits = 0
    bool_list = []
    for i in range(0, rows):
        if truth[i][0] == x or truth[i][0] == y:
            hits += 1
            bool_list.append(True)
        else:
            bool_list.append(False)
    print("#" + str(x) + "'s" + " + #" + str(y) + "'s:", hits)
    reduced_train = np.compress(bool_list, train, axis=0)
    reduced_truth = np.compress(bool_list, truth, axis=0)
    return reduced_train, reduced_truth


if __name__ == '__main__':
    main()
