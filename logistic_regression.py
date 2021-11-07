import numpy as np
import scipy.io
import util
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
    train_fea1, train_gnd1 = util.reduce_data(train_fea1, train_gnd1, 4, 8)
    test_fea1, test_gnd1 = util.reduce_data(test_fea1, test_gnd1, 4, 8)

    print("shape of new train data:", train_fea1.shape)
    print("shape of new ground truth:", train_gnd1.shape)

    # data values from [0, 255] to [0, 1]
    train_fea1 = np.divide(train_fea1, 255)
    test_fea1 = np.divide(test_fea1, 255)

    train_gnd1_input = np.ravel(train_gnd1)
    for x in range(1)
    model = LogisticRegression(C=.01, penalty="l2", solver="liblinear", tol=0.1)
    model.fit(train_fea1, train_gnd1_input)

    score = model.score(test_fea1, test_gnd1)
    print("accuracy on validation set:", score)
    bestScore = 0


if __name__ == '__main__':
    main()
