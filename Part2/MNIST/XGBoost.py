import scipy.io
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt


def main():
    train, test = get_data('MNIST.mat')
    print(train.num_row())


def get_data(file):
    print("Loading data...")
    dictionary_data = scipy.io.loadmat(file)
    train_X = np.array(dictionary_data['train_fea'])
    train_Y = np.array(dictionary_data['train_gnd'])
    test_X = np.array(dictionary_data['test_fea'])
    test_Y = np.array(dictionary_data['test_gnd'])

    dtrain = xgb.DMatrix(train_X, train_Y)
    dtest = xgb.DMatrix(test_X, test_Y)

    return dtrain, dtest


if __name__ == '__main__':
    main()