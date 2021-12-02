import scipy.io
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt


def main():
    train, test = get_data('MNIST.mat')
    param = {'num_class': 10, 'max_depth': 2, 'eta': 1, 'objective': 'multi:softmax'}
    num_round = 2
    bst = xgb.train(param, train, num_round)
    # make prediction
    preds = bst.predict(test)
    print(preds)

    # for saving later
    # bst.save_model('model_file_name.json')


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