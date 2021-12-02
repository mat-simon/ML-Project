import scipy.io
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt


def main():
    train, test = get_data('MNIST.mat')
    for value in test.get_label():
        if value not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            print("wtf?", value)

    param = {'num_class': 10, 'max_depth': 2, 'eta': 1, 'objective': 'multi:softmax'}
    num_round = 12
    early_stopping = 50
    eval_list = [(train, "train"), (test, "test")]
    bst = xgb.train(param, train, num_round, evals=eval_list, early_stopping_rounds=early_stopping, verbose_eval=True)

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

    # replace label "10" from Matlab with label of 0
    for i in range(len(train_Y)):
        if train_Y[i] == 10:
            train_Y[i] = 0
    for i in range(len(test_Y)):
        if test_Y[i] == 10:
            test_Y[i] = 0

    dtrain = xgb.DMatrix(train_X, train_Y)
    dtest = xgb.DMatrix(test_X, test_Y)

    return dtrain, dtest


if __name__ == '__main__':
    main()