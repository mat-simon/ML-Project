import scipy.io
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def main():
    print("Loading data...")
    dictionary_data = scipy.io.loadmat('data/MNIST-LeNet5.mat')
    lenet_data = scipy.io.loadmat('data/MNIST-LeNet5.mat')
    train_x = np.array(dictionary_data['train_fea'])
    train_y = np.array(dictionary_data['train_gnd'])
    test_x = np.array(dictionary_data['test_fea'])
    test_y = np.array(dictionary_data['test_gnd'])
    train_80x, val_x, train_80y, val_y = train_test_split(train_x,
                                                          train_y,
                                                          test_size=0.2,
                                                          random_state=0)
    print(f"train_x: {train_x.shape}")
    print(f"test_x: {test_x.shape}")
    # replace label "10" from Matlab with label of 0
    for i in range(len(train_y)):
        if train_y[i] == 10:
            train_y[i] = 0
    for i in range(len(test_y)):
        if test_y[i] == 10:
            test_y[i] = 0
    train = xgb.DMatrix(train_x, train_y)
    train_80x = xgb.DMatrix(train_80x, train_80y)
    test = xgb.DMatrix(test_x, test_y)

    params = {'num_class': 10, 'max_depth': 2, 'eta': .4, 'objective': 'multi:softmax', 'gpu_id': 0,
              'tree_method': 'gpu_hist'}
    num_round = 5000
    #params['max_depth'] = 5
    #params['min_child_weight'] = 1

    # grid = [
    #     (max_depth)
    #     for max_depth in range(3, 12)
    # ]

    min_merror = float("Inf")
    best_params = None
    for eta in [.4]:
        print("CV with eta={}".format(eta))
        # update params
        params['eta'] = eta
        # Run CV
        cv_results = xgb.cv(
            params,
            train,
            num_boost_round=num_round,
            seed=8,
            nfold=5,
            metrics=['merror'],
            early_stopping_rounds=50,
            verbose_eval=10
        )
        # Update best score
        mean_merror = min(cv_results['test-merror-mean'])
        boost_rounds = cv_results['test-merror-mean'].argmin()
        print("\tmerror {} for {} rounds\n".format(mean_merror, boost_rounds))
        if mean_merror < min_merror:
            min_merror = mean_merror
            best_params = eta
            print("Best params: {}, merror: {}".format(best_params, min_merror))

        cv_results.save_model("model_leNet5", format="cbm")


if __name__ == '__main__':
    main()
