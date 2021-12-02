import scipy.io
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt


def main():
    train, test = get_data('MNIST.mat')

    params = {'num_class': 10, 'max_depth': 5, 'eta': .3, 'objective': 'multi:softmax', 'gpu_id': 0,
              'tree_method': 'gpu_hist'}
    num_round = 5000

    grid = [
        (max_depth, min_child_weight)
        for max_depth in range(3, 12)
        for min_child_weight in range(1, 10)
    ]

    min_merror = float("Inf")
    best_params = None
    for max_depth, min_child_weight in grid:
        print("CV with max_depth={}, min_child_weight={}".format(
            max_depth,
            min_child_weight))
        params['max_depth'] = max_depth
        params['min_child_weight'] = min_child_weight
        cv_results = xgb.cv(
            params,
            train,
            num_boost_round=num_round,
            seed=8,
            nfold=5,
            metrics='merror',
            verbose_eval=False,
            early_stopping_rounds=10
        )

        mean_merror = min(cv_results['test-merror-mean'])
        boost_rounds = cv_results['test-merror-mean'].index(mean_merror)
        print("\tmerror {} for {} rounds".format(mean_merror, boost_rounds))
        if mean_merror < min_merror:
            min_merror = mean_merror
            best_params = (max_depth, min_child_weight)
            print("Best params: {}, {}, merror: {}".format(best_params[0], best_params[1], min_merror))


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
