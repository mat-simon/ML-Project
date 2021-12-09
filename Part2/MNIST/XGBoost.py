import scipy.io
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt


def main():
    train, test = get_data('MNIST.mat')

    params = {'num_class': 10, 'max_depth': 5, 'eta': .3, 'objective': 'multi:softmax', 'gpu_id': 0,
              'tree_method': 'gpu_hist'}
    num_round = 5000
    params['max_depth'] = 5
    params['min_child_weight'] = 1

    # grid = [
    #     (max_depth)
    #     for max_depth in range(3, 12)
    # ]

    min_merror = float("Inf")
    best_params = None
    for eta in [.3]:
        print("CV with eta={}".format(eta))
        # We update our parameters
        params['eta'] = eta
        # Run and time CV
        cv_results = xgb.cv(
            params,
            train,
            num_boost_round=num_round,
            seed=8,
            nfold=5,
            metrics=['merror'],
            early_stopping_rounds=10
        )  # Update best score
        mean_merror = min(cv_results['test-merror-mean'])
        boost_rounds = cv_results['test-merror-mean'].index(mean_merror)
        print("\tmerror {} for {} rounds\n".format(mean_merror, boost_rounds))
        if mean_merror < min_merror:
            min_merror = mean_merror
            best_params = eta
            print("Best params: {}, merror: {}".format(best_params, min_merror))

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
