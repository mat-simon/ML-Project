from catboost import CatBoostClassifier, Pool
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import numpy as np
import scipy.io
from numpy import genfromtxt
#import seaborn as sns
import matplotlib.pyplot as plt


def main():
    print("Loading data...")
    dictionary_data = scipy.io.loadmat('data/MNIST-LeNet5.mat')
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
    print(f"train_x: {train_x.shape}")
    print(f"val_x: {val_x.shape}")

    train_pool = Pool(
        data=train_80x, label=train_80y,
        cat_features=None, weight=None,
        thread_count=-1
    )
    val_pool = Pool(
        data=val_x, label=val_y,
        cat_features=None, weight=None,
        thread_count=-1
    )

    model = CatBoostClassifier(
            iterations=8000,
            learning_rate=.1,
            depth=9,
            l2_leaf_reg=3,
            od_type='IncToDec',
            od_pval=.001,
            task_type='GPU',
            verbose=100
        )
    model.fit(train_pool, eval_set=val_pool)
    #print(model.get_evals_result())
    print(model.score(val_pool))
    print(model.score(test_x, test_y))
    # model.fit(train_pool, eval_set=val_pool)
    # print(model.get_evals_result())
    # print(model.score(val_pool))
    # print("")

    #
    # # SKlearn gridcv
    # params = {
    #     'iterations': [390, 400, 410],
    #     'learning_rate': [.14, .15, .16],
    #     'depth': [12]
    # }
    # model = GridSearchCV(CatBoostRegressor(
    #     task_type='GPU',
    #     verbose=False
    # ), params, verbose=True)
    # model.fit(train_x, train_y)
    # print(f"Best estimator: {model.best_params_})")
    # print("mean y of train", train_y.average())
    # print(f"RMSE: {model.best_score_}")
    # print("Test score:", model.score(test_x, test_y))


if __name__ == '__main__':
    main()
