from catboost import CatBoostRegressor, Pool
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import numpy as np
from numpy import genfromtxt
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    csv_np = genfromtxt('data/train.csv', delimiter=',')
    full_x = csv_np[1:, :-1]
    full_y = csv_np[1:, -1]
    print(f"full_x: {full_x.shape}")

    # split data into train and test
    train_x, test_x, train_y, test_y = train_test_split(full_x,
                                                        full_y,
                                                        test_size=0.25,
                                                        random_state=42)
    print(f"test_x: {test_x.shape}")
    # split train further into train and validate
    train_x, val_x, train_y, val_y = train_test_split(train_x,
                                                      train_y,
                                                      test_size=0.25,
                                                      random_state=0)
    print(f"train_x: {train_x.shape}")
    print(f"val_x: {val_x.shape}")

    train_pool = Pool(
        data=train_x, label=train_y,
        cat_features=None, weight=None,
        thread_count=-1
    )
    val_pool = Pool(
        data=val_x, label=val_y,
        cat_features=None, weight=None,
        thread_count=-1
    )

    model = CatBoostRegressor(
        iterations=100000,
        # learning_rate=1,
        # depth=12,
        od_type='IncToDec',
        od_pval=.001,
        task_type='GPU',
        verbose=100
    )
    model.fit(train_pool, eval_set=val_pool)
    print(model.get_evals_result())
    print(model.score(val_pool))
    print("")

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