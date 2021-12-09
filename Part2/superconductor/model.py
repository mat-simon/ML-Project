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
    train_full = csv_np[1:, :-1]
    truth_full = csv_np[1:, -1]
    train_x, test_x, train_y, test_y = train_test_split(train_full,
                                                        truth_full,
                                                        test_size=0.2,
                                                        random_state=42)
    print("train:", train_x.shape)
    print("test:", test_x.shape)

    train_pool = Pool(
        data=train_x, label=train_y,
        cat_features=None, weight=None,
        thread_count=-1
    )

    # model = CatBoostRegressor(
    #     iterations=1000,
    #     learning_rate=1,
    #     depth=10,
    #     od_type='IncToDec',
    #     od_pval=.01,
    #     task_type='GPU',
    #     verbose=100
    # )
    # model.fit(train_pool)
    # print("test score:", model.score(test_x, test_y))

    params = {
        'iterations': [1000],
        'depth': [14]
    }
    model = GridSearchCV(CatBoostRegressor(
        task_type='GPU',
        verbose=10
    ), params)
    model.fit(train_x, train_y)
    print(f"Best estimator: {model.best_params_})")
    print(f"MultiRMSE: {model.best_score_}")
    print("Test score:", model.score(test_x, test_y))


if __name__ == '__main__':
    main()