from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from numpy import genfromtxt
import csv


def main():
    csv_np = genfromtxt('data/train.csv', delimiter=',')
    full_x = csv_np[1:, :-1]
    full_y = csv_np[1:, -1]

    file = open('data/train.csv')
    type(file)
    csvreader = csv.reader(file)
    features = next(csvreader)
    print(features)

    print(f"full_x: {full_x.shape}")

    # split data into train and test
    train_x, test_x, train_y, test_y = train_test_split(full_x,
                                                        full_y,
                                                        test_size=0.25,
                                                        random_state=42)
    print(f"test_x: {test_x.shape}")
    # split train further into train and validate
    train_80_x, val_x, train_80_y, val_y = train_test_split(train_x,
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
    train_80_pool = Pool(
        data=train_80_x, label=train_80_y,
        cat_features=None, weight=None,
        thread_count=-1
    )
    val_pool = Pool(
        data=val_x, label=val_y,
        cat_features=None, weight=None,
        thread_count=-1
    )

    model = CatBoostRegressor(
        iterations=450,
        learning_rate=.1,
        depth=12,
        # od_type='IncToDec',
        # od_pval=.001,
        l2_leaf_reg=3,
        task_type='GPU',
        verbose=50
    )
    model.fit(train_80_pool, eval_set=val_pool)
    print(f"model params: {model.get_params()}")
    # print(model.get_evals_result())
    print(f"val R^2: {model.score(val_pool)}")

    print(f"feature size: {model.get_feature_importance().shape}")
    print(f"features: {model.get_feature_importance()}")

    # # SKlearn gridcv
    # params = {
    #     'iterations': [400, 450],
    #     'learning_rate': [.15],
    #     'depth': [5]
    # }
    # cat = CatBoostRegressor(
    #     l2_leaf_reg=3,
    #     task_type='GPU',
    #     verbose=100
    # )
    # model = GridSearchCV(cat, params, verbose=True)
    # model.fit(train_x, train_y)
    # print(f"Best params: {model.best_params_})")
    # print(f"R2: {model.best_score_}")
    # print(f"model.RMSE: {model.error_score}")
    # print(f"test R2: {model.score(test_x, test_y)}")
    # print(f"evals_result: {cat.get_evals_result()}")
    # # print(f"feature importance: {cat.get_feature_importance()}")


if __name__ == '__main__':
    main()
