import scipy.io
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
import matplotlib.pyplot as plt


def main():
    csv_np = genfromtxt('data/train.csv', delimiter=',')
    full_x = csv_np[1:, :-1]
    full_y = csv_np[1:, -1]
    print(f"full_x: {full_x.shape}")
    print(min(full_y))
    print(max(full_y))
    print(np.average(full_y))

    # split data into train and test
    train_x, val_x, train_y, val_y = train_test_split(full_x,
                                                        full_y,
                                                        test_size=0.25,
                                                        random_state=42)

    train_x, test_x, train_y, test_y = train_test_split(val_x,
                                                      val_y,
                                                      test_size=0.25,
                                                      random_state=42)
    print(f"test_x: {val_x.shape}")
    # # split train further into train and validate
    # train_x, val_x, train_y, val_y = train_test_split(train_x,
    #                                                   train_y,
    #                                                   test_size=0.25,
    #                                                   random_state=0)
    # print(f"train_x: {train_x.shape}")
    # print(f"val_x: {val_x.shape}")

    train = xgb.DMatrix(train_x, train_y)

    regressor_results = xgb.XGBRegressor(max_depth=5, n_estimators=700, n_jobs=2,
    objective='reg:squarederror', booster='gbtree',
    random_state=42, learning_rate=0.19)

    # params = {'num_class': 10, 'max_depth': 5, 'eta': .3, 'objective': 'multi:softmax', 'gpu_id': 0,
    #           'tree_method': 'gpu_hist'}
    # num_round = 5000
    # params['max_depth'] = 5
    # params['min_child_weight'] = 1

    # grid = [
    #     (max_depth)
    #     for max_depth in range(3, 12)
    # ]

    # min_merror = float("Inf")
    # best_params = None
    # # for eta in [.3]:
    # #     print("CV with eta={}".format(eta))
    # #     # update params
    # #     params['eta'] = eta
    # #     # Run CV
    # #     regressor_results = xgb.XGBRegressor(
    # #         params,
    # #         train,
    # #         num_boost_round=num_round,
    # #         seed=8,
    # #         nfold=5,
    # #         early_stopping_rounds=50,
    # #         verbose_eval=10
    # #     )
        # Update best score

    regressor_results.fit(train_x, train_y)

    y_pred = regressor_results.predict(val_x)  # Predictions
    y_true = val_y  # True values

    MSE = mse(y_true, y_pred)
    RMSE = np.sqrt(MSE)

    R_squared = r2_score(y_true, y_pred)

    print("\nRMSE: ", RMSE)
    print()
    print("R-Squared: ", R_squared)

    print(regressor_results.score(test_x, test_y))
        # mean_RMSE = min(regressor_results['test-RMSE-mean'])
        # boost_rounds = regressor_results['test-RMSE-mean'].argmin()
        # print("\tRMSE {} for {} rounds\n".format(mean_RMSE, boost_rounds))
        # if mean_RMSE < min_RMSE:
        #     min_RMSE = mean_RMSE
        #     best_params = eta
        #     print("Best params: {}, RMSE: {}".format(best_params, min_RMSE))


if __name__ == '__main__':
    main()