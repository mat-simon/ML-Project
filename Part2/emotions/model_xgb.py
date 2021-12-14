import scipy.io
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd


def main():
    # Load dataset
    dataset = pd.read_csv("fer2013.csv")
    # Preview the first 5 lines of the loaded data
    dataset.head()

    # Create numpy arrays from pandas dataframe
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    i = 0
    for rows in dataset.pixels:
        if dataset.Usage[i] == "Training":
            train_x.append(np.fromstring(rows, dtype=int, sep=' '))
            train_y.append(dataset.emotion[i])
        else:
            test_x.append(np.fromstring(rows, dtype=int, sep=' '))
            test_y.append(dataset.emotion[i])
        i += 1

    train_full = np.array(train_x)
    truth_full = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    # Create a validation set using sklean split
    train_x, val_x, train_y, val_y = train_test_split(train_full,
                                                      truth_full,
                                                      test_size=0.2,
                                                      random_state=42)
    print("Shape of training and testing data:")
    print("training_x:", train_x.shape, "training_y:", train_y.shape)
    print("validation_x:", val_x.shape, "validation_y:", val_y.shape)

    train = xgb.DMatrix(train_x, train_y)
    val = xgb.DMatrix(val_x, val_y)
    test = xgb.DMatrix(test_x, test_y)

    params = {
        'objective': 'multi:softmax',
        'num_class': 7,
        'max_depth': 6,
        'eval_metric': 'merror',
        'eta': .3,
        'gpu_id': 0,
        'tree_method': 'gpu_hist'
    }
    model = xgb.train(params, train, 10)

    print("train error", accuracy_score(train_y, model.predict(train)))
    print("test error:", accuracy_score(val_y, model.predict(val)))


if __name__ == '__main__':
    main()
