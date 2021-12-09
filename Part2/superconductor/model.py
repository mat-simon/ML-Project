from catboost import CatBoostRegressor, Pool, cv, CatBoost
from sklearn import metrics
from sklearn.model_selection import train_test_split
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




if __name__ == '__main__':
    main()