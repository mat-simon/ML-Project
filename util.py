"""
Declare common infrastructure that may be needed for RF and LR
"""
import numpy as np


def reduce_data(train, truth, x, y):
    # reduces full data to just data of digits x and y
    rows = truth.shape[0]

    hits = 0
    bool_list = []
    for i in range(0, rows):
        if i < 100:
            print(truth[i][0])
        if truth[i][0] == x or truth[i][0] == y:
            hits += 1
            bool_list.append(True)
        else:
            bool_list.append(False)
    print("#" + str(x) + "'s" + " + #" + str(y) + "'s:", hits)
    reduced_train = np.compress(bool_list, train, axis=0)
    reduced_truth = np.compress(bool_list, truth, axis=0)
    return reduced_train, reduced_truth
