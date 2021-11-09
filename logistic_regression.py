import numpy as np
import scipy.io
import util
# import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
# from sklearn.preprocessing import StandardScaler


def main():
    dictionary_data = scipy.io.loadmat('MNISTmini.mat')

    train_fea1 = np.array(dictionary_data['train_fea1'])
    train_gnd1 = np.array(dictionary_data['train_gnd1'])
    test_fea1 = np.array(dictionary_data['test_fea1'])
    test_gnd1 = np.array(dictionary_data['test_gnd1'])

    # delete rows that are not 4 or 8 (our assigned digits to classify)
    train_fea1, train_gnd1 = util.reduce_data(train_fea1, train_gnd1, 4, 8)
    test_fea1, test_gnd1 = util.reduce_data(test_fea1, test_gnd1, 4, 8)

    print("shape of train data:", train_fea1.shape)
    print("shape of test data:", test_fea1.shape)

    ### preprocessing steps
    # data values from [0, 255] to [0, 1]
    # train_fea1 = np.divide(train_fea1, 255)
    ###

    train_gnd1_input = np.ravel(train_gnd1)

    # model = LogisticRegressionCV(Cs=10, n_jobs=-1, penalty='elasticnet', solver='saga', tol=0.001, l1_ratios=[0.0, 0.25, 0.5, 0.75, 1])
    # model.fit(train_fea1, train_gnd1_input)

    # Cs = [x / 1 for x in range(1, 500)]
    Cs = [.00000001, .0000001, .000001, .00001, .0001, .001,  .01, .1, 1, 10, 100, 1000, 10000]

    params = {'C': Cs}
    model = GridSearchCV(LogisticRegression(penalty='l2', solver='liblinear', max_iter=500, tol=0.00001), params)
    model.fit(train_fea1, train_gnd1_input)
    print(f"Best liblinear estimator: {model.best_estimator_})")
    print(f"Best liblinear: {model.best_params_} with acc {model.best_score_}")
    print("CV test score:", model.score(test_fea1, test_gnd1))

    params = {'C': Cs, 'l1_ratio': [.1, .2, .3, .4, .5, .6, .7, .8, .9]}
    model = GridSearchCV(LogisticRegression(penalty='elasticnet', solver='saga', max_iter=10000, tol=0.00001), params)
    model.fit(train_fea1, train_gnd1_input)
    print(f"Best saga: {model.best_params_} with acc {model.best_score_}")
    print("CV test score:", model.score(test_fea1, test_gnd1))

    # print('Max auc_roc:', model.scores_[8].mean(axis=0).max()) # best avg score across folds for LogRegCV()


if __name__ == '__main__':
    main()
