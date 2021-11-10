import numpy as np
import scipy.io
import util
# import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LogisticRegressionCV
# from sklearn.model_selection import GridSearchCV
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

    # 4's: [0-999], 8's: [5842-6841]
    # take first 1000 rows of 4's and first 1000 rows of 8's to create the training set
    train_X = train_fea1[np.r_[0:1000, 5842:6842], :]
    train_Y = train_gnd1[np.r_[0:1000, 5842:6842], :]
    # take second 1000 rows of 4's and second 1000 rows of 8's to create validation set
    validation_X = train_fea1[np.r_[1000:2000, 6842:7842], :]
    validation_Y = train_gnd1[np.r_[1000:2000, 6842:7842], :]

    print("shape of train set:", train_X.shape)
    print("shape of validation set:", validation_X.shape)

    #     # data values from [0, 255] to [0, 1]
    #     #train_fea1 = np.divide(train_fea1, 255)
    #     #test_fea1 = np.divide(test_fea1, 255)

    penalty_arr = ["l2"]
    best_model = ''
    best_acc = 0

    for penalty in penalty_arr:
        for i in range(1, 30000):
            c = i / 1000
            model = LogisticRegression(C=c, penalty=penalty, solver='liblinear', tol=0.0001)
            model.fit(train_X, train_Y.ravel())
            accuracy = model.score(validation_X, validation_Y)
            if accuracy > best_acc:
                best_model = "C = " + str(c) + ", penalty = " + penalty + ", solver = 'liblinear', tol = 0.0001\n"
                best_acc = accuracy
    print("best accuracy on validation set: ", best_acc)
    print("parameters for best model: ", best_model)

    ### preprocessing steps
    # data values from [0, 255] to [0, 1]
    # train_fea1 = np.divide(train_fea1, 255)
    ###


if __name__ == '__main__':
    main()

    ### Ignore below, just for learning GridSearchCV

    # model = LogisticRegressionCV(Cs=10, n_jobs=-1, penalty='elasticnet', solver='saga', tol=0.001, l1_ratios=[0.0, 0.25, 0.5, 0.75, 1])
    # model.fit(train_fea1, train_gnd1_input)

    # Cs = [x / 1 for x in range(1, 500)]
    # Cs = [.00000001, .0000001, .000001, .00001, .0001, .001,  .01, .1, 1, 10, 100, 1000, 10000]
    #
    # params = {'C': Cs}
    # model = GridSearchCV(LogisticRegression(penalty='l2', solver='liblinear', max_iter=500, tol=0.00001), params)
    # model.fit(train_fea1, train_gnd1_input)
    # print(f"Best liblinear estimator: {model.best_estimator_})")
    # print(f"Best liblinear: {model.best_params_} with acc {model.best_score_}")
    # print("CV test score:", model.score(test_fea1, test_gnd1))
    #
    # params = {'C': Cs, 'l1_ratio': [.1, .2, .3, .4, .5, .6, .7, .8, .9]}
    # model = GridSearchCV(LogisticRegression(n_jobs=-1, penalty='elasticnet', solver='saga', max_iter=10000, tol=0.00001), params)
    # model.fit(train_fea1, train_gnd1_input)
    # print(f"Best saga: {model.best_params_} with acc {model.best_score_}")
    # print("CV test score:", model.score(test_fea1, test_gnd1))

    # print('Max auc_roc:', model.scores_[8].mean(axis=0).max()) # best avg score across folds for LogRegCV()
