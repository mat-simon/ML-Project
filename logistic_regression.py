import numpy as np
import scipy.io
import util
# import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
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
    # test_fea1 = np.divide(test_fea1, 255)
    ###

    train_gnd1_input = np.ravel(train_gnd1)

    # model = LogisticRegression(C=48.8, penalty="l2", solver="sag", tol=0.001, max_iter=1000)
    # model.fit(train_fea1, train_gnd1_input)
    # train_score = model.score(train_fea1, train_gnd1)
    # val_score = model.score(test_fea1, test_gnd1)
    # print("LR score on train data:", train_score)
    # print("LR score on validation data:", val_score)


    Cs = [x / 10 for x in range(1, 305)]
    model = LogisticRegressionCV(Cs=Cs, n_jobs=-1, penalty="l2", solver="liblinear", tol=.001)
    model.fit(train_fea1, train_gnd1_input)
    # print('Max:', model.scores_[1].mean(axis=0).max())
    print("CV train score:", model.score(train_fea1, train_gnd1))
    print("CV validation score:", model.score(test_fea1, test_gnd1))

    # ax = plt.gca()
    # rfc_disp = RocCurveDisplay.from_estimator(rfc, X_test, y_test, ax=ax, alpha=0.8)
    # svc_disp.plot(ax=ax, alpha=0.8)


if __name__ == '__main__':
    main()
