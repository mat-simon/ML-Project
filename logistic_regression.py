import util
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay
# from sklearn.linear_model import LogisticRegressionCV
# from sklearn.model_selection import GridSearchCV
# from sklearn.preprocessing import StandardScaler


def main():
    # get data after the preprocessing
    train_X, train_Y, validation_X, validation_Y, test_fea1, test_gnd1 = util.get_data(4, 8)
    print("shape of train set:", train_X.shape)
    print("shape of validation set:", validation_X.shape, "\n")

    # trying out standard scalar
    # scalar = StandardScaler()
    # scalar.fit(train_X)
    # scalar.transform(train_X)
    # scalar.fit(validation_X)
    # scalar.transform(validation_X)

    # Fix tolerance, graph C vs accuracy
    c_train_acc = []
    c_validation_acc = []
    c_arr = [.00000001, .0000001, .00001, .0001, .001, .01, .1, 1, 10, 100]
    for c in c_arr:
        model = LogisticRegression(C=c, penalty='l2', solver='liblinear', tol=.01)
        model.fit(train_X, train_Y.ravel())
        train_acc = model.score(train_X, train_Y)
        val_acc = model.score(validation_X, validation_Y)
        c_train_acc.append(train_acc * 100)
        c_validation_acc.append(val_acc * 100)
    fig, ax = plt.subplots()
    plt.xscale("log")
    ax.set_xlabel("1/λ")
    ax.set_ylabel("Accuracy %")
    ax.set_title("Accuracy vs 1/λ (C)")
    ax.plot(c_arr, c_train_acc, label="train", marker='o')
    ax.plot(c_arr, c_validation_acc, label="validation", marker='o')
    ax.legend()
    plt.show()

    # Fix C, graph tolerance vs accuracy
    t_train_acc = []
    t_validation_acc = []
    tol_arr = [1, 0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    for tol in tol_arr:
        model = LogisticRegression(C=0.01, penalty='l2', solver='liblinear', tol=tol)
        model.fit(train_X, train_Y.ravel())
        train_acc = model.score(train_X, train_Y)
        val_acc = model.score(validation_X, validation_Y)
        t_train_acc.append(train_acc * 100)
        t_validation_acc.append(val_acc * 100)
    fig, ax = plt.subplots()
    plt.xscale("log")
    ax.set_xlabel("Tolerance")
    ax.set_ylabel("Accuracy %")
    ax.set_title("Accuracy vs Tolerance, fixed 1/λ = .01")
    ax.plot(tol_arr, t_train_acc, label="train", marker='o')
    ax.plot(tol_arr, t_validation_acc, label="validation", marker='o')
    ax.legend()
    plt.show()

    # small grid search (zooming in based on graphs above) over C and tol for best model overall
    tol_arr = [x/100 for x in range(1, 50, 1)]
    c_arr = [x/10000 for x in range(1, 100, 1)]
    best_params = ''
    best_acc = 0
    best_model = None
    print("training LogisticRegression...")
    for c in c_arr:
        for tol in tol_arr:
            model = LogisticRegression(C=c, penalty='l2', solver='liblinear', tol=tol)
            model.fit(train_X, train_Y.ravel())
            val_acc = model.score(validation_X, validation_Y)
            if val_acc > best_acc:
                best_model = model
                best_params = "C = " + str(c) + ", penalty = L2, solver = 'liblinear', tol = " + str(tol)
                best_acc = val_acc
    # report best model and final accuracy on test set
    print("parameters for best model: ", best_params)
    print("best model validation accuracy: ", best_acc)
    print(f"accuracy on test set: {best_model.score(test_fea1, test_gnd1)}")

    # ROC curve on test data
    RocCurveDisplay.from_predictions(test_gnd1.ravel(), model.predict(test_fea1), pos_label=8)
    plt.show()


if __name__ == '__main__':
    main()

### Ignore below, code for learning GridSearchCV()

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
