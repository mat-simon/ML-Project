import numpy as np
import scipy.io
import math
import util
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier


def render_acc_trees_graph(train, truth, test, test_truth):
    clfs = []
    num_trees = list(range(1, 200))

    for i in range(1, 200):
        clf = RandomForestClassifier(n_estimators=i, n_jobs=-1)
        clf.fit(train, truth)
        clfs.append(clf)
    train_scores = [1 - clf.score(train, truth) for clf in clfs]
    test_scores = [1 - clf.score(test, test_truth) for clf in clfs]
    fig, ax = plt.subplots()
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("Error")
    ax.set_title("Error vs number of trees for training and testing sets")
    ax.plot(num_trees, train_scores, marker="o", label="train",
            drawstyle="steps-post")
    ax.plot(num_trees, test_scores, marker="o", label="test",
            drawstyle="steps-post")
    ax.legend()
    fig.savefig("error_trees.png")


def prune_tree(train, truth):
    print("Pruning Tree - Finding best alpha...")
    clf = DecisionTreeClassifier(random_state=0)
    path = clf.cost_complexity_pruning_path(train, truth)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    fig, ax = plt.subplots()
    ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o",
            drawstyle="steps-post")
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")
    fig.savefig("example1.png")

    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(train, truth)
        clfs.append(clf)
    print(
        "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
            clfs[-1].tree_.node_count, ccp_alphas[-1]
        )
    )
    return ccp_alphas[-1]


def feature_importance(train, truth):
    sel = SelectFromModel(RandomForestClassifier(n_estimators=1))
    sel.fit(train, truth)
    np_fi = np.array(sel.get_support())
    print(np_fi.reshape(10, 10))
    # selected_feat = train.columns[(sel.get_support())]
    # len(selected_feat)
    # print(selected_feat)
    # pd.series(sel.estimator_, feature_importances_,.ravel()).hist()

    return sel

def main():
    dictionary_data = scipy.io.loadmat('MNISTmini.mat')
    # print("dictionary keys:", dictionary_data.keys())

    train_fea1 = np.array(dictionary_data['train_fea1'])
    # print("shape of train_fea1 np array:", train_fea1.shape)

    train_gnd1 = np.array(dictionary_data['train_gnd1'])
    # print("shape of train_gnd1:", train_gnd1.shape)

    test_fea1 = np.array(dictionary_data['test_fea1'])
    # print("shape of test_fea1:", test_fea1.shape)

    test_gnd1 = np.array(dictionary_data['test_gnd1'])
    # print("shape of test_gnd1:", test_gnd1.shape)

    # delete rows that are not 4 or 8 (our assigned digits to classify)
    train_fea1, train_gnd1 = util.reduce_data(train_fea1, train_gnd1, 4, 8)
    test_fea1, test_gnd1 = util.reduce_data(test_fea1, test_gnd1, 4, 8)
    # num_features = math.log(train_gnd1.shape[0], 2)
    # feature_importance(train_fea1, train_gnd1.ravel())

    # alpha = prune_tree( train_fea1, train_gnd1.ravel())

    render_acc_trees_graph(train_fea1, train_gnd1.ravel(), test_fea1, test_gnd1.ravel())
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    clf.fit(train_fea1, train_gnd1.ravel())


    result = clf.score(test_fea1, test_gnd1.ravel())
    print(result)

    # names = list(range(1, 100))
    #
    # print("Features sorted by score:")
    # print(sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), names), reverse=True))
    #
    # print("Best accuracy by training:")
    # best_accuracy = 0
    # best_accuracy_inx = 0
    # for i in range(1, 100):
    #     clf = RandomForestClassifier(n_estimators=1, max_features=i)
    #     clf.fit(train_fea1, train_gnd1.ravel())
    #     result = clf.score(test_fea1, test_gnd1.ravel())
    #     if best_accuracy < result:
    #         best_accuracy = result
    #         best_accuracy_inx = i
    #     # print(i, result)
    #
    # print("Best accuracy:", best_accuracy_inx, best_accuracy)


if __name__ == '__main__':
    main()
