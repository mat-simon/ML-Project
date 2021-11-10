import numpy as np
import scipy.io
import util
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def graph_acc_trees(train, truth, test, test_truth):
    clfs = []
    num_trees = list(range(1, 201))

    for i in range(1, 11):
        clf = RandomForestClassifier(n_estimators=i, n_jobs=-1)
        clf.fit(train, truth)
        clfs.append(clf)
    train_scores = [1 - clf.score(train, truth) for clf in clfs]
    test_scores = [1 - clf.score(test, test_truth) for clf in clfs]
    fig, ax = plt.subplots()
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("Error")
    ax.set_title("Error vs number of trees for training and testing sets")
    ax.plot(num_trees, train_scores, label="train",
            drawstyle="steps-post")
    ax.plot(num_trees, test_scores, label="test",
            drawstyle="steps-post")
    ax.legend()
    fig.savefig("error_trees2.png")


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
    print("Feature importance for every pixel:")
    sel = SelectFromModel(RandomForestClassifier(n_estimators=100))
    sel.fit(train, truth)
    np_fi = np.array(sel.get_support())
    print(np_fi.reshape(10, 10))
    return sel


def train_model(train, truth, test, test_truth):
    """
    Trains a single model using the RandomForestClassifier and prints score
    result
    Modify hyperparameters for best accuracy
    :param train: numpy array with the MNIST image data for training
    :param truth: array of 8s and 4s true values for training data
    :param test: numpy array with the MNIST image data for testing
    :param test_truth: array of 8s and 4s true values for test data
    :return: trained model
    """
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    clf.fit(train, truth)
    result = clf.score(test, test_truth)
    print("Score: ", result)

    return clf


def get_data():
    print("Loading data...")
    try:
        dictionary_data = scipy.io.loadmat('MNISTmini.mat')
        train_fea1 = np.array(dictionary_data['train_fea1'])
        train_gnd1 = np.array(dictionary_data['train_gnd1'])
        test_fea1 = np.array(dictionary_data['test_fea1'])
        test_gnd1 = np.array(dictionary_data['test_gnd1'])
        print("success")
    except Exception as e:
        print(e)

    print("Reducing data to 8s and 4s...")
    try:
        train_fea1, train_gnd1 = util.reduce_data(train_fea1, train_gnd1, 4, 8)
        test_fea1, test_gnd1 = util.reduce_data(test_fea1, test_gnd1, 4, 8)
        print("success")
    except Exception as e:
        print(e)

    return train_fea1, train_gnd1.ravel(), test_fea1, test_gnd1.ravel()


def feature_heatmap(model):
    feature_scores = np.array(model.feature_importances_).reshape(10, 10)
    fig, ax = plt.subplots()
    im = ax.imshow(feature_scores)
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    fig.savefig('heatmap2.png')

def main():

    train, truth, test, test_truth = get_data()
    # feature_importance(train, truth)
    clf = train_model(train, truth, test, test_truth)
    # feature_heatmap(clf)

    params = {'n_estimators': [1, 2, 3, 5, 10, 20, 50, 100, 200],
              'max_features': [x for x in range(2, 25)],
              # 'criterion': ['gini', 'entropy']
              }

    model = GridSearchCV(RandomForestClassifier(n_jobs=-1), params)
    model.fit(train, truth)
    print(f"Best estimator: {model.best_estimator_})")
    print(f"Accuracy = {model.best_score_}")
    print("Test score:", model.score(test, test_truth))


if __name__ == '__main__':
    main()
