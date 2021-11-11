import numpy as np
import scipy.io
import util
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.model_selection import GridSearchCV


def graph_acc_trees(train_X, train_Y, validation_X, validation_Y):
    # Fix depth, graph number of trees vs accuracy
    clfs = []
    num_trees = list(range(1, 501))
    for i in num_trees:
        clf = RandomForestClassifier(n_estimators=i, n_jobs=-1)
        clf.fit(train_X, train_Y)
        clfs.append(clf)
    train_scores = [clf.score(train_X, train_Y)*100 for clf in clfs]
    validation_scores = [clf.score(validation_X, validation_Y)*100 for clf in clfs]
    fig, ax = plt.subplots()
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("Accuracy %")
    ax.set_title("Number of trees vs Accuracy")
    ax.plot(num_trees, train_scores, label="train", marker='o', markersize=2)
    ax.plot(num_trees, validation_scores, label="validation", marker='o', markersize=2)
    ax.legend()
    fig.savefig("accuracy_trees.png")
    plt.show()


def prune_tree(train, truth):
    print("Pruning Tree - Finding best alpha...")
    clf = DecisionTreeClassifier(random_state=0)
    path = clf.cost_complexity_pruning_path(train, truth)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    fig, ax = plt.subplots()
    ax.plot(ccp_alphas[:-1], impurities[:-1], #marker="o",
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


def feature_importance(model):
    print("Feature importance for every pixel:")
    # fi = np.array(SelectFromModel(model).get_support())
    # print(fi.reshape(10, 10))
    return model


def train_model(train_X, train_Y, validation_X, validation_Y):
    """
    Trains models using the RandomForestClassifier and prints validation scores and models
    Modify hyperparameters for best accuracy
    :param train_X: numpy array with the MNIST image data for training
    :param train_Y: array of ground truth values for training
    :param validation_X: numpy array with the MNIST image data for validation
    :param validation_Y: array of ground truth values for validation
    :return: trained model
    """
    n_estimators_arr = [5] #[x for x in range(1, 300, 1)]
    max_features = ['sqrt', 'log2']
    max_depth = [10] #[x for x in range(1, 50)]
    best_params = ''
    best_acc = 0
    best_model = None
    acc_dict = {}
    for n in n_estimators_arr:
        for max_fea in max_features:
            for depth in max_depth:
                # print("n:", n, "max_fea:", max_fea, "depth:", depth)
                model = RandomForestClassifier(n_estimators=n, max_features=max_fea, n_jobs=-1, max_depth=depth)
                model.fit(train_X, train_Y)
                accuracy = model.score(validation_X, validation_Y)
                acc_dict[accuracy] = [f'n = {n}', f'max_fea = {max_fea}', f'depth = {depth}']
                if accuracy > best_acc:
                    best_model = model
                    best_params = "n_estimators = " + str(n) + ", max_features = " + str(max_fea) + ", max depth = " + str(depth)
                    best_acc = accuracy
    print("parameters for best model: ", best_params)
    print("best accuracy on validation set: ", best_acc)
    temp = sorted(acc_dict, reverse=True)
    acc_dict = {key: acc_dict[key] for key in temp}
    print("acc_dict", acc_dict)
    return model


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

    print("Reducing train data to first 1000 rows of 4s and 8s...")
    print("Reducing validation data to second 1000 rows of 4s and 8s...")
    try:
        train_fea1, train_gnd1 = util.reduce_data(train_fea1, train_gnd1, 4, 8)
        test_fea1, test_gnd1 = util.reduce_data(test_fea1, test_gnd1, 4, 8)
        # 4's: [0-999], 8's: [5842-6841]
        train_X = train_fea1[np.r_[0:1000, 5842:6842], :]
        train_Y = train_gnd1[np.r_[0:1000, 5842:6842], :]
        validation_X = train_fea1[np.r_[1000:2000, 6842:7842], :]
        validation_Y = train_gnd1[np.r_[1000:2000, 6842:7842], :]
        print("success\n")
    except Exception as e:
        print(e)
    return train_X, train_Y.ravel(), validation_X, validation_Y, test_fea1, test_gnd1


def feature_heatmap(model):
    feature_scores = np.array(model.feature_importances_).reshape(10, 10).T
    fig, ax = plt.subplots()
    heatmap = ax.imshow(feature_scores)
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_title('Pixel importance')
    plt.colorbar(heatmap)
    fig.savefig('heatmap.png')

def main():
    train_X, train_Y, validation_X, validation_Y, test_fea1, test_gnd1 = get_data()
    model = train_model(train_X, train_Y, validation_X, validation_Y)
    graph_acc_trees(train_X, train_Y, validation_X, validation_Y)
    model = feature_importance(model)
    feature_heatmap(model)


if __name__ == '__main__':
    main()

### Ignore below, just for learning GridSearchCV

# params = {'n_estimators': [x for x in range(200, 500)],
    #           'max_features': [x for x in range(2, 33)],
    #           # 'criterion': ['gini', 'entropy']
    #           }
    #
    # model = GridSearchCV(RandomForestClassifier(n_jobs=-1), params)
    # model.fit(train_X, train_Y)
    # print(f"Best estimator: {model.best_estimator_})")
    # print(f"Accuracy = {model.best_score_}")
    # print("Test score:", model.score(test, test_truth))