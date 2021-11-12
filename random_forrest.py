import numpy as np
import scipy.io
from sklearn.metrics import RocCurveDisplay

import util
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.model_selection import GridSearchCV


def graph_acc_trees(train_X, train_Y, validation_X, validation_Y):
    # Fix depth and max_features, graph number of trees vs accuracy
    clfs = []
    num_trees = list(range(1, 501, 5))
    for i in num_trees:
        clf = RandomForestClassifier(n_estimators=i, n_jobs=-1)
        clf.fit(train_X, train_Y)
        clfs.append(clf)
    train_scores = [clf.score(train_X, train_Y)*100 for clf in clfs]
    validation_scores = [clf.score(validation_X, validation_Y)*100 for clf in clfs]
    fig, ax = plt.subplots()
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("Accuracy %")
    ax.set_title("N trees vs Accuracy, fix max_features='log2', depth=None")
    ax.plot(num_trees, train_scores, label="train", marker='o', markersize=2)
    ax.plot(num_trees, validation_scores, label="validation", marker='o', markersize=2)
    ax.legend()
    fig.savefig("accuracy_trees.png")
    plt.show()

def graph_acc_features(train_X, train_Y, validation_X, validation_Y):
    clfs = []
    max_features = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    best_acc = [0, 0]
    for i in max_features:
        clf = RandomForestClassifier(n_estimators=3, max_features=i, n_jobs=-1)
        clf.fit(train_X, train_Y)
        clfs.append(clf)
        score = clf.score(validation_X, validation_Y) * 100
        if best_acc[1] < score:
            best_acc[0] = i
            best_acc[1] = score
    train_scores = [clf.score(train_X, train_Y) * 100 for clf in clfs]
    validation_scores = [clf.score(validation_X, validation_Y) * 100 for clf in clfs]
    fig, ax = plt.subplots()
    ax.set_xlabel("Max features")
    ax.set_ylabel("Accuracy %")
    ax.set_title("Max Features vs Accuracy, fix n_estimators=3, max_depth=None")
    ax.plot(max_features, train_scores, label="train", marker='o')
    ax.plot(max_features, validation_scores, label="validation", marker='o')
    high_score = round(best_acc[1], 2)
    plt.annotate(str(high_score), (best_acc[0], best_acc[1]))
    ax.legend()
    fig.savefig("accuracy_features.png")
    plt.show()

def graph_acc_depth(train_X, train_Y, validation_X, validation_Y):
    clfs = []
    num_depth = list(range(1, 100))
    best_acc = [0, 0]
    for i in num_depth:
        clf = RandomForestClassifier(n_estimators=3, max_features='log2', max_depth=i, n_jobs=-1)
        clf.fit(train_X, train_Y)
        clfs.append(clf)
        score = clf.score(validation_X, validation_Y)*100
        if best_acc[1] < score:
            best_acc[0] = i
            best_acc[1] = score
    train_scores = [clf.score(train_X, train_Y)*100 for clf in clfs]
    validation_scores = [clf.score(validation_X, validation_Y)*100 for clf in clfs]
    fig, ax = plt.subplots()
    ax.set_xlabel("Max depth")
    ax.set_ylabel("Accuracy %")
    ax.set_title("Max Depth vs Accuracy")
    ax.plot(num_depth, train_scores, label="train", marker='o', markersize=2)
    ax.plot(num_depth, validation_scores, label="validation", marker='o', markersize=2)
    ax.legend()
    plt.annotate(str(best_acc[1]), (best_acc[0], best_acc[1]))
    fig.savefig("accuracy_depth.png")
    plt.show()


def feature_importance(model):
    fi = np.array(SelectFromModel(model).get_support())
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
    n_estimators_arr = [x for x in range(1, 200, 1)]
    max_features = ['log2', 'sqrt', 20, 30, 40, 50, 60]
    max_depth = [None]
    best_params = ''
    best_acc = 0
    best_model = None
    model_dict = {}
    for n in n_estimators_arr:
        for max_fea in max_features:
            for depth in max_depth:
                print("n:", n, "max_fea:", max_fea, "depth:", depth)
                model = RandomForestClassifier(n_estimators=n, max_features=max_fea, max_depth=depth, n_jobs=-1)
                model.fit(train_X, train_Y)
                accuracy = model.score(validation_X, validation_Y)
                model_dict[accuracy] = [f'n = {n}', f'max_fea = {max_fea}', f'depth = {depth}']
                if accuracy > best_acc:
                    best_model = model
                    best_params = "n_estimators = " + str(n) + ", max_features = " + str(max_fea) + ", max depth = " + str(depth)
                    best_acc = accuracy
    print("parameters for best model: ", best_params)
    print("best accuracy on validation set: ", best_acc)
    temp = sorted(model_dict, reverse=True)
    model_dict = {key: model_dict[key] for key in temp}
    print("model_dict", model_dict)
    return best_model


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
    # model = train_model(train_X, train_Y, validation_X, validation_Y)
    # graph_acc_trees(train_X, train_Y, validation_X, validation_Y)
    # graph_acc_features(train_X, train_Y, validation_X, validation_Y)
    graph_acc_depth(train_X, train_Y, validation_X, validation_Y)
    # print(f"accuracy on test set: {model.score(test_fea1, test_gnd1)}")
    # model = feature_importance(model)
    # feature_heatmap(model)

    # RocCurveDisplay.from_estimator(model.fit(train_X, train_Y), test_fea1, test_gnd1)
    # RocCurveDisplay.from_predictions(test_gnd1.ravel(), model.predict(test_fea1), pos_label=8)
    # plt.show()


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