from catboost import CatBoostClassifier, Pool, cv
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt


def generate_metrics(model, test_x, test_y):
    # make predictions
    expected_y = test_y
    predicted_y = model.predict(test_x)
    classes = ['angry', 'disgust', 'fear',
               'happy', 'neutral', 'sad', 'surprise']
    # summarize the fit of the model
    print("Summary:")
    print(metrics.classification_report(expected_y, predicted_y,
                                        target_names=classes))
    print("Confusion matrix:")
    cm = metrics.confusion_matrix(expected_y, predicted_y)
    print(metrics.confusion_matrix(expected_y, predicted_y))
    sns.heatmap(cm, annot=True)
    plt.savefig('confusion_matrix.png')


def get_accuracy(model, test_x, test_y):
    expected_y = test_y
    predicted_y = predicted_y = model.predict(test_x).reshape(5757, )

    i = 0
    hit = 0
    for prediction in predicted_y:
        if prediction == expected_y[i]:
            hit += 1
        i += 1
    return (hit / i) * 100


def main():
    print("Loading data..")
    train_x = np.array(np.load('data/train_x.npy'))
    train_y = np.array(np.load('data/train_y.npy'))
    # train_x = np.array(np.load('data/training_x.npy'))
    # train_y = np.array(np.load('data/training_y.npy'))
    print("train_x:", train_x.shape)
    print("train_y:", train_y.shape)
    test_x = np.array(np.load('data/test_x.npy'))
    test_y = np.array(np.load('data/test_y.npy'))

    # train_80_x, val_20_x, train_80_y, val_20_y = train_test_split(train_x,
    #                                                               train_y,
    #                                                               test_size=0.2,
    #                                                               random_state=42)
    # Pre-processing data
    le = preprocessing.LabelEncoder()
    le.fit(train_y)
    train_labels_encoded = le.transform(train_y)
    le.fit(test_y)
    test_labels_encoded = le.transform(test_y)

    train_y, test_y = train_labels_encoded, test_labels_encoded

    train_pool = Pool(
        data=train_x, label=train_y,
        cat_features=None, weight=None,
        thread_count=-1
    )
    # train_80_pool = Pool(
    #     data=train_80_x, label=train_80_y,
    #     cat_features=None, weight=None,
    #     thread_count=-1
    # )
    # val_20_pool = Pool(
    #     data=val_20_x, label=val_20_y,
    #     cat_features=None, weight=None,
    #     thread_count=-1
    # )

    # simple_model = CatBoostClassifier(
    #     iterations=9000,
    #     loss_function='MultiClass',
    #     od_type='IncToDec',
    #     od_pval=.001,
    #     task_type='GPU'
    # )
    # simple_model.fit(train_80_pool, eval_set=val_20_pool, verbose=100)
    # print(simple_model.score(test_x, test_y))

    # Catboost gridsearch
    # param_grid = {'iterations': [4000],
    #               'task_type': ['GPU'],
    #               'depth': [6, 8, 11],
    #               'verbose': [100]}
    # simple_model.grid_search(
    #     param_grid,
    #     train_x,
    #     train_y,
    #     cv=5,
    #     shuffle=True,
    #     verbose=10
    # )

    # Catboost cv
    # params = {
    #     'loss_function': 'MultiClass',
    #     'iterations': 8000,
    #     'task_type': 'GPU'
    #
    # }
    # # print("Training model...")
    # model = cv(
    #     params=params,
    #     pool=train_pool,
    #     fold_count=5,
    #     shuffle=True,
    #     partition_random_seed=0,
    #     verbose=100
    # )

    # SKlearn GridSearchCV
    params = {
        'iterations': [2000],
        'depth': [6, 8, 10]
    }
    model = GridSearchCV(CatBoostClassifier(
        task_type='GPU',
        od_type='IncToDec',
        od_pval=.0001,
        verbose=500),
        params)
    model.fit(train_x, train_y)
    print(f"Best estimator: {model.best_params_})")
    print(f"Score: {model.best_score_}")
    print("Test score:", model.score(test_x, test_y))

    # uncomment to save model
    # model.save_model("model_v3.cbm")
    # generate_metrics(model, test_x, test_y)


if __name__ == '__main__':
    main()
