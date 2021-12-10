from catboost import CatBoostClassifier, Pool
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, train_test_split
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


def main():
    print("Loading data..")
    # Create numpy arrays from pandas dataframe
    dataset = pd.read_csv("data/fer2013.csv")
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    i = 0
    for rows in dataset.pixels:
        if dataset.Usage[i] == "Training":
            train_x.append(np.fromstring(rows, dtype=int, sep=' '))
            train_y.append(dataset.emotion[i])
        else:
            test_x.append(np.fromstring(rows, dtype=int, sep=' '))
            test_y.append(dataset.emotion[i])
        i += 1
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    print(f"full_x: {train_x.shape}")
    print(f"test_x: {test_x.shape}")

    train_x, val_x, train_y, val_y = train_test_split(train_x,
                                                      train_y,
                                                      test_size=0.25,
                                                      random_state=42)
    print(f"train_x: {train_x.shape}")
    print(f"val_x: {val_x.shape}")
    le = preprocessing.LabelEncoder()
    le.fit(train_y)
    train_labels_encoded = le.transform(train_y)
    le.fit(val_y)
    val_labels_encoded = le.transform(val_y)
    le.fit(test_y)
    test_labels_encoded = le.transform(test_y)
    train_y, val_y, test_y = train_labels_encoded, val_labels_encoded, test_labels_encoded

    train_pool = Pool(
        data=train_x, label=train_y,
        cat_features=None, weight=None,
        thread_count=-1
    )
    val_pool = Pool(
        data=val_x, label=val_y,
        cat_features=None, weight=None,
        thread_count=-1
    )

    # model = CatBoostClassifier(
    #     iterations=1000,
    #     # learning_rate=1,
    #     # depth=12,
    #     # od_type='IncToDec',
    #     # od_pval=.001,
    #     task_type='GPU',
    #     verbose=False
    # )
    # model.fit(train_pool, eval_set=val_pool)
    # print(model.get_evals_result())
    # print(model.score(val_pool))

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

    # SKlearn GridSearchCV
    params = {
        'iterations': [2000],
        'depth': [6, 8, 10]
    }
    model = GridSearchCV(CatBoostClassifier(
        task_type='GPU',
        od_type='IncToDec',
        learning_rate=.15,
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
