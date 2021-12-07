from catboost import CatBoostClassifier, Pool, cv, CatBoost
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import numpy as np
from sklearn import preprocessing
import seaborn as sns


def generate_metrics(model, X_test, y_test):
    # make predictions
    expected_y = y_test
    predicted_y = model.predict(X_test)

    # summarize the fit of the model
    print("Summary:")
    print(metrics.classification_report(expected_y, predicted_y))
    print("Confusion matrix:")
    cm = metrics.confusion_matrix(expected_y, predicted_y)
    print(metrics.confusion_matrix(expected_y, predicted_y))
    sns.heatmap(cm, annot=True)


def main():
    # Load data
    X_train = np.array(np.load('train_x.npy'))
    y_train = np.array(np.load('train_y.npy'))
    X_test = np.array(np.load('test_x.npy'))
    y_test = np.array(np.load('test_y.npy'))
    print("Shape of training and testing data")
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    # Pre-processing data
    print("Encoding labels to numerical values...")
    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    train_labels_encoded = le.transform(y_train)
    print(train_labels_encoded)
    le.fit(y_test)
    test_labels_encoded = le.transform(y_test)
    print(test_labels_encoded)

    y_train, y_test = train_labels_encoded, test_labels_encoded
    X_train, X_test = X_train.astype('float32'), X_test.astype('float32')
    # X_train, X_test = X_train / 255.0, X_test / 255.0

    cat_features = list(range(0, X_train.shape[1]))
    # print(X_train)

    # emotions_dataset = Pool(data=X_train,
    #                         label=y_train,
    #                         cat_features=cat_features,
    #                         thread_count=-1,)

    print("Training model...")
    # params = {"iterations": 1000,
    #           "loss_function": "MultiClass"}
    # scores = cv(emotions_dataset,
    #             params,
    #             fold_count=10)
    # print(scores)
    model = CatBoostClassifier(iterations=1000,
                               loss_function='MultiClass',
                               task_type='GPU',
                               devices='0:1')

    model.fit(X_train, y_train)

    generate_metrics(model, X_test, y_test)

    return


if __name__ == '__main__':
    main()
