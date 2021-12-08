from catboost import CatBoostClassifier, Pool, cv, CatBoost
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
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

    train_full = np.array(np.load('data/training_x.npy'))
    truth_full = np.array(np.load('data/training_y.npy'))
    print(train_full.shape, truth_full.shape)
    # test_x = np.array(np.load('data/test_x.npy'))
    # test_y = np.array(np.load('data/test_y.npy'))
    train_x, val_x, train_y, val_y = train_test_split(train_full,
                                                      truth_full,
                                                      test_size=0.2,
                                                      random_state=42)
    print("Shape of training and testing data:")
    print(train_x.shape, train_y.shape)
    print(val_x.shape, val_y.shape)

    # Pre-processing data
    print("Encoding labels to numerical values...")
    le = preprocessing.LabelEncoder()
    le.fit(train_y)
    train_labels_encoded = le.transform(train_y)
    print(train_labels_encoded)
    le.fit(val_y)
    validation_labels_encoded = le.transform(val_y)
    print(validation_labels_encoded)

    train_y, val_y = train_labels_encoded, validation_labels_encoded

    print("Creating Pool from training dataset...")
    train_data = Pool(data=train_x, label=train_y,
                      cat_features=None, weight=None,
                      thread_count=-1)

    print("Training model...")
    model = CatBoostClassifier(iterations=1000,
                               loss_function='MultiClass',
                               task_type='GPU',
                               devices='0:1')

    model.fit(train_data, verbose=10)

    # uncomment to save model
    # model.save_model("model_v3.cbm")

    generate_metrics(model, val_x, val_y)
    get_accuracy(model, val_x, val_y)


if __name__ == '__main__':
    main()
