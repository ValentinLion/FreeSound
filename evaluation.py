import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def svmPredictions(X, y, test_data, i2c):
    classifier = svm.SVC(kernel="linear")
    classify(X, y, test_data, i2c, classifier)


def neighborsPredictions(neighbors, X, y, test_data, i2c):
    classifier = KNeighborsClassifier(n_neighbors=neighbors)
    classify(X, y, test_data, i2c, classifier)


def XGBPredictions(X, y, test_data, i2c):
    classifier = GradientBoostingClassifier(max_depth=5, learning_rate=0.05, n_estimators=100,
                                            random_state=0)
    classify(X, y, test_data, i2c, classifier)


def randomForestPredictions(X, y, test_data, i2c):
    classifier = RandomForestClassifier()

    classify(X, y, test_data, i2c, classifier)


def classify(X, y, test_data, i2c, classifier):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, shuffle=True)

    fname_test = X_test[:, 0]
    X_train = np.delete(X_train, 0, axis=1)
    X_test = np.delete(X_test, 0, axis=1)

    X = np.delete(X, 0, axis=1)

    classifier = fitClassifier(classifier, X_train, X_test, y_train, y_test)

    seeError(classifier, X_test, y_test, i2c, fname_test)

    return getPredictions(classifier, X, y, test_data, i2c)


def getPredictions(classifier, X, y, test_data, i2c):
    classifier.fit(X, y)

    return probaToLabels(classifier.predict_proba(test_data.drop('label', axis=1).values), i2c, k=3)


def fitClassifier(classifier, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    print "estimation : ", classifier.score(X_test, y_test)

    return classifier


def probaToLabels(preds, i2c, k=3):
    ans = []
    ids = []

    for p in preds:
        idx = np.argsort(p)[:: 1]
        ids.append([i for i in idx[:k]])
        ans.append(' '.join([i2c[i] for i in idx[:k]]))

    return ans


def idToLabel(preds, i2c):
    ans = []

    for p in preds:
        ans.append(i2c[p])

    return ans


def transformLabel(train_data):
    X = train_data.drop('label', axis=1).values
    labels = np.sort(np.unique(train_data.label.values))
    c2i = {}
    i2c = {}
    for i, c in enumerate(labels):
        c2i[c] = i
        i2c[i] = c
    y = np.array([c2i[x] for x in train_data.label.values])

    return X, y, i2c


def seeError(classifier, X_test, y_test, i2c, fname_test):
    y_predict = probaToLabels(classifier.predict_proba(X_test), i2c, k=3)

    df_error = pd.DataFrame()

    df_error["fname"] = fname_test
    df_error["correct_answer"] = idToLabel(y_test, i2c)
    df_error["answer"] = y_predict

    # df_error = df_error.drop(df_error[df_error.correct_answer in df_error.answer].index)

    for index, ligne in enumerate(df_error["correct_answer"]):
        if (ligne in df_error["answer"][index]):
            df_error = df_error.drop([index])

    #     if (ligne == 0):
    #         df_error = df_error.drop([index])
    #
    # df_error["correct_answer"] = idToLabel(df_error["correct_answer"], i2c)
    # df_error["answer"] = idToLabel(df_error["answer"], i2c)
    # df_error = df_error.drop(["diff"], axis=1)

    df_error.sort_values(["correct_answer"]).to_csv("error.csv", index=False)
