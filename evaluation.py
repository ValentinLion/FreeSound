import pandas as pd

import numpy as np
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def svmPredictions(X, y, test_data, i2c):
    classifier = svm.SVC(kernel="poly")
    printEstimation(classifier, X, y)
    return getPredictions(classifier, X, y, test_data, i2c)


def neighborsPredictions(neighbors, X, y, test_data, i2c):
    classifier = KNeighborsClassifier(n_neighbors=neighbors)
    printEstimation(classifier, X, y)
    return getPredictions(classifier, X, y, test_data, i2c)


def XGBPredictions(X, y, test_data, i2c):
    classifier = GradientBoostingClassifier(max_depth=5, learning_rate=0.05, n_estimators=100,
                                            random_state=0)
    printEstimation(classifier, X, y)
    return getPredictions(classifier, X, y, test_data, i2c)


def randomForestPredictions(X, y, test_data, i2c):
    classifier = RandomForestClassifier(bootstrap=False, max_depth=70,
                                        max_features='sqrt', max_leaf_nodes=None,
                                        min_impurity_decrease=0.0, min_impurity_split=None,
                                        min_samples_leaf=2, min_samples_split=8,
                                        min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=-1,
                                        oob_score=False, random_state=None, verbose=0, warm_start=False)

    printEstimation(classifier, X, y, i2c)
    return getPredictions(classifier, X, y, test_data, i2c)


def getPredictions(classifier, X, y, test_data, i2c):
    classifier.fit(X, y)

    return probaToLabels(classifier.predict_proba(test_data.drop('label', axis=1).values), i2c, k=3)


def printEstimation(classifier, X, y, i2c):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, shuffle=True)
    classifier.fit(X_train, y_train)
    seeError(classifier, X_test, y_test, i2c)
    print "estimation : ", classifier.score(X_test, y_test)


def probaToLabels(preds, i2c, k=3):
    ans = []
    ids = []

    for p in preds:
        idx = np.argsort(p)[::-1]
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


def seeError(classifier, X_test, y_test, i2c):
    y_predict = classifier.predict(X_test)

    df_error = pd.DataFrame()

    df_error["correct_answer"] = y_test
    df_error["answer"] = y_predict
    df_error["diff"] = df_error["correct_answer"] - df_error["answer"]

    for index, ligne in enumerate(df_error["diff"]):
        if (ligne == 0):
            df_error = df_error.drop([index])

    df_error["correct_answer"] = idToLabel(df_error["correct_answer"], i2c)
    df_error["answer"] = idToLabel(df_error["answer"], i2c)
    df_error = df_error.drop(["diff"], axis=1)

    df_error.to_csv("error.csv", index=False)
