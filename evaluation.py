import numpy as np
import pandas as pd
from scipy.stats import randint as sp_randint
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
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
    classifier = RandomForestClassifier(bootstrap=False, max_depth=70,
                                        max_features='sqrt', max_leaf_nodes=None,
                                        min_impurity_decrease=0.0, min_impurity_split=None,
                                        min_samples_leaf=2, min_samples_split=8,
                                        min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=-1,
                                        oob_score=False, random_state=None, verbose=0, warm_start=False)

    return classify(X, y, test_data, i2c, classifier)


def randomizedSearchCV(X, y):
    X = np.delete(X, 0, axis=1)

    clf = RandomForestClassifier(n_estimators=20)

    param_dist = {"max_depth": [3, None],
                  "max_features": sp_randint(1, 11),
                  "min_samples_split": sp_randint(2, 11),
                  "min_samples_leaf": sp_randint(1, 11),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    # run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search)

    print random_search.fit(X, y)

    param_grid = {"max_depth": [3, None],
                  "max_features": [1, 3, 10],
                  "min_samples_split": [2, 3, 10],
                  "min_samples_leaf": [1, 3, 10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    # run grid search
    grid_search = GridSearchCV(clf, param_grid=param_grid)
    print grid_search.fit(X, y)


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
    classifier = classifier.fit(X, y)

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
    y_predict = probaToLabels(classifier.predict_proba(X_test), i2c, k=10)

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

    df_error.sort_values(["correct_answer"]).to_csv("csv/error.csv", index=False)
