from sklearn.ensemble import RandomForestClassifier


def RandomForest(X_train, y_train, X_test, y_test):
    rfc = RandomForestClassifier(n_estimators=150)
    rfc.fit(X_train, y_train)

    print(rfc.score(X_test, y_test))
