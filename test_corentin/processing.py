import librosa
import numpy as np 
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm


def clean_filename(fname, string):   
    file_name = fname.split('/')[1]
    if file_name[:2] == '__':        
        file_name = string + file_name
    return file_name

def get_mfcc(name, path, rate):
    b, _ = librosa.core.load(path + name, sr = rate)
    assert _ == rate
    try:
        gmm = librosa.feature.mfcc(b, sr = rate, n_mfcc=20)
        return pd.Series(np.hstack((np.mean(gmm, axis=1), np.std(gmm, axis=1))))
    except:
        print('bad file')
        return pd.Series([0]*40)

def toCSV( data, path, filename):
    data.to_csv(path+filename, index=False)

def importCSV(path, filename):
    return pd.read_csv(path+filename)

def apply_mfcc(trainCSV, path_train_audio,path_test_audio,rate):
    audio_test_files = os.listdir(path_test_audio)
    train_data = pd.DataFrame()
    train_data['fname'] = trainCSV['fname']
    test_data = pd.DataFrame()
    test_data['fname'] = audio_test_files

    train_data = train_data['fname'].progress_apply(get_mfcc, path=path_train_audio, rate=rate)
    print('done loading train mfcc')
    test_data = test_data['fname'].progress_apply(get_mfcc, path=path_test_audio, rate=rate)
    print('done loading test mfcc')
    train_data['label'] = trainCSV['label']
    test_data['label'] = np.zeros((len(audio_test_files)))
    
    print(train_data.head())

    return train_data, test_data

def transform_data(train_data):
    X = train_data.drop('label', axis=1)
    X = X.values
    labels = np.sort(np.unique(train_data.label.values))
    c2i = {}
    i2c = {}
    for i, c in enumerate(labels):
        c2i[c] = i
        i2c[i] = c
    y = np.array([c2i[x] for x in train_data.label.values])
    return X,y,i2c

def proba2labels(preds, i2c, k=3):
    ans = []
    ids = []
    for p in preds:
        idx = np.argsort(p)[::-1]
        ids.append([i for i in idx[:k]])
        ans.append(' '.join([i2c[i] for i in idx[:k]]))

    return ans

def randomForestPredictions(X,y,test_data,i2c):
    classifier = RandomForestClassifier(bootstrap=False, max_depth=70,
           max_features='sqrt', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=2, min_samples_split=8,
           min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=-1,
           oob_score=False, random_state=None, verbose=0, warm_start=False)
    printEstimation(classifier,X,y)
    return getPredictions(classifier,X,y,test_data,i2c)


def svmPredictions(X,y,test_data,i2c):
    classifier = svm.SVC(kernel="poly")
    printEstimation(classifier,X,y)
    return getPredictions(classifier,X,y,test_data,i2c)

def neighborsPredictions(neighbors,X,y,test_data,i2c):
    classifier = KNeighborsClassifier(n_neighbors=neighbors)
    printEstimation(classifier,X,y)
    return getPredictions(classifier,X,y,test_data,i2c)

def XGBPredictions(X,y,test_data,i2c):
    classifier = GradientBoostingClassifier(max_depth=5, learning_rate=0.05, n_estimators=100,
                    random_state=0)
    printEstimation(classifier,X,y)
    return getPredictions(classifier,X,y,test_data,i2c)

def getPredictions(classifier,X,y,test_data,i2c):
    classifier.fit(X, y)
    return proba2labels(classifier.predict_proba(test_data.drop('label', axis = 1).values), i2c, k=3)


def predictionsToCSV(predictions,path_test_audio,path,name):
    audio_test_files = os.listdir(path_test_audio)
    subm = pd.DataFrame()
    subm['fname'] = audio_test_files
    subm['label'] = predictions
    subm.to_csv(path+name, index=False)

def printEstimation(classifier,X,y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10, shuffle = True)
    classifier.fit(X_train, y_train)
    print "estimation : ", classifier.score(X_val, y_val)
