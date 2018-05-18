# https://www.kaggle.com/amlanpraharaj/random-forest-using-mfcc-features/notebook

import numpy as np 
import pandas as pd
import os
import librosa
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm, tqdm_pandas

tqdm.pandas()



SAMPLE_RATE = 44100

path_train_audio = "../FreeSoundData/audio_train/"
path_test_audio = "../FreeSoundData/audio_test/"
path_csvfiles = "./"

#loading data
audio_train_files = os.listdir(path_train_audio)
audio_test_files = os.listdir(path_test_audio)

train = pd.read_csv(path_csvfiles+'train.csv')
submission = pd.read_csv(path_csvfiles+'sample_submission.csv')

def clean_filename(fname, string):   
    file_name = fname.split('/')[1]
    if file_name[:2] == '__':        
        file_name = string + file_name
    return file_name

#returns mfcc features with mean and standard deviation along time
def get_mfcc(name, path):
    b, _ = librosa.core.load(path + name, sr = SAMPLE_RATE)
    assert _ == SAMPLE_RATE
    try:
        gmm = librosa.feature.mfcc(b, sr = SAMPLE_RATE, n_mfcc=20)
        return pd.Series(np.hstack((np.mean(gmm, axis=1), np.std(gmm, axis=1))))
    except:
        print('bad file')
        return pd.Series([0]*40)
		
#preparing data
train_data = pd.DataFrame()
train_data['fname'] = train['fname']
test_data = pd.DataFrame()
test_data['fname'] = audio_test_files

train_data = train_data['fname'].progress_apply(get_mfcc, path=path_train_audio)
print('done loading train mfcc')
test_data = test_data['fname'].progress_apply(get_mfcc, path=path_test_audio)
print('done loading test mfcc')

train_data['label'] = train['label']
test_data['label'] = np.zeros((len(audio_test_files)))

print(train_data.head())

X = train_data.drop('label', axis=1)
feature_names = list(X.columns)
X = X.values
labels = np.sort(np.unique(train_data.label.values))
num_class = len(labels)
c2i = {}
i2c = {}
for i, c in enumerate(labels):
    c2i[c] = i
    i2c[i] = c
y = np.array([c2i[x] for x in train_data.label.values])

rfc = RandomForestClassifier(n_estimators = 150)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10, shuffle = True)
rfc.fit(X_train, y_train)

def proba2labels(preds, i2c, k=3):
    ans = []
    ids = []
    for p in preds:
        idx = np.argsort(p)[::-1]
        ids.append([i for i in idx[:k]])
        ans.append(' '.join([i2c[i] for i in idx[:k]]))

    return ans, ids
	
#checking the accuracy of the model
print(rfc.score(X_val, y_val))
rfc.fit(X, y)
str_preds, _ = proba2labels(rfc.predict_proba(test_data.drop('label', axis = 1).values), i2c, k=3)

# Prepare submission
subm = pd.DataFrame()
subm['fname'] = audio_test_files
subm['label'] = str_preds
subm.to_csv(path_csvfiles+'submission.csv', index=False)
