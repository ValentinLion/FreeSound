#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path
import pandas as pd

from sklearn import model_selection
from sklearn import preprocessing

import audio_processing
import description
import evaluation

if (os.path.isfile("train_mfcc.csv")):
    df_dataframe = pd.read_csv("train_mfcc.csv", sep=",")

else:
    df_dataframe = pd.read_csv("train.csv", sep=",")
    # Ajout de la colonne mfcc
    # #df_dataframe = df_dataframe.head()
    df_dataframe = audio_processing.addMFCCFromFile(df_dataframe, "/home/valentin/Téléchargements/audio_train/")

df_submission = pd.read_csv("sample_submission.csv", sep=",")

description.description(df_dataframe)

# Transformation des donnees qui ne sont pas dans un format numerique
le = preprocessing.LabelEncoder()
df_dataframe.fname = le.fit_transform(df_dataframe.fname)
df_dataframe.label = le.fit_transform(df_dataframe.label)

# Suppression des attribut inutile
df_dataframe = df_dataframe.drop(['manually_verified'], axis=1)

# Suppression de la colonne "label" dans le training set
X = df_dataframe.drop(['label'], axis=1).values

# Recuperation de la colonne "label"
y = df_dataframe['label'].values

# Separation des tuples en training set(70%) et test set(30%)
# avec crossvalidation
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

evaluation.RandomForest(X_train, y_train, X_test, y_test)

# submission.fake_submission(df_training, df_test)
print("Finish")



print("Finish")
