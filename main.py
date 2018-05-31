#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

import audio_processing
import evaluation

useAudioAnalysCSV = True
rate = 44100

path_audio_train = "/home/valentin/Téléchargements/audio_train/"
path_audio_test = "/home/valentin/Téléchargements/audio_test/"

# description.description(pd.read_csv("csv/train.csv", sep=","))

if (useAudioAnalysCSV):
    df_dataframe = pd.read_csv("csv/train_analys_all.csv", sep=",", index_col=False)
    df_test = pd.read_csv("csv/sample_submission_analys_all.csv", sep=",", index_col=False)
else:
    df_dataframe = pd.read_csv("csv/train.csv", sep=",", index_col=False)
    df_dataframe = audio_processing.apply_audio_analys(df_dataframe, path_audio_train, rate)

    df_dataframe.to_csv("csv/train_analys.csv", sep=",", index=False)

    df_test = pd.read_csv("csv/sample_submission.csv", sep=",", index_col=False)
    df_test = audio_processing.apply_audio_analys(df_test, path_audio_test, rate)

    df_test.to_csv("csv/sample_submission_analys.csv", sep=",", index=False)

df_dataframe_without_manually_verified = df_dataframe.drop('manually_verified', axis=1)

df_test_without_fname = df_test.drop('fname', axis=1)

X, y, i2c = evaluation.transformLabel(df_dataframe_without_manually_verified)

# evaluation.randomizedSearchCV(X,y)
#exit()

preds = evaluation.randomForestPredictions(X, y, df_test_without_fname, i2c)

df_submission = pd.DataFrame()
df_submission['fname'] = df_test['fname']
df_submission['label'] = preds
df_submission.to_csv("csv/submission.csv", index=False, index_label=False)

print("Finish")