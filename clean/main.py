#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

import audio_processing
import description
import evaluation

useMFCCCSV = True
rate = 44100

path_audio_train = "/home/valentin/Téléchargements/audio_train/"
path_audio_test = "/home/valentin/Téléchargements/audio_test/"

description.description(pd.read_csv("train.csv", sep=","))

if (useMFCCCSV):
    df_dataframe = pd.read_csv("train_mfcc.csv", sep=",")
    df_test = pd.read_csv("sample_submission_mfcc.csv", sep=",")

else:
    df_dataframe = pd.read_csv("train.csv", sep=",")
    df_dataframe = audio_processing.apply_mfcc(df_dataframe, path_audio_train, rate)

    df_dataframe.to_csv("train_mfcc.csv", sep=",")

    df_test = pd.read_csv("sample_submission.csv", sep=",")
    df_test = audio_processing.apply_mfcc(df_test, path_audio_test, rate)

    df_test.to_csv("sample_submission_mfcc.csv", sep=",")

df_dataframe_without_fname_manually_verified = df_dataframe.drop('fname', axis=1).drop('manually_verified', axis=1)
df_test_without_fname = df_test.drop('fname', axis=1)

X, y, i2c = evaluation.transformLabel(df_dataframe_without_fname_manually_verified)
preds = evaluation.randomForestPredictions(X, y, df_test_without_fname, i2c)

df_submission = pd.DataFrame()
df_submission['fname'] = df_test['fname']
df_submission['label'] = preds
df_submission.to_csv("submission.csv", index=False)

print("Finish")
