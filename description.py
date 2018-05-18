#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd


def description(df_training):
    # Nb data
    print "Number of data : " + str(df_training.shape[0])

    # Labels
    labels = df_training.label.unique()
    print "Number of labels : " + str(len(labels))
    print labels

    # Missing data
    missing_data = df_training.isnull().sum().sort_values(ascending=False)
    tab_missing_data = pd.concat([missing_data], axis=1, keys=["Total"])
    print(tab_missing_data)

    # Manually Verified
    data_manually_verified = df_training["manually_verified"].value_counts();
    labels_manually_verified = data_manually_verified.index

    plt.pie(data_manually_verified, labels=labels_manually_verified, startangle=90, autopct='%.1f%%')
    plt.title("Data manually verified 0:No 1:Yes")
    plt.show()

    # # MFCC
    #
    # for row in df_training.iterrows():
    #
    #     if (row[1].label == 'Knock'):
    #         file_path = "/home/valentin/Téléchargements/audio_train/" + row[1].fname
    #         y, sr = librosa.load(file_path)
    #
    #         mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    #
    #         plt.figure(figsize=(10, 4))
    #         librosa.display.specshow(mfccs, x_axis='time')
    #         plt.colorbar()
    #         plt.title('MFCC')
    #         plt.tight_layout()
    #         plt.show()
