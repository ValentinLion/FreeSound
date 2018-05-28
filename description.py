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

    # Distribution
    df_display = df_training[["fname", "label", "manually_verified"]]

    category_group = df_display.groupby(['label', 'manually_verified']).count()
    plot = category_group.unstack().reindex(category_group.unstack().sum(axis=1).sort_values().index) \
        .plot(kind='bar', stacked=True, title="Nombre d'echantillons selon le type de son")
    plot.set_xlabel("Type")
    plot.set_ylabel("Nombre")
    plt.show()

    # Missing data
    missing_data = df_training.isnull().sum().sort_values(ascending=False)
    tab_missing_data = pd.concat([missing_data], axis=1, keys=["Total"])

    print("Missing Data")
    print(tab_missing_data)

    # Manually Verified
    data_manually_verified = df_training["manually_verified"].value_counts();
    labels_manually_verified = data_manually_verified.index

    plt.pie(data_manually_verified, labels=labels_manually_verified, startangle=90, autopct='%.1f%%')
    plt.title("Data manually verified 0:No 1:Yes")
    plt.show()
