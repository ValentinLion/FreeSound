#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

import numpy as np


def fake_submission(df_training, df_test):
    proba = []
    labels = df_training.label.unique()

    for label in labels:
        print("label:" + label)
        print((df_training.label == label).sum())
        print((df_training.label == label).mean())
        proba.append((df_training.label == label).mean())

    soumission_proba = []

    for i in range(0, df_test.fname.count()):
        soumission_proba.append(
            np.random.choice(labels, p=proba) + " " + np.random.choice(labels, p=proba) + " " + np.random.choice(labels,
                                                                                                                 p=proba))

    results = pd.DataFrame({'fname': df_test["fname"]})
    results["label"] = soumission_proba

    results.to_csv(path_or_buf="predict.csv", sep=',', index=False)
