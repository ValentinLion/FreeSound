#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd

df_training = pd.read_csv("train.csv", sep=",")

list_test = os.listdir("/home/valentin/Téléchargements/audio_test")

labels = df_training.label.unique()

proba = []

for label in labels:
    print("label:" + label)
    print((df_training.label == label).sum())
    print((df_training.label == label).mean())
    proba.append((df_training.label == label).mean())

soumission_proba = []

for i in range(0, len(list_test)):
    soumission_proba.append(
        np.random.choice(labels, p=proba) + " " + np.random.choice(labels, p=proba) + " " + np.random.choice(labels,
                                                                                                             p=proba))

results = pd.DataFrame({'fname': list_test})
results["label"] = soumission_proba

results.to_csv(path_or_buf="predict.csv", sep=',', index=False)
