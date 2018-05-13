#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

import description
import submission

df_training = pd.read_csv("train.csv", sep=",")
df_test = pd.read_csv("sample_submission.csv", sep=",")

submission.fake_submission(df_training, df_test)
print("Finish")

description.description(df_training)

print("Finish")
