#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

df = pd.read_csv("train.csv", sep=",")
print df["label"].count()
df_test = df.groupby(['label']).size()
print df_test
#df_test.to_csv('statistics.csv', sep=',', index=False)

np.random.seed(0)
df = pd.DataFrame({'state': ['CA', 'WA', 'CO', 'AZ'] * 3,
               'office_id': list(range(1, 7)) * 2,
               'sales': [np.random.randint(100000, 999999) for _ in range(12)]})

state_office = df.groupby(['state', 'office_id']).agg({'sales': 'sum'})
state = df.groupby(['state']).agg({'sales': 'sum'})
state_office.div(state, level='state') * 100
print state_office
