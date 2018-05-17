import numpy as np

import os
import shutil

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import wave

matplotlib.style.use('ggplot')

train = pd.read_csv("../FreeSoundData/train.csv")
test = pd.read_csv("../FreeSoundData/sample_submission.csv")

path = "../FreeSoundData"

listLabels = pd.unique(train["label"])
prediction = test.drop('label',1)

prediction['label'] = prediction['fname'].apply(lambda f: listLabels[np.random.randint(0, high=listLabels.size-1)]) 

prediction.to_csv("prediction.csv", sep=',', index=False)
