import pandas as pd

import librosa
import numpy as np
from sklearn import preprocessing
from tqdm import tqdm

tqdm.pandas()


def get_mfcc(name, path, rate):
    n_mfcc = 20

    file, _ = librosa.core.load(path + name, sr=rate)

    if (len(file) != 0):
        mfcc = librosa.feature.mfcc(file, sr=rate, n_mfcc=n_mfcc)
        ft2 = librosa.feature.zero_crossing_rate(file)[0]
        ft3 = librosa.feature.spectral_rolloff(file)[0]
        ft4 = librosa.feature.spectral_centroid(file)[0]
        ft5 = librosa.feature.spectral_contrast(file)[0]
        ft6 = librosa.feature.spectral_bandwidth(file)[0]
        return pd.Series(np.hstack((preprocessing.scale(np.mean(mfcc, axis=1)), np.mean(ft2), np.mean(ft3), np.mean(ft4), np.mean(ft5), np.mean(ft6))))
    return pd.Series([0] * n_mfcc+5)


def apply_mfcc(df, path, rate):
    features = pd.DataFrame(df['fname'].progress_apply(get_mfcc, path=path, rate=rate))

    return pd.concat([df, features], axis=1)
