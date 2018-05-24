import pandas as pd

import librosa
import numpy as np
from tqdm import tqdm

tqdm.pandas()


def get_mfcc(name, path, rate):
    n_mfcc = 20

    file, _ = librosa.core.load(path + name, sr=rate)

    if (len(file) != 0):
        mfcc = librosa.feature.mfcc(file, sr=rate, n_mfcc=n_mfcc)
        return pd.Series(np.hstack((np.mean(mfcc, axis=1), np.std(mfcc, axis=1))))

    return pd.Series([0] * n_mfcc * 2)


def apply_mfcc(df, path, rate):
    features = pd.DataFrame(df['fname'].progress_apply(get_mfcc, path=path, rate=rate))

    return pd.concat([df, features], axis=1)
