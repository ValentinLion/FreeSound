import pandas as pd

import librosa
import numpy as np
from tqdm import tqdm

tqdm.pandas()


def get_fft(name, path, rate):
    n_fft = 20

    file, _ = librosa.core.load(path + name, sr=rate)

    if (len(file) != 0):
        fft = librosa.stft(file, n_fft=n_fft)
        return pd.Series(np.std(fft, axis=1))

    return pd.Series([0] * ((n_fft / 2) + 1))

def get_mfcc(name, path, rate):
    n_mfcc = 20

    file, _ = librosa.core.load(path + name, sr=rate)

    if (len(file) != 0):
        mfcc = librosa.feature.mfcc(file, sr=rate, n_mfcc=n_mfcc)
        return pd.Series(np.hstack((np.mean(mfcc, axis=1), np.std(mfcc, axis=1))))

    return pd.Series([0] * n_mfcc * 2)


def apply_audio_analys(df, path, rate):
    features_mfcc = pd.DataFrame(df['fname'].progress_apply(get_mfcc, path=path, rate=rate))
    # features_stft = pd.DataFrame(df['fname'].progress_apply(get_fft, path=path, rate=rate))

    # return pd.concat([df, features_mfcc, features_stft], axis=1)
    return pd.concat([df, features_mfcc], axis=1)
