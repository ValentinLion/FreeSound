import numpy as np
import pandas as pd

import librosa
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
        ft2 = librosa.feature.zero_crossing_rate(file)[0]
        ft3 = librosa.feature.spectral_rolloff(file)[0]
        ft4 = librosa.feature.spectral_centroid(file)[0]
        ft5 = librosa.feature.spectral_contrast(file)[0]
        ft6 = librosa.feature.spectral_bandwidth(file)[0]
        ft1_trunc = np.hstack((np.mean(mfcc, axis=1), np.std(mfcc, axis=1)))
        ft2_trunc = np.hstack((np.mean(ft2), np.std(ft2)))
        ft3_trunc = np.hstack((np.mean(ft3), np.std(ft3)))
        ft4_trunc = np.hstack((np.mean(ft4), np.std(ft4)))
        ft5_trunc = np.hstack((np.mean(ft5), np.std(ft5)))
        ft6_trunc = np.hstack((np.mean(ft6), np.std(ft6)))
        return pd.Series(np.hstack((ft1_trunc, ft2_trunc, ft3_trunc, ft4_trunc, ft5_trunc, ft6_trunc)))

    return pd.Series([0] * n_mfcc+5 *2)


def get_mfcc(file, rate, n_mfcc):
    try:
        ft = librosa.feature.mfcc(file, sr=rate, n_mfcc=n_mfcc)
        ft_trunc = np.hstack((np.mean(ft, axis=1), np.std(ft, axis=1), np.max(ft, axis=1), np.min(ft, axis=1)))
        return ft_trunc

    except:
        return pd.Series([0] * 80)


def get_features(name, path, rate):
    n_mfcc = 20

    file, _ = librosa.core.load(path + name, sr=rate)

    ft1 = get_mfcc(file, rate, n_mfcc)

    return pd.Series(np.hstack(ft1))


def apply_audio_analys(df, path, rate):
    features = pd.DataFrame(df['fname'].progress_apply(get_features, path=path, rate=rate))
    # features_stft = pd.DataFrame(df['fname'].progress_apply(get_fft, path=path, rate=rate))

    # return pd.concat([df, features_mfcc, features_stft], axis=1)
    return pd.concat([df, features], axis=1)
