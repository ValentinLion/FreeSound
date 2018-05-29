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


def get_mfcc(file, rate, n_mfcc):
    try:
        ft = librosa.feature.mfcc(file, sr=rate, n_mfcc=n_mfcc)
        ft_trunc = np.hstack((np.mean(ft, axis=1), np.std(ft, axis=1), np.max(ft, axis=1), np.min(ft, axis=1)))
        return ft_trunc

    except:
        return pd.Series([0] * 80)

def get_zero_crossing(file):
    try:
        ft = librosa.feature.zero_crossing_rate(file)[0]
        ft_trunc = np.hstack((np.mean(ft), np.std(ft)))
        return ft_trunc

    except:
        return pd.Series([0] * 8)

def get_spectral_rolloff(file):
    try:
        ft = librosa.feature.spectral_rolloff(file)[0]
        ft_trunc = np.hstack((np.mean(ft), np.std(ft)))
        return ft_trunc

    except:
        return pd.Series([0] * 8)

def get_spectral_centroid(file):
    try:
        ft = librosa.feature.spectral_centroid(file)[0]
        ft_trunc = np.hstack((np.mean(ft), np.std(ft)))
        return ft_trunc

    except:
        return pd.Series([0] * 8)

def get_spectral_contrast(file):
    try:
        ft = librosa.feature.spectral_contrast(file)[0]
        ft_trunc = np.hstack((np.mean(ft), np.std(ft)))
        return ft_trunc

    except:
        return pd.Series([0] * 8)


def get_spectral_bandwidth(file):
    try:
        ft = librosa.feature.spectral_bandwidth(file)[0]
        ft_trunc = np.hstack((np.mean(ft), np.std(ft)))
        return ft_trunc

    except:
        return pd.Series([0] * 8)

def get_features(name, path, rate):
    n_mfcc = 20

    file, _ = librosa.core.load(path + name, sr=rate)

    ft1 = get_mfcc(file, rate, n_mfcc)
    ft2 = get_zero_crossing(file)
    ft3 = get_spectral_centroid(file)
    ft4 = get_spectral_contrast(file)
    ft5 = get_spectral_bandwidth(file)
    ft6 = get_spectral_rolloff(file)

    return pd.Series(np.hstack(ft1, ft2, ft3, ft4, ft5, ft6))


def apply_audio_analys(df, path, rate):
    features = pd.DataFrame(df['fname'].progress_apply(get_features, path=path, rate=rate))
    # features_stft = pd.DataFrame(df['fname'].progress_apply(get_fft, path=path, rate=rate))

    # return pd.concat([df, features_mfcc, features_stft], axis=1)
    return pd.concat([df, features], axis=1)
