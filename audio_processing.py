import librosa
import numpy as np

HZ = 44100
N_MFCC = 20


def addMFCCFromFile(df, path):
    for i in range(0, N_MFCC):
        df["mfcc" + str(i)] = np.nan

    index = 0

    for row in df.iterrows():

        file_path = path + row[1].fname
        y, _ = librosa.load(file_path)

        mfcc = librosa.feature.mfcc(y=y, sr=HZ, n_mfcc=N_MFCC)

        for i in range(0, N_MFCC):
            df.loc[index, "mfcc" + str(i)] = mfcc.item(i)

        index = index + 1

        print index

    df.to_csv(path_or_buf="train_mfcc.csv", sep=',', index=False)

    return df
