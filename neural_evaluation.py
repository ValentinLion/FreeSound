from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.models import Sequential
import numpy as np
import evaluation

def createModel(X,y):
    X = preprocessing.scale(X)
    X=np.expand_dims(X, axis=2)
    print(X.shape)
    X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2, random_state=10, shuffle=True)
    model = get_model() 
    model.fit(X, y,epochs=150, batch_size=5)
    #print("estimation : ", model.evaluate(X_val, y_val))
    
def get_model():
    inp = Input(shape=(21, 1))
    img_1 = Convolution1D(64, kernel_size=1, activation=activations.relu, padding="valid")(inp)
    img_1 = Convolution1D(32, kernel_size=1, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=16)(img_1)
    """
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=1, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=1, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=1)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=1, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=1, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=1)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(256, kernel_size=1, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(256, kernel_size=1, activation=activations.relu, padding="valid")(img_1)
    """
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    dense_1 = Dense(64, activation=activations.relu)(img_1)
    dense_1 = Dense(41, activation=activations.sigmoid)(dense_1)
    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.0001)

    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['acc'])
    model.summary()
    return model