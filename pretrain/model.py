import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.layers import *

def model1(input_shape, name="embd_model"):
    I = Input(shape=(input_shape))
    X = Reshape((-1, input_shape[-1], 1))(I)
    X = BatchNormalization()(X)
    X = Conv2D(filters=96, kernel_size=(11,5), strides=(4,1))(X)
    X = ReLU()(X)
    X = BatchNormalization()(X)
    X = MaxPool2D(pool_size=(3,3), strides=(2,2))(X)
    X = Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding="same")(X)
    X = ReLU()(X)
    X = BatchNormalization()(X)
    X = MaxPool2D(pool_size=(3,3), strides=(2,2))(X)
    X = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="same")(X)
    X = ReLU()(X)
    X = BatchNormalization()(X)
    X = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="same")(X)
    X = ReLU()(X)
    X = BatchNormalization()(X)
    X = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same")(X)
    X = ReLU()(X)
    X = BatchNormalization()(X)
    X = MaxPool2D(pool_size=(3,3), strides=(2,2))(X)
    X = Flatten()(X)
    X = Dense(4096)(X)
    X = ReLU()(X)
    X = Dropout(0.5)(X)
    out = Dense(256)(X)
    return K.Model(inputs = I, outputs= out, name=name)


def embd_model(input_shape, name="embd_model"):
    # better than alexnet
    I = Input(shape=(input_shape))
    X = BatchNormalization()(I)
    X = Bidirectional(LSTM(128, return_sequences=True))(X)
    X = BatchNormalization()(X)
    X = Bidirectional(LSTM(128, return_sequences=True))(X)
    X = BatchNormalization()(X)
    X = Bidirectional(LSTM(128, return_sequences=True))(X)
    X = GlobalAveragePooling1D()(X)
    out = Dense(256)(X)
    return K.Model(inputs = I, outputs= out, name=name)


