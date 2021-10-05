from __future__ import print_function
import keras
from keras import Model
from keras.layers import Input, Dense, Lambda, Dropout, Flatten, Reshape, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv1D

from keras.optimizers import Adam
import numpy as np



def tcn_block(x_in, dil_rate_1, dil_rate_2, n_filters, k_size, dropout_rate):

  x = x_in
  x = Conv1D(n_filters, k_size, activation=None, padding='same', dilation_rate=dil_rate_1)(x)
  x = Activation('relu')(x)
  x = Dropout(dropout_rate)(x)

  x = Conv1D(n_filters, k_size, activation=None, padding='same', dilation_rate=dil_rate_2)(x)
  x = Activation('relu')(x)
  x = Dropout(dropout_rate)(x)


  x_short = Conv1D(n_filters, 1, activation=None, padding='same')(x_in)

  x = keras.layers.add([x_short,x])
  return x


def build_tcn(input_shape, n_classes):
  input_x = Input(input_shape)
  x = input_x

  x = tcn_block(x, 1, 1, 32, 7, 0.0)
  x = tcn_block(x, 1, 1, 64, 5, 0.2)
  x = tcn_block(x, 2, 2, 64, 5, 0.2)
  x = tcn_block(x, 4, 1, 64, 5, 0.2)

  x = Dense(64, activation='relu')(x)
  x = Dropout(0.8)(x)

  x_softmax = Dense(n_classes, activation='softmax')(x)

  model = Model(input_x, x_softmax)
  model.compile(loss='categorical_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])
  return model

def build_rnn(input_shape, n_classes):
  input_x = Input(input_shape)

  from keras.layers.recurrent import LSTM, GRU

  x = input_x
  x = tcn_block(x, 1, 1, 32, 7, 0.0)
  x = GRU(64, return_sequences=True, recurrent_dropout=0.2, dropout=0.2)(x)
  x = GRU(64, return_sequences=True, recurrent_dropout=0.2, dropout=0.2)(x)
  x = Dense(64, activation='relu')(x)
  x = Dropout(0.5)(x)
  x_softmax = Dense(n_classes, activation='softmax')(x)

  model = Model(input_x, x_softmax)
  model.compile(loss='categorical_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])
  return model

#
# def build_spectrum_cnn(input_shape, n_classes):
#   input_x = Input(input_shape)
#   x = Conv2D(32, (7,7), activation='relu', padding='valid')(input_x)
#   x = Conv2D(32, (3, 3), activation='relu', padding='valid')(x)
#   x = MaxPooling2D()(x)
#   x = Conv2D(32, (3, 3), activation='relu', padding='valid')(x)
#   x = MaxPooling2D()(x)
#   x = Conv2D(32, (3, 3), activation='relu', padding='valid')(x)
#   x = MaxPooling2D()(x)
#   x = Conv2D(32, (3, 3), activation='relu', padding='valid')(x)
#   x = MaxPooling2D()(x)
#
#   x = Flatten()(x)
#   x = Dense(64, activation='relu')(x)
#   x = Dropout(0.5)(x)
#   x = Dense(64, activation='relu')(x)
#
#   x_softmax = Dense(n_classes, activation='softmax')(x)
#
#   model = Model(input_x, x_softmax)
#   model.compile(loss='categorical_crossentropy',
#                 optimizer=Adam(),
#                 metrics=['accuracy'])
#
#   return model



