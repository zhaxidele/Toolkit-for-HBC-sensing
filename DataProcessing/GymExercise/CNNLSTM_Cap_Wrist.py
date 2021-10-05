

#import matplotlib.pyplot as plt
import csv
import os
import fnmatch
import sys
import time
from multiprocessing import Queue
from multiprocessing import Process
import pandas as pd
from scipy import signal
import numpy as np

import tensorflow as tf
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, Dense, Reshape, Conv1D
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import EarlyStopping

from keras.preprocessing import sequence
from keras.datasets import imdb
from keras import regularizers
from keras import layers

from keras.models import Input,Model

from keras.layers import LSTM, Permute
from keras.layers.wrappers import TimeDistributed
import pickle as cp
from importlib import reload
import numpy as np
import csv

import pickle as cPickle
#import cPickle as cp
#import theano.tensor as T
#from sliding_window import sliding_window
#import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as metrics
from sklearn.utils import class_weight
from keras.optimizers import SGD
import time
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

Header_9 = ["LocalSerie", "Win_Num", "Win_In_Num", "Object", "Day", "Workout", "Workout_time", "Position", "A_x", "A_y","A_z", "G_x", "G_y", "G_z", "C_1"]

Name_dic = {1:'Chentianyu', 2:'Cuihaiyang', 3:'Dangfeng', 4:'Hanyue', 5:'Songshiqi', 6:'Wangxueying', 7:'Yanzehui', 8:'Yujia', 9:'Zhangyu', 10:'Zhangzeyu', 11:'Test'}
Name_dic_r = {'Chentianyu':1, 'Cuihaiyang':2, 'Dangfeng':3, 'Hanyue':4, 'Songshiqi':5, 'Wangxueying':6, 'Yanzehui':7, 'Yujia':8, 'Zhangyu':9, 'Zhangzeyu':10, 'Test':11}

Workout = {1:"Adductor",2:"ArmCurl",3:"BenchPress",4:"LegCurl",5:"LegPress",6:"Riding",7:"RopeSkipping",8:"Running",9:"SquatConcrete",10:"SquatRubber",11:"SquatWood",12:"StairClimber",13:"Walking"}
Workout_1 = {1:"Adductor",2:"ArmCurl",3:"BenchPress",4:"LegCurl",5:"LegPress",6:"Riding",7:"RopeSkipping",8:"Running",9:"Squat",10:"StairClimber",11:"Walking",12:"Null"}
Workout_r_2 = {"Adductor":1, "ArmCurl":2, "BenchPress":3, "LegCurl":4,"LegPress":5, "Null":6, "Riding":7, "RopeSkipping":8, "Running":9, "Squat":10, "StairClimber":11, "Walking":12}

Position = {1:"leg",2:"pocket",3:"wrist"}

inputdir = "data/"
outputdir = "data/output/"



# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 7

# Hardcoded number of classes in the gesture recognition problem
NUM_CLASSES = 12

# Hardcoded length of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_LENGTH = 80

# Length of the input sequence after convolutional operations
FINAL_SEQUENCE_LENGTH = 10

# Hardcoded step of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_STEP = 40

# Batch Size
BATCH_SIZE = 128

# Number filters convolutional layers
NUM_FILTERS = 64

# Size filters convolutional layers
FILTER_SIZE = 3

# Number of unit in the long short-term recurrent layers
NUM_UNITS_LSTM = 32

EPOCH_SIZE = 8000


def most_common(lst):
    return max(set(lst), key=lst.count)

def fold(n, data):
    if n == 1:
        train = data.loc[[2, 3, 4, 5, 6, 7, 8, 9, 10], :]
        test = data.loc[[1], :]
    elif n== 2:
        train = data.loc[[1, 3, 4, 5, 6, 7, 8, 9, 10], :]
        test = data.loc[[2], :]
    elif n== 3:
        train = data.loc[[1, 2, 4, 5, 6, 7, 8, 9, 10], :]
        test = data.loc[[3], :]
    elif n== 4:
        train = data.loc[[1, 2, 3, 5, 6, 7, 8, 9, 10], :]
        test = data.loc[[4], :]
    elif n== 5:
        train = data.loc[[1, 2, 3, 4, 6, 7, 8, 9, 10], :]
        test = data.loc[[5], :]
    elif n== 6:
        train = data.loc[[1, 2, 3, 4, 5, 7, 8, 9, 10], :]
        test = data.loc[[6], :]
    elif n== 7:
        train = data.loc[[1, 2, 3, 4, 5, 6, 8, 9, 10], :]
        test = data.loc[[7], :]
    elif n== 8:
        train = data.loc[[1, 2, 3, 4, 5, 6, 7, 9, 10], :]
        test = data.loc[[8], :]
    elif n== 9:
        train = data.loc[[1, 2, 3, 4, 5, 6, 7, 8, 10], :]
        test = data.loc[[9], :]
    elif n== 10:
        train = data.loc[[1, 2, 3, 4, 5, 6, 7, 8, 9], :]
        test = data.loc[[10], :]
    return train, test

def Xy_TrainTest(file, fold_n):

    X_train = np.array([])
    y_train = np.array([])
    X_test = np.array([])
    y_test = np.array([])

    data_session = pd.read_csv(inputdir + file)
    data_session = pd.DataFrame(data_session)

    data_session = data_session.set_index("Object")
    print(data_session.shape)

    print("Fold = ", fold_n)
    train, test = fold(fold_n, data_session)

    train = train.reset_index()
    test = test.reset_index()

    train = train.drop(["LocalSerie", "Win_Num", "Win_In_Num", "Object", "Day", "Workout_time", "Position"], axis=1)
    test = test.drop(["LocalSerie", "Win_Num", "Win_In_Num", "Object","Day", "Workout_time", "Position"], axis=1)

    print(train.shape)
    print(test.shape)
    print(train.head(5))
    print(test.head(5))

    X_train = train[["A_x","A_y","A_z","G_x","G_y","G_z","C_1"]].to_numpy()
    X_test= test[["A_x", "A_y", "A_z", "G_x", "G_y", "G_z", "C_1"]].to_numpy()

    for i in range(0,train.shape[0], 80):
        #X_train = np.append(X_train, train.iloc[i:i+80,1:8].to_numpy().ravel())
        y_train = np.append(y_train, most_common(list(train.iloc[i:(i + 80), 0].to_numpy().ravel())))

    for i in range(0, test.shape[0], 80):
        #X_test = np.append(X_test, test.iloc[i:i + 80, 1:8].to_numpy().ravel())
        y_test = np.append(y_test, most_common(list(test.iloc[i:(i + 80), 0].to_numpy().ravel())))

    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_test shape: ", y_test.shape)

    #pd.DataFrame(y_train).to_csv("y_train.csv")
    #pd.DataFrame(y_test).to_csv("y_test.csv")

    X_train = X_train.reshape(-1, 80, 7)
    #np.save(str(fold_n) + "_X_Train.npy", X_train)
    print("X_train shape: ", X_train.shape)
    #pd.DataFrame(X_train).to_csv(outputdir + "X_train.csv")

    y_train = y_train.reshape(-1, 1)
    #np.save(str(fold_n) + "_y_Train.npy", y_train)
    print("y_train shape: ", y_train.shape)
    #pd.DataFrame(y_train).to_csv(outputdir + "y_train.csv")

    X_test = X_test.reshape(-1, 80, 7)
    #np.save(str(fold_n) + "_X_Test.npy", X_test)
    print("X_test shape: ", X_test.shape)
    #pd.DataFrame(X_test).to_csv(outputdir + "X_test.csv")

    y_test = y_test.reshape(-1, 1)
    #np.save(str(fold_n) + "_y_Test.npy", y_test)
    print("y_test shape: ", y_test.shape)
    #pd.DataFrame(y_test).to_csv(outputdir + "y_test.csv")

    return X_train, y_train, X_test, y_test

### create sequence with length of look_back
def create_dataset(dataset_x, dataset_y, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset_x)-look_back-1):
        a = dataset_x[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset_y[i + look_back -1])
    return np.array(dataX), np.array(dataY)



#######  sample weigth
def generate_sample_weights(training_data, class_weight_dictionary):
    sample_weights = [class_weight_dictionary[np.where(one_hot_row == 1)[0][0]] for one_hot_row in training_data]
    return np.asarray(sample_weights)

def read_save():


    for k in range(3,4):  ##Position

        filelist = sorted(os.listdir(inputdir))
        A_Name = "Deep_" + Position[k] + "_All.csv"

        for file in filelist:
            if fnmatch.fnmatch(file, A_Name):
                print(file)

                for fold_num in range(2, 11):

                    start = time.time()

                    X_train, y_train, X_test, y_test = Xy_TrainTest(file, fold_num)
                    print()

                    ## one hot encoding
                    y_train = keras.utils.to_categorical(np.array(y_train) - 1)
                    y_test = keras.utils.to_categorical(np.array(y_test) - 1)

                    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
                    print("n_timesteps: ", n_timesteps)
                    print("n_features: ", n_features)
                    print("n_outputs: ", n_outputs)

                    print(X_train.shape)
                    print(y_train.shape)
                    n_steps, n_length = 4, 20
                    X_train = X_train.reshape((X_train.shape[0], n_steps, n_length, n_features))
                    X_test = X_test.reshape((X_test.shape[0], n_steps, n_length, n_features))
                    print(X_train.shape)
                    print(y_train.shape)


                    ## give weight to training classes
                    ## class weight
                    #y_integers = np.argmax(y_train, axis=1)
                    #class_weights = class_weight.compute_class_weight('balanced', np.unique(y_integers), y_integers)
                    #d_class_weights = dict(enumerate(class_weights))



                    print(" ..after reshape for LSTM (training): X_train inputs {0}, y_train targets {1}".format(X_train.shape, y_train.shape))
                    print(" ..after reshape for LSTM (testing): X_test inputs {0}, y_test targets {1}".format(X_test.shape, y_test.shape))

                    class_weight_dictionary = {0: 12.0, 1: 12.0, 2: 12.0, 3: 12.0, 4: 12.0, 5: 1.0, 6: 8.0, 7: 36.0, 8: 8.0, 9: 8.0, 10: 8.0, 11: 8.0}
                    sample_weights = generate_sample_weights(y_train, class_weight_dictionary)

                    print(sample_weights)
                    print(sample_weights.shape)
                    pd.DataFrame(sample_weights).to_csv("sampleweight.csv")
                    pd.DataFrame(y_train).to_csv("ytrain.csv")

                    model = Sequential()
                    model.add(TimeDistributed(Conv1D(NUM_FILTERS, kernel_size=FILTER_SIZE, activation='relu', padding='same'), input_shape=(None,n_length,n_features)))
                    model.add(Dropout(0.3))
                    model.add(TimeDistributed(Conv1D(NUM_FILTERS, kernel_size=FILTER_SIZE, activation='relu', padding='same')))
                    model.add(Dropout(0.3))
                    model.add(TimeDistributed(Conv1D(NUM_FILTERS, kernel_size=FILTER_SIZE, activation='relu', padding='same')))
                    model.add(Dropout(0.3))
                    model.add(TimeDistributed(Conv1D(NUM_FILTERS, kernel_size=FILTER_SIZE, activation='relu', padding='same')))
                    model.add(Dropout(0.3))
                    model.add(TimeDistributed(Flatten()))
                    model.add(LSTM(units=NUM_UNITS_LSTM, stateful=False, return_sequences=True))
                    model.add(Dropout(0.3))
                    model.add(LSTM(units=NUM_UNITS_LSTM))
                    model.add(Dense(NUM_CLASSES, activation='softmax'))
                    print(model.summary())

                    #opt = SGD(lr=0.001, momentum=0.9)
                    opt = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)
                    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics = ['accuracy'])

                    # patient early stopping
                    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=500)

                    #history = model.fit(X_train, y_train, batch_size=64, epochs=5, verbose=1, validation_data=(X_test, y_test))
                    #history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=600, verbose=1, validation_split=0.2, class_weight = d_class_weights )

                    # class weight
                    #history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH_SIZE, verbose=1, validation_split=0.2, class_weight = d_class_weights, callbacks=[es])
                    # sample weight
                    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH_SIZE, verbose=1, validation_split=0.2, sample_weight=generate_sample_weights(y_train, class_weight_dictionary),callbacks=[es])
                    pd.DataFrame(history.history).to_csv(outputdir + Position[k] + "__" + str(fold_num) + "_train_history.csv")

                    model.save_weights('weights.h5')

                    score = model.evaluate(X_test, y_test, verbose=1)
                    print('Test loss:', score[0])
                    print('Test accuracy:', score[1])

                    predictions = model.predict(X_test)
                    matrix = metrics.confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
                    print(matrix)

                    pd.DataFrame(score).to_csv(outputdir + Position[k] + "__" + str(fold_num) + "_test_score.csv")
                    pd.DataFrame(predictions).to_csv(outputdir + Position[k] + "__" + str(fold_num) + "_y_predictions.csv")
                    pd.DataFrame(y_test).to_csv(outputdir + Position[k] + "__" + str(fold_num) + "_y_test.csv")

                    end = time.time()
                    print(end - start)


if __name__ == '__main__':

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.05
    # config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    set_session(tf.Session(config=config))


    Process = Process(target=read_save(), args=(), ).start()

