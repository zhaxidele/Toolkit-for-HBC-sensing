import theano
import lasagne
import time
import pandas as pd
import numpy as np
#import pickle as cPickle
import pickle as cp
#import cPickle as cp
import theano.tensor as T
from sliding_window import sliding_window
import sys
import csv
from sklearn.model_selection import train_test_split
from lasagne.init import Orthogonal
import os
import fnmatch
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import keras


inputdir = "data/deep_new/"
outputdir = "data/output_mean/"

#inputdir = "Data/"
#outputdir = "Output_DeepConvLSTM/"


# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 1

# Hardcoded number of classes in the gesture recognition problem
NUM_CLASSES = 12

# Hardcoded length of the sliding window mechanism employed to segment the data
#SLIDING_WINDOW_LENGTH = 40
SLIDING_WINDOW_LENGTH = 24

# Length of the input sequence after convolutional operations
FINAL_SEQUENCE_LENGTH = 8
#FINAL_SEQUENCE_LENGTH = 24
#FINAL_SEQUENCE_LENGTH = 32

# Hardcoded step of the sliding window mechanism employed to segment the data
#SLIDING_WINDOW_STEP = 20
SLIDING_WINDOW_STEP = 12

# Batch Size
BATCH_SIZE = 100
#BATCH_SIZE = 32

# Number filters convolutional layers
NUM_FILTERS = 64
#NUM_FILTERS = 32

# Size filters convolutional layers
FILTER_SIZE = 5
#FILTER_SIZE = 3

# Number of unit in the long short-term recurrent layers
NUM_UNITS_LSTM = 128
#NUM_UNITS_LSTM = 16

num_epochs = 500


Position = {1:"leg",2:"pocket",3:"wrist"}
Workout_r_2 = {"Adductor":1, "ArmCurl":2, "BenchPress":3, "LegCurl":4,"LegPress":5, "Null":6, "Riding":7, "RopeSkipping":8, "Running":9, "Squat":10, "StairClimber":11, "Walking":12}
classes = ["Adductor", "ArmCurl", "BenchPress", "LegCurl","LegPress", "Null", "Riding", "RopeSkipping", "Running", "Squat", "StairClimber", "Walking"]


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




def Xy_TrainTest_IMU(file, fold_n):

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

    train = train[train["Win_In_Num"] < 41]
    train = train[train["Win_In_Num"] < 41]

    #train = train.drop(["Win_In_Num"] > 40, axis=0)
    #test = test.drop(["Win_In_Num"] > 40, axis=0)

    train = train.drop(["LocalSerie", "Win_Num", "Win_In_Num", "Object", "Day", "Workout_time", "Position", "C_1"], axis=1)
    test = test.drop(["LocalSerie", "Win_Num", "Win_In_Num", "Object","Day", "Workout_time", "Position", "C_1"], axis=1)

    print(train.shape)
    print(test.shape)
    print(train.head(5))
    print(test.head(5))

    X_train = train[["A_x","A_y","A_z","G_x","G_y","G_z"]].to_numpy()
    X_test= test[["A_x", "A_y", "A_z", "G_x", "G_y", "G_z"]].to_numpy()

    y_train = train.iloc[:, 0].to_numpy().ravel() - 1
    y_test = test.iloc[:, 0].to_numpy().ravel() - 1


    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    #print(y_test)

    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_test shape: ", y_test.shape)

    return X_train, y_train, X_test, y_test


def load_dataset(filename):

    with open(filename, 'rb') as f:
    #f = file(filename, "rb")
        data = cp.load(f, encoding='latin1')
        f.close()

    #print(data)

    X_train, y_train = data[0]
    X_test, y_test = data[1]

    print(" ..from file {}".format(filename))
    print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    return X_train, y_train, X_test, y_test



def df_normalise_columnly(df):

    headers = df.columns.values.tolist()
    df_Normal = pd.DataFrame()
    for m in range(len(headers)):
        # print(data_session[Header_10[m]])
        df_N = df[headers[m]]

        # Normalise
        max_value = df_N.max()
        min_value = df_N.min()
        df_tem_N = (df_N - min_value) / (max_value - min_value)
        df_Normal = pd.concat([df_Normal, df_tem_N], axis=1)
        #df_Normal.loc[:, m] = df_tem_N

    return df_Normal



def Xy_TrainTest_Cap(file, fold_n):
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

    train = train[train["Win_In_Num"] < 41]
    train = train[train["Win_In_Num"] < 41]

    # train = train.drop(["Win_In_Num"] > 40, axis=0)
    # test = test.drop(["Win_In_Num"] > 40, axis=0)

    train = train.drop(["LocalSerie", "Win_Num", "Win_In_Num", "Object", "Day", "Workout_time", "Position", "A_x", "A_y", "A_z", "G_x", "G_y", "G_z"], axis=1)
    test = test.drop(["LocalSerie", "Win_Num", "Win_In_Num", "Object", "Day", "Workout_time", "Position", "A_x", "A_y", "A_z", "G_x", "G_y", "G_z"], axis=1)

    print(train.shape)
    print(test.shape)
    print(train.head(5))
    print(test.head(5))

    X_train = train[["C_1"]].to_numpy()
    X_test = test[["C_1"]].to_numpy()

    y_train = train.iloc[:, 0].to_numpy().ravel() - 1
    y_test = test.iloc[:, 0].to_numpy().ravel() - 1

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    # print(y_test)

    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_test shape: ", y_test.shape)




    return X_train, y_train, X_test, y_test


def Xy_TrainTest_Cap_IMU(file, fold_n):
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

    train = train[train["Win_In_Num"] < 41]
    train = train[train["Win_In_Num"] < 41]

    # train = train.drop(["Win_In_Num"] > 40, axis=0)
    # test = test.drop(["Win_In_Num"] > 40, axis=0)

    train = train.drop(["LocalSerie", "Win_Num", "Win_In_Num", "Object", "Day", "Workout_time", "Position"], axis=1)
    test = test.drop(["LocalSerie", "Win_Num", "Win_In_Num", "Object", "Day", "Workout_time", "Position"], axis=1)

    print(train.shape)
    print(test.shape)
    print(train.head(5))
    print(test.head(5))

    X_train = train[["A_x", "A_y", "A_z", "G_x", "G_y", "G_z", "C_1"]].to_numpy()
    X_test = test[["A_x", "A_y", "A_z", "G_x", "G_y", "G_z", "C_1"]].to_numpy()

    y_train = train.iloc[:, 0].to_numpy().ravel() - 1
    y_test = test.iloc[:, 0].to_numpy().ravel() - 1

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    # print(y_test)

    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_test shape: ", y_test.shape)


    return X_train, y_train, X_test, y_test




def load_data():

    for k in range(1,4):  ##Position

        filelist = sorted(os.listdir(inputdir))
        A_Name = "Deep_" + Position[k] + "_All.csv"
        print(A_Name)

        for file in filelist:
            if fnmatch.fnmatch(file, A_Name):
                print(file)

                for fold_num in range(1, 11):
                    start = time.time()
                    X_train, y_train, X_test, y_test = Xy_TrainTest_Cap(file, fold_num)

                    input_var = T.tensor4('inputs')
                    target_var = T.ivector('targets')

                    #assert NB_SENSOR_CHANNELS == X_train.shape[1]
                    #print(NB_SENSOR_CHANNELS)

                    def opp_sliding_window(data_x, data_y, ws, ss):
                        data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
                        data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
                        return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)

                    # Sensor data is segmented using a sliding window mechanism
                    X_test, y_test = opp_sliding_window(X_test, y_test, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
                    print(" ..after sliding window (testing): inputs {0}, targets {1}".format(X_test.shape, y_test.shape))
                    # print(X_test.shape)
                    # Data is reshaped since the input of the network is a 4 dimension tensor
                    X_test = X_test.reshape((-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))



                    # Sensor data is segmented using a sliding window mechanism
                    X_train, y_train = opp_sliding_window(X_train, y_train, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
                    print(" ..after sliding window (training): inputs {0}, targets {1}".format(X_train.shape, y_train.shape))
                    X_train = X_train.reshape((-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))
                    print(" ..after reshaping sliding window (training): inputs {0}, targets {1}".format(X_train.shape, y_train.shape))


                    '''
                    ##### dealing with imbalanced training dataset
                    print("Before resample: ")
                    print("X_train shape: ", X_train.shape)
                    print("y_train shape: ", y_train.shape)
                    print("X_test shape: ", X_test.shape)
                    print("y_test shape: ", y_test.shape)
                    # over = SMOTE(sampling_strategy="all")
                    over = SMOTE(sampling_strategy="all")
                    # under = RandomUnderSampler(sampling_strategy="all")
                    #under = RandomUnderSampler(sampling_strategy="majority")
                    # steps = [('o', over), ('u', under)]
                    # pipeline = Pipeline(steps=steps)
                    # X_train, y_train = pipeline.fit_resample(X_train, y_train.ravel())
                    X_train = X_train.reshape(X_train.shape[0], -1)
                    X_train, y_train = over.fit_resample(X_train, y_train.ravel())
                    X_train = X_train.reshape((-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))
                    print("After resample: ")
                    print("X_train shape: ", X_train.shape)
                    print("y_train shape: ", y_train.shape)
                    print("X_test shape: ", X_test.shape)
                    print("y_test shape: ", y_test.shape)
                    ##### dealing with imbalanced training dataset
                    '''

                    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.33, random_state = 42)

                    net = {}
                    net['input'] = lasagne.layers.InputLayer((BATCH_SIZE, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))
                    net['conv1/5x1'] = lasagne.layers.Conv2DLayer(net['input'], NUM_FILTERS, (FILTER_SIZE, 1), W=Orthogonal(gain=1.0))
                    net['conv2/5x1'] = lasagne.layers.Conv2DLayer(net['conv1/5x1'], NUM_FILTERS, (FILTER_SIZE, 1), W=Orthogonal(gain=1.0))
                    net['conv3/5x1'] = lasagne.layers.Conv2DLayer(net['conv2/5x1'], NUM_FILTERS, (FILTER_SIZE, 1), W=Orthogonal(gain=1.0))
                    net['conv4/5x1'] = lasagne.layers.Conv2DLayer(net['conv3/5x1'], NUM_FILTERS, (FILTER_SIZE, 1), W=Orthogonal(gain=1.0))
                    net['shuff'] = lasagne.layers.DimshuffleLayer(net['conv4/5x1'], (0, 2, 1, 3))
                    net['lstm1'] = lasagne.layers.LSTMLayer(net['shuff'], NUM_UNITS_LSTM, nonlinearity=lasagne.nonlinearities.tanh)
                    net['lstm2'] = lasagne.layers.LSTMLayer(net['lstm1'], NUM_UNITS_LSTM, nonlinearity=lasagne.nonlinearities.tanh)
                    # In order to connect a recurrent layer to a dense layer, it is necessary to flatten the first two dimensions
                    # to cause each time step of each sequence to be processed independently (see Lasagne docs for further information)
                    net['shp1'] = lasagne.layers.ReshapeLayer(net['lstm2'], (-1, NUM_UNITS_LSTM))
                    #net['dropout_5'] = lasagne.layers.DropoutLayer(net['shp1'], p=0.5)    ### 2021.02.25
                    net['prob'] = lasagne.layers.DenseLayer(net['shp1'], NUM_CLASSES, nonlinearity=lasagne.nonlinearities.softmax)
                    # Tensors reshaped back to the original shape
                    net['shp2'] = lasagne.layers.ReshapeLayer(net['prob'], (BATCH_SIZE, FINAL_SEQUENCE_LENGTH, NUM_CLASSES))
                    # Last sample in the sequence is considered
                    net['output'] = lasagne.layers.SliceLayer(net['shp2'], -1, 1)


                    def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
                        assert len(inputs) == len(targets)
                        if shuffle:
                            indices = np.arange(len(inputs))
                            np.random.shuffle(indices)
                        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
                            if shuffle:
                                excerpt = indices[start_idx:start_idx + batchsize]
                            else:
                                excerpt = slice(start_idx, start_idx + batchsize)
                            yield inputs[excerpt], targets[excerpt]


                    # create loss function
                    prediction = lasagne.layers.get_output(net['output'])
                    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)

                    ##########################
                    #######  sample weight, failed.
                    #def generate_sample_weights(training_data, class_weight_dictionary):
                    #    sample_weights = [class_weight_dictionary[np.where(one_hot_row == 1)[0][0]] for one_hot_row in training_data]
                    #    return np.asarray(sample_weights)
                    #class_weight_dictionary = {0: 12.0, 1: 12.0, 2: 12.0, 3: 12.0, 4: 12.0, 5: 1.0, 6: 8.0, 7: 36.0, 8: 8.0, 9: 8.0, 10: 8.0, 11: 8.0}
                    #targets_one_hot = keras.utils.to_categorical(np.array(target_var))
                    #print(y_train_one_hot)
                    #sample_weight = generate_sample_weights(prediction, class_weight_dictionary)
                    #print(sample_weight)
                    #loss = loss*sample_weight
                    ##########################
                    #loss = lasagne.objectives.aggregate(loss, weights=sample_weight, mode = "normalized_sum")
                    #loss = lasagne.objectives.aggregate(loss, weights=sample_weight, mode="normalized_sum")

                    #weights_per_label = theano.shared(lasagne.utils.floatX([0.0876, 0.0876, 0.0876, 0.0876, 0.0876, 0.0073, 0.0584, 0.2628, 0.0584, 0.0584, 0.0584, 0.0584]))
                    weights_per_label = theano.shared(lasagne.utils.floatX([87.6, 87.6, 87.6, 87.6, 87.6, 7.3, 58.4, 262.8, 58.4, 58.4, 58.4, 58.4]))
                    weights = weights_per_label[target_var]
                    #loss = lasagne.objectives.aggregate(loss, weights=weights, mode="normalized_sum")
                    loss = lasagne.objectives.aggregate(loss, weights=weights, mode="mean")

                    #loss = loss.mean() + 1e-4 * lasagne.regularization.regularize_network_params(net['output'], lasagne.regularization.l2)
                    #loss = loss.mean()


                    # create parameter update expressions
                    params = lasagne.layers.get_all_params(net['output'], trainable=True)
                    #updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.0001, momentum=0.9)
                    updates = lasagne.updates.rmsprop(loss, params, learning_rate=0.005, rho=0.9)
                    # compile training function that updates parameters and returns training loss
                    train_fn = theano.function([net['input'].input_var, target_var], loss, updates=updates)


                    test_prediction = lasagne.layers.get_output(net['output'], deterministic=True)
                    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
                    test_loss = test_loss.mean()
                    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)
                    val_fn = theano.function([net['input'].input_var, target_var], [test_loss, test_acc])

                    #test_prediction = lasagne.layers.get_output(net['output'], deterministic=True)
                    #test_fn = theano.function([net['input'].input_var], [T.argmax(test_prediction, axis=1)])

                    '''
                    # train network (assuming you've got some training data in numpy arrays)
                    for epoch in range(num_epochs):
                        loss = 0
                        train_batches = 0
                        for batch in iterate_minibatches(X_train, y_train, BATCH_SIZE, shuffle=True):
                            input_batch, target_batch = batch
                            loss += train_fn(input_batch, target_batch)
                            train_batches += 1
                        print("Epoch %d: Loss %g" % (epoch + 1, loss / train_batches))
                    '''

                    looping_non_done = True

                    val_loss_list = []
                    val_acc_list = []
                    train_loss_list = []
                    train_acc_list = []

                    for epoch in range(num_epochs):

                        if looping_non_done:

                            # In each epoch, we do a full pass over the training data:
                            train_err = 0
                            train_batches = 0
                            start_time = time.time()
                            for batch in iterate_minibatches(X_train, y_train, BATCH_SIZE, shuffle=True):
                                inputs, targets = batch
                                train_err += train_fn(inputs, targets)
                                train_batches += 1


                            # And a full pass over the validation data:
                            val_err = 0
                            val_acc = 0
                            val_batches = 0
                            for batch in iterate_minibatches(X_val, y_val, BATCH_SIZE, shuffle=False):
                                inputs, targets = batch
                                err, acc = val_fn(inputs, targets)
                                val_err += err
                                val_acc += acc
                                val_batches += 1

                            def list_differenct(a_list):
                                new_list = []
                                for i in range(len(a_list)-1):
                                    new_list.append(a_list[i] - a_list[i+1])
                                return new_list


                            train_loss_list.append(train_err / train_batches)
                            val_acc_list.append(val_acc / val_batches)
                            val_loss_list.append(val_err / val_batches)

                            ### early stopping method one, better than method two
                            if len(val_loss_list) > 20:
                                if all(element < 0.01 for element in list_differenct(val_loss_list)[-3:]):
                                    looping_non_done = False

                            ### early stopping method two
                            #if len(val_loss_list) > 40:
                            #    #if all(element < 0.01 for element in list_differenct(val_loss_list)[-5:]):
                            #    if (val_loss_list[-11] - val_loss_list[-1]) < 0.01 and (val_loss_list[-12] - val_loss_list[-2]) < 0.01  and (val_loss_list[-13] - val_loss_list[-3]) < 0.01:
                            #        looping_non_done = False


                            # Then we print the results for this epoch:
                            #print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
                            print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time), "  training loss:\t\t{:.6f}".format(train_err / train_batches), "validation loss:\t\t{:.6f}".format(val_err / val_batches), "  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))
                            #print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
                            #print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

                        else:
                            break


                    # Compilation of theano functions
                    # Obtaining the probability distribution over classes
                    test_prediction = lasagne.layers.get_output(net['output'], deterministic=True)
                    # Returning the predicted output for the given minibatch
                    test_fn = theano.function([net['input'].input_var], [T.argmax(test_prediction, axis=1)])

                    # Classification of the testing data
                    print("Processing {0} instances in mini-batches of {1}".format(X_test.shape[0], BATCH_SIZE))
                    test_pred = np.empty((0))
                    test_true = np.empty((0))
                    start_time = time.time()
                    for batch in iterate_minibatches(X_test, y_test, BATCH_SIZE, shuffle=False):
                        inputs, targets = batch
                        y_pred, = test_fn(inputs)
                        test_pred = np.append(test_pred, y_pred, axis=0)
                        test_true = np.append(test_true, targets, axis=0)

                    # Results presentation
                    print("||Results||")
                    print("\tTook {:.3f}s.".format(time.time() - start_time))
                    import sklearn.metrics as metrics
                    print("\tTest fscore:\t{:.4f} ".format(metrics.f1_score(test_true, test_pred, average='weighted')))

                    pd.DataFrame(test_pred).to_csv(outputdir + "Position_" + Position[k] + "_DeepConvLSTM_Cap_Fold_" + str(fold_num) + "_y_predictions.csv")
                    pd.DataFrame(test_true).to_csv(outputdir + "Position_" + Position[k] + "_DeepConvLSTM_Cap_Fold_" + str(fold_num) + "_y_test.csv")



if __name__ == '__main__':
    load_data()
