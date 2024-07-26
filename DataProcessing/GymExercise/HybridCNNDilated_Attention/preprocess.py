
import numpy as np
import scipy.io as sio
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import pandas as pd

# We need the following function to load and preprocess the High Gamma Dataset
# from preprocess_HGD import load_HGD_data

#%%

def most_common(lst):
    return max(set(lst), key=lst.count)


def fold(n, data):
    index =[x for x in range(1, 11)]
    index.pop(n-1)
    train = data.loc[index, :]
    test = data.loc[[n], :]
    return train, test


def load_data_Gym (data_path, subject, dataset, sensor = "combine"):   # sensor = combine / imu / cap
    data_session = pd.read_csv(data_path)
    data_session = pd.DataFrame(data_session)

    data_session = data_session.set_index("Object")
    print(data_session.head)

    print("Fold = ", subject)
    train, test = fold(subject, data_session)

    train = train.reset_index()
    test = test.reset_index()

    train = train.drop(["LocalSerie", "Win_Num", "Win_In_Num", "Object", "Day", "Workout_time", "Position"], axis=1)
    test = test.drop(["LocalSerie", "Win_Num", "Win_In_Num", "Object", "Day", "Workout_time", "Position"], axis=1)

    #X_test = np.array()
    #channel = 0
    if sensor == "cap": ## Cap only
        X_train = train["C_1"].to_numpy()
        X_test= test["C_1"].to_numpy()
        channel = 1
    elif sensor == "imu": ## IMU only
        X_train = train[["A_x", "A_y", "A_z", "G_x", "G_y", "G_z"]].to_numpy()
        X_test = test[["A_x", "A_y", "A_z", "G_x", "G_y", "G_z"]].to_numpy()
        channel = 6
    elif sensor == "combine": ## Cap and IMU
        X_train = train[["A_x", "A_y", "A_z", "G_x", "G_y", "G_z", "C_1"]].to_numpy()
        X_test = test[["A_x", "A_y", "A_z", "G_x", "G_y", "G_z", "C_1"]].to_numpy()
        channel = 7

    y_train = np.array([])
    y_test = np.array([])
    for i in range(0, train.shape[0], 80):
        # X_train = np.append(X_train, train.iloc[i:i+80,1:8].to_numpy().ravel())
        y_train = np.append(y_train, most_common(list(train.iloc[i:(i + 80), 0].to_numpy().ravel())))

    for i in range(0, test.shape[0], 80):
        # X_test = np.append(X_test, test.iloc[i:i + 80, 1:8].to_numpy().ravel())
        y_test = np.append(y_test, most_common(list(test.iloc[i:(i + 80), 0].to_numpy().ravel())))

    # pd.DataFrame(y_train).to_csv("y_train.csv")
    # pd.DataFrame(y_test).to_csv("y_test.csv")

    X_train = X_train.reshape(-1, 80, channel)
    # np.save(str(fold_n) + "_X_Train.npy", X_train)
    print("X_train shape: ", X_train.shape)
    print(X_train)
    # pd.DataFrame(X_train).to_csv(outputdir + "X_train.csv")

    y_train = y_train.reshape(-1) - 1
    # np.save(str(fold_n) + "_y_Train.npy", y_train)
    print("y_train shape: ", y_train.shape)
    # pd.DataFrame(y_train).to_csv(outputdir + "y_train.csv")

    X_test = X_test.reshape(-1, 80, channel)
    # np.save(str(fold_n) + "_X_Test.npy", X_test)
    print("X_test shape: ", X_test.shape)
    # pd.DataFrame(X_test).to_csv(outputdir + "X_test.csv")

    y_test = y_test.reshape(-1) - 1
    # np.save(str(fold_n) + "_y_Test.npy", y_test)
    print("y_test shape: ", y_test.shape)
    print(np.unique(y_test))

    # Get the unique values and their counts
    unique_values, counts = np.unique(y_train, return_counts=True)
    # Print the results
    for value, count in zip(unique_values, counts):
        print(f"{value} occurs {count} times")
    print(counts)
    # pd.DataFrame(y_test).to_csv(outputdir + "y_test.csv")

    return X_train, y_train, X_test, y_test


#%%
def get_data(path, subject, isShuffle = True):

    X_train, y_train, X_test, y_test = load_data_Gym(path, subject, dataset, sensor="cap")   ## sensor="combine"  "imu"  "cap"

    # shuffle the data 
    if isShuffle:
        X_train, y_train = shuffle(X_train, y_train,random_state=42)
        #X_test, y_test = shuffle(X_test, y_test,random_state=42)

    # Prepare training data     
    N_tr, N_ch, T = X_train.shape 
    X_train = X_train.reshape(N_tr, 1, N_ch, T)
    y_train_onehot = to_categorical(y_train)
    # Prepare testing data 
    N_tr, N_ch, T = X_test.shape 
    X_test = X_test.reshape(N_tr, 1, N_ch, T)
    y_test_onehot = to_categorical(y_test)

    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot

