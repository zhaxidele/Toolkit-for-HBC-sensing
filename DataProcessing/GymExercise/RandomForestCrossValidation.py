

################################################################################################################################################
# This code deals with a feature-acstraction finished data set, which has a too large volumn to be uploaded.  Contact the authors if you need it.
################################################################################################################################################


import matplotlib.pyplot as plt
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
from scipy.signal import butter,lfilter,freqz
import math
from scipy.fftpack import fft
from scipy.stats import iqr, entropy
#from entropy import *
from spectrum import *
from scipy.stats import pearsonr, skew, kurtosis
from sklearn.preprocessing import MinMaxScaler

from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton,  QAction, QLineEdit, QMessageBox,QInputDialog,QComboBox, QLabel
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


# spot check on raw data
from numpy import dstack
from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


Header_5 = ["Object", "Day", "Workout", "Sensor_Position",  "A_x", "A_y","A_z", "G_x", "G_y", "G_z", "Body_Capacitance"]


#Feature_Cap = ['t_Cap_mean', 't_Cap_std', 't_Cap_mad', 't_Cap_min', 't_Cap_max', 't_Cap_energy', 't_Cap_iqr_range', 't_Cap_entropy', 't_Cap_AR_estimate_0', 't_Cap_AR_estimate_1', 't_Cap_AR_estimate_2', 't_Cap_AR_estimate_3', 'f_Cap_mean', 'f_Cap_std', 'f_Cap_mad', 'f_Cap_min', 'f_Cap_max', 'f_Cap_energy', 'f_Cap_iqr_range', 'f_Cap_entropy', 'f_Cap_maxindex', 'f_Cap_skewness', 'f_Cap_kurtosis', 'f_Cap__0', 'f_Cap__1', 'f_Cap__2', 'f_Cap__3', 'f_Cap__4', 'f_Cap__5', 'f_Cap__6', 'f_Cap__7', 'f_Cap__8', 'f_Cap__9', 't_Cap_Jerk_mean', 't_Cap_Jerk_std', 't_Cap_Jerk_mad', 't_Cap_Jerk_min', 't_Cap_Jerk_max', 't_Cap_Jerk_energy', 't_Cap_Jerk_iqr_range', 't_Cap_Jerk_entropy', 't_Cap_Jerk_AR_estimate_0', 't_Cap_Jerk_AR_estimate_1', 't_Cap_Jerk_AR_estimate_2', 't_Cap_Jerk_AR_estimate_3', 'f_Cap_Jerk_mean', 'f_Cap_Jerk_std', 'f_Cap_Jerk_mad', 'f_Cap_Jerk_min', 'f_Cap_Jerk_max', 'f_Cap_Jerk_energy', 'f_Cap_Jerk_iqr_range', 'f_Cap_Jerk_entropy', 'f_Cap_Jerk_maxindex', 'f_Cap_Jerk_skewness', 'f_Cap_Jerk_kurtosis', 'f_Cap_Jerk__0', 'f_Cap_Jerk__1', 'f_Cap_Jerk__2', 'f_Cap_Jerk__3', 'f_Cap_Jerk__4', 'f_Cap_Jerk__5', 'f_Cap_Jerk__6', 'f_Cap_Jerk__7', 'f_Cap_Jerk__8','f_Cap_Jerk__9']

#Feature_IMU = ['t_Acc_X_mean', 't_Acc_X_std', 't_Acc_X_mad', 't_Acc_X_min', 't_Acc_X_max', 't_Acc_X_energy', 't_Acc_X_iqr_range', 't_Acc_X_entropy', 't_Acc_X_AR_estimate_0', 't_Acc_X_AR_estimate_1', 't_Acc_X_AR_estimate_2', 't_Acc_X_AR_estimate_3', 'f_Acc_X_mean', 'f_Acc_X_std', 'f_Acc_X_mad', 'f_Acc_X_min', 'f_Acc_X_max', 'f_Acc_X_energy', 'f_Acc_X_iqr_range', 'f_Acc_X_entropy', 'f_Acc_X_maxindex', 'f_Acc_X_skewness', 'f_Acc_X_kurtosis', 'f_Acc_X__0', 'f_Acc_X__1', 'f_Acc_X__2', 'f_Acc_X__3', 'f_Acc_X__4', 'f_Acc_X__5', 'f_Acc_X__6', 'f_Acc_X__7', 'f_Acc_X__8', 'f_Acc_X__9', 't_Acc_Jerk_X_mean', 't_Acc_Jerk_X_std', 't_Acc_Jerk_X_mad', 't_Acc_Jerk_X_min', 't_Acc_Jerk_X_max', 't_Acc_Jerk_X_energy', 't_Acc_Jerk_X_iqr_range', 't_Acc_Jerk_X_entropy', 't_Acc_Jerk_X_AR_estimate_0', 't_Acc_Jerk_X_AR_estimate_1', 't_Acc_Jerk_X_AR_estimate_2', 't_Acc_Jerk_X_AR_estimate_3', 'f_Acc_Jerk_X_mean', 'f_Acc_Jerk_X_std', 'f_Acc_Jerk_X_mad', 'f_Acc_Jerk_X_min', 'f_Acc_Jerk_X_max', 'f_Acc_Jerk_X_energy', 'f_Acc_Jerk_X_iqr_range', 'f_Acc_Jerk_X_entropy', 'f_Acc_Jerk_X_maxindex', 'f_Acc_Jerk_X_skewness', 'f_Acc_Jerk_X_kurtosis', 'f_Acc_Jerk_X__0', 'f_Acc_Jerk_X__1', 'f_Acc_Jerk_X__2', 'f_Acc_Jerk_X__3', 'f_Acc_Jerk_X__4', 'f_Acc_Jerk_X__5', 'f_Acc_Jerk_X__6', 'f_Acc_Jerk_X__7', 'f_Acc_Jerk_X__8', 'f_Acc_Jerk_X__9', 't_Acc_Y_mean', 't_Acc_Y_std', 't_Acc_Y_mad', 't_Acc_Y_min', 't_Acc_Y_max', 't_Acc_Y_energy', 't_Acc_Y_iqr_range', 't_Acc_Y_entropy', 't_Acc_Y_AR_estimate_0', 't_Acc_Y_AR_estimate_1', 't_Acc_Y_AR_estimate_2', 't_Acc_Y_AR_estimate_3', 'f_Acc_Y_mean', 'f_Acc_Y_std', 'f_Acc_Y_mad', 'f_Acc_Y_min', 'f_Acc_Y_max', 'f_Acc_Y_energy', 'f_Acc_Y_iqr_range', 'f_Acc_Y_entropy', 'f_Acc_Y_maxindex', 'f_Acc_Y_skewness', 'f_Acc_Y_kurtosis', 'f_Acc_Y__0', 'f_Acc_Y__1', 'f_Acc_Y__2', 'f_Acc_Y__3', 'f_Acc_Y__4', 'f_Acc_Y__5', 'f_Acc_Y__6', 'f_Acc_Y__7', 'f_Acc_Y__8', 'f_Acc_Y__9', 't_Acc_Jerk_Y_mean', 't_Acc_Jerk_Y_std', 't_Acc_Jerk_Y_mad', 't_Acc_Jerk_Y_min', 't_Acc_Jerk_Y_max', 't_Acc_Jerk_Y_energy', 't_Acc_Jerk_Y_iqr_range', 't_Acc_Jerk_Y_entropy', 't_Acc_Jerk_Y_AR_estimate_0', 't_Acc_Jerk_Y_AR_estimate_1', 't_Acc_Jerk_Y_AR_estimate_2', 't_Acc_Jerk_Y_AR_estimate_3', 'f_Acc_Jerk_Y_mean', 'f_Acc_Jerk_Y_std', 'f_Acc_Jerk_Y_mad', 'f_Acc_Jerk_Y_min', 'f_Acc_Jerk_Y_max', 'f_Acc_Jerk_Y_energy', 'f_Acc_Jerk_Y_iqr_range', 'f_Acc_Jerk_Y_entropy', 'f_Acc_Jerk_Y_maxindex', 'f_Acc_Jerk_Y_skewness', 'f_Acc_Jerk_Y_kurtosis', 'f_Acc_Jerk_Y__0', 'f_Acc_Jerk_Y__1', 'f_Acc_Jerk_Y__2', 'f_Acc_Jerk_Y__3', 'f_Acc_Jerk_Y__4', 'f_Acc_Jerk_Y__5', 'f_Acc_Jerk_Y__6', 'f_Acc_Jerk_Y__7', 'f_Acc_Jerk_Y__8', 'f_Acc_Jerk_Y__9', 't_Acc_Z_mean', 't_Acc_Z_std', 't_Acc_Z_mad', 't_Acc_Z_min', 't_Acc_Z_max', 't_Acc_Z_energy', 't_Acc_Z_iqr_range', 't_Acc_Z_entropy', 't_Acc_Z_AR_estimate_0', 't_Acc_Z_AR_estimate_1', 't_Acc_Z_AR_estimate_2', 't_Acc_Z_AR_estimate_3', 'f_Acc_Z_mean', 'f_Acc_Z_std', 'f_Acc_Z_mad', 'f_Acc_Z_min', 'f_Acc_Z_max', 'f_Acc_Z_energy', 'f_Acc_Z_iqr_range', 'f_Acc_Z_entropy', 'f_Acc_Z_maxindex', 'f_Acc_Z_skewness', 'f_Acc_Z_kurtosis', 'f_Acc_Z__0', 'f_Acc_Z__1', 'f_Acc_Z__2', 'f_Acc_Z__3', 'f_Acc_Z__4', 'f_Acc_Z__5', 'f_Acc_Z__6', 'f_Acc_Z__7', 'f_Acc_Z__8', 'f_Acc_Z__9', 't_Acc_Jerk_Z_mean', 't_Acc_Jerk_Z_std', 't_Acc_Jerk_Z_mad', 't_Acc_Jerk_Z_min', 't_Acc_Jerk_Z_max', 't_Acc_Jerk_Z_energy', 't_Acc_Jerk_Z_iqr_range', 't_Acc_Jerk_Z_entropy', 't_Acc_Jerk_Z_AR_estimate_0', 't_Acc_Jerk_Z_AR_estimate_1', 't_Acc_Jerk_Z_AR_estimate_2', 't_Acc_Jerk_Z_AR_estimate_3', 'f_Acc_Jerk_Z_mean', 'f_Acc_Jerk_Z_std', 'f_Acc_Jerk_Z_mad', 'f_Acc_Jerk_Z_min', 'f_Acc_Jerk_Z_max', 'f_Acc_Jerk_Z_energy', 'f_Acc_Jerk_Z_iqr_range', 'f_Acc_Jerk_Z_entropy', 'f_Acc_Jerk_Z_maxindex', 'f_Acc_Jerk_Z_skewness', 'f_Acc_Jerk_Z_kurtosis', 'f_Acc_Jerk_Z__0', 'f_Acc_Jerk_Z__1', 'f_Acc_Jerk_Z__2', 'f_Acc_Jerk_Z__3', 'f_Acc_Jerk_Z__4', 'f_Acc_Jerk_Z__5', 'f_Acc_Jerk_Z__6', 'f_Acc_Jerk_Z__7', 'f_Acc_Jerk_Z__8', 'f_Acc_Jerk_Z__9', 't_Gyro_X_mean', 't_Gyro_X_std', 't_Gyro_X_mad', 't_Gyro_X_min', 't_Gyro_X_max', 't_Gyro_X_energy', 't_Gyro_X_iqr_range', 't_Gyro_X_entropy', 't_Gyro_X_AR_estimate_0', 't_Gyro_X_AR_estimate_1', 't_Gyro_X_AR_estimate_2', 't_Gyro_X_AR_estimate_3', 'f_Gyro_X_mean', 'f_Gyro_X_std', 'f_Gyro_X_mad', 'f_Gyro_X_min', 'f_Gyro_X_max', 'f_Gyro_X_energy', 'f_Gyro_X_iqr_range', 'f_Gyro_X_entropy', 'f_Gyro_X_maxindex', 'f_Gyro_X_skewness', 'f_Gyro_X_kurtosis', 'f_Gyro_X__0', 'f_Gyro_X__1', 'f_Gyro_X__2', 'f_Gyro_X__3', 'f_Gyro_X__4', 'f_Gyro_X__5', 'f_Gyro_X__6', 'f_Gyro_X__7', 'f_Gyro_X__8', 'f_Gyro_X__9', 't_Gyro_Jerk_X_mean', 't_Gyro_Jerk_X_std', 't_Gyro_Jerk_X_mad', 't_Gyro_Jerk_X_min', 't_Gyro_Jerk_X_max', 't_Gyro_Jerk_X_energy', 't_Gyro_Jerk_X_iqr_range', 't_Gyro_Jerk_X_entropy', 't_Gyro_Jerk_X_AR_estimate_0', 't_Gyro_Jerk_X_AR_estimate_1', 't_Gyro_Jerk_X_AR_estimate_2', 't_Gyro_Jerk_X_AR_estimate_3', 'f_Gyro_Jerk_X_mean', 'f_Gyro_Jerk_X_std', 'f_Gyro_Jerk_X_mad', 'f_Gyro_Jerk_X_min', 'f_Gyro_Jerk_X_max', 'f_Gyro_Jerk_X_energy', 'f_Gyro_Jerk_X_iqr_range', 'f_Gyro_Jerk_X_entropy', 'f_Gyro_Jerk_X_maxindex', 'f_Gyro_Jerk_X_skewness', 'f_Gyro_Jerk_X_kurtosis', 'f_Gyro_Jerk_X__0', 'f_Gyro_Jerk_X__1', 'f_Gyro_Jerk_X__2', 'f_Gyro_Jerk_X__3', 'f_Gyro_Jerk_X__4', 'f_Gyro_Jerk_X__5', 'f_Gyro_Jerk_X__6', 'f_Gyro_Jerk_X__7', 'f_Gyro_Jerk_X__8', 'f_Gyro_Jerk_X__9', 't_Gyro_Y_mean', 't_Gyro_Y_std', 't_Gyro_Y_mad', 't_Gyro_Y_min', 't_Gyro_Y_max', 't_Gyro_Y_energy', 't_Gyro_Y_iqr_range', 't_Gyro_Y_entropy', 't_Gyro_Y_AR_estimate_0', 't_Gyro_Y_AR_estimate_1', 't_Gyro_Y_AR_estimate_2', 't_Gyro_Y_AR_estimate_3', 'f_Gyro_Y_mean', 'f_Gyro_Y_std', 'f_Gyro_Y_mad', 'f_Gyro_Y_min', 'f_Gyro_Y_max', 'f_Gyro_Y_energy', 'f_Gyro_Y_iqr_range', 'f_Gyro_Y_entropy', 'f_Gyro_Y_maxindex', 'f_Gyro_Y_skewness', 'f_Gyro_Y_kurtosis', 'f_Gyro_Y__0', 'f_Gyro_Y__1', 'f_Gyro_Y__2', 'f_Gyro_Y__3', 'f_Gyro_Y__4', 'f_Gyro_Y__5', 'f_Gyro_Y__6', 'f_Gyro_Y__7', 'f_Gyro_Y__8', 'f_Gyro_Y__9', 't_Gyro_Jerk_Y_mean', 't_Gyro_Jerk_Y_std', 't_Gyro_Jerk_Y_mad', 't_Gyro_Jerk_Y_min', 't_Gyro_Jerk_Y_max', 't_Gyro_Jerk_Y_energy', 't_Gyro_Jerk_Y_iqr_range', 't_Gyro_Jerk_Y_entropy', 't_Gyro_Jerk_Y_AR_estimate_0', 't_Gyro_Jerk_Y_AR_estimate_1', 't_Gyro_Jerk_Y_AR_estimate_2', 't_Gyro_Jerk_Y_AR_estimate_3', 'f_Gyro_Jerk_Y_mean', 'f_Gyro_Jerk_Y_std', 'f_Gyro_Jerk_Y_mad', 'f_Gyro_Jerk_Y_min', 'f_Gyro_Jerk_Y_max', 'f_Gyro_Jerk_Y_energy', 'f_Gyro_Jerk_Y_iqr_range', 'f_Gyro_Jerk_Y_entropy', 'f_Gyro_Jerk_Y_maxindex', 'f_Gyro_Jerk_Y_skewness', 'f_Gyro_Jerk_Y_kurtosis', 'f_Gyro_Jerk_Y__0', 'f_Gyro_Jerk_Y__1', 'f_Gyro_Jerk_Y__2', 'f_Gyro_Jerk_Y__3', 'f_Gyro_Jerk_Y__4', 'f_Gyro_Jerk_Y__5', 'f_Gyro_Jerk_Y__6', 'f_Gyro_Jerk_Y__7', 'f_Gyro_Jerk_Y__8', 'f_Gyro_Jerk_Y__9', 't_Gyro_Z_mean', 't_Gyro_Z_std', 't_Gyro_Z_mad', 't_Gyro_Z_min', 't_Gyro_Z_max', 't_Gyro_Z_energy', 't_Gyro_Z_iqr_range', 't_Gyro_Z_entropy', 't_Gyro_Z_AR_estimate_0', 't_Gyro_Z_AR_estimate_1', 't_Gyro_Z_AR_estimate_2', 't_Gyro_Z_AR_estimate_3', 'f_Gyro_Z_mean', 'f_Gyro_Z_std', 'f_Gyro_Z_mad', 'f_Gyro_Z_min', 'f_Gyro_Z_max', 'f_Gyro_Z_energy', 'f_Gyro_Z_iqr_range', 'f_Gyro_Z_entropy', 'f_Gyro_Z_maxindex', 'f_Gyro_Z_skewness', 'f_Gyro_Z_kurtosis', 'f_Gyro_Z__0', 'f_Gyro_Z__1', 'f_Gyro_Z__2', 'f_Gyro_Z__3', 'f_Gyro_Z__4', 'f_Gyro_Z__5', 'f_Gyro_Z__6', 'f_Gyro_Z__7', 'f_Gyro_Z__8', 'f_Gyro_Z__9', 't_Gyro_Jerk_Z_mean', 't_Gyro_Jerk_Z_std', 't_Gyro_Jerk_Z_mad', 't_Gyro_Jerk_Z_min', 't_Gyro_Jerk_Z_max', 't_Gyro_Jerk_Z_energy', 't_Gyro_Jerk_Z_iqr_range', 't_Gyro_Jerk_Z_entropy', 't_Gyro_Jerk_Z_AR_estimate_0', 't_Gyro_Jerk_Z_AR_estimate_1', 't_Gyro_Jerk_Z_AR_estimate_2', 't_Gyro_Jerk_Z_AR_estimate_3', 'f_Gyro_Jerk_Z_mean', 'f_Gyro_Jerk_Z_std', 'f_Gyro_Jerk_Z_mad', 'f_Gyro_Jerk_Z_min', 'f_Gyro_Jerk_Z_max', 'f_Gyro_Jerk_Z_energy', 'f_Gyro_Jerk_Z_iqr_range', 'f_Gyro_Jerk_Z_entropy', 'f_Gyro_Jerk_Z_maxindex', 'f_Gyro_Jerk_Z_skewness', 'f_Gyro_Jerk_Z_kurtosis', 'f_Gyro_Jerk_Z__0', 'f_Gyro_Jerk_Z__1', 'f_Gyro_Jerk_Z__2', 'f_Gyro_Jerk_Z__3', 'f_Gyro_Jerk_Z__4', 'f_Gyro_Jerk_Z__5', 'f_Gyro_Jerk_Z__6', 'f_Gyro_Jerk_Z__7', 'f_Gyro_Jerk_Z__8', 'f_Gyro_Jerk_Z__9', 't_Acc_Mag_mean', 't_Acc_Mag_std', 't_Acc_Mag_mad', 't_Acc_Mag_min', 't_Acc_Mag_max', 't_Acc_Mag_energy', 't_Acc_Mag_iqr_range', 't_Acc_Mag_entropy', 't_Acc_Mag_AR_estimate_0', 't_Acc_Mag_AR_estimate_1', 't_Acc_Mag_AR_estimate_2', 't_Acc_Mag_AR_estimate_3', 'f_Acc_Mag_mean', 'f_Acc_Mag_std', 'f_Acc_Mag_mad', 'f_Acc_Mag_min', 'f_Acc_Mag_max', 'f_Acc_Mag_energy', 'f_Acc_Mag_iqr_range', 'f_Acc_Mag_entropy', 'f_Acc_Mag_maxindex', 'f_Acc_Mag_skewness', 'f_Acc_Mag_kurtosis', 'f_Acc_Mag__0', 'f_Acc_Mag__1', 'f_Acc_Mag__2', 'f_Acc_Mag__3', 'f_Acc_Mag__4', 'f_Acc_Mag__5', 'f_Acc_Mag__6', 'f_Acc_Mag__7', 'f_Acc_Mag__8', 'f_Acc_Mag__9', 't_Acc_Jerk_Mag_mean', 't_Acc_Jerk_Mag_std', 't_Acc_Jerk_Mag_mad', 't_Acc_Jerk_Mag_min', 't_Acc_Jerk_Mag_max', 't_Acc_Jerk_Mag_energy', 't_Acc_Jerk_Mag_iqr_range', 't_Acc_Jerk_Mag_entropy', 't_Acc_Jerk_Mag_AR_estimate_0', 't_Acc_Jerk_Mag_AR_estimate_1', 't_Acc_Jerk_Mag_AR_estimate_2', 't_Acc_Jerk_Mag_AR_estimate_3', 'f_Acc_Jerk_Mag_mean', 'f_Acc_Jerk_Mag_std', 'f_Acc_Jerk_Mag_mad', 'f_Acc_Jerk_Mag_min', 'f_Acc_Jerk_Mag_max', 'f_Acc_Jerk_Mag_energy', 'f_Acc_Jerk_Mag_iqr_range', 'f_Acc_Jerk_Mag_entropy', 'f_Acc_Jerk_Mag_maxindex', 'f_Acc_Jerk_Mag_skewness', 'f_Acc_Jerk_Mag_kurtosis', 'f_Acc_Jerk_Mag__0', 'f_Acc_Jerk_Mag__1', 'f_Acc_Jerk_Mag__2', 'f_Acc_Jerk_Mag__3', 'f_Acc_Jerk_Mag__4', 'f_Acc_Jerk_Mag__5', 'f_Acc_Jerk_Mag__6', 'f_Acc_Jerk_Mag__7', 'f_Acc_Jerk_Mag__8', 'f_Acc_Jerk_Mag__9', 't_Gyro_Mag_mean', 't_Gyro_Mag_std', 't_Gyro_Mag_mad', 't_Gyro_Mag_min', 't_Gyro_Mag_max', 't_Gyro_Mag_energy', 't_Gyro_Mag_iqr_range', 't_Gyro_Mag_entropy', 't_Gyro_Mag_AR_estimate_0', 't_Gyro_Mag_AR_estimate_1', 't_Gyro_Mag_AR_estimate_2', 't_Gyro_Mag_AR_estimate_3', 'f_Gyro_Mag_mean', 'f_Gyro_Mag_std', 'f_Gyro_Mag_mad', 'f_Gyro_Mag_min', 'f_Gyro_Mag_max', 'f_Gyro_Mag_energy', 'f_Gyro_Mag_iqr_range', 'f_Gyro_Mag_entropy', 'f_Gyro_Mag_maxindex', 'f_Gyro_Mag_skewness', 'f_Gyro_Mag_kurtosis', 'f_Gyro_Mag__0', 'f_Gyro_Mag__1', 'f_Gyro_Mag__2', 'f_Gyro_Mag__3', 'f_Gyro_Mag__4', 'f_Gyro_Mag__5', 'f_Gyro_Mag__6', 'f_Gyro_Mag__7', 'f_Gyro_Mag__8', 'f_Gyro_Mag__9', 't_Gyro_Jerk_Mag_mean', 't_Gyro_Jerk_Mag_std', 't_Gyro_Jerk_Mag_mad', 't_Gyro_Jerk_Mag_min', 't_Gyro_Jerk_Mag_max', 't_Gyro_Jerk_Mag_energy', 't_Gyro_Jerk_Mag_iqr_range', 't_Gyro_Jerk_Mag_entropy', 't_Gyro_Jerk_Mag_AR_estimate_0', 't_Gyro_Jerk_Mag_AR_estimate_1', 't_Gyro_Jerk_Mag_AR_estimate_2', 't_Gyro_Jerk_Mag_AR_estimate_3', 'f_Gyro_Jerk_Mag_mean', 'f_Gyro_Jerk_Mag_std', 'f_Gyro_Jerk_Mag_mad', 'f_Gyro_Jerk_Mag_min', 'f_Gyro_Jerk_Mag_max', 'f_Gyro_Jerk_Mag_energy', 'f_Gyro_Jerk_Mag_iqr_range', 'f_Gyro_Jerk_Mag_entropy', 'f_Gyro_Jerk_Mag_maxindex', 'f_Gyro_Jerk_Mag_skewness', 'f_Gyro_Jerk_Mag_kurtosis', 'f_Gyro_Jerk_Mag__0', 'f_Gyro_Jerk_Mag__1', 'f_Gyro_Jerk_Mag__2', 'f_Gyro_Jerk_Mag__3', 'f_Gyro_Jerk_Mag__4', 'f_Gyro_Jerk_Mag__5', 'f_Gyro_Jerk_Mag__6', 'f_Gyro_Jerk_Mag__7', 'f_Gyro_Jerk_Mag__8', 'f_Gyro_Jerk_Mag__9', 't_Acc_X_Y_correlation', 't_Acc_X_Z_correlation', 't_Acc_Y_Z_correlation', 't_Acc_Jerk_X_Y_correlation', 't_Acc_Jerk_X_Z_correlation', 't_Acc_Jerk_Y_Z_correlation', 't_Gyro_X_Y_correlation', 't_Gyro_X_Z_correlation', 't_Gyro_Y_Z_correlation', 't_Gyro_Jerk_X_Y_correlation', 't_Gyro_Jerk_X_Z_correlation', 't_Gyro_Jerk_Y_Z_correlation', 't_Acc_X_Y_Z_SMA', 't_Acc_Jerk_X_Y_Z_SMA', 't_Gyro_X_Y_Z_SMA', 't_Gyro_Jerk_X_Y_Z_SMA', 'f_Acc_X_Y_Z_SMA', 'f_Acc_Jerk_X_Y_Z_SMA', 'f_Gyro_X_Y_Z_SMA', 'f_Gyro_Jerk_X_Y_Z_SMA']



Kept_Features = []
To_be_removed_Features = []

#### three kinds of squat
class_names = np.array(["Adductor", "ArmCurl", "BenchPress", "LegCurl", "LegPress", "Null", "Riding", "RopeSkipping", "Running", "SquatConcrete", "SquatRubber", "SquatWood", "StairClimber", "Walking"])
#### combined squats
class_names_1 = np.array(["Adductor", "ArmCurl", "BenchPress", "LegCurl", "LegPress", "Null", "Riding", "RopeSkipping", "Running", "Squat", "StairClimber", "Walking"])



Day = {1:"1",2:"2",3:"3",4:"4",5:"5"}
Workout = {1:"Adductor",2:"ArmCurl",3:"BenchPress",4:"LegCurl",5:"LegPress",6:"Riding",7:"RopeSkipping",8:"Running",9:"SquatConcrete",10:"SquatRubber",11:"SquatWood",12:"StairClimber",13:"Walking"}
Workout_time = {1:"1",2:"2",3:"3"}
Position = {1:"leg",2:"pocket",3:"wrist"}

WindowSize = {1:"2_Seconds",2:"4_Seconds",3:"6_Seconds"}
Normalize = {1:"N", 2:"SN"}

yhat = []
yhat_all = []
y_test_all = []
y_name = ""


outputdir = "Data/"
inputdir = "Data/" 


def define_models(depth, tree_num):
    models = dict()
    # nonlinear models
    #models['knn'] = KNeighborsClassifier(n_neighbors=7)
    #models['cart'] = DecisionTreeClassifier()
    #models['svm'] = SVC()
    #models['bayes'] = GaussianNB()
    # ensemble models
    #models['bag'] = BaggingClassifier(n_estimators=100)

    #models['rf'] = RandomForestClassifier(n_estimators=100)
    models['rf'] = RandomForestClassifier(n_estimators=tree_num, max_depth = depth, oob_score=True)


    #random_classifier = RandomForestClassifier()
    #parameters = {'max_features': np.arange(5, 10), 'n_estimators': [500], 'min_samples_leaf': [10, 50, 100, 200, 500]}
    #random_grid = GridSearchCV(random_classifier, parameters, cv=5)


    #models['et'] = ExtraTreesClassifier(n_estimators=100)
    #models['gbm'] = GradientBoostingClassifier(n_estimators=100)
    #print('Defined %d models' % len(models))
    return models


# evaluate a single model
def evaluate_model(trainX, trainy, testX, testy, model):
    global yhat, yhat_all, y_name
    # fit the model
    model.fit(trainX, trainy)
    # make predictions
    yhat = model.predict(testX)

    trainy_pred = model.predict(trainX)

    yhat_all.append(yhat)
    pd.DataFrame(yhat).to_csv(outputdir_51 + y_name + "_predict.csv")
    # evaluate predictions
    accuracy_test = accuracy_score(testy, yhat)
    accuracy_train = accuracy_score(trainy, trainy_pred)

    hammingloss_test = hamming_loss(testy, yhat)
    hammingloss_train = hamming_loss(trainy, trainy_pred)

    #print("train accruacy: ", accuracy_train)
    #return accuracy * 100.0
    return accuracy_test, accuracy_train, hammingloss_test, hammingloss_train


# evaluate a dict of models {name:object}, returns {name:score}
def evaluate_models(trainX, trainy, testX, testy, models):
    results = dict()
    for name, model in models.items():
        # evaluate the model
        #results[name] = evaluate_model(trainX, trainy, testX, testy, model)
        # show process

        accuracy_test, accuracy_train, hammingloss_test, hammingloss_train = evaluate_model(trainX, trainy, testX, testy, model)
        #print('>%s: %.3f' % (name, results[name]))
    #return results
    return accuracy_test, accuracy_train, hammingloss_test, hammingloss_train


# print and plot the results
def summarize_results(results, maximize=True):
    # create a list of (name, mean(scores)) tuples
    mean_scores = [(k,v) for k,v in results.items()]
    # sort tuples by mean score
    mean_scores = sorted(mean_scores, key=lambda x: x[1])
    # reverse for descending order (e.g. for accuracy)
    if maximize:
        mean_scores = list(reversed(mean_scores))
        #print()
    #for name, score in mean_scores:
        #print('Name=%s, Score=%.3f' % (name, score))


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    accuracy = accuracy_score(y_true, y_pred)
    F1_note = f1_score(y_true, y_pred, average='macro')
    title = "Macro F1: " + str(round(F1_note,2)) + "  " + "Accuracy: " + str(round(accuracy*100.0,2))
    #print(F1_note)
    #if not title:
        #if normalize:
        #    title = 'Normalized confusion matrix'
        #else:
        #    title = 'Confusion matrix, without normalization'
        #title = "Macro F1: " + str(F1_note)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #print(y_true)
    #print(y_pred)
    #print(classes)
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=6)
    fig.tight_layout()
    return ax


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

m = 0
Wrist_m = False

def read_save():
    global data_session, yhat, yhat_all, y_name, m, Wrist_m

    grid_acc = np.array([])
    grid_acc_all = np.array([0.0]*6)

    for k in range(1, 4):  ##Position

            ## output one file
            A_Name = "Combined_All_4_Seconds" + "_" + Position[k] + "_" + "N.csv"

            filelist = sorted(os.listdir(inputdir))
            filelist.sort(key=lambda x: os.path.getctime(inputdir+x))

            for file in filelist:
                if fnmatch.fnmatch(file, A_Name):
                    #print(file)

                    data_session = pd.read_csv(inputdir + file)
                    data_session=pd.DataFrame(data_session)
                    row_session, column_session = data_session.shape
                    #print(row_session, column_session)

                    data_session = data_session.drop(["Day", "Workout_time", "Position"], axis=1)

                    #data_session = data_session.drop(To_be_removed_Features, axis=1)
                    #data_session = data_session.drop(Feature_Cap, axis=1)
                    #data_session = data_session.drop(list(set(Feature_Cap)-set(To_be_removed_Features)), axis=1)

                    Feature_Cap = []
                    Feature_IMU = []
                    feature_list = data_session.columns.tolist()
                    for nn in range(len(feature_list)):
                        if "Cap" in feature_list[nn]:
                            Feature_Cap.append(feature_list[nn])
                        elif "Acc" in feature_list[nn]:
                            Feature_IMU.append(feature_list[nn])
                        elif "Gyro" in feature_list[nn]:
                            Feature_IMU.append(feature_list[nn])

                    #data_session = data_session.drop(Feature_Cap, axis=1)
                    #data_session = data_session.drop(Feature_IMU, axis=1)

                    data_session = data_session.set_index("Object")
                    data_session = data_session.iloc[0:-1,:]
                    #print(data_session.loc[[1,2,3,4,5],:])

                    for i in range(1,11):

                        print("Fold = ", i)
                        train, test = fold(i, data_session)

                        #train_1 = data_session.loc[[1,2,3,4,5,6,7,8,9],:]
                        #test_1 = data_session.loc[[10], :]


                        X_train = np.array(train.iloc[:, train.columns != 'Workout'])
                        y_train = np.array(train.iloc[:, train.columns == 'Workout'])

                        X_test = np.array(test.iloc[:, test.columns != 'Workout'])
                        y_test = np.array(test.iloc[:, test.columns == 'Workout'])

                        y_name = Position[k] + "_Fold_" + str(i)
                        print(y_name)
                        pd.DataFrame(y_test).to_csv(outputdir_51 + y_name + "_actual.csv")

                        y_test_all.append(y_test)
                        #X = np.array(data_session.iloc[:, data_session.columns != 'Workout'])
                        #y = np.array(data_session.iloc[:, data_session.columns == 'Workout'])
                        #print('Shape of X: {}'.format(X.shape))
                        #print('Shape of y: {}'.format(y.shape))
                        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)      ### choose 30% samples randomly as test set.
                        #print("Number transactions X_train dataset: ", X_train.shape)
                        #print("Number transactions y_train dataset: ", y_train.shape)
                        #print("Number transactions X_test dataset: ", X_test.shape)
                        #print("Number transactions y_test dataset: ", y_test.shape)


                        #print('Before OverSampling, the shape of train_X: {}'.format(X_train.shape))
                        #print('Before OverSampling, the shape of train_y: {} \n'.format(y_train.shape))

                        sm = SMOTE(sampling_strategy="all",random_state=2)
                        X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

                        #print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
                        #print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))


                        # get model list

                        #for depth in range(4,16):
                        for depth in range(60, 61):
                            #for tree_num in (20, 60, 100, 140, 180, 220):
                            tree_num = 60
                            models = define_models(depth, tree_num)
                            # evaluate models
                            accuracy_test, accuracy_train, hammingloss_test, hammingloss_train = evaluate_models(X_train_res, y_train_res, X_test, y_test, models)
                            # summarize results
                            #summarize_results(results)
                            #print("depth: ", depth)
                            #print("tree_num: ", tree_num)
                            #print("train_accuracy: ", accuracy_train)
                            #print("test_accuracy: ", accuracy_test)
                            #print("hammingloss_train: ", hammingloss_train)
                            #print("hammingloss_test: ", hammingloss_test)

                            grid_acc = np.hstack((depth, tree_num, accuracy_train, accuracy_test, hammingloss_train, hammingloss_test))
                            grid_acc_all = np.vstack((grid_acc_all, grid_acc))
                            grid_acc = np.array([])

                        #pd.DataFrame(grid_acc_all).to_csv(outputdir_40 + "Grid_Accuracy.csv", header=["Depth", "Tree_num", "Train_accuracy", "Test_accuracy", "hammingloss_train", "hammingloss_test"])


                        np.set_printoptions(precision=2)

                        # Plot non-normalized confusion matrix
                        plot_confusion_matrix(y_test, yhat, classes=class_names_1, normalize=False, title=True)
                        plt.savefig(outputdir_52 + "Combined_All_4_Seconds" + "_" + Position[k] + "_" + "N"+ "_" + "NonNormalized_" + str(i) + ".png", format='png', dpi=900)
                        plt.clf()

                        # Plot normalized confusion matrix
                        plot_confusion_matrix(y_test, yhat, classes=class_names_1, normalize=True, title=True)
                        plt.savefig(outputdir_52 + "Combined_All_4_Seconds"  + "_" + Position[k] + "_" + "N" + "_" + "Normalized_" + str(i) + ".png", format='png', dpi=900)
                        #plt.show()
                        plt.clf()

                    ##### combined confusion matrix after cross validation
                    # Plot normalized confusion matrix
                    #plot_confusion_matrix(y_test_all.ravel(), yhat_all.ravel(), classes=class_names_1, normalize=True, title=True)
                    #plt.savefig(outputdir_52 + "Combined_All_4_Seconds"  + "_" + Position[k] + "_" + "N" + "_" + "Normalized_All.png", format='png', dpi=900)
                    #plt.clf()
                    # Plot normalized confusion matrix
                    #plot_confusion_matrix(y_test_all.ravel(), yhat_all.ravel(), classes=class_names_1, normalize=False, title=True)
                    #plt.savefig(outputdir_52 + "Combined_All_4_Seconds"  + "_" + Position[k] + "_" + "N" + "_" + "NonNormalized_All.png", format='png', dpi=900)
                    #plt.clf()


if __name__ == '__main__':
    Process = Process(target=read_save(), args=(), ).start()
    #Process = Process(target=read_save_2021_01(), args=(), ).start()
