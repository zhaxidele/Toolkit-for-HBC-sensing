from __future__ import print_function
import numpy as np

from util import make_sliding_window_in_place
from sklearn.preprocessing import minmax_scale
import math
from scipy.fftpack import fft
from scipy.stats import iqr
from spectrum import *
from scipy.stats import pearsonr, skew, kurtosis
#from entropy import app_entropy

def jerk(data, frequency):
    m = []
    for i in range(1,len(data)):
        m.append((data[i] - data[i-1]) * frequency)
    return m


def mag(X,Y,Z):
    euclidean = []
    for i in range(len(X)):
        euclidean.append(math.sqrt(X[i]**2+Y[i]**2+Z[i]**2))
    return euclidean


def fft_W(X):
    return fft(X,20)[0:10]     # sample rate = 20 hz, so the maximal frequency component is 10 Hz.  remove the symetric part,  return an array with 10 elements
    #return fft(X)[0:61]


def mean_w(X):
    return np.mean(X)


def std_w(X):
    return np.std(X)


def min_w(X):
    return np.min(X)


def max_w(X):
    return np.max(X)


def mad_w(X):
    axis = None
    return np.mean(np.absolute(X - np.mean(X, axis)), axis)


def energy_w(X):
    energy = 0
    for i in range(len(X)):
        energy = X[i]**2 + energy
    return energy/len(X)


def iqr_range_w(X):
    return iqr(X, rng=(25, 75), interpolation='midpoint')


#def entropy_w(X):
#    #return spectral_entropy(X,20,normalize=False)          # https://raphaelvallat.com/entropy/build/html/index.html     Sample Rate=20,  and normalize the spectral entropy between 0 and 1.
#    return app_entropy(X, order=2, metric='chebyshev')      # Approximate entropy


def AR_estimate_w(X):
    AR, P, k = arburg(X, 4)      # BURG method of AR model estimate with order 4   http://thomas-cokelaer.info/software/spectrum/html/user/ref_param.html#module-burg
    return np.real(AR)


def correlation_w(X,Y):          # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.pearsonr.html
    c, p = pearsonr(X, Y)
    return c


def SMA_w(X,Y,Z):
    Area = 0
    for i in range(len(X)):
        Area = abs(X[i]) + abs(Y[i]) + abs(Z[i]) + Area
    return Area/len(X)


def maxindex_w(f_X):         # index of the frequency component with largest magnitude         https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html#numpy.argmax
    return np.argmax(f_X)


def fre_mean_w(X):                #  http://luscinia.sourceforge.net/page26/page35/page35.html
    sum = 0
    divide = 0
    for i in range(len(X)):
        sum = i * 20*log10(abs(X[i])) + sum
        divide = 20*log10(abs(X[i])) + divide
    return sum/divide


def skewness_w(X):           #https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.stats.skew.html
    return skew(X)


def kurtosis_w(X):
    return kurtosis(X)


def band_energy_w(X):                    # return an array with 6 elements, the bin size is 1 Hz, namely 0-1, 1-2, 2-3, 3-4, 4-5, 5-6
    energy = []
    for i in range(len(X)):
        energy.append(X[i]**2)
    return energy


def preprocess_capacitance(arr_data, sampling_rate_original = 10.0, sampling_rate_final = 10.0):
    arr_times = np.arange(len(arr_data)) * ( 1.0 / float(sampling_rate_original))

    # Go to micro volt.
    arr_x = np.array(arr_data, np.float)
    arr_x *= (float(10**6) / float(2**24)) * 3.3

    # Work in changes in amplitude.

    arr_x_changes = np.diff(arr_x, axis=0)


    # Remove outliers.
    max_ampl = 3.0 * 10 ** 3
    arr_x_changes[arr_x_changes > max_ampl] = max_ampl
    arr_x_changes[arr_x_changes < -max_ampl] = -max_ampl

    # Interpolate so that we have the final sampling rate.
    times_interp = np.arange(start=0.0, stop=arr_times[-1], step=1.0 / sampling_rate_final)
    arr_x_interp = np.interp(times_interp, arr_times[1:], arr_x_changes.flat)
    return arr_x_interp.reshape((len(arr_x_interp), 1))

def capacitance_min_max(arr_data):
    # arr_x = np.array(arr_data, np.float)
    # arr_x *= (float(10 ** 6) / float(2 ** 24)) * 3.3
    #
    # arr_scaled = minmax_scale(arr_x, feature_range=(-1.0, 1.0))
    #
    # from matplotlib import pyplot as plt
    # plt.plot(arr_scaled, "b")
    # plt.plot(arr_scaled, "b")

    arr_x = preprocess_capacitance(arr_data)
    arr_scaled = minmax_scale(arr_x, feature_range=(-1.0, 1.0))

    # from matplotlib import pyplot as plt
    # plt.plot(arr_scaled, "b")
    # # plt.plot(arr_data, "r")
    # plt.show()

    return arr_scaled



def calculate_acc_to_norm(arr_data):
    norm = np.sqrt(np.square(arr_data[:,0]) + np.square(arr_data[:,1]) + np.square(arr_data[:,2]))
    return norm.reshape( (-1,1) )



def relevant_part_of_the_spectrum(dt_ws):
    periods = np.array(1 / dt_ws.columns)
    i_rel_freq = 0
    while periods[i_rel_freq] < 4.6:
        i_rel_freq += 1
    ws = np.array(dt_ws)[:, 0:i_rel_freq]
    return ws


def band_separated_features_from_spectrum(dt_ws, c_splits = 4):
      periods = np.array(1 / dt_ws.columns)
      i_rel_freq = 0
      while periods[i_rel_freq] < 4.6:
          i_rel_freq += 1

      ws = np.array(dt_ws)[:,0:i_rel_freq]
      n, c = ws.shape
      a_w = np.zeros( (n, c_splits*2) )

      for i in range(c_splits):
          w_relevant = ws[:, i/c_splits : (c*(i+1))/c_splits]
          a_w[:, 2*i] =  np.mean(w_relevant, axis=1)
          a_w[:, 2*i + 1] = np.std(w_relevant, axis=1)
      return a_w


def mean_across_all_spectrum(ws):
    periods = np.array(1 / ws.columns)
    i_rel_freq = 0
    while periods[i_rel_freq] < 4.6:
        i_rel_freq += 1

    a_w = np.mean(np.array(ws)[:,0:i_rel_freq], axis=1).reshape( (-1,1) )
    return a_w



def mean_squashed_bands(dt_ws, n_out_bands = 6, w_size = 100, w_step =10):
    ws = relevant_part_of_the_spectrum(dt_ws)
    ws = np.real(ws)

    n_times, n_original_bands = ws.shape
    bands = np.zeros( (n_times, n_out_bands) )
    band_step = n_original_bands / n_out_bands
    for i in range(n_out_bands):
        w_relevant = ws[:,  band_step * i: band_step * (i + 1)]
        bands[:, i] = np.mean(w_relevant)
        # res[:, i*2 + 1] = np.std(w_relevant)

    res = np.zeros(bands.shape)
    i = 0
    while (i + w_size) < len(ws):
        res[i:i + w_step] = np.mean(bands[i:i + w_size], axis=0)
        i += w_step
    return ws

    return res



def squash_dividing_bands(X_win, n_out_bands = 6):
    n_instances, _, n_channels = X_win.shape
    band_step = n_channels / n_out_bands
    res = np.zeros( (n_instances, n_out_bands) )
    for i in range(n_out_bands):
        arr_band = np.mean( X_win[:, :,  band_step * i: band_step * (i + 1)], axis=1)
        res[:,i] = np.mean(arr_band, axis=1)
    return res


def squash_all_bands(X_win):
    # return  np.mean(X_win, axis=1)

    n_inst, win_size, n_channels = X_win.shape

    X_new = X_win.reshape( (n_inst,  win_size/10, 10, n_channels)  )

    X_means = np.mean(X_new, axis=2)
    f_mean = np.mean(X_means, axis=1)
    f_std = np.std(X_means, axis=1)

    return np.hstack([f_mean, f_std])








def sneaky_squash(X_win):
    X_rel = X_win[:, :, 7:23]
    return squash_all_bands(X_rel)

'''
################# add features #####################
Variation:
time domain:  Acc(mag)/Charge,  Acc_Jerk/Charge_Jerk (first_diff as below),
frequency domain: f_Acc/f_Charge, f_Acc_Jerk/f_Charge_Jerk,

Features:
time domain: mean(), std(), mad(), max(), min(), sma(), energy(), iqr(), entropy(), arCoeff()
frequency domain: mean(), std(), mad(), max(), min(), sma(), energy(), iqr(), maxInds(), skewness(), kurtosis(), bandsEnergy()

####################################################
'''
t_function = [mean_w, std_w, mad_w, min_w, max_w, energy_w, iqr_range_w, AR_estimate_w]      #### 11 features
f_function = [mean_w, std_w, mad_w, min_w, max_w, energy_w, iqr_range_w, maxindex_w, skewness_w, kurtosis_w]  ### 10 features

def feature_gen(data, type):
    Content = []
    if type == "time_domain":
        for j in range(len(t_function)):
            if t_function[j] == AR_estimate_w:
                R0, R1, R2, R3 = t_function[j](data)
                Content.append(R0)
                Content.append(R1)
                Content.append(R2)
                Content.append(R3)
            else:
                Content.append(t_function[j](data))
    elif type == "frequency_domain":
        for j in range(len(f_function)):
            Content.append(f_function[j](data))
            '''
            if t_function[j] == band_energy_w:
                R0, R1, R2, R3, R4, R5, R6, R7, R8, R9 = f_function[j](data)
                Content.append(R0)
                Content.append(R1)
                Content.append(R2)
                Content.append(R3)
                Content.append(R4)
                Content.append(R5)
                Content.append(R6)
                Content.append(R7)
                Content.append(R8)
                Content.append(R9)
            else:
                Content.append(f_function[j](data))
            '''
    return Content


def calculate_window_features(X_win):
    feats = np.zeros( (len(X_win), 63) )
    win_size = float(len(X_win[0]))
    win_size_minus_one = win_size - 1.0
    for x_i, x_f in enumerate(X_win):
        x_f = x_f.flat
        first_diff = np.diff(x_f)
        second_diff = np.diff(first_diff)
        f_X_f = np.real(fft_W(x_f))
        f_first_diff = np.real(fft_W(first_diff))
        f_second_diff = np.real(fft_W(second_diff))

        feats[x_i] = feature_gen(x_f, "time_domain") + feature_gen(first_diff, "time_domain") + feature_gen(second_diff, "time_domain") + feature_gen(f_X_f, "frequency_domain") + feature_gen(f_first_diff, "frequency_domain") + feature_gen(f_second_diff, "frequency_domain")

        '''
        feats[x_i, 0] = np.mean(x_f)
        feats[x_i, 1] = np.std(x_f)
        feats[x_i, 2] = np.sum(np.square(x_f)) / win_size
        feats[x_i, 3] = len(np.where(np.diff(np.sign(x_f)))[0]) / win_size_minus_one
        feats[x_i, 4] = np.max(x_f) - np.min(x_f)

        feats[x_i, 5] = np.mean(first_diff)
        feats[x_i, 6] = np.std(first_diff)

        feats[x_i, 7] = np.mean(second_diff)
        feats[x_i, 8] = np.std(second_diff)
        '''

    return feats


# def calculate_window_features(X_win):
#     feats = np.zeros( (len(X_win), 2) )
#     win_size = float(len(X_win[0]))
#     win_size_minus_one = win_size - 1.0
#     for x_i, x_f in enumerate(X_win):
#         feats[x_i, 0] = np.mean(x_f)
#         feats[x_i, 1] = np.std(x_f)
#     return feats



# def calculate_window_features(X_win):
#     feats = np.zeros( (len(X_win), 1) )
#     win_size = float(len(X_win[0]))
#     win_size_minus_one = win_size - 1.0
#     for x_i, x_f in enumerate(X_win):
#         feats[x_i, 0] = np.mean(x_f)
#     return feats



# def calculate_window_features(X_win):
#     feats = np.zeros( (len(X_win), 4) )
#     win_size = float(len(X_win[0]))
#     win_size_minus_one = win_size - 1.0
#     for x_i, x_f in enumerate(X_win):
#         feats[x_i, 0] = np.mean(x_f)
#         feats[x_i, 1] = np.std(x_f)
#         feats[x_i, 2] = np.sum(np.square(x_f)) / win_size
#         feats[x_i, 3] = len(np.where(np.diff(np.sign(x_f)))[0]) / win_size_minus_one
#     return feats

# def calculate_window_features(X_win):
#     feats = np.zeros( (len(X_win), 5) )
#     win_size = float(len(X_win[0]))
#     win_size_minus_one = win_size - 1.0
#     for x_i, x_f in enumerate(X_win):
#         feats[x_i, 0] = np.mean(x_f)
#         feats[x_i, 1] = np.std(x_f)
#         feats[x_i, 2] = np.min(x_f)
#         feats[x_i, 3] = np.max(x_f)
#         feats[x_i, 4] = np.max(x_f) - np.min(x_f)
#     return feats




def get_only_valid_windows(lst_arr_sensor_data, arr_data_is_unreliable, arr_y, win_size, win_step, strategy):

    # Get the sensor sliding window.
    lst_win_data = []
    for arr_sensor_data in lst_arr_sensor_data:
        X_data, _ = make_sliding_window_in_place(arr_sensor_data, None, win_size, win_step, None)
        lst_win_data.append(X_data)


    arr_labels_unreliable = np.zeros(len(arr_y))
    arr_labels_unreliable[np.where(arr_y == "REMOVE")] = 1.0
    arr_data_is_unreliable += arr_labels_unreliable

    # Get indexes of windows that have only reliable points.
    X_win_reliable, y_data = make_sliding_window_in_place(np.array(arr_data_is_unreliable).reshape( (-1,1) ),
                                                          arr_y, win_size, win_step, strategy)

    X_win_reliable = np.sum(X_win_reliable, axis=1).reshape( (len(X_win_reliable),) )
    arr_i_reliable = X_win_reliable == 0.0

    # print("y", len(y_data), len(arr_i_reliable), [ len(X_data) for X_data in lst_win_data])
    return [X_data[arr_i_reliable] for X_data in lst_win_data] , np.array(y_data)[arr_i_reliable]




