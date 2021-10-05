# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 00:55:15 2018

@author: jward
"""


import pycwt
from pycwt.helpers import rect, fft, fft_kwargs
from scipy.signal import convolve2d

import pandas as pd
import os
import numpy as np
import itertools
import pycwt

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.style.use('seaborn-poster')


def smooth( W, dt, dj, scales, deltaj0):
    """Smoothing function used in coherence analysis.
    Hack adaptation to let smoothing work with non-morlet wavelets
    Parameters
    ----------
    W :
    dt :
    dj :
    scales :
    deltaj0: taken from mother wavelet class object
    Returns
    -------
    T :
    """
    # The smoothing is performed by using a filter given by the absolute
    # value of the wavelet function at each scale, normalized to have a
    # total weight of unity, according to suggestions by Torrence &
    # Webster (1999) and by Grinsted et al. (2004).
    m, n = W.shape

    # Filter in time.
    k = 2 * np.pi * fft.fftfreq(fft_kwargs(W[0, :])['n'])
    k2 = k ** 2
    snorm = scales / dt
    # Smoothing by Gaussian window (absolute value of wavelet function)
    # using the convolution theorem: multiplication by Gaussian curve in
    # Fourier domain for each scale, outer product of scale and frequency
    F = np.exp(-0.5 * (snorm[:, np.newaxis] ** 2) * k2)  # Outer product
    smooth = fft.ifft(F * fft.fft(W, axis=1, **fft_kwargs(W[0, :])),
                      axis=1,  # Along Fourier frequencies
                      **fft_kwargs(W[0, :], overwrite_x=True))
    T = smooth[:, :n]  # Remove possibly padded region due to FFT

    if np.isreal(W).all():
        T = T.real

    # Filter in scale. For the Morlet wavelet it's simply a boxcar with
    # 0.6 width.
    wsize = deltaj0 / dj * 2
    win = rect(np.int(np.round(wsize)), normalize=True)
    T = convolve2d(T, win[:, np.newaxis], 'same')  # Scales are "vertical"

    return T


def wt( D, dj=1/12, s0=-1, J=-1,  wavelet='morlet', normalize=True ):
    
    """Wavelet transform for a single continuous vector y1
    Parameters
    ----------
    D : pandas Series with continuous timeseries index 
    dj : float, optional
        Spacing between discrete scales. Default value is 1/12.
        Smaller values will result in better scale resolution, but
        slower calculation and plot.
    s0 : float, optional
        Smallest scale of the wavelet. Default value is 2*dt.
    J : float, optional
        Number of scales less one. Scales range from s0 up to
        s0 * 2**(J * dj), which gives a total of (J + 1) scales.
        Default is J = (log2(N*dt/so))/dj.
        
    """
    
    dt = pd.Timedelta( D.index[1] - D.index[0] )
    dt = 1e5/dt.value
    assert(dt==0.01)

    wavelet = pycwt.wavelet._check_parameter_wavelet(wavelet)
    
    # Checking some input parameters
    if s0 == -1:
        # Number of scales
        s0 = 2 * dt / wavelet.flambda()
    if J == -1:
        # Number of scales
        J = np.int(np.round(np.log2(y1.size * dt / s0) / dj))

    # Makes sure input signals are numpy arrays.
    y1 = np.array( D.T.values )
    # Calculates the standard deviation of both input signals.
    std1 = y1.std()
    # Normalizes both signals, if appropriate.
    if normalize:
        y1_normal = (y1 - y1.mean()) / std1
    else:
        y1_normal = y1

    # Calculates the CWT of the time-series making sure the same parameters
    # are used in both calculations.
    _kwargs = dict(dj=dj, s0=s0, J=J, wavelet=wavelet)
    W1, sj, freq, coi, _, _ = pycwt.wavelet.cwt(y1_normal, dt, **_kwargs)
    scales1 = np.ones([1, y1.size]) * sj[:, None]
    
    W = pd.DataFrame(data=W1.T, columns=sj, index=D.index)
    
    return W, coi # {'W':W1, 'scales':sj, 'freq':freq, 'coi':coi }


def wcompare( W1, W2, dj  ):
    """Wavelet coherence transform (WCT) and cross-wavelet comparison

    The WCT finds regions in time frequency space where the two time
    series co-vary, but do not necessarily have high power.
    
    Uses pre-computed wavelets (they must have matching sizes),
    and using the same parameters dt, dj, sj (explain!!!!)
    
    W1, W2:  matching input wavelet transforms in timeseries vs. scales
    dj : float
        Spacing between discrete scales. Default value is 1/12.
        Smaller values will result in better scale resolution, but
        slower calculation and plot.
    
    
    """
    
    wavelet = pycwt.wavelet._check_parameter_wavelet('morlet')
    
    assert(W1.shape==W2.shape)
    sj = W1.columns.values 
    dt = W1.index[1]-W1.index[0]
    dt = 1e5/dt.value
    data_len = W1.shape[0]
    scales = np.ones([1, data_len]) * sj[:, None]
    
    #print('%s' % (scales))
    
    _W1 = W1.T.values
    _W2 = W2.T.values
    
    S1 = smooth(np.abs(_W1) ** 2 / scales, dt, dj, sj, wavelet.deltaj0)
    S2 = smooth(np.abs(_W2) ** 2 / scales, dt, dj, sj, wavelet.deltaj0)

    # cross-wavelet transform 
    _W12 = _W1 * _W2.conj()

    #! Using a local adapted version of this to allow use with non-Morlet wavelets CHECK!
    S12 = smooth(_W12 / scales, dt, dj, sj, wavelet.deltaj0)
        
    _WCT = np.abs(S12) ** 2 / (S1 * S2)
    
    W12 = pd.DataFrame(data=_W12.T, columns=sj, index=W1.index)
    WCT = pd.DataFrame(data=_WCT.T, columns=sj, index=W1.index)
    
    
    return  WCT, W12
    

def wct(y1, y2, dt, dj=1 / 12, s0=-1, J=-1, sig=True,
        significance_level=0.95, wavelet='morlet', normalize=True, **kwargs):
    """Wavelet coherence transform (WCT).

    The WCT finds regions in time frequency space where the two time
    series co-vary, but do not necessarily have high power.

    Parameters
    ----------
    y1, y2 : numpy.ndarray, list
        Input signals.
    dt : float
        Sample spacing.
    dj : float, optional
        Spacing between discrete scales. Default value is 1/12.
        Smaller values will result in better scale resolution, but
        slower calculation and plot.
    s0 : float, optional
        Smallest scale of the wavelet. Default value is 2*dt.
    J : float, optional
        Number of scales less one. Scales range from s0 up to
        s0 * 2**(J * dj), which gives a total of (J + 1) scales.
        Default is J = (log2(N*dt/so))/dj.
    significance_level (float, optional) :
        Significance level to use. Default is 0.95.
    normalize (boolean, optional) :
        If set to true, normalizes CWT by the standard deviation of
        the signals.

    Returns
    -------
    Adapted by J.Ward to also return scales

    See also
    --------
    cwt, xwt
    """
    wavelet = pycwt.wavelet._check_parameter_wavelet(wavelet)

    # Checking some input parameters
    if s0 == -1:
        # Number of scales
        s0 = 2 * dt / wavelet.flambda()
    if J == -1:
        # Number of scales
        J = np.int(np.round(np.log2(y1.size * dt / s0) / dj))

    # Makes sure input signals are numpy arrays.
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    # Calculates the standard deviation of both input signals.
    std1 = y1.std()
    std2 = y2.std()
    # Normalizes both signals, if appropriate.
    if normalize:
        y1_normal = (y1 - y1.mean()) / std1
        y2_normal = (y2 - y2.mean()) / std2
    else:
        y1_normal = y1
        y2_normal = y2

    # Calculates the CWT of the time-series making sure the same parameters
    # are used in both calculations.
    _kwargs = dict(dj=dj, s0=s0, J=J, wavelet=wavelet)
    W1, sj, freq, coi, _, _ = pycwt.wavelet.cwt(y1_normal, dt, **_kwargs)
    W2, sj, freq, coi, _, _ = pycwt.wavelet.cwt(y2_normal, dt, **_kwargs)

    scales1 = np.ones([1, y1.size]) * sj[:, None]
    scales2 = np.ones([1, y2.size]) * sj[:, None]

    # Smooth the wavelet spectra before truncating.
    S1 = wavelet.smooth(np.abs(W1) ** 2 / scales1, dt, dj, sj)
    S2 = wavelet.smooth(np.abs(W2) ** 2 / scales2, dt, dj, sj)

    # Now the wavelet transform coherence
    W12 = W1 * W2.conj()
    scales = np.ones([1, y1.size]) * sj[:, None]
    S12 = wavelet.smooth(W12 / scales, dt, dj, sj)
    WCT = np.abs(S12) ** 2 / (S1 * S2)
    aWCT = np.angle(W12)

    # Calculates the significance using Monte Carlo simulations with 95%
    # confidence as a function of scale.
    a1, b1, c1 = pycwt.wavelet.ar1(y1)
    a2, b2, c2 = pycwt.wavelet.ar1(y2)
    if sig:
        sig = pycwt.wavelet.wct_significance(a1, a2, dt=dt, dj=dj, s0=s0, J=J,
                                             significance_level=significance_level,
                                             wavelet=wavelet, **kwargs)
    else:
        sig = np.asarray([0])

    return WCT, aWCT, W12, sj, coi, freq, sig


def mwct(D, dt, dj=1 / 12.0, s0=-1, J=-1, sig=True,
         significance_level=0.95, wavelet='morlet', normalize=True, **kwargs):
    ' accepts and array of equal length np.array signals to be analysed'

    nSignals = len(D)
    if nSignals < 2:
        raise 'Not enough signals in D'

    # first signal in array
    y1 = D[0]
    y1 = np.asarray(y1)

    if normalize:
        y1_normal = (y1 - y1.mean()) / y1.std()
    else:
        y1_normal = y1

    # Checking some input parameters
    wavelet = pycwt.wavelet._check_parameter_wavelet(wavelet)
    if s0 == -1:
        # Number of scales
        s0 = 2 * dt / wavelet.flambda()
    if J == -1:
        # Number of scales
        J = np.int(np.round(np.log2(y1.size * dt / s0) / dj))
    _kwargs = dict(dj=dj, s0=s0, J=J, wavelet=wavelet)

    # calculate the initial wavelet
    W1, sj, freq, coi, _, _ = pycwt.wavelet.cwt(y1_normal, dt, **_kwargs)
    scales = np.ones([1, y1.size]) * sj[:, None]  # assumes all signals are same length
    S1 = wavelet.smooth(np.abs(W1) ** 2 / scales, dt, dj, sj)

    W12 = W1
    S12 = S1
    SS = S1.copy()

    for y2 in D[1:]:
        y2 = np.asarray(y2)
        if normalize:
            y2_normal = (y2 - y2.mean()) / y2.std()
        else:
            y2_normal = y2

        W2, sj, freq, coi, _, _ = pycwt.wavelet.cwt(y2_normal, dt, **_kwargs)
        S2 = wavelet.smooth(np.abs(W2) ** 2 / scales, dt, dj, sj)

        SS = SS * S2

        # recursively calculate cross-wavelet
        W12 = W12 * W2.conj()

    S12 = wavelet.smooth(W12 / scales, dt, dj, sj)  # combined, smoothed

    WCT = np.abs(S12) ** nSignals / (SS)
    aWCT = np.angle(W12)

    return WCT, aWCT, W12, sj, coi, freq, sig


def all_combinations(D, keys, data_path, day):  # fn_store = 'all_wct_combinations.h5' ):
    # only processes right-wrist data, but can handle N different people simultaneously

    fn_store = os.path.join(data_path, 'all_wct_combinations_%s.h5' % day)

    dt = 0.05  # seconds  3.1709792e-8 #years
    s0 = 10 * dt  # Starting scale, in this case 2 * 0.05 s = 0.1s
    dj = 1 / 8  # Twelve sub-octaves per octaves
    J = 8 / dj  # Seven powers of two with dj sub-octaves

    for pair in itertools.combinations(keys, 2):
        # devices = ['R-'+a for a in pair]
        Darray = D.loc[:, pair].fillna(0, axis=0).transpose().values
        WCT, aWCT, WXT, scales, coi, freqs, sig95 = mwct(Darray, dt, dj=dj, s0=s0, J=J, wavelet='morlet', sig=False)

        # annoyingly, we cant use '-' within a tag name for storing purposes, so strip any instances if they occur
        tag = 'W_%s_%s' % (pair[0], pair[1])
        tag = tag.replace('-', 'x')

        print('storing wct for: %s' % tag)
        pd.DataFrame(WCT.transpose(), columns=freqs, index=D.index).to_hdf(fn_store, tag)

        tag = tag.replace('W_', 'X_')
        print('(and storing xwt for: %s)' % tag)
        pd.DataFrame(WXT.transpose(), columns=freqs, index=D.index).to_hdf(fn_store, tag)

        # W[pair] = pd.DataFrame( WCT.transpose(), columns = freqs, index = D.index)

    # return W, Darray


def wavelet_coherence(d1, d2):
    # d1, d2 should be pandas Series

    dt = 0.05  # seconds  3.1709792e-8 #years
    mother = pycwt.Morlet(6)
    s0 = 10 * dt  # Starting scale, in this case 2 * 0.05 s = 0.1s
    dj = 1 / 8  # Twelve sub-octaves per octaves
    J = 8 / dj  # Seven powers of two with dj sub-octaves

    D = pd.DataFrame()
    D[d1.name] = d1
    D[d2.name] = d2

    D = D.dropna()

    Darray = D.transpose().values
    WCT, aWCT, WXT, scales, coi, freqs, sig95 = mwct(Darray, dt, dj=dj, s0=s0, J=J, wavelet='morlet', sig=False)

    return pd.DataFrame(WCT.transpose(), columns=freqs, index=D.index)


def cross_wavelet(d1, d2):
    # d1, d2 should be pandas Series

    dt = 0.05  # seconds  3.1709792e-8 #years
    mother = pycwt.Morlet(6)
    s0 = 10 * dt  # Starting scale, in this case 2 * 0.05 s = 0.1s
    dj = 1 / 8  # Twelve sub-octaves per octaves
    J = 8 / dj  # Seven powers of two with dj sub-octaves

    D = pd.DataFrame()
    D[d1.name] = d1
    D[d2.name] = d2

    D = D.dropna()

    Darray = D.transpose().values
    WCT, aWCT, WXT, scales, coi, freqs, sig95 = mwct(Darray, dt, dj=dj, s0=s0, J=J, wavelet='morlet', sig=False)

    return pd.DataFrame(WXT.transpose(), columns=freqs, index=D.index)


def plot_wavelet_coherence(d, w, title):
    t = w.index
    label = 'wavelet coherence'

    d = d.loc[t, :]

    power = np.abs(w.T.values) ** 2
    period = 1 / w.columns

    # plt.close('all')
    plt.ioff()
    figprops = dict(figsize=(11, 8), dpi=72)
    fig = plt.figure(**figprops)

    # First sub-plot, the original time series anomaly and inverse wavelet
    # transform.
    ax = plt.axes([0.1, 0.8, 0.85, 0.15])  # [left, bottom, width, height]`
    # ax.plot(t, iwave, '-', linewidth=1, color=[0.5, 0.5, 0.5])

    ax.plot(t, d, linewidth=1.5)

    # ax.plot(t,D2, 'b', linewidth=1.5)
    ax.set_title('a) {}'.format(title))
    ax.set_ylabel(r'{} [{}]'.format(label, 'normalised'))

    # Second sub-plot, the normalized wavelet power spectrum and significance
    # level contour lines and cone of influece hatched area. Note that period
    # scale is logarithmic.
    bx = plt.axes([0.1, 0.45, 0.85, 0.35], sharex=ax)
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    med = int(np.floor(np.log2(np.median(power))))
    st = abs(np.floor(np.log2(np.nanmean(np.nanstd(power, 0)))))
    llevels = np.arange(med - 2 * st, med + 2 * st, 1)

    p = bx.contourf(t, np.log2(period), np.log2(power), llevels,  # np.log2(levels),
                    extend='both', cmap=plt.cm.viridis)
    # plt.colorbar(p)

    bx.set_ybound(np.log2(min(period)), max(np.log2(period)))

    bx.set_title('b) {} Wavelet Coherence Power Spectrum (Morlett)'.format(label))
    bx.set_ylabel('Period (seconds)')

    Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                            np.ceil(np.log2(period.max())))
    bx.set_yticks(np.log2(Yticks))
    bx.set_yticklabels(Yticks)

    plt.show()

    return ax


def plot_wavelet_example(D, W, tag='W_RxGab_RxD5'):
    k = tag
    devs = k.replace('x', '-').split('_')[1:]  # ['R-'+a for a in k.split('_')[1:] ]

    d_pair = D.loc[:, devs].dropna()
    W[k] = W[k].loc[d_pair.index, :]
    ax = plot_wavelet_coherence(d_pair, W[k], k)

    R = d_pair[devs[0]].rolling(window=200).corr(other=d_pair[devs[1]])  # 10s

    # test inverse of WCT
    dt = 0.05  # seconds  3.1709792e-8 #years

    s0 = 10 * dt  # Starting scale, in this case 2 * 0.05 s = 0.1s
    dj = 1 / 8  # Twelve sub-octaves per octaves
    J = 8 / dj  # Seven powers of two with dj sub-octaves

    Darray = D.loc[:, devs].transpose().values
    WCT, aWCT, WXT, scales, coi, freqs, sig95 = mwct(Darray, dt, dj=dj, s0=s0, J=J, wavelet='morlet', sig=False)

    w = WCT

    # plot these outputs     comparing inverse wct with correlation and summed coherence
    plt.ion()
    fig = plt.figure(**dict(figsize=(11, 8), dpi=72))
    bx = plt.axes([0.1, 0.30, 0.85, 0.25])
    W[k].mean(axis=1).plot(ax=bx)

    cx = plt.axes([0.1, 0.05, 0.85, 0.25], sharex=bx)
    R.plot(ax=cx)
    dx = plt.axes([0.1, 0.65, 0.85, 0.30])

    # iw= pycwt.wavelet.icwt(w[2:],scales[2:],dt=0.05,dj=1/8)
    # dx.plot( R.index, np.abs(iw) )


def create_mean_combinations_file(data_path, day):
    # load pre-computed wavelet combinations, take the real component, and average over 3 different frequency bands
    mW = pd.DataFrame()
    with pd.HDFStore(os.path.join(data_path, 'all_wct_combinations_%s.h5' % day)) as store:
        for k in store.keys():
            if k.startswith('/A'):
                del store[k]  # remove these to save space, no longer needed.
            else:
                W = store[k]
                # select only frequencies with periods between 0.2s and 40s
                (periods, freqs) = zip(
                    *([(1 / f, f) for f in list(W.columns) if 1 / f < 40 and 1 / f > .5]))  # and 1/f > 1.5
                mW[k[1:] + '-a'] = W.loc[:, freqs].apply(np.real).mean(axis=1)
                (periods, freqs) = zip(*([(1 / f, f) for f in list(W.columns) if 1 / f > 5]))  # low freq
                mW[k[1:] + '-l'] = W.loc[:, freqs].apply(np.real).mean(axis=1)
                (periods, freqs) = zip(*([(1 / f, f) for f in list(W.columns) if 1 / f <= 5]))  # high freq
                mW[k[1:] + '-h'] = W.loc[:, freqs].apply(np.real).mean(axis=1)
    # save mean wavelets in a pickle
    mW.to_pickle(os.path.join(data_path, 'all_mean_wct_combinations_%s.pickle' % day))


def load_mean_combinations_files(data_path, day):
    return pd.read_pickle(os.path.join(data_path, 'all_mean_wct_combinations_%s.pickle' % day))


def load_wavelet_files(data_path, day, get_keys=None):
    W = {}
    with pd.HDFStore(os.path.join(data_path, 'all_wct_combinations_%s.h5' % day)) as store:

        print('Keys in store: %s' % store.keys())
        if get_keys is None:
            for k in store.keys():
                W[k[1:]] = store[k]  # remove the '/' from the store key
        else:
            for k in get_keys:
                W[k] = store[k]

    return W


if __name__ == "__main__":

    if False:
        DO_DAYS = [9]
        day = 9

        #     for day in DO_DAYS:

        # D =
        # all_combinations(D, wrists, fn_store = os.path.join(data_path, day) )
        # calculate the mean over all frequencies of the wavelet transform
        # create_mean_combinations_file(data_path, day)

        get_keys = 'W_RxGab_RxOll'
        W = load_wavelet_files(data_path, day, get_keys=[get_keys])
        [D, wrists] = prepare_data_for_wavelet_creation(data_path, day)
        plot_wavelet_example(D, W, tag=get_keys)

    if False:
        freqs = WCT.columns
        t = WCT.index  # np.array(np.arange(1,len(w)+1))#[x.value/10e8 for x in w.index]) # in seconds

        power = np.abs(WCT).values.T ** 2
        Xpower = np.abs(WXT).values.T ** 2
        period = 1 / freqs
        levels = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32]

        figprops = dict(figsize=(11, 8), dpi=72)
        f = plt.figure(**figprops)
        ax = plt.axes([0.05, 0.58, 0.85, 0.4])
        cx = plt.axes([0.05, 0.05, 0.85, 0.4], sharex=ax)

        p1 = cx.contourf(t, np.log2(period), np.log2(power), np.log2(levels),
                         extend='both', cmap=plt.cm.viridis)

        p2 = cx.contourf(t, np.log2(period), np.log2(Xpower), np.log2(levels),
                         extend='both', cmap=plt.cm.viridis)


