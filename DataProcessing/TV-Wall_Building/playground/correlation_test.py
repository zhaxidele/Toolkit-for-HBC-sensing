import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import recognition.igroups_wtc as iwtc
from parser import read_files
from time_converter import video_time_to_index


def plot_wavelet_coherence(data, wavelet_coherence, sample_rate=10, coherence=True):
    t = wavelet_coherence.index
    power = np.abs(wavelet_coherence.T.values) ** 2
    period = 1 / wavelet_coherence.columns

    # plt.close('all')
    plt.ioff()

    fig, axarr = plt.subplots(3, 1, sharex=True, figsize=(16, 8), dpi=72, gridspec_kw={'height_ratios':[1, 1, 2]})

    names = data.columns.values


    for i, n in enumerate(names):
        axarr[i].plot(t / sample_rate, data[n])
        axarr[i].set_title(n)
        axarr[i].grid()


    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    med = int(np.floor(np.log2(np.median(power))))
    st = abs(np.floor(np.log2(np.nanmean(np.nanstd(power, 0)))))
    llevels = np.arange(med - 2 * st, med + 2 * st, 1)

    p = axarr[2].contourf(t/sample_rate, np.log2(period), np.log2(power), llevels,  # np.log2(levels),
                    extend='both', cmap=plt.cm.viridis)
    # plt.colorbar(p)

    axarr[2].set_ybound(np.log2(min(period)), max(np.log2(period)))

    if coherence:
        axarr[2].set_title('Wavelet Coherence Power Spectrum (Morlett)')
    else:
        axarr[2].set_title('Cross Wavelet Power Spectrum (Morlett)')
    axarr[2].set_ylabel('Period (seconds)')

    Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                            np.ceil(np.log2(period.max())))
    axarr[2].set_yticks(np.log2(Yticks))
    axarr[2].set_yticklabels(Yticks)
    axarr[2].set_xlabel('Time [s]')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    root = '/Users/hevesi/Downloads/2_Data_Synchronized'
    d = read_files(root)

    start = "04:00"
    end = "06:10"
    print(d.columns.values)

    signal_1 = "P2_Wrist_charge"
    signal_2 = "P2_Wrist_g_x"
    normalise = True
    global_scale = True
    plot_type = 'coherece' # or  'cross_wavelet'
    period_filter = 32


    start_index = video_time_to_index(start, 10)
    end_index = video_time_to_index(end, 10)
    if not global_scale:
        d = d.loc[start_index:end_index]

    data = np.vstack([d[signal_1].values, d[signal_2].values])
    WCT, aWCT, WXT, scales, coi, freqs, sig95 = iwtc.mwct(data, 0.1, normalize=normalise)

    if plot_type == 'coherence':
        W = pd.DataFrame(WCT.transpose(), columns=freqs, index=d.index)
    else:
        W = pd.DataFrame(WXT.transpose(), columns=freqs, index=d.index)

    W = W.loc[:, 2**np.log2((1 / freqs)) < period_filter]

    plot_wavelet_coherence(d[[signal_1, signal_2]].loc[start_index:end_index], W.loc[start_index:end_index], coherence=plot_type=='coherence')

