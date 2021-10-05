from __future__ import print_function

import glob
import itertools
import json
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
from preprocess import calculate_acc_to_norm, preprocess_capacitance

from data import get_exp_data, get_user_map
from definitions import COH_SAVE_DIR
from definitions import RAW_DATA_DIR


def format_i_to_time(value, tick_number):
    from data import from_t_in_seconds_to_string
    t_in_sec = value / 10.0
    return from_t_in_seconds_to_string(t_in_sec)


def plot_wavelet_spectrum(ax, wavelet_coherence, str_pre_title , sample_rate=10.0 ,coherence=True):
    t = wavelet_coherence.index
    power = np.abs(wavelet_coherence.T.values) ** 2
    period = 1 / wavelet_coherence.columns

    med = int(np.floor(np.log2(np.median(power))))
    st = abs(np.floor(np.log2(np.nanmean(np.nanstd(power, 0)))))
    llevels = np.arange(med - 2 * st, med + 2 * st, 1)

    ax.contourf(t , np.log2(period), np.log2(power), llevels,  # np.log2(levels),
                      extend='both', cmap=plt.cm.viridis)

    ax.set_ybound(np.log2(min(period)), max(np.log2(period)))

    if coherence:
        ax.set_title(str_pre_title + ' Wavelet Coherence Power Spectrum (Morlett)')
    else:
        ax.set_title(str_pre_title + ' Cross Wavelet Power Spectrum (Morlett)')
    ax.set_ylabel('Period (seconds)')

    Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                            np.ceil(np.log2(period.max())))
    ax.set_yticks(np.log2(Yticks))
    ax.set_yticklabels(Yticks)
    # ax.set_xlabel('Time h.m.s.s/30')
    # plt.tight_layout()




global I_USER, EXP_NOW
global ax1, ax2

def change_person_plot(event):
    global I_USER, EXP_NOW

    I_USER = (I_USER + 1) % 3
    USR_NOW = "P" + str(I_USER + 1)

    try:
        print(event)
        path_coh = COH_SAVE_DIR + "|".join(
            [USR_NOW, USR_NOW, EXP_NOW, "Right_Wrist", "Charge", "Right_Wrist", "Charge"]) + "_coh.pkl"
        w_coh = pd.read_pickle(path_coh)
        plot_wavelet_spectrum(ax1, w_coh, USR_NOW + " norm_acc -> charge", coherence=True)

        path_corr = COH_SAVE_DIR + "|".join(
            [USR_NOW, USR_NOW, EXP_NOW, "Right_Wrist", "Charge", "Right_Wrist", "Charge"]) + "_corr.pkl"
        w_corr = pd.read_pickle(path_corr)
        plot_wavelet_spectrum(ax2, w_corr, USR_NOW + " norm_acc -> charge", coherence=False)
    except:
        print("Probably no W file")



def plot_label_predictions_debug(y_gt, y_pred, cl_probs, lst_ordered_classes):

    f, axarr = plt.subplots(2, 1, sharex=True)
    axarr[0].xaxis.set_major_formatter(plt.FuncFormatter(format_i_to_time))

    clr_map = {"NULL": "silver", "WALKING": "b", "CARRY ALONE": "g", "CARRY TOGETHER": "r", "LIFT" : "y" ,
               "DROP" : "c", "REMOVE" : "w"}

    def add_lbl_boxes(user_lbls, ax, v_y_axis):
        last_y, i_start = user_lbls[0], 0
        for i, y in enumerate(user_lbls):
            if y != last_y:
                clr = clr_map[last_y]
                rect = Rectangle((i_start, v_y_axis*10), i - i_start, 8, color=clr, label=y)
                ax.add_artist(rect)
                i_start, last_y = i, y
        # The last label.
        rect = Rectangle((i_start, v_y_axis * 10), i - i_start, 8, color=clr, label=y)
        ax.add_artist(rect)

    add_lbl_boxes(y_gt, axarr[0], 1)
    add_lbl_boxes(y_pred, axarr[0], 2)

    # axarr[0].plot()
    for cl_i, cl_name in enumerate(lst_ordered_classes):
        axarr[1].plot(cl_probs[:, cl_i], label=cl_name, alpha = 0.5, color = clr_map[cl_name])
    axarr[1].legend()

    lst_lbl_legend = [Patch(color=clr_map[nm], label=nm) for nm in clr_map]
    axarr[0].legend(handles=lst_lbl_legend, loc=0, fontsize=8)
    axarr[0].set_ylim((0, 40))
    axarr[0].text(-375, 12, "True")
    axarr[0].text(-375, 22, "Predicted")


    plt.show()




def plot_label_predictions():
    lst_placement = ["Right_Wrist"]
    lst_type = ["Charge", "acc"]
    map_users_in_exp = get_user_map()
    for exp_dir in glob.glob(RAW_DATA_DIR + "/*/"):
        exp_nm = exp_dir.split("/")[-2]
        # map P -> (Snsr_name -> Snsr_vals, Lbls)
        map_exp_data = get_exp_data(exp_nm, lst_placement, lst_type)

        lst_lbl_patches = []
        clr_map = {"NULL": "white", "WALKING": "b", "CARRY ALONE": "g", "CARRY TOGETHER": "r", "LOST DATA": "k"}

        map_my_data = {}
        i_user = 0
        f, axarr = plt.subplots(3, 1, sharex=True)
        axarr[0].xaxis.set_major_formatter(plt.FuncFormatter(format_i_to_time))

        for usr, user_data in map_exp_data.items():
            axarr[i_user].set_title(usr)

            try:
                map_data, user_lbls = user_data
                arr_reliability = map_data["Right_Wrist|acc"][1] + map_data["Right_Wrist|Charge"][1]
                user_lbls = np.array(user_lbls)

                axarr[i_user].plot(calculate_acc_to_norm(map_data["Right_Wrist|acc"][0]).flat, alpha=0.5, label=usr + "_acc")


                # Add labels.
                last_y, i_start = user_lbls[0], 0
                for i, y in enumerate(user_lbls):
                    if y != last_y:
                        clr = clr_map[last_y] if arr_reliability[i_start] == 0.0 else 'k'
                        rect = Rectangle((i_start, 10), i - i_start, 8, color=clr, label=y)
                        axarr[i_user].add_artist(rect)
                        lst_lbl_patches.append(rect)
                        i_start, last_y = i, y
            except:
                print("ERROR", exp_nm, usr)

            # Move on.
            i_user += 1




        for ax in axarr:
            lst_lbl_legend = [Patch(color=clr_map[nm], label=nm) for nm in clr_map]
            ax.legend(handles=lst_lbl_legend, loc=0, fontsize=8)
            ax.set_ylim((0, 40))

        # axarr[1].set_ylim((5, 20))
        # # axarr[3].set_ylim((0.0, 3.0))
        # # axarr[4].set_ylim((0.0, 3.0))
        #
        # axarr[0].text(-975, 12, "P1")
        # axarr[0].text(-975, 22, "P2")
        # axarr[0].text(-975, 32, "P3")

        plt.show()



def convert_to_group_lbls(user_lbls_A, user_lbls_B):
      lbls_both = []
      for la, lb in zip(user_lbls_A, user_lbls_B):
        if la == "REMOVE" or lb == "REMOVE":
          lbls_both.append("REMOVE")
        elif la == "CARRY TOGETHER" and lb == "CARRY TOGETHER":
          lbls_both.append("CARRY TOGETHER")
        elif la == "LIFT" and lb == "LIFT":
          lbls_both.append("LIFT")
        elif la == "DROP" and lb == "DROP":
          lbls_both.append("DROP")
        else:
          lbls_both.append("NULL")
      lbls_both = np.array(lbls_both)
      return lbls_both




def plot_all_sessions():
    global I_USER, EXP_NOW
    global ax1, ax2

    lst_placement = ["Right_Wrist"]
    lst_type = ["Charge", "acc"]
    map_users_in_exp = get_user_map()
    for exp_dir in glob.glob(RAW_DATA_DIR + "/*/"):
        exp_nm = exp_dir.split("/")[-2]
        # map P -> (Snsr_name -> Snsr_vals, Lbls)
        map_exp_data = get_exp_data(exp_nm, "paper_hard", lst_placement, lst_type)

        lst_users = map_users_in_exp[exp_nm]
        grp_id = "_".join(sorted(lst_users))


        f, axarr = plt.subplots(5, 1, sharex=True)
        f.suptitle(exp_nm)
        axarr[0].xaxis.set_major_formatter(plt.FuncFormatter(format_i_to_time))

        # plt.show()



        axarr[0].set_title("Labels")
        axarr[1].set_title("Norm Acc")
        axarr[2].set_title("Charge")

        ax1 = axarr[3]
        ax2 = axarr[4]
        I_USER = 0
        EXP_NOW = exp_nm

        lbls_ordered = None

        lst_lbl_patches = []

        # clr_map = { "NULL" : "white", "WALKING" : "b",  "CARRY ALONE" : "g", "CARRY TOGETHER" : "r", "LOST DATA" : "k"}
        clr_map = {"NULL": "silver", "WALKING": "b", "CARRY ALONE": "g", "CARRY TOGETHER": "r", "LIFT": "y",
         "DROP": "c", "LOST DATA" : "k", "REMOVE" : "w"}


        map_my_data = {}

        for usr, user_data in map_exp_data.items():
            map_data, user_lbls = user_data
            user_lbls = np.array(user_lbls)

            print(usr)
            print(map_data.keys())


            try:
                acc_data = map_data["Right_Wrist|acc"][0]
                cap_data = map_data["Right_Wrist|Charge"][0]

                map_my_data[usr] = (preprocess_capacitance(cap_data), calculate_acc_to_norm(acc_data))
                arr_reliability = map_data["Right_Wrist|acc"][1] + map_data["Right_Wrist|Charge"][1]


                axarr[2].plot(preprocess_capacitance(cap_data).flat, alpha=0.5, label= usr + "_cap")
                axarr[1].plot(calculate_acc_to_norm(acc_data).flat, alpha=0.5, label= usr + "_acc")


                # i_coh_plot = 3
                # path_coh = COH_SAVE_DIR + "|".join(
                #     ["P1", "P2", exp_nm, "Right_Wrist", "Charge", "Right_Wrist", "Charge"] ) + "_coh.pkl"
                # w_coh = pd.read_pickle(path_coh)
                # plot_wavelet_spectrum(axarr[i_coh_plot], w_coh,  "Coh" ,  coherence=True)
                #
                # path_corr = COH_SAVE_DIR + "|".join(
                #     ["P1", "P2", exp_nm, "Right_Wrist", "Charge", "Right_Wrist", "Charge"]) + "_corr.pkl"
                # w_corr = pd.read_pickle(path_corr)
                # plot_wavelet_spectrum(axarr[i_coh_plot + 1], w_corr,  "Corr" , coherence=False)




                # Add labels.
                from matplotlib.patches import Patch
                from matplotlib.patches import Rectangle

                last_y, i_start = user_lbls[0], 0
                usr_n = int(usr[1])
                for i, y in enumerate(user_lbls):
                    if y != last_y:
                      clr = clr_map[last_y] if arr_reliability[i_start] == 0.0 else 'k'
                      rect = Rectangle( (i_start, usr_n*10),  i - i_start, 8, color=clr, label=y)
                      axarr[0].add_artist(rect)
                      lst_lbl_patches.append(rect)
                      i_start, last_y = i, y


            except:
                print("MISSING", exp_nm, usr)


        def calc_coef_means(w_path, labels=None):
            w_dt = pd.read_pickle(w_path)
            periods = np.array(1 / w_dt.columns)
            # from preprocess import relevant_part_of_the_spectrum
            # ws  relevant_part_of_the_spectrum(w_dt)

            a = np.real(np.array(w_dt))
            # a = np.abs(np.array(w_dt))

            if labels is not None:
                ls = np.unique(labels)
                map_per_cl = { l : [] for l in ls}

            corr, w_size, w_step = [], 10, 10
            corr_t = []
            i = 0
            while (i  + w_size) < len(a):
                corr.append( np.mean( a[i:i + w_size], axis=0) )
                corr_t.append(i)

                if labels is not None:
                    cl, cl_count = np.unique(labels[i:i + w_size], return_counts=True)
                    cl_selected = cl[np.argmax(cl_count)]
                    map_per_cl[cl_selected].append(np.mean( a[i:i + w_size], axis=0))

                i += w_step


            # Statistics.
            # plt.clf()
            #
            # i_rel_freq = 0
            # while periods[i_rel_freq] < 4.6:
            #     i_rel_freq += 1
            # gl_arr = np.array(corr)[:, 0:i_rel_freq]
            # plt.xscale("log")
            # plt.errorbar( periods[0:i_rel_freq]  + np.random.normal(0.01, 0.01, len(periods[0:i_rel_freq])),
            #               np.mean(gl_arr, axis=0), yerr= np.std(gl_arr, axis=0),
            #                 label="GLOBAL", alpha=0.4, fmt='--o--')
            #
            #
            # # print("GLOBAL", "MEAN", np.mean(gl_arr, axis=0), "STD", np.std(gl_arr, axis=0) )
            # print("GLOBAL", np.mean(gl_arr), np.std(gl_arr) )
            # for cl_name, cl_lst_wins in map_per_cl.items():
            #     cl_data = np.array(cl_lst_wins)[:, 0:i_rel_freq]
            #     # print(cl_name, np.mean(cl_data, axis=0), np.std(cl_data, axis=0) )
            #     print(cl_name, np.mean(cl_data), np.std(cl_data))
            #
            #     plt.errorbar( periods[0:i_rel_freq] + np.random.normal(0.01, 0.01, len(periods[0:i_rel_freq])),
            #                   np.mean(cl_data, axis=0), yerr=np.std(cl_data, axis=0),
            #                       label=cl_name, alpha=0.4, fmt='--o--')
            # plt.legend()
            # plt.show()






            return np.array(corr_t), np.array(corr), periods

        try:
            path_coh = COH_SAVE_DIR + "|".join(
                ["P1", "P1", exp_nm, "Right_Wrist", "norm_acc", "Right_Wrist", "Charge"] ) + "_angl.pkl"

            lbls = np.array(map_exp_data["P1"][1])
            # lbls = convert_to_group_lbls(map_exp_data["P1"][1], map_exp_data["P2"][1])

            corr_t, corr, periods = calc_coef_means(path_coh, lbls)
            axarr[3].set_title("COH P1 norm acc -> P1 norm acc" )

            for per_i, per_v in enumerate(periods):
                freq = 1.0 / per_v
                if per_v < 4.6:
                    axarr[3].plot(corr_t, corr[:,per_i],  alpha=0.5, color=str(freq*100.0), label= str(freq)[0:4])
        except:
            pass





        # path_coh = COH_SAVE_DIR + "|".join(
        #     ["P1", "P1", exp_nm, "Right_Wrist", "norm_acc", "Right_Wrist", "norm_acc"]) + "_corr.pkl"
        # corr_t, corr, periods = calc_coef_means(path_coh)
        # axarr[4].set_title("CORR P1 norm acc -> P1 norm acc")
        # for per_i, per_v in enumerate(periods):
        #     freq = 1.0 / per_v
        #     if per_v < 4.6:
        #         axarr[4].plot(corr_t, corr[:, per_i], alpha=0.5, color=str(freq), label=str(freq)[0:4])
















        # axarr[0].set_yticks(range(len(arr_y_unique)), lbls_ordered)

        # pc = PatchCollection(lst_lbl_patches, alpha=0.5)
        # axarr[0].add_collection(pc)
        # artists = ax.errorbar(xdata, ydata, xerr=xerror, yerr=yerror,
        #                       fmt='None', ecolor='k')

        # try:
        #     P1_cap, P1_acc = map_my_data["P1"]
        #     P2_cap, P2_acc = map_my_data["P2"]
        #
        #     from scipy.signal import correlate
        #     from sklearn.preprocessing import minmax_scale
        #     from scipy.stats import pearsonr
        #
        #     def calc_corr_coefs(a, b):
        #         corr, w_size, w_step = [], 100, 1
        #         corr_t = []
        #         i = 0
        #         while (i  + w_size) < len(a):
        #             corr_t.append(i)
        #             corr.append(  pearsonr(a[i:i + w_size], b[i:i + w_size])[0] )
        #             i += w_size
        #         return corr_t, corr
        #
        #
        #     t, c = calc_corr_coefs(P1_acc, P1_cap)
        #     axarr[3].plot( t, c, alpha=0.5, label= "corr(p1_acc, p1_cap)" )
        #
        #     t, c = calc_corr_coefs(P2_acc, P2_cap)
        #     axarr[3].plot(t, c, alpha=0.5, label="corr(p2_acc, p2_cap)")
        #
        #     t, c = calc_corr_coefs(P1_acc, P2_cap)
        #     # axarr[4].plot(t, c, alpha=0.5, label="corr(p1_acc, p2_cap)")
        #
        #     t, c = calc_corr_coefs(P2_acc, P1_cap)
        #     # axarr[4].plot(t, c, alpha=0.5, label="corr(p2_acc, p1_cap)")
        #
        #     t, c = calc_corr_coefs(P1_acc, P2_acc)
        #     axarr[4].plot(t, c, alpha=0.5, label="corr(p1_acc, p2_acc)")
        #
        #     t, c = calc_corr_coefs(P1_cap, P2_cap)
        #     axarr[4].plot(t, c, alpha=0.5, label="corr(p1_cap, p2_cap)")
        #
        #     print("Done")
        # except Exception as e:
        #     print("Exception!!", e)











        lst_lbl_legend = [Patch(color=clr_map[nm], label=nm)  for nm in clr_map]
        axarr[0].legend(handles=lst_lbl_legend, loc=0, fontsize=8)
        axarr[0].set_ylim( (0,60) )
        axarr[1].set_ylim((5, 20))
        # axarr[3].set_ylim((0.0, 3.0))
        # axarr[4].set_ylim((0.0, 3.0))

        axarr[0].text(-975, 12, "P1")
        axarr[0].text(-975, 22, "P2")
        axarr[0].text(-975, 32, "P3")


        # axarr[0].legend(loc='upper right')

        for ax in axarr[1:]:
            ax.legend(loc='upper right')


        # from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
        # button_ax = plt.axes([0, 0, 1, 1])
        # ip = InsetPosition(axarr[0], [0.6, 0.7, 0.2, 0.1])  # posx, posy, width, height
        # button_ax.set_axes_locator(ip)
        # bnext = Button(button_ax, 'Next User')
        # bnext.on_clicked(change_person_plot)

        plt.show()
        # plt.clf()


def plot_performance(path_file):

    res_map = json.load( open(path_file, 'r') )
    map_pts = defaultdict(list)
    str_title = "???"
    str_classes = None
    for k, lst_data in res_map.items():
        if k == u'features':
            str_title = "\n".join(lst_data)
        else:
            # if str_classes is None:
            #     str_classes = ",".join(lst_data[0][1])

            map_pts[k].extend(v[0] for v in lst_data)
    # str_title += "\n" + str_classes

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.set_title(str_title)

    data_to_plot, names_to_plot = [], []

    i = 0.0
    for nm, vs in map_pts.items():
        data_to_plot.append(vs)
        names_to_plot.append(nm)

        x = np.random.normal(i+1.0, 0.04, len(vs))
        ax.scatter(x, vs, alpha = 0.5)
        i += 1.0



    bp = ax.boxplot(data_to_plot)
    ax.set_xticklabels(names_to_plot)

    # plt.show()
    plt.savefig(path_file.replace(".json", ".png"))
    plt.clf()


import os
def plot_performance_across_combinations(lst_path_files, str_title):

    map_feats = {}
    map_feats["<minmax>Right_Wrist|Charge"] = "wrist_cap"
    map_feats["<norm>Right_Wrist|acc"] = "wrist_norm_acc"
    map_feats["<norm>Left_Calf|acc"] = "calf_norm_acc"
    map_feats["<coh_raw_relevant>(<norm>Right_Wrist|acc,<raw>Right_Wrist|Charge)"] = "coh_acc_cap"
    map_feats["<angl_raw_relevant>(<norm>Right_Wrist|acc,<raw>Right_Wrist|Charge)"] = "angl_acc_cap"


    map_feats["<P1><minmax>Right_Wrist|Charge"] = "P1 wrist cap"
    map_feats["<P2><minmax>Right_Wrist|Charge"] = "P2_wrist cap"
    map_feats["<P1><norm>Right_Wrist|acc"] = "P1 wrist acc"
    map_feats["<P2><norm>Right_Wrist|acc"] = "P2 wrist acc"
    map_feats["<P1><norm>Left_Calf|acc"] = "P1 calf acc"
    map_feats["<P2><norm>Left_Calf|acc"] = "P2 calf acc"
    map_feats["<coh_raw_relevant>(<P1><raw>Right_Wrist|Charge,<P2><raw>Right_Wrist|Charge)"] = "coh P1 wrist cap P2 wrist cap"
    map_feats["<angl_raw_relevant>(<P1><raw>Right_Wrist|Charge,<P2><raw>Right_Wrist|Charge)"] = "angl P1 wrist cap P2 wrist cap"

    def from_features_to_name(lst_feats):
        str_nm = ""
        for str_f in lst_feats:
            str_nm += map_feats[str_f] + "\n"

        return str_nm


    map_pts = defaultdict(list)
    for path_res in lst_path_files:
        res_map = json.load(open(path_res, 'r'))
        nm_trial = from_features_to_name(res_map[u'features'])
        for k, lst_data in res_map.items():
            if k != u'features':
                for v in lst_data:
                    arr_cm = np.array(v[2])
                    gt_cnt = np.sum(arr_cm, axis=1)
                    if v[0] < 0.3:
                        print("NOW", nm_trial, v[0], v[3])
                        print(arr_cm)


                    if not np.any(gt_cnt == 0):
                        map_pts[nm_trial].append(v[0])
                    else:
                        print("DISCARDED", v[3])
                        # print(arr_cm)





    fig = plt.figure(1 )#figsize=(16, 16))
    ax = fig.add_subplot(111)
    # ax.set_title(str_title)

    data_to_plot, names_to_plot = [], []


    clr_colors = []

    to_plot = [v for v in map_pts.items() ]
    print(to_plot)

    # to_plot = [to_plot[0], to_plot[1], to_plot[2], to_plot[4], to_plot[3] ]
    # print(to_plot)

    i = 0.0
    for nm, vs in to_plot:
        data_to_plot.append(vs)
        names_to_plot.append(nm)

        x = np.random.normal(i + 1.0, 0.04, len(vs))
        # ax.scatter(x, vs, alpha=0.5)
        ax.scatter(vs, x, alpha=0.5)
        i += 1.0

    bp = ax.boxplot(data_to_plot, vert=False)
    plt.xlabel('Macro F1 Score')
    ax.set_yticklabels(names_to_plot) #fontdict={'fontsize': 7})

    # locs, labels = plt.xticks()
    # plt.setp(labels, rotation=275)

    # plt.show()
    plt.savefig("./" + str_title + ".png", bbox_inches='tight')
    plt.clf()





def plot_confusion_matrix(cfm, str_title, lst_ordered_classes, res_path, normalize=True, cmap=plt.cm.Blues):

  # Normalize if needed.
  if normalize:
    cfm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]

  #  Print.
  np.set_printoptions(precision=2)
  plt.figure()
  plt.title(str_title)

  plt.imshow(cfm, interpolation='nearest', cmap=cmap)
  plt.colorbar()
  tick_marks = np.arange(len(lst_ordered_classes))
  plt.xticks(tick_marks, lst_ordered_classes, rotation=45)
  plt.yticks(tick_marks, lst_ordered_classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cfm.max() / 2.
  for i, j in itertools.product(range(cfm.shape[0]), range(cfm.shape[1])):
    plt.text(j, i, format(cfm[i, j], fmt), fontdict={'fontsize': 17},
             horizontalalignment="center",
             color="white" if cfm[i, j] > thresh else "black")


  plt.ylabel('True label')
  plt.xlabel('Predicted label')

  plt.tight_layout()
  plt.savefig(res_path, bbox_inches='tight')
  # plt.show()


def plot_joined_cfm(path_res, res_path):

    joined_cfm = None
    all_classes = None
    all_valid_f1s = []

    res_map = json.load(open(path_res, 'r'))
    for k, lst_data in res_map.items():
        if k != u'features':
            for v in lst_data:
                arr_cm = np.array(v[2])
                joined_cfm = arr_cm if joined_cfm is None else (joined_cfm + arr_cm)
                all_classes = np.array(v[1])

                gt_cnt = np.sum(arr_cm, axis=1)
                if not np.any(gt_cnt == 0):
                    all_valid_f1s.append(v[0])


    n_classes = len(all_classes)
    correct = np.array([joined_cfm[i,i] for i in xrange(n_classes)], dtype=np.float)
    tp_plus_fneg = np.array([ np.sum(joined_cfm[:,i]) for i in xrange(n_classes)], dtype=np.float)
    tp_plus_fpos = np.array([ np.sum(joined_cfm[i,:]) for i in xrange(n_classes)], dtype=np.float)

    precision = correct / tp_plus_fpos
    recall = correct / tp_plus_fneg
    f1_per_cl = 2.0 * (precision * recall) / (precision + recall)
    f1 = np.mean(f1_per_cl)

    f1_singles_mean = np.mean(all_valid_f1s)
    f1_singles_srd = np.std(all_valid_f1s)
    str_title = "Combined Macro F1 " + str(f1)[0:4] + ", Mean " + str(f1_singles_mean)[0:4] \
        + " Std " + str(f1_singles_srd)[0:4]


    plot_confusion_matrix(joined_cfm, str_title, all_classes, res_path)
    plt.clf()







if __name__ == '__main__':
    # plot_all_sessions()
    # exit(0)
    #
    # for baseline_path in glob.glob("CL_*coh-*.json"):
    #     nm_out = "./" + os.path.split(baseline_path)[-1].split(".")[0] + "_CFM.png"
    #     plot_joined_cfm(baseline_path, nm_out)
    # exit(0)

    lst_single_baselines = [
        "/Users/sizhenbian/venv_2.7/Cap/capacitive-data_analysis/recognition/CL_calf_acc-paper_res.json",
        "/Users/sizhenbian/venv_2.7/Cap/capacitive-data_analysis/recognition/CL_all_base-paper_res.json",
        "/Users/sizhenbian/venv_2.7/Cap/capacitive-data_analysis/recognition/CL_wrist_Charge-paper_res.json",
        "/Users/sizhenbian/venv_2.7/Cap/capacitive-data_analysis/recognition/CL_wrist_acc-paper_res.json",
        "/Users/sizhenbian/venv_2.7/Cap/capacitive-data_analysis/recognition/CL_wrist-paper_res.json",
        "/Users/sizhenbian/venv_2.7/Cap/capacitive-data_analysis/recognition/CL_acc_all-paper_res.json"
    ]

    for baseline_path in lst_single_baselines:
        nm_out = "./" + os.path.split(baseline_path)[-1].split(".")[0] + "_CFM.png"
        try:
            plot_joined_cfm(baseline_path, nm_out)
        except:
            print("ERROR", baseline_path)
    plot_performance_across_combinations(lst_single_baselines,"SINGLE_USER_EASY_CLASSES")

    lst_hard_baselines = [
        "/Users/sizhenbian/venv_2.7/Cap/capacitive-data_analysis/recognition/CL_calf_acc-paper_hard_res.json",
        "/Users/sizhenbian/venv_2.7/Cap/capacitive-data_analysis/recognition/CL_all_base-paper_hard_res.json",
        "/Users/sizhenbian/venv_2.7/Cap/capacitive-data_analysis/recognition/CL_wrist_Charge-paper_hard_res.json",
        "/Users/sizhenbian/venv_2.7/Cap/capacitive-data_analysis/recognition/CL_wrist_acc-paper_hard_res.json",
        "/Users/sizhenbian/venv_2.7/Cap/capacitive-data_analysis/recognition/CL_wrist-paper_hard_res.json",
        "/Users/sizhenbian/venv_2.7/Cap/capacitive-data_analysis/recognition/CL_acc_all-paper_hard_res.json"
    ]
    plot_performance_across_combinations(lst_hard_baselines, "SINGLE_USER_HARD_CLASSES")
    for baseline_path in lst_hard_baselines:
        nm_out = "./" + os.path.split(baseline_path)[-1].split(".")[0] + "_CFM.png"
        try:
            plot_joined_cfm(baseline_path, nm_out)
        except:
            print("ERROR", baseline_path)

    lst_group_baselines = [
        "/Users/sizhenbian/venv_2.7/Cap/capacitive-data_analysis/recognition/CL_group_calf_acc-paper_res.json",
        "/Users/sizhenbian/venv_2.7/Cap/capacitive-data_analysis/recognition/CL_group_all_all-paper_res.json",
        "/Users/sizhenbian/venv_2.7/Cap/capacitive-data_analysis/recognition/CL_group_wrist_cap-paper_res.json",
        "/Users/sizhenbian/venv_2.7/Cap/capacitive-data_analysis/recognition/CL_group_wrist_acc-paper_res.json",
        "/Users/sizhenbian/venv_2.7/Cap/capacitive-data_analysis/recognition/CL_group_wrist_all-paper_res.json",
        "/Users/sizhenbian/venv_2.7/Cap/capacitive-data_analysis/recognition/CL_group_acc_all-paper_res.json"
    ]
    for baseline_path in lst_group_baselines:
        nm_out = "./" + os.path.split(baseline_path)[-1].split(".")[0] + "_CFM.png"
        plot_joined_cfm(baseline_path, nm_out)
    plot_performance_across_combinations(lst_group_baselines, "GROUP_USER_EASY_CLASSES")


    lst_group_hard_baselines = [
        "/Users/sizhenbian/venv_2.7/Cap/capacitive-data_analysis/recognition/CL_group_calf_acc-paper_hard_res.json",
        "/Users/sizhenbian/venv_2.7/Cap/capacitive-data_analysis/recognition/CL_group_all_all-paper_hard_res.json",
        "/Users/sizhenbian/venv_2.7/Cap/capacitive-data_analysis/recognition/CL_group_wrist_cap-paper_hard_res.json",
        "/Users/sizhenbian/venv_2.7/Cap/capacitive-data_analysis/recognition/CL_group_wrist_acc-paper_hard_res.json",
        "/Users/sizhenbian/venv_2.7/Cap/capacitive-data_analysis/recognition/CL_group_wrist_all-paper_hard_res.json",
        "/Users/sizhenbian/venv_2.7/Cap/capacitive-data_analysis/recognition/CL_group_acc_all-paper_hard_res.json"
    ]
    for baseline_path in lst_group_hard_baselines:
        nm_out = "./" + os.path.split(baseline_path)[-1].split(".")[0] + "_CFM.png"
        try:
            plot_joined_cfm(baseline_path, nm_out)
        except:
            print("ERROR", baseline_path)

    plot_performance_across_combinations(lst_group_hard_baselines, "GROUP_USER_HARD_CLASSES")

    exit(0)


    # lst_group_baselines = [
    #     "/home/vitor/git/capacitive-data_analysis/recognition/DL_group_calf_acc-paper_res.json",
    #     "/home/vitor/git/capacitive-data_analysis/recognition/DL_group_all_all-paper_res.json",
    #     "/home/vitor/git/capacitive-data_analysis/recognition/DL_group_wrist_cap-paper_res.json",
    #     "/home/vitor/git/capacitive-data_analysis/recognition/DL_group_wrist_acc-paper_res.json",
    #     "/home/vitor/git/capacitive-data_analysis/recognition/DL_group_wrist_all-paper_res.json"
    # ]
    # for baseline_path in lst_group_baselines:
    #     nm_out = "./" + os.path.split(baseline_path)[-1].split(".")[0] + "_CFM.png"
    #     plot_joined_cfm(baseline_path, nm_out)
    # plot_performance_across_combinations(lst_group_baselines, "GROUP_USER_EASY_CLASSES")
    #
    #
    # lst_group_hard_baselines = [
    #     "/home/vitor/git/capacitive-data_analysis/recognition/DL_group_calf_acc-paper_hard_res.json",
    #     "/home/vitor/git/capacitive-data_analysis/recognition/DL_group_all_all-paper_hard_res.json",
    #     "/home/vitor/git/capacitive-data_analysis/recognition/DL_group_wrist_cap-paper_hard_res.json",
    #     "/home/vitor/git/capacitive-data_analysis/recognition/DL_group_wrist_acc-paper_hard_res.json",
    #     "/home/vitor/git/capacitive-data_analysis/recognition/DL_group_wrist_all-paper_hard_res.json"
    # ]
    # for baseline_path in lst_group_hard_baselines:
    #     nm_out = "./" + os.path.split(baseline_path)[-1].split(".")[0] + "_CFM.png"
    #     try:
    #         plot_joined_cfm(baseline_path, nm_out)
    #     except:
    #         print("ERROR", baseline_path)
    #
    # plot_performance_across_combinations(lst_group_hard_baselines, "GROUP_USER_HARD_CLASSES")
    # exit(0)
    #
    #
    #
    # lst_single_baselines = [
    #     "/home/vitor/git/capacitive-data_analysis/recognition/DL_calf_acc-paper_res.json",
    #     "/home/vitor/git/capacitive-data_analysis/recognition/DL_all_base-paper_res.json",
    #     "/home/vitor/git/capacitive-data_analysis/recognition/DL_wrist_Charge-paper_res.json",
    #     "/home/vitor/git/capacitive-data_analysis/recognition/DL_wrist_acc-paper_res.json",
    #     "/home/vitor/git/capacitive-data_analysis/recognition/DL_wrist-paper_res.json"
    # ]
    #
    # for baseline_path in lst_single_baselines:
    #     nm_out = "./" + os.path.split(baseline_path)[-1].split(".")[0] + "_CFM.png"
    #     try:
    #         plot_joined_cfm(baseline_path, nm_out)
    #     except:
    #         print("ERROR", baseline_path)
    #
    # # plot_performance_across_combinations(lst_single_baselines,"SINGLE_USER_EASY_CLASSES")
    #
    # lst_hard_baselines = [
    #     "/home/vitor/git/capacitive-data_analysis/recognition/DL_calf_acc-paper_hard_res.json",
    #     "/home/vitor/git/capacitive-data_analysis/recognition/DL_all_base-paper_hard_res.json",
    #     "/home/vitor/git/capacitive-data_analysis/recognition/DL_wrist_Charge-paper_hard_res.json",
    #     "/home/vitor/git/capacitive-data_analysis/recognition/DL_wrist_acc-paper_hard_res.json",
    #     "/home/vitor/git/capacitive-data_analysis/recognition/DL_wrist-paper_hard_res.json"
    # ]
    # plot_performance_across_combinations(lst_hard_baselines, "SINGLE_USER_HARD_CLASSES")
    # for baseline_path in lst_hard_baselines:
    #     nm_out = "./" + os.path.split(baseline_path)[-1].split(".")[0] + "_CFM.png"
    #     try:
    #         plot_joined_cfm(baseline_path, nm_out)
    #     except:
    #         print("ERROR", baseline_path)


    # plot_performance_across_combinations("./CL_wrist*paper_res.json", "Z_WRIST_SINGLE_EASY")
    # plot_performance_across_combinations("./CL_wrist*paper_hard_res.json", "Z_WRIST_SINGLE_HARD")
    #
    #
    # plot_performance_across_combinations("./CL_all*paper_res.json", "Z_ALL_SINGLE_EASY")
    # plot_performance_across_combinations("./CL_all*paper_hard_res.json", "Z_ALL_SINGLE_HARD")



    # plot_performance_across_combinations("./CL**paper_res.json", "Z_MIX_SINGLE_EASY")
    # plot_performance_across_combinations("./CL_*_*paper_hard_res.json", "Z_MIX_SINGLE_HARD")
    #
    # plot_performance_across_combinations("./CL_wrist_*paper_res.json", "Z_WRIST_SINGLE_EASY")
    # plot_performance_across_combinations("./CL_wrist_*paper_hard_res.json", "Z_WRIST_SINGLE_HARD")
    # plot_performance_across_combinations("./CL_all_*paper_res.json", "Z_ALL_SINGLE_EASY")
    # plot_performance_across_combinations("./CL_all_*paper_hard_res.json", "Z_ALL_SINGLE_HARD")
    #
    # plot_performance_across_combinations("./CL_group_wrist_*paper_res.json", "Z_WRIST_GROUP_EASY")
    # plot_performance_across_combinations("./CL_group_wrist_*paper_hard_res.json", "Z_WRIST_GROUP_HARD")
    # plot_performance_across_combinations("./CL_group_all_*paper_res.json", "Z_ALL_GROUP_EASY")
    # plot_performance_across_combinations("./CL_group_all_*paper_hard_res.json", "Z_ALL_GROUP_HARD")



    exit(0)


    # plot_all_sessions()
    # exit(0)

    for path_res in glob.glob("./*.json"):
        try:
            plot_performance(path_res)
        except:
            print("EXCEPTION", path_res)

    # plot_label_predictions()
    #