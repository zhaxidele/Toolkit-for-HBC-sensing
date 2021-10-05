from __future__ import print_function


import tensorflow as tf
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.2,
                              visible_device_list="1"),
)
sess = tf.Session(config=config)
import keras
keras.backend.set_session(sess)


from test import run_DL_test_new





def baselines_single1(win_size=50):
    str_pref = "" if win_size == 50 else str(win_size)
    for labl_typ in ["paper", "paper_hard"]:
            run_DL_test_new(str_pref+"DL_wrist_Charge-" + labl_typ, labl_typ,
                                 lst_feats=["<minmax>Right_Wrist|Charge"],
                                 win_size=win_size, win_step=10, strategy='all', do_plot=False, pairwise=False)
            run_DL_test_new(str_pref+"DL_wrist_acc-" + labl_typ, labl_typ,
                                 lst_feats=["<norm>Right_Wrist|acc"],
                                 win_size=win_size, win_step=10, strategy='all', do_plot=False, pairwise=False)

    # baselines_single1()

def baselines_single2(win_size=50):
    str_pref = "" if win_size == 50 else str(win_size)
    for labl_typ in ["paper", "paper_hard"]:
            run_DL_test_new(str_pref+"DL_calf_acc-" + labl_typ, labl_typ,
                                 lst_feats=["<norm>Left_Calf|acc"],
                                 win_size=win_size, win_step=10, strategy='all', do_plot=False, pairwise=False)
            run_DL_test_new(str_pref+"DL_wrist-" + labl_typ, labl_typ,
                                 lst_feats=["<minmax>Right_Wrist|Charge", "<norm>Right_Wrist|acc"],
                                 win_size=win_size, win_step=10, strategy='all', do_plot=False, pairwise=False)
            run_DL_test_new(str_pref+"DL_all_base-" + labl_typ, labl_typ,
                                 lst_feats=["<minmax>Right_Wrist|Charge", "<norm>Right_Wrist|acc",
                                            "<norm>Left_Calf|acc"],
                                 win_size=win_size, win_step=10, strategy='all', do_plot=False, pairwise=False)

    # baselines_single2()





def baselines_group1(win_size=50):
    str_pref = "" if win_size == 50 else str(win_size)
    for labl_typ in ["paper", "paper_hard"]:
            run_DL_test_new(str_pref+"DL_group_wrist_all-" + labl_typ, labl_typ,
                                 lst_feats=["<P1><minmax>Right_Wrist|Charge", "<P1><norm>Right_Wrist|acc",
                                            "<P2><minmax>Right_Wrist|Charge", "<P2><norm>Right_Wrist|acc"],
                                 win_size=win_size, win_step=10, strategy='all', do_plot=False, pairwise=True)
            run_DL_test_new(str_pref+"DL_group_wrist_cap-" + labl_typ, labl_typ,
                                 lst_feats=["<P1><minmax>Right_Wrist|Charge", "<P2><minmax>Right_Wrist|Charge"],
                                 win_size=win_size, win_step=10, strategy='all', do_plot=False, pairwise=True)
    # baselines_group1()


def baselines_group2(win_size=50):
    str_pref = "" if win_size == 50 else str(win_size)
    for labl_typ in ["paper", "paper_hard"]:
            run_DL_test_new(str_pref+"DL_group_wrist_acc-" + labl_typ, labl_typ,
                                 lst_feats=["<P1><norm>Right_Wrist|acc", "<P2><norm>Right_Wrist|acc"],
                                 win_size=win_size, win_step=10, strategy='all', do_plot=False, pairwise=True)

            run_DL_test_new(str_pref + "DL_group_calf_acc-" + labl_typ, labl_typ,
                                 lst_feats=["<P1><norm>Left_Calf|acc", "<P2><norm>Left_Calf|acc"],
                                 win_size=win_size, win_step=10, strategy='all', do_plot=False, pairwise=True)
            run_DL_test_new(str_pref + "DL_group_all_all-" + labl_typ, labl_typ,
                            lst_feats=["<P1><minmax>Right_Wrist|Charge", "<P1><norm>Right_Wrist|acc",
                                       "<P2><minmax>Right_Wrist|Charge", "<P2><norm>Right_Wrist|acc",
                                       "<P1><norm>Left_Calf|acc", "<P2><norm>Left_Calf|acc"],
                            win_size=win_size, win_step=10, strategy='all', do_plot=False, pairwise=True)

    # baselines_group2()


# baselines_single1()
# baselines_single2()
# baselines_group1()
baselines_group2()











#
# def run_baselines_group_1():
#     run_DL_test_new("W_base_DL_group_all_wrist",
#                     ["<P1><diff>Right_Wrist|Charge", "<P1><norm>Right_Wrist|acc",
#                      "<P2><diff>Right_Wrist|Charge", "<P2><norm>Right_Wrist|acc"],
#                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=True)
#
#     run_DL_test_new("W_base_DL_group_wrist_Charge",
#                     ["<P1><diff>Right_Wrist|Charge", "<P2><diff>Right_Wrist|Charge"],
#                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=True)
#
#     run_DL_test_new("W_base_DL_group_calf_acc",
#                     ["<P1><norm>Left_Calf|acc", "<P2><norm>Left_Calf|acc"],
#                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=True)
#
# # run_baselines_group_1()
#
# def run_baselines_group_2():
#     run_DL_test_new("W_base_DL_group_wrist_acc",
#                     ["<P1><norm>Right_Wrist|acc", "<P2><norm>Right_Wrist|acc"],
#                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=True)
#
#     run_DL_test_new("W_base_DL_group_all_sensors",
#                     ["<P1><diff>Right_Wrist|Charge", "<P1><norm>Right_Wrist|acc", "<P1><norm>Left_Calf|acc",
#                      "<P2><diff>Right_Wrist|Charge", "<P2><norm>Right_Wrist|acc", "<P2><norm>Left_Calf|acc"],
#                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=True)
#
# # run_baselines_group_2()
#
#
# def run_baselines_single_1():
#     run_DL_test_new("W_base_DL_all_wrist",
#                     ["<diff>Right_Wrist|Charge", "<norm>Right_Wrist|acc"],
#                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=False)
#
#     run_DL_test_new("W_base_DL_wrist_Charge",
#                     ["<diff>Right_Wrist|Charge"],
#                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=False)
#
#     run_DL_test_new("W_base_DL_calf_acc",
#                     ["<norm>Left_Calf|acc"],
#                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=False)
#
# # run_baselines_single_1()
#
# def run_baselines_single_2():
#     run_DL_test_new("W_base_DL_wrist_acc",
#                     ["<norm>Right_Wrist|acc"],
#                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=False)
#
#     run_DL_test_new("W_base_DL_all_sensors",
#                     ["<diff>Right_Wrist|Charge", "<norm>Right_Wrist|acc", "<norm>Left_Calf|acc"],
#                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=False)

# run_baselines_single_2()

# # run_baselines_single_1()
#
#
# def run_corr_single_1():
#     run_DL_test_new("n_corr_DL_wrist_relevant",
#                     ["<diff>Right_Wrist|Charge", "<norm>Right_Wrist|acc",
#                      "<corr_raw_relevant>(<norm>Right_Wrist|acc,<raw>Right_Wrist|Charge)"],
#                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=False)
#
#     run_DL_test_new("n_corr_DL_wrist_mean",
#                     ["<diff>Right_Wrist|Charge", "<norm>Right_Wrist|acc",
#                      "<corr_mean>(<norm>Right_Wrist|acc,<raw>Right_Wrist|Charge)"],
#                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=False)
#
#     run_DL_test_new("n_corr_DL_wrist_bands",
#                     ["<diff>Right_Wrist|Charge", "<norm>Right_Wrist|acc",
#                      "<corr_bands>(<norm>Right_Wrist|acc,<raw>Right_Wrist|Charge)"],
#                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=False)
#
# # run_corr_single_1()
#
#
# def run_corr_angl_single_1():
#     run_DL_test_new("n_corr_DL_wrist_relevant_angle",
#                     ["<diff>Right_Wrist|Charge", "<norm>Right_Wrist|acc",
#                      "<corr_raw_relevant>(<norm>Right_Wrist|acc,<raw>Right_Wrist|Charge)",
#                      "<angl_raw_relevant>(<norm>Right_Wrist|acc,<raw>Right_Wrist|Charge)"],
#                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=False)
#
#
#     run_DL_test_new("n_corr_DL_wrist_mean_angle",
#                     ["<diff>Right_Wrist|Charge", "<norm>Right_Wrist|acc",
#                      "<corr_mean>(<norm>Right_Wrist|acc,<raw>Right_Wrist|Charge)"
#                      "<corr_mean>(<norm>Right_Wrist|acc,<raw>Right_Wrist|Charge)",
#                      "<angl_mean>(<norm>Right_Wrist|acc,<raw>Right_Wrist|Charge)"
#                      "<angl_mean>(<norm>Right_Wrist|acc,<raw>Right_Wrist|Charge)"
#                      ],
#                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=False)
#
#     run_DL_test_new("n_corr_DL_wrist_bands_angle",
#                     ["<diff>Right_Wrist|Charge", "<norm>Right_Wrist|acc",
#                      "<corr_bands>(<norm>Right_Wrist|acc,<raw>Right_Wrist|Charge)",
#                      "<angl_bands>(<norm>Right_Wrist|acc,<raw>Right_Wrist|Charge)"],
#                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=False)
#
# # run_corr_angl_single_1()
#
#
#
#
#
#
# # run_baselines_group_1()
#
#
#
# def run_corr_relevant_group_1():
#     run_DL_test_new("n_base_DL_group_just_corr_user_acc_cap",
#                     ["<corr_raw_relevant>(<P1><norm>Right_Wrist|acc,<P2><raw>Right_Wrist|Charge)"],
#                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=True)
#
#     run_DL_test_new("n_base_DL_group_just_corr_user_caps",
#                     ["<corr_raw_relevant>(<P1><raw>Right_Wrist|Charge,<P2><raw>Right_Wrist|Charge)"],
#                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=True)
#
#     run_DL_test_new("n_base_DL_group_corr_acc_charge_and_charge_charge",
#                     ["<corr_raw_relevant>(<P1><norm>Right_Wrist|acc,<P2><raw>Right_Wrist|Charge)",
#                      "<corr_raw_relevant>(<P1><raw>Right_Wrist|Charge,<P2><raw>Right_Wrist|Charge)"],
#                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=True)
#
#     run_DL_test_new("n_base_DL_group_corr_all_wrist_cross",
#                     ["<corr_raw_relevant>(<P1><norm>Right_Wrist|acc,<P2><raw>Right_Wrist|Charge)",
#                      "<corr_raw_relevant>(<P1><raw>Right_Wrist|Charge,<P2><raw>Right_Wrist|Charge)",
#                      "<corr_raw_relevant>(<P1><norm>Right_Wrist|acc,<P2><raw>Right_Wrist|acc)"],
#                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=True)
#
#     run_DL_test_new("n_base_DL_group_corr_all_wrist_cross_plus_intra",
#                     ["<corr_raw_relevant>(<P1><norm>Right_Wrist|acc,<P2><raw>Right_Wrist|Charge)",
#                      "<corr_raw_relevant>(<P1><raw>Right_Wrist|Charge,<P2><raw>Right_Wrist|Charge)",
#                      "<corr_raw_relevant>(<P1><norm>Right_Wrist|acc,<P2><raw>Right_Wrist|acc)",
#                      "<corr_raw_relevant>(<P1><norm>Right_Wrist|acc,<P1><raw>Right_Wrist|Charge)",
#                      "<corr_raw_relevant>(<P2><norm>Right_Wrist|acc,<P2><raw>Right_Wrist|Charge)"],
#                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=True)
#
# # run_corr_relevant_group_1()
#
#
#
# def run_corr_mean_group_1():
#     run_DL_test_new("n_base_DL_group_just_corr_user_acc_cap_mean",
#                     ["<corr_mean>(<P1><norm>Right_Wrist|acc,<P2><raw>Right_Wrist|Charge)"],
#                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=True)
#
#     run_DL_test_new("n_base_DL_group_just_corr_user_caps_mean",
#                     ["<corr_mean>(<P1><raw>Right_Wrist|Charge,<P2><raw>Right_Wrist|Charge)"],
#                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=True)
#
#     run_DL_test_new("n_base_DL_group_corr_acc_charge_and_charge_charge_mean",
#                     ["<corr_mean>(<P1><norm>Right_Wrist|acc,<P2><raw>Right_Wrist|Charge)",
#                      "<corr_mean>(<P1><raw>Right_Wrist|Charge,<P2><raw>Right_Wrist|Charge)"],
#                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=True)
#
#     run_DL_test_new("n_base_DL_group_corr_all_wrist_cross_mean",
#                     ["<corr_mean>(<P1><norm>Right_Wrist|acc,<P2><raw>Right_Wrist|Charge)",
#                      "<corr_mean>(<P1><raw>Right_Wrist|Charge,<P2><raw>Right_Wrist|Charge)",
#                      "<corr_mean>(<P1><norm>Right_Wrist|acc,<P2><raw>Right_Wrist|acc)"],
#                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=True)
#
#     run_DL_test_new("n_base_DL_group_corr_all_wrist_cross_plus_intra_mean",
#                     ["<corr_mean>(<P1><norm>Right_Wrist|acc,<P2><raw>Right_Wrist|Charge)",
#                      "<corr_mean>(<P1><raw>Right_Wrist|Charge,<P2><raw>Right_Wrist|Charge)",
#                      "<corr_mean>(<P1><norm>Right_Wrist|acc,<P2><raw>Right_Wrist|acc)",
#                      "<corr_mean>(<P1><norm>Right_Wrist|acc,<P1><raw>Right_Wrist|Charge)",
#                      "<corr_mean>(<P2><norm>Right_Wrist|acc,<P2><raw>Right_Wrist|Charge)"],
#                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=True)

# run_corr_mean_group_1()









# from test import run_DL_test
#
#
# def do_DL_baselines_1():
#
#     run_DL_test("DL_baseline_cap", win_size=50, win_step=10, strategy='all',
#                 lst_placement=["Right_Wrist"],
#                 lst_type=["Charge"], use_coh=False, use_corr=False, use_bands=False)
#
# def do_DL_baselines_2():
#     run_DL_test("DL_baseline_acc", win_size=50, win_step=10, strategy='all',
#                 lst_placement=["Right_Wrist"],
#                 lst_type=["acc"], use_coh=False, use_corr=False, use_bands=False)
#
# def do_DL_baselines_3():
#     run_DL_test("DL_baseline_acc_cap", win_size=50, win_step=10, strategy='all',
#                 lst_placement=["Right_Wrist"],
#                 lst_type=["Charge", "acc"], use_coh=False, use_corr=False, use_bands=False)
#
#
#
# def do_DL_just_some_spec_feats_1():
#     lst_type = ["Charge", "acc"]
#     run_DL_test("DL_add_corr", win_size=50, win_step=10, strategy='all',
#                 lst_placement=["Right_Wrist"],
#                 lst_type=lst_type, use_coh=False, use_corr=True, use_bands=False)
#
# def do_DL_just_some_spec_feats_2():
#     lst_type = ["Charge", "acc"]
#     run_DL_test("DL_add_coh", win_size=50, win_step=10, strategy='all',
#                 lst_placement=["Right_Wrist"],
#                 lst_type=lst_type, use_coh=True, use_corr=False, use_bands=False)
#
# def do_DL_just_some_spec_feats_3():
#     lst_type = ["Charge", "acc"]
#     run_DL_test("DL_add_both", win_size=50, win_step=10, strategy='all',
#                 lst_placement=["Right_Wrist"],
#                 lst_type=lst_type, use_coh=True, use_corr=True, use_bands=False)
#
#
#
# def do_DL_bands_just_some_spec_feats_1():
#     lst_type = ["Charge", "acc"]
#     run_DL_test("DL_bands_add_corr", win_size=50, win_step=10, strategy='all',
#                 lst_placement=["Right_Wrist"],
#                 lst_type=lst_type, use_coh=False, use_corr=True, use_bands=True)
#
# def do_DL_bands_just_some_spec_feats_2():
#     lst_type = ["Charge", "acc"]
#     run_DL_test("DL__bands_add_coh", win_size=50, win_step=10, strategy='all',
#                 lst_placement=["Right_Wrist"],
#                 lst_type=lst_type, use_coh=True, use_corr=False, use_bands=True)
#
# def do_DL_bands_just_some_spec_feats_3():
#     lst_type = ["Charge", "acc"]
#     run_DL_test("DL__bands_add_both", win_size=50, win_step=10, strategy='all',
#                 lst_placement=["Right_Wrist"],
#                 lst_type=lst_type, use_coh=True, use_corr=True, use_bands=True)



# do_DL_bands_just_some_spec_feats_1()
# do_DL_bands_just_some_spec_feats_2()
# do_DL_bands_just_some_spec_feats_3()
#
#
# do_DL_baselines_2()
# do_DL_baselines_2()
# do_DL_baselines_3()
# #
# do_DL_just_some_spec_feats_1()
# do_DL_just_some_spec_feats_2()
# do_DL_just_some_spec_feats_3()
