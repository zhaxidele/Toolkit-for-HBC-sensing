import numpy as np

from sklearn.metrics import f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, minmax_scale

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix

from util import get_sample_ws_from_multiple_y_per_seg, convert_lbls_from_one_per_ts_to_one_per_ts
from util import smooth_predictions_using_soft_voting, smooth_predictions_using_hard_voting, get_cl_from_probabilities


from data import gen_leave_one_group_out_new_version, gen_leave_pairs_of_one_group_out_new_version
from preprocess import calculate_window_features, squash_all_bands, squash_dividing_bands, sneaky_squash


def run_classic_test_new(str_test_name, lbl_type, lst_feats, lst_fn_for_feats, win_size, win_step, strategy, do_plot=False, pairwise=False):
  map_res = { "features" : [str(f) for f in lst_feats] }


  if pairwise:
      generator = gen_leave_pairs_of_one_group_out_new_version(lst_feats, lbl_type, win_size=win_size, win_step=win_step, strategy=strategy)
  else:
     generator = gen_leave_one_group_out_new_version(lst_feats, lbl_type, win_size=win_size, win_step=win_step, strategy=strategy)


  for test_grp_id, lst_x_train, lst_y_train, lst_x_test, lst_y_test, lst_exp_nms in generator:
    print "----------------", "TEST", test_grp_id, len(lst_x_train), len(lst_y_train), len(lst_x_test), len(lst_y_test)

    def do_all_prep(lst_x_train, lst_x_test, lst_fn_for_feats):
      def extract_win_feats(lst_X, lst_fn):
        # For each day.
        lst_feat_for_day = []
        for lst_day_X in lst_X:
            lst_feats_data = []
            for arr_snsr, fn_snsr in zip(lst_day_X, lst_fn):
                lst_feats_data.append(fn_snsr(arr_snsr))
            feats_day = np.concatenate(lst_feats_data, axis=-1)
            lst_feat_for_day.append(feats_day)

        return lst_feat_for_day

      lst_x_train = extract_win_feats(lst_x_train, lst_fn_for_feats)
      lst_x_test = extract_win_feats(lst_x_test, lst_fn_for_feats)

      X_train = np.vstack(lst_x_train)

      prep = StandardScaler().fit(X_train)
      X_train_prepped = prep.transform(X_train)
      lst_x_test_prepped = [prep.transform(x) for x in lst_x_test]

      return X_train_prepped, lst_x_test_prepped

    X_train, lst_x_test = do_all_prep(lst_x_train, lst_x_test, lst_fn_for_feats)

    y_train = np.vstack(lst_y_train)
    arr_cl, cl_cnts = np.unique(y_train, return_counts=True)
    all_classes = list(arr_cl)
    print("CLASSES", all_classes, cl_cnts, cl_cnts / float(np.sum(cl_cnts)))

    sample_ws = get_sample_ws_from_multiple_y_per_seg(y_train, all_classes)
    Y_train = convert_lbls_from_one_per_ts_to_one_per_ts(y_train)


    # model = RandomForestClassifier().fit(X_train, Y_train, sample_weight = sample_ws)
    model = LogisticRegression().fit(X_train, Y_train, sample_weight = sample_ws)

    y_train_pred = model.predict(X_train)
    print "SCORE TRAINING"
    f1 = f1_score(Y_train, y_train_pred, all_classes, average="macro")
    print "F1:", f1
    print all_classes
    cfm = confusion_matrix(Y_train, y_train_pred, all_classes)
    print cfm


    map_res[test_grp_id] = []
    for X_test, y_test, exp_nm in zip(lst_x_test, lst_y_test, lst_exp_nms):
        y_gt = convert_lbls_from_one_per_ts_to_one_per_ts(y_test)

        cl_probs = model.predict_proba(X_test)
        cl_probs = smooth_predictions_using_soft_voting(cl_probs, window_size=6)
        # cl_probs = smooth_predictions_using_hard_voting(cl_probs, window_size=20, all_classes=model.classes_)

        y_pred = get_cl_from_probabilities(cl_probs, model.classes_)
        # y_pred = model.predict(X_test)

        if do_plot:
            from plot import plot_label_predictions_debug
            plot_label_predictions_debug(y_gt, y_pred, cl_probs, model.classes_)


        f1 = f1_score(y_gt, y_pred, all_classes, average="macro")
        print exp_nm, "F1:", f1
        print all_classes
        cfm = confusion_matrix(y_gt, y_pred, all_classes)
        print cfm

        map_res[test_grp_id].append( [f1,  all_classes, [[c for c in lin] for lin in cfm], exp_nm] )

        import json
        json.dump(map_res, open("./" +str_test_name+ "_res.json", "w"))


def run_DL_test_new(str_test_name, lbl_type, lst_feats, win_size, win_step, strategy, do_plot=False, pairwise=False):
  map_res = { "features" : [str(f) for f in lst_feats]  }

  if pairwise:
      generator = gen_leave_pairs_of_one_group_out_new_version(lst_feats, lbl_type, win_size = win_size , win_step = win_step,
                                                      strategy = strategy)
  else:
      generator = gen_leave_one_group_out_new_version(lst_feats, lbl_type, win_size = win_size , win_step = win_step,
                                                      strategy = strategy)

  for test_grp_id, lst_x_train, lst_y_train, lst_x_test, lst_y_test, lst_exp_nms in generator:
    print "----------------", "TEST", test_grp_id, len(lst_x_train), len(lst_y_train), len(lst_x_test), len(lst_y_test)

    y_train = np.vstack(lst_y_train)
    arr_cl, cl_cnts = np.unique(y_train, return_counts=True)
    all_classes = list(arr_cl)
    print("CLASSES", all_classes, cl_cnts, cl_cnts / float(np.sum(cl_cnts)))

    X_train = []
    for lst_day_features in lst_x_train:
        X_train.append(np.concatenate(lst_day_features, axis=-1))
    X_train = np.vstack(X_train)
    lst_x_test = [np.concatenate(lst_day_features, axis=-1) for lst_day_features in lst_x_test]

    from model import build_tcn, build_rnn
    model = build_tcn( (X_train.shape[-2], X_train.shape[-1]) , len(all_classes))
    # model = build_rnn((X_train.shape[-2], X_train.shape[-1]), len(all_classes))

    # Scaling prep.
    def learn_prep(X, c_i):
      X_f = X[:,:, c_i].reshape( (-1,1) )
      prep = StandardScaler().fit(X_f)
      return prep

    def do_all_prep(X_train_in, lst_X_test_in):

      X_train = X_train_in
      X_train_prepped = np.zeros(X_train.shape)
      lst_X_test_prepped = [np.zeros(x.shape) for x in lst_X_test_in]

      for c_i in range(X_train.shape[2]):
        prep_ci = learn_prep(X_train, c_i)
        X_train_prepped[:,:,c_i] = prep_ci.transform(X_train[:,:,c_i].reshape((-1,1)) ).reshape(
          (X_train_prepped.shape[0],X_train_prepped.shape[1])
        )


        for X_test, X_test_prepped in zip(lst_x_test, lst_X_test_prepped):
            X_test_prepped[:,:,c_i]  = prep_ci.transform(X_test[:,:,c_i].reshape((-1,1)) ).reshape(
              (X_test_prepped.shape[0], X_test_prepped.shape[1])
            )

      return X_train_prepped, lst_X_test_prepped

    X_train, lst_x_test = do_all_prep(X_train, lst_x_test)

    from train import train_classification
    from util import make_one_label_per_timestep_one_hot, get_cl_for_timestep_one_hot

    Y_train = make_one_label_per_timestep_one_hot(y_train, all_classes)

    model = train_classification(model, X_train, Y_train,
                  "test=" + str_test_name + "|gr=" + test_grp_id , "./", "./",
                  batch_size = 16, epochs = 300, patience = 15, all_classes = all_classes)

    map_res[test_grp_id] = []
    for X_test, y_test, exp_nm in zip(lst_x_test, lst_y_test, lst_exp_nms):
        y_gt = convert_lbls_from_one_per_ts_to_one_per_ts(y_test)


        cl_probs =  np.sum( model.predict(X_test), axis=1)
        cl_probs = smooth_predictions_using_soft_voting(cl_probs, window_size=6)
        y_pred = get_cl_from_probabilities(cl_probs, all_classes)
        # y_pred = get_cl_for_timestep_one_hot(model.predict(X_test), all_classes)

        if do_plot:
            from plot import plot_label_predictions_debug
            plot_label_predictions_debug(y_gt, y_pred, cl_probs, all_classes)


        f1 = f1_score(y_gt, y_pred, all_classes, average="macro")
        print exp_nm, "F1:", f1
        print all_classes
        cfm = confusion_matrix(y_gt, y_pred, all_classes)
        print cfm

        map_res[test_grp_id].append([f1, all_classes, [[c for c in lin] for lin in cfm], exp_nm])

    import json
    json.dump(map_res, open("./" + str_test_name + "_res.json", "w"))







def do_missing_baselines(win_size=50):
    str_pref = "" if win_size == 50 else str(win_size)
    for labl_typ in ["paper", "paper_hard"]:
        run_classic_test_new(str_pref + "CL_acc_all-" + labl_typ, labl_typ,
                             lst_feats=["<norm>Right_Wrist|acc",
                                        "<norm>Left_Calf|acc"],
                             lst_fn_for_feats=[calculate_window_features, calculate_window_features,
                                               calculate_window_features,
                                               sneaky_squash, sneaky_squash, sneaky_squash],
                             win_size=win_size, win_step=10, strategy='all', do_plot=False, pairwise=False)

        run_classic_test_new(str_pref + "CL_group_acc_all-" + labl_typ, labl_typ,
                             lst_feats=["<P1><norm>Right_Wrist|acc", "<P2><norm>Right_Wrist|acc",
                                        "<P1><norm>Left_Calf|acc", "<P2><norm>Left_Calf|acc"],
                             lst_fn_for_feats=[calculate_window_features, calculate_window_features,
                                               calculate_window_features, calculate_window_features,
                                               calculate_window_features, calculate_window_features,
                                               squash_dividing_bands, squash_dividing_bands],
                             win_size=win_size, win_step=10, strategy='all', do_plot=False, pairwise=True)


def do_classic_baselines(win_size=50):
    import gc

    str_pref = "" if win_size == 50 else str(win_size)


    def baselines_single():
        for labl_typ in ["paper", "paper_hard"]:
            run_classic_test_new(str_pref+"CL_wrist_Charge-" + labl_typ, labl_typ,
                                 lst_feats=["<minmax>Right_Wrist|Charge"],
                                 lst_fn_for_feats=[calculate_window_features, calculate_window_features,
                                                   calculate_window_features,
                                                   sneaky_squash, sneaky_squash, sneaky_squash],
                                 win_size=win_size, win_step=10, strategy='all', do_plot=False, pairwise=False)
            run_classic_test_new(str_pref+"CL_wrist_acc-" + labl_typ, labl_typ,
                                 lst_feats=["<norm>Right_Wrist|acc"],
                                 lst_fn_for_feats=[calculate_window_features, calculate_window_features,
                                                   calculate_window_features,
                                                   sneaky_squash, sneaky_squash, sneaky_squash],
                                 win_size=win_size, win_step=10, strategy='all', do_plot=False, pairwise=False)
            run_classic_test_new(str_pref+"CL_calf_acc-" + labl_typ, labl_typ,
                                 lst_feats=["<norm>Left_Calf|acc"],
                                 lst_fn_for_feats=[calculate_window_features, calculate_window_features,
                                                   calculate_window_features,
                                                   sneaky_squash, sneaky_squash, sneaky_squash],
                                 win_size=win_size, win_step=10, strategy='all', do_plot=False, pairwise=False)
            run_classic_test_new(str_pref+"CL_wrist-" + labl_typ, labl_typ,
                                 lst_feats=["<minmax>Right_Wrist|Charge", "<norm>Right_Wrist|acc"],
                                 lst_fn_for_feats=[calculate_window_features, calculate_window_features,
                                                   calculate_window_features,
                                                   sneaky_squash, sneaky_squash, sneaky_squash],
                                 win_size=win_size, win_step=10, strategy='all', do_plot=False, pairwise=False)
            run_classic_test_new(str_pref+"CL_all_base-" + labl_typ, labl_typ,
                                 lst_feats=["<minmax>Right_Wrist|Charge", "<norm>Right_Wrist|acc",
                                            "<norm>Left_Calf|acc"],
                                 lst_fn_for_feats=[calculate_window_features, calculate_window_features,
                                                   calculate_window_features,
                                                   sneaky_squash, sneaky_squash, sneaky_squash],
                                 win_size=win_size, win_step=10, strategy='all', do_plot=False, pairwise=False)

    try:
        baselines_single()
        gc.collect()
    except:
        print("ERROR SINGLE")

    def baselines_group():
        for labl_typ in ["paper", "paper_hard"]:
            run_classic_test_new(str_pref+"CL_group_wrist_all-" + labl_typ, labl_typ,
                                 lst_feats=["<P1><minmax>Right_Wrist|Charge", "<P1><norm>Right_Wrist|acc",
                                            "<P2><minmax>Right_Wrist|Charge", "<P2><norm>Right_Wrist|acc"],
                                 lst_fn_for_feats=[calculate_window_features, calculate_window_features,
                                                   calculate_window_features, calculate_window_features,
                                                   squash_dividing_bands, squash_dividing_bands,
                                                   squash_dividing_bands, squash_dividing_bands],
                                 win_size=win_size, win_step=10, strategy='all', do_plot=False, pairwise=True)
            run_classic_test_new(str_pref+"CL_group_wrist_cap-" + labl_typ, labl_typ,
                                 lst_feats=["<P1><minmax>Right_Wrist|Charge", "<P2><minmax>Right_Wrist|Charge"],
                                 lst_fn_for_feats=[calculate_window_features, calculate_window_features,
                                                   calculate_window_features, calculate_window_features,
                                                   squash_dividing_bands, squash_dividing_bands,
                                                   squash_dividing_bands, squash_dividing_bands],
                                 win_size=win_size, win_step=10, strategy='all', do_plot=False, pairwise=True)
            run_classic_test_new(str_pref+"CL_group_wrist_acc-" + labl_typ, labl_typ,
                                 lst_feats=["<P1><norm>Right_Wrist|acc", "<P2><norm>Right_Wrist|acc"],
                                 lst_fn_for_feats=[calculate_window_features, calculate_window_features,
                                                   calculate_window_features, calculate_window_features,
                                                   squash_dividing_bands, squash_dividing_bands,
                                                   squash_dividing_bands, squash_dividing_bands],
                                 win_size=win_size, win_step=10, strategy='all', do_plot=False, pairwise=True)

            run_classic_test_new(str_pref + "CL_group_calf_acc-" + labl_typ, labl_typ,
                                 lst_feats=["<P1><norm>Left_Calf|acc", "<P2><norm>Left_Calf|acc"],
                                 lst_fn_for_feats=[calculate_window_features, calculate_window_features,
                                                   calculate_window_features, calculate_window_features,
                                                   calculate_window_features, calculate_window_features,
                                                   squash_dividing_bands, squash_dividing_bands],
                                 win_size=win_size, win_step=10, strategy='all', do_plot=False, pairwise=True)

            run_classic_test_new(str_pref + "CL_group_all_all-" + labl_typ, labl_typ,
                                 lst_feats=["<P1><minmax>Right_Wrist|Charge", "<P1><norm>Right_Wrist|acc",
                                            "<P2><minmax>Right_Wrist|Charge", "<P2><norm>Right_Wrist|acc",
                                            "<P1><norm>Left_Calf|acc", "<P2><norm>Left_Calf|acc"],
                                 lst_fn_for_feats=[calculate_window_features, calculate_window_features,
                                                   calculate_window_features, calculate_window_features,
                                                   calculate_window_features, calculate_window_features,
                                                   squash_dividing_bands, squash_dividing_bands],
                                 win_size=win_size, win_step=10, strategy='all', do_plot=False, pairwise=True)




    try:
        baselines_group()
        gc.collect()
    except:
        print("ERROR GROUP")


def do_specs_single():
    import gc
    def spec_single():
        for nm_fn, squash_fn in zip(["sneaky", "mean"], [sneaky_squash, squash_all_bands]):
            for labl_typ in ["paper", "paper_hard"]:
                run_classic_test_new("CL_wrist_none_coh-" + nm_fn + "-" + labl_typ, labl_typ,
                                     lst_feats=[
                                                "<coh_raw_relevant>(<norm>Right_Wrist|acc,<raw>Right_Wrist|Charge)"],
                                     lst_fn_for_feats=[squash_fn, squash_fn, squash_fn],
                                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=False)

                run_classic_test_new("CL_wrist_none_angl-" + nm_fn + "-" + labl_typ, labl_typ,
                                     lst_feats=["<angl_raw_relevant>(<norm>Right_Wrist|acc,<raw>Right_Wrist|Charge)"],
                                     lst_fn_for_feats=[squash_fn, squash_fn, squash_fn],
                                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=False)

                run_classic_test_new("CL_wrist_none_coh-angl-" + nm_fn + "-" + labl_typ, labl_typ,
                                     lst_feats=["<coh_raw_relevant>(<norm>Right_Wrist|acc,<raw>Right_Wrist|Charge)",
                                                "<angl_raw_relevant>(<norm>Right_Wrist|acc,<raw>Right_Wrist|Charge)"],
                                     lst_fn_for_feats=[squash_fn, squash_fn, squash_fn],
                                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=False)


                run_classic_test_new("CL_wrist_all_coh-" + nm_fn + "-" + labl_typ, labl_typ,
                            lst_feats=["<minmax>Right_Wrist|Charge", "<norm>Right_Wrist|acc",
                                       "<coh_raw_relevant>(<norm>Right_Wrist|acc,<raw>Right_Wrist|Charge)"],
                            lst_fn_for_feats=[calculate_window_features, calculate_window_features,
                                                       squash_fn, squash_fn, squash_fn],
                            win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=False)

                run_classic_test_new("CL_wrist_all_coh-angl-" + nm_fn + "-" + labl_typ, labl_typ,
                                     lst_feats=["<minmax>Right_Wrist|Charge", "<norm>Right_Wrist|acc",
                                                "<coh_raw_relevant>(<norm>Right_Wrist|acc,<raw>Right_Wrist|Charge)",
                                                "<angl_raw_relevant>(<norm>Right_Wrist|acc,<raw>Right_Wrist|Charge)"],
                                     lst_fn_for_feats=[calculate_window_features, calculate_window_features,
                                                       squash_fn, squash_fn, squash_fn],
                                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=False)

                run_classic_test_new("CL_all_coh-" + nm_fn + "-" + labl_typ, labl_typ,
                                     lst_feats=["<minmax>Right_Wrist|Charge", "<norm>Right_Wrist|acc",
                                                "<norm>Left_Calf|acc",
                                                "<coh_raw_relevant>(<norm>Right_Wrist|acc,<raw>Right_Wrist|Charge)"
                                                ],
                                     lst_fn_for_feats=[calculate_window_features, calculate_window_features,
                                                       calculate_window_features,
                                                       squash_fn, squash_fn, squash_fn],
                                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=False)

                run_classic_test_new("CL_all_coh-angl-" + nm_fn + "-" + labl_typ, labl_typ,
                                     lst_feats=["<minmax>Right_Wrist|Charge", "<norm>Right_Wrist|acc","<norm>Left_Calf|acc",
                                                "<coh_raw_relevant>(<norm>Right_Wrist|acc,<raw>Right_Wrist|Charge)",
                                                "<angl_raw_relevant>(<norm>Right_Wrist|acc,<raw>Right_Wrist|Charge)"],
                                     lst_fn_for_feats=[calculate_window_features, calculate_window_features,
                                                       calculate_window_features,
                                                       squash_fn, squash_fn, squash_fn],
                                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=False)

    try:
        spec_single()
        gc.collect()
    except:
        print("ERROR SINGLE")

def do_specs_group():
    import gc
    def spec_group():
        for nm_fn, squash_fn in zip(["sneaky", "mean"], [sneaky_squash, squash_all_bands]):
            for labl_typ in ["paper", "paper_hard"]:
                run_classic_test_new("CL_group_wrist_all_coh-" + nm_fn + "-" + labl_typ, labl_typ,
                            lst_feats=[ "<P1><minmax>Right_Wrist|Charge", "<P1><norm>Right_Wrist|acc",
                                        "<P2><minmax>Right_Wrist|Charge", "<P2><norm>Right_Wrist|acc",
                                        "<coh_raw_relevant>(<P1><raw>Right_Wrist|Charge,<P2><raw>Right_Wrist|Charge)"],
                            lst_fn_for_feats=[calculate_window_features, calculate_window_features,
                                              calculate_window_features, calculate_window_features,
                                             squash_fn, squash_fn, squash_fn, squash_fn, squash_fn, squash_fn],
                            win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=True)

                run_classic_test_new("CL_group_wrist_all_coh-angl" + nm_fn + "-" + labl_typ, labl_typ,
                                     lst_feats=["<P1><minmax>Right_Wrist|Charge", "<P1><norm>Right_Wrist|acc",
                                                "<P2><minmax>Right_Wrist|Charge", "<P2><norm>Right_Wrist|acc",
                                                "<coh_raw_relevant>(<P1><raw>Right_Wrist|Charge,<P2><raw>Right_Wrist|Charge)",
                                                "<angl_raw_relevant>(<P1><raw>Right_Wrist|Charge,<P2><raw>Right_Wrist|Charge)"],
                                     lst_fn_for_feats=[calculate_window_features, calculate_window_features,
                                                       calculate_window_features, calculate_window_features,
                                                       squash_fn, squash_fn, squash_fn, squash_fn, squash_fn,
                                                       squash_fn],
                                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=True)

                run_classic_test_new("CL_group_all_all_coh-" + nm_fn + "-" + labl_typ, labl_typ,
                                     lst_feats=["<P1><minmax>Right_Wrist|Charge", "<P1><norm>Right_Wrist|acc",
                                                "<P2><minmax>Right_Wrist|Charge", "<P2><norm>Right_Wrist|acc",
                                                "<coh_raw_relevant>(<P1><raw>Right_Wrist|Charge,<P2><raw>Right_Wrist|Charge)"],
                                     lst_fn_for_feats=[calculate_window_features, calculate_window_features,
                                                       calculate_window_features, calculate_window_features,
                                                       calculate_window_features, calculate_window_features,
                                                       squash_fn, squash_fn, squash_fn, squash_fn, squash_fn,
                                                       squash_fn],
                                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=True)

                run_classic_test_new("CL_group_wrist_all_coh-angl" + nm_fn + "-" + labl_typ, labl_typ,
                                     lst_feats=["<P1><minmax>Right_Wrist|Charge", "<P1><norm>Right_Wrist|acc",
                                                "<P2><minmax>Right_Wrist|Charge", "<P2><norm>Right_Wrist|acc",
                                                "<coh_raw_relevant>(<P1><raw>Right_Wrist|Charge,<P2><raw>Right_Wrist|Charge)",
                                                "<angl_raw_relevant>(<P1><raw>Right_Wrist|Charge,<P2><raw>Right_Wrist|Charge)"],
                                     lst_fn_for_feats=[calculate_window_features, calculate_window_features,
                                                       calculate_window_features, calculate_window_features,
                                                       squash_fn, squash_fn, squash_fn, squash_fn, squash_fn,
                                                       squash_fn],
                                     win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=True)

    spec_group()

    try:

        gc.collect()
    except Exception as e:
        print("ERROR SINGLE")
        print("EXCEPTION", e, type(e))
        import sys, os
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(exc_type, exc_tb.tb_lineno)


if __name__ == '__main__':
    # win_size = 50
    # for labl_typ in ["paper", "paper_hard"]:
    #     run_DL_test_new("" + "DL_group_all_all-" + labl_typ, labl_typ,
    #                     lst_feats=["<P1><minmax>Right_Wrist|Charge", "<P1><norm>Right_Wrist|acc",
    #                                "<P2><minmax>Right_Wrist|Charge", "<P2><norm>Right_Wrist|acc",
    #                                "<P1><norm>Left_Calf|acc", "<P2><norm>Left_Calf|acc"],
    #                     win_size=win_size, win_step=10, strategy='all', do_plot=False, pairwise=True)
    # exit(0)
    # do_specs_group()
    # do_specs_single()
    do_classic_baselines()
    do_missing_baselines()

    # do_missing_baselines()

    exit(0)


    # for labl_typ in ["paper", "paper_hard"]:
    #     run_classic_test_new("Z---CL_wrist_acc-" + labl_typ, labl_typ,
    #                          lst_feats=["<norm>Right_Wrist|acc"],
    #                          lst_fn_for_feats=[calculate_window_features, calculate_window_features,
    #                                            calculate_window_features,
    #                                            sneaky_squash, sneaky_squash, sneaky_squash],
    #                          win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=False)



    # do_specs_group()
    # exit(0)










    # def baselines_group():
    #     for labl_typ in ["pape", "paper_hard"]:
    #         run_classic_test_new("CL_group_wrist_acc-"+labl_typ, labl_typ,
    #             lst_feats= ["<P1><minmax>Right_Wrist|Charge", "<P1><norm>Right_Wrist|acc",
    #                         "<P2><minmax>Right_Wrist|Charge", "<P2><norm>Right_Wrist|acc",
    #                         "<coh_raw_relevant>(<P1><norm>Right_Wrist|acc,<P2><norm>Right_Wrist|acc)",
    #                         "<angl_raw_relevant>(<P1><norm>Right_Wrist|acc,<P2><norm>Right_Wrist|acc)",
    #                         "<coh_raw_relevant>(<P1><raw>Right_Wrist|Charge,<P2><raw>Right_Wrist|Charge)",
    #                         "<angl_raw_relevant>(<P1><raw>Right_Wrist|Charge,<P2><raw>Right_Wrist|Charge)"],
    #             lst_fn_for_feats=[calculate_window_features, calculate_window_features,
    #                               calculate_window_features, calculate_window_features,
    #                               squash_dividing_bands, squash_dividing_bands,
    #                               squash_dividing_bands, squash_dividing_bands],
    #             win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=True)
    # try:
    #     baselines_group()
    #     gc.collect()
    # except:
    #     print("ERROR GROUP")




    #

    # run_classic_test_new("Z-LS_base_CL_group_coh",
    #                                           lst_feats= [
    #                                            "<P1><minmax>Right_Wrist|Charge", "<P1><norm>Right_Wrist|acc",
    #                                            "<P2><minmax>Right_Wrist|Charge", "<P2><norm>Right_Wrist|acc",
    #
    #                                            "<coh_raw_relevant>(<P1><raw>Right_Wrist|Charge,<P2><raw>Right_Wrist|Charge)",
    #                                            "<angl_raw_relevant>(<P1><raw>Right_Wrist|Charge,<P2><raw>Right_Wrist|Charge)",
    #                                            "<corr_raw_relevant>(<P1><raw>Right_Wrist|Charge,<P2><raw>Right_Wrist|Charge)",
    #                                            #
    #                                            "<coh_raw_relevant>(<P1><norm>Right_Wrist|acc,<P2><norm>Right_Wrist|acc)",
    #                                            "<angl_raw_relevant>(<P1><norm>Right_Wrist|acc,<P2><norm>Right_Wrist|acc)",
    #                                            "<corr_raw_relevant>(<P1><norm>Right_Wrist|acc,<P2><norm>Right_Wrist|acc)",
    #
    #                                            ],
    #                                           lst_fn_for_feats=[
    #                                               calculate_window_features, calculate_window_features,
    #                                               calculate_window_features, calculate_window_features,
    #                                               squash_all_bands, squash_all_bands, squash_all_bands,
    #                                               squash_all_bands, squash_all_bands, squash_all_bands
    #                                           ],
    #                                           win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=True)



    # run_classic_test_new("Z_base_CL_group_coh_all_wrist_cross_mean",
    #                                           lst_feats= ["<P1><minmax>Right_Wrist|Charge", "<P1><norm>Right_Wrist|acc",
    #                                            "<P2><minmax>Right_Wrist|Charge", "<P2><norm>Right_Wrist|acc",
    #                                            "<coh_raw_relevant>(<P1><norm>Right_Wrist|acc,<P2><norm>Right_Wrist|acc)",
    #                                            "<angl_raw_relevant>(<P1><norm>Right_Wrist|acc,<P2><norm>Right_Wrist|acc)",
    #                                            "<coh_raw_relevant>(<P1><raw>Right_Wrist|Charge,<P2><raw>Right_Wrist|Charge)",
    #                                            "<angl_raw_relevant>(<P1><raw>Right_Wrist|Charge,<P2><raw>Right_Wrist|Charge)"],
    #                                           lst_fn_for_feats=[calculate_window_features, calculate_window_features,
    #                                                             calculate_window_features, calculate_window_features,
    #                                                             squash_dividing_bands, squash_dividing_bands,
    #                                                             squash_dividing_bands, squash_dividing_bands],
    #                                           win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=True)

    # run_classic_test_new("Z_base_CL_group_coh_all_wrist_cross_mean",
    #                      lst_feats= ["<P1><minmax>Right_Wrist|Charge", "<P1><norm>Right_Wrist|acc",
    #                       "<P2><minmax>Right_Wrist|Charge", "<P2><norm>Right_Wrist|acc",
    #                       "<coh_raw_relevant>(<P1><norm>Right_Wrist|acc,<P2><norm>Right_Wrist|acc)",
    #                       "<angl_raw_relevant>(<P1><norm>Right_Wrist|acc,<P2><norm>Right_Wrist|acc)",
    #                       "<coh_raw_relevant>(<P1><raw>Right_Wrist|Charge,<P2><raw>Right_Wrist|Charge)",
    #                       "<angl_raw_relevant>(<P1><raw>Right_Wrist|Charge,<P2><raw>Right_Wrist|Charge)"],
    #                      lst_fn_for_feats=[calculate_window_features, calculate_window_features,
    #                                        calculate_window_features, calculate_window_features,
    #                                        squash_dividing_bands, squash_dividing_bands,
    #                                        squash_dividing_bands, squash_dividing_bands],
    #                      win_size=50, win_step=10, strategy='all', do_plot=False, pairwise=True)



    exit(0)
















