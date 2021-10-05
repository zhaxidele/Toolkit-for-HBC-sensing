from __future__ import print_function

import csv
import glob
import itertools
import os
import re
from collections import defaultdict, namedtuple

import numpy as np
import pandas as pd

#import recognition.igroups_wtc as iwtc
#from recognition.definitions import RAW_DATA_DIR, COH_SAVE_DIR

import igroups_wtc as iwtc
from definitions import RAW_DATA_DIR, COH_SAVE_DIR

# matplotlib.use('Agg')




map_activities = {
  1  : "the start and stop steps",
  2  : "doing nothing",
  3  : "walk alone with nothing",
  31 : "walk alone and taking something alone (some pieces of stuff)",
  32 : "walk alone and taking something alone (TV base)",
  33 : "walk and carry TV together with another person",
  41 : "touch and lift from box",
  42 : "touch and lift from ground",
  43 : "touch and lift from TV wall",
  51 : "drop into box",
  52 : "drop to ground",
  53 : "drop into TV wall",
  61 : "turn the screw with normal screw driver",
  62 : "turn the screw with electric screw driver",
  7  : "out of camera",
  0  : "no definition"
}
def get_class_label(lbl_id, strategy):
  """
  Get activity classes in a correct and consistent manner according to some strategy.

  :param lbl_id: How it is in the labeling file.
  :param strategy: How to join the labels.
    pure : as in the labeling file.
    presentation : as in igroup presentation.
    paper : as in the paper.
    paper_hard : the version that includes also lifting and dropping.
  :return:
  """

  if strategy == "pure":
    return map_activities[lbl_id]
  elif strategy == "presentation":
    # lbl_txt = map_activities[lbl_id]
    if lbl_id == 3 or lbl_id == 32 or lbl_id == 33:
      # Allowed are "walk alone with nothing", "walk alone and taking something alone (TV base)"
      #  and "walk and carry TV together with another person".
      return map_activities[lbl_id]
    else:
      return "NULL"
  elif strategy == "paper":
    if lbl_id == 7 or lbl_id == 0 or lbl_id == 1:
      return "REMOVE"
    elif lbl_id == 2:
      return "NULL"
    elif lbl_id == 3 or lbl_id == 31:
        return "WALKING"
    elif lbl_id == 32:
        return "CARRY ALONE"
    elif lbl_id == 33:
        return "CARRY TOGETHER"
    else:
      return "NULL"
  elif strategy == "paper_hard":
    if lbl_id == 7 or lbl_id == 0 or lbl_id == 1:
      return "REMOVE"
    elif lbl_id == 2:
      return "NULL"
    elif lbl_id == 3 or lbl_id == 31:
        return "WALKING"
    elif lbl_id == 32:
        return "CARRY ALONE"
    elif lbl_id == 33:
        return "CARRY TOGETHER"
    elif lbl_id == 41 or lbl_id == 42 or lbl_id == 43:
      return "LIFT"
    elif lbl_id == 51 or lbl_id == 52 or lbl_id == 53:
      return "DROP"
    else:
      return "NULL"

  raise Exception("Strategy Unknown for class labels! " + strategy)





def get_user_map():
  """

  :param exp_nm: the experiment id.
  :return: A map with mappings such as P1 -> (Nm1, Nm2, Nm3)
  """
  with open(os.path.join(RAW_DATA_DIR, "Names.csv")) as f_persons:
    reader = csv.DictReader(f_persons)
    map_users = { row["Ex"] : (row["P1_B"], row["P2_G"], row["P3_Y"])  for row in reader }
  return map_users


def get_all_sensors_for_users(exp_nm):
  """
  :param exp_nm: The experiment id.
  :return: A map person_id -> body_position -> sensor_type -> (sensor_data, arr_mask_reliability) )
  """
  nested_dict = lambda: defaultdict(nested_dict)
  map_result = nested_dict()

  map_clr_to_p_id = {"B" : "P1", "G" : "P2", "Y" : "P3" }
  dt_notice = pd.read_csv(os.path.join(RAW_DATA_DIR, exp_nm, exp_nm + "_Notice.csv"))
  notice_header = dt_notice[ dt_notice.columns[0] ]

  for path_sensor_data in glob.glob( os.path.join(RAW_DATA_DIR, exp_nm, "Data_Synchronised")  + "/*.TXT"):
    lst_file_info = os.path.split(path_sensor_data)[-1].split(".")[0].split("_")

    # Determine sensor location.
    if len(lst_file_info) == 3:
      person_color, sensor_loc_code, sensor_type = lst_file_info
      is_in_right_side = True
    else:
      person_color, sensor_loc_code, _, sensor_type = lst_file_info
      is_in_right_side = False
    str_side = "Right" if is_in_right_side else "Left"
    if sensor_loc_code == "W":
      sensor_loc = str_side + "_Wrist"
    elif sensor_loc_code == "C":
      sensor_loc = str_side + "_Calf"

    # Determine person.
    person_id = map_clr_to_p_id[person_color]

    # Get all relevant notices.
    arr_notice_col = np.array(dt_notice[person_id + "_" + person_color])
    # Skip files that are not fine.
    if os.path.split(path_sensor_data)[-1] not in arr_notice_col:
      print("INFO", exp_nm, os.path.split(path_sensor_data)[-1], "not reliable!")
      continue


    # Get sensor data.
    dt_sensr_data = pd.read_csv(path_sensor_data, sep=' ')

    # Mark the times where the sensor is not realiable
    arr_not_reliable = np.zeros(len(dt_sensr_data))
    if type(arr_notice_col[-1]) == str:
      s, e = arr_notice_col[-1].split("-")
      arr_not_reliable[int(s) : int(e)] = 1.0


    # Fill for each sensor source.
    if sensor_type == "Charge":
      sensr_data = np.array(dt_sensr_data.iloc[:,3]).reshape( (-1,1) )
      map_result[person_id][sensor_loc][sensor_type] = (sensr_data, arr_not_reliable)
    elif sensor_type == "Frequency":
      sensr_data = np.array(dt_sensr_data.iloc[:,2]).reshape( (-1,1) )
      map_result[person_id][sensor_loc][sensor_type] = (sensr_data, arr_not_reliable)
    elif sensor_type == "BNO055":
      acc_data = np.array(dt_sensr_data.iloc[:, [2,3,4] ])
      map_result[person_id][sensor_loc]["acc"] = (acc_data, arr_not_reliable)

      gyro_data = np.array(dt_sensr_data.iloc[:, [5, 6, 7]])
      map_result[person_id][sensor_loc]["gyro"] = (gyro_data, arr_not_reliable)

      mag_data = np.array(dt_sensr_data.iloc[:, [8, 9, 10]])
      map_result[person_id][sensor_loc]["mag"] = (mag_data, arr_not_reliable)

      eul_data = np.array(dt_sensr_data.iloc[:, [11, 12, 13]])
      map_result[person_id][sensor_loc]["eul"] = (eul_data, arr_not_reliable)

  return map_result



def get_ground_label(lbl_txt):
  """
  Convert labels to a consistent / interpretable value across all experiments.

  Ground	Describe
  1	      fabric
  2	      carpet
  3	      concret

  :param lbl_txt: The label in the labeling file.
  :return: The readable label for the ground.
  """
  if lbl_txt == 1:
    return "fabric"
  elif lbl_txt == 2:
    return "carpet"
  elif lbl_txt == 3:
    return "concret"
  raise Exception("Ground label unknown!  "+lbl_txt)


def parse_labeling_times(str_time):
  """
  03.02.17
  Minute.Second.Second/30

  :param str_time: As in the labeling file.
  :return: float time in seconds
  """
  lst_t_info = str_time.split(".")
  if len(lst_t_info) == 3:
    m, s, fr = lst_t_info
    t_in_sec = float(m)*60.0 + float(s) + (float(fr) / 30.0)
  elif len(lst_t_info) == 4:
    h, m, s, fr = lst_t_info
    t_in_sec =  float(h)*3600.0 + float(m) * 60.0 + float(s) + (float(fr) / 30.0)
  else:
    print(str_time)
    raise("Stange time format: " + str_time)

  return t_in_sec



def from_t_in_seconds_to_string(t_in_seconds):
  h = int(t_in_seconds / 3600)
  t_in_seconds -= h * 3600
  m = int(t_in_seconds / 60)
  t_in_seconds -= m * 60
  s = int(t_in_seconds)
  t_in_seconds -= s
  fr = int(t_in_seconds * 30.0)

  return ".".join( [ str(v) for v in [h, m, s, fr] ])



def get_activity_labels(exp_nm, t_sec_per_label = 0.1, strategy="paper_hard"):
  """

  :param exp_nm:
  :param label_type:if len()
  :param t_sec_per_label:
  :return: map with P1 -> arr_lbls
  """

  # Determine exp range.
  start_in_sec, end_in_sec = np.inf, -np.inf
  map_data_parsed = {}
  for usr, usr_clr in zip(["P1", "P2", "P3"], ["B", "G", "Y"]):

    path_lbl_file = os.path.join(RAW_DATA_DIR, exp_nm, exp_nm + "_" + usr + "_" + usr_clr + ".csv")
    if os.path.exists(path_lbl_file):
      with open(path_lbl_file, 'r') as file_lbl:
        reader = csv.DictReader(file_lbl)
        map_data_parsed[usr] = [ (parse_labeling_times(row["Start"]), parse_labeling_times(row["End"]),
                           get_class_label(int(row["Activity"]), strategy) ) for row in reader ]

        my_start_in_sec, my_end_in_sec = map_data_parsed[usr][0][0], map_data_parsed[usr][-1][1]
        start_in_sec = min(start_in_sec, my_start_in_sec)
        end_in_sec = max(end_in_sec, my_end_in_sec)
    else:
      map_data_parsed[usr] = []




  map_labels_per_user = defaultdict(list)
  for usr, usr_clr in zip(["P1", "P2", "P3"], ["B", "G", "Y"]):
    data_parsed = map_data_parsed[usr]
    # print("------------------------>", usr, len(data_parsed), exp_nm)
    lst_active_acts, i_parsed = [], 0
    lst_labels = map_labels_per_user[usr]


    # print("GOING TO START LBLS  <<< ", usr)
    for t_lbl in np.arange(start=start_in_sec, stop=end_in_sec, step=t_sec_per_label):

      # print(from_t_in_seconds_to_string(t_lbl))

      # Move needle adding valid.
      while i_parsed < len(data_parsed) and data_parsed[i_parsed][0] <= t_lbl:
        lst_active_acts.append(data_parsed[i_parsed])
        # print("ADDED", from_t_in_seconds_to_string(t_lbl), data_parsed[i_parsed][2],
        #                        from_t_in_seconds_to_string(data_parsed[i_parsed][0]),
        #                        from_t_in_seconds_to_string(data_parsed[i_parsed][1]))
        # print("NOW POINTING", i_parsed, data_parsed[i_parsed])
        i_parsed += 1

      # Remove from active the past ones.
      # lst_active_acts = [act_line for act_line in lst_active_acts if t_lbl < act_line[1]]
      lst_active_acts_new = []
      for act in lst_active_acts:
          if t_lbl < act[1]:
              lst_active_acts_new.append(act)
          # else:
          #     print("REMOVE", from_t_in_seconds_to_string(t_lbl),
          #           from_t_in_seconds_to_string(act[0]), "->", from_t_in_seconds_to_string(act[1]), act[2] )
      lst_active_acts = lst_active_acts_new


      # Add for usr its label.
      if len(lst_active_acts) > 1:
        # print("MANY")
        # for a in lst_active_acts:
        #     print( a[2], from_t_in_seconds_to_string(a[0]), from_t_in_seconds_to_string(a[1]) )
        # print("----")
        lst_active_acts = [lst_active_acts[-1]]
        usr_lbl = lst_active_acts[-1][2]
      elif len(lst_active_acts) == 0:
        # print("ERROR!", exp_nm, usr, "NO ACTIVITY", from_t_in_seconds_to_string(t_lbl) ,"I am at act line", i_parsed)
        usr_lbl = "NULL"
      else:
        usr_lbl = lst_active_acts[0][2]

      lst_labels.append(usr_lbl)
  return map_labels_per_user


def get_ground_labels(exp_nm, t_sec_per_label = 0.1):
  map_labels_per_user = {"P1" : [], "P2" : [], "P3" : []}
  path_lbl_file = os.path.join(RAW_DATA_DIR, exp_nm, exp_nm + "_Ground.csv")
  with open(path_lbl_file, 'r') as file_lbl:
    reader = csv.DictReader(file_lbl)
    data_parsed_all = [
      (
        parse_labeling_times(row["Start"]), parse_labeling_times(row["End"]),
        get_ground_label(int(row["Ground"])), "P" + str(int(row["Person"]))
      ) for row in reader
    ]


  for user_nm in map_labels_per_user.keys():
    data_parsed = [ l for l in data_parsed_all if l[3] == user_nm]
    start_in_sec, end_in_sec = data_parsed[0][0], data_parsed[-1][1]
    lst_active_acts, i_parsed = [], 0

    for t_lbl in np.arange(start=start_in_sec, stop=end_in_sec, step=t_sec_per_label):
      # Move needle adding valid.
      while i_parsed < len(data_parsed) and data_parsed[i_parsed][0] <= t_lbl:
        lst_active_acts.append(data_parsed[i_parsed])
        i_parsed += 1

      # Remove from active the past ones.
      lst_active_acts = [act_line for act_line in lst_active_acts if t_lbl < act_line[1]]

      # Add for usr its label.
      if len(lst_active_acts) == 0:
        if len(map_labels_per_user[user_nm]) > 0:
              map_labels_per_user[user_nm].append(map_labels_per_user[user_nm][-1])
        else:
          map_labels_per_user[user_nm].append("fabric")
      else:
        usr_ground = lst_active_acts[-1][2]
        map_labels_per_user[user_nm].append(usr_ground)
  return map_labels_per_user



def get_exp_data(exp_nm, lbl_type, lst_placement = ["Right_Wrist"], lst_type = ["Charge", "acc"]):
  """

 :param exp_nm:
 :param lst_placement:
 :param lst_type:
 :return: map P -> (Snsr_name -> Snsr_vals, Lbls) where all data is synchronized.
 """
  # P1 -> Placement -> Type -> arr_snsr_data
  map_usrs_lbl_in_day =  get_activity_labels(exp_nm, strategy=lbl_type)

  # "P1" -> lst_lbls
  map_usrs_data_in_day = get_all_sensors_for_users(exp_nm)


  # Crop end of sensors
  lbls_p1, lbls_p2, lbls_p3  = map_usrs_lbl_in_day["P1"], map_usrs_lbl_in_day["P2"], map_usrs_lbl_in_day["P3"]
  fn_all_null = lambda i : lbls_p1[i] == "NULL" and lbls_p2[i] == "NULL" and lbls_p3[i] == "NULL"
  i_l_real_start, i_l_real_end = 0, len(lbls_p1) - 1
  # while fn_all_null(i_l_real_start):
  #     i_l_real_start += 1
  # while fn_all_null(i_l_real_end):
  #     i_l_real_end -= 1


  map_result = {"P1" : [{}, lbls_p1[i_l_real_start:i_l_real_end]],
                "P2" : [{}, lbls_p2[i_l_real_start:i_l_real_end]],
                "P3" : [{}, lbls_p3[i_l_real_start:i_l_real_end]]}



  # Add the sensor data.
  for usr_nm, map_usr_data in map_usrs_data_in_day.items():
    min_usr_snsr_len = len(map_result[usr_nm][1])
    for str_side in lst_placement:
      for str_type in lst_type:
        if str_type not in map_usr_data[str_side]:
          continue
        sensr_data = map_usr_data[str_side][str_type][0][i_l_real_start:i_l_real_end,:]
        arr_reliable = map_usr_data[str_side][str_type][1][i_l_real_start:i_l_real_end]
        min_usr_snsr_len = min(min_usr_snsr_len, len(sensr_data))
        map_result[usr_nm][0][str_side + "|" + str_type] = [sensr_data, arr_reliable]

    # Crop labels if sensors are missing in the end of labels.
    if min_usr_snsr_len != len(map_result[usr_nm][1]):
        # Crop labels
        map_result[usr_nm][1] = map_result[usr_nm][1][0:min_usr_snsr_len]
        # Crop sensors.
        for nm_snsr in map_result[usr_nm][0].keys():
            map_result[usr_nm][0][nm_snsr] = (map_result[usr_nm][0][nm_snsr][0][0:min_usr_snsr_len],
                                              map_result[usr_nm][0][nm_snsr][1][0:min_usr_snsr_len])
  return map_result



def gen_jaime_ws(params):
  sig_1_v, sig_2_v, path_file = params

  print(path_file, sig_1_v.shape, sig_2_v.shape)

  if not os.path.exists(path_file + "_coh.pkl") or not os.path.exists(path_file + "_corr.pkl")  \
          or not os.path.exists(path_file + "_angl.pkl"):
    # plot_type = 'coherece', global_scale = True

    normalise = True
    global_scale = True

    # data = np.vstack([sig_1_v.flatten(), sig_2_v.flatten()])
    data = [sig_1_v.flatten(), sig_2_v.flatten()]
    WCT, aWCT, WXT, scales, coi, freqs, sig95 = iwtc.mwct(data, 0.1, dj=0.1, normalize=normalise)
    WCoh = pd.DataFrame(WCT.transpose(), columns=freqs, index=np.arange(len(sig_1_v)))
    WCorr = pd.DataFrame(WXT.transpose(), columns=freqs, index=np.arange(len(sig_1_v)))
    WAng = pd.DataFrame(aWCT.transpose(), columns=freqs, index=np.arange(len(sig_1_v)))

    # np.save(path_file + "_coh",  WCoh)
    # np.save(path_file + "_corr", WCorr)

    WCoh.to_pickle(path_file + "_coh.pkl")
    WCorr.to_pickle(path_file + "_corr.pkl")
    WAng.to_pickle(path_file + "_angl.pkl")


def gen_w_global(dir_save):
  # Load all experiments.
  map_groups = defaultdict(list)
  for exp_dir in glob.glob(RAW_DATA_DIR + "/*/"):
    exp_nm = exp_dir.split("/")[-2]
    # map P -> (Snsr_name -> Snsr_vals, Lbls)
    map_exp_data = get_exp_data(exp_nm, lst_placement = ["Right_Wrist"], lst_type = ["Charge", "acc"])

    for usr_1, user_data_1 in map_exp_data.items():
      for usr_2, user_data_2 in map_exp_data.items():
        map_data_1, user_lbls_1 = user_data_1
        map_data_2, user_lbls_2 = user_data_2

        try:
          acc_data_1 = calculate_acc_to_norm(map_data_1["Right_Wrist|acc"][0])
          cap_data_1 = map_data_1["Right_Wrist|Charge"][0]

          acc_data_2 = calculate_acc_to_norm(map_data_2["Right_Wrist|acc"][0])
          cap_data_2 = map_data_2["Right_Wrist|Charge"][0]

          for loc1, loc2 in [ ["Right_Wrist", "Right_Wrist"] ]:
            path_file = dir_save + "|".join( [usr_1, usr_2, exp_nm, loc1, "norm_acc", loc2, "Charge"] )
            gen_jaime_ws([acc_data_1, cap_data_2 , path_file])

            path_file = dir_save + "|".join([usr_1, usr_2, exp_nm, loc1, "norm_acc", loc2, "norm_acc"])
            gen_jaime_ws([acc_data_1, acc_data_2, path_file])

            path_file = dir_save + "|".join([usr_1, usr_2, exp_nm, loc1, "Charge", loc2, "Charge"])
            gen_jaime_ws([cap_data_1, cap_data_2, path_file])


        except Exception as e:
          print("EXCEPTION", usr_1, usr_2, exp_nm, e, type(e))
          continue







from preprocess import get_only_valid_windows
from preprocess import calculate_acc_to_norm, preprocess_capacitance
from preprocess import band_separated_features_from_spectrum, mean_across_all_spectrum, relevant_part_of_the_spectrum
from preprocess import capacitance_min_max


def load_group_windows_new(lst_features, lbl_type, win_size = 50 , win_step = 10, strategy = 'all',
                            lst_placement = ["Right_Wrist"], lst_type = ["Charge", "acc"]):
  # Load all experiments.
  map_groups = defaultdict(list)
  map_users_in_exp = get_user_map()
  for exp_dir in glob.glob(RAW_DATA_DIR + "/*/"):
    exp_nm = exp_dir.split("/")[-2]
    # map P -> (Snsr_name -> Snsr_vals, Lbls)
    map_exp_data = get_exp_data(exp_nm, lbl_type, lst_placement, lst_type)

    lst_users = map_users_in_exp[exp_nm]
    grp_id = "_".join(sorted(lst_users))

    for usr, user_data in map_exp_data.items():
      map_data, user_lbls = user_data
      user_lbls = np.array(user_lbls)
      lst_arr_sensor_data = []
      arr_reliability = np.zeros(len(user_lbls))
      try:
          for feat in lst_features:
            if type(feat) == SingleFeature:
              snsr_data, arr_reliable = map_data[feat.loc + "|" + feat.typ]
              if feat.prep != 'raw':
                if feat.typ == 'acc' and feat.prep == 'norm':
                    snsr_data = calculate_acc_to_norm(snsr_data)
                elif feat.typ == 'gyro' and feat.prep == 'norm':
                  snsr_data = calculate_acc_to_norm(snsr_data)
                elif feat.typ == 'Charge' and feat.prep == 'diff':
                  snsr_data = preprocess_capacitance(snsr_data)
                elif feat.typ == 'Charge' and feat.prep == 'minmax':
                  snsr_data = capacitance_min_max(snsr_data)
                else:
                  raise Exception("Unknown prep " + feat.prep, feat.typ)
              # Add this single sensor
              lst_arr_sensor_data.append(snsr_data)
              arr_reliability += arr_reliable
            elif type(feat) == MultiFeature:

              def translate(t, p):
                if p == "norm" and t == "acc":
                  return "norm_acc"
                if p == "raw" and t == "Charge":
                  return "Charge"
                return None
              src_nm = translate(feat.src_typ, feat.src_prep)
              dest_nm = translate(feat.dest_typ, feat.dest_prep)

              ext = None
              if feat.proc.startswith("coh"):
                ext = "_coh.pkl"
              elif feat.proc.startswith("corr"):
                ext = "_corr.pkl"
              elif feat.proc.startswith("angl"):
                ext = "_angl.pkl"

              path_ws = os.path.join(COH_SAVE_DIR,
                                     "|".join([usr, usr, exp_nm, feat.src_loc, src_nm, feat.dest_loc, dest_nm])
                                     + ext)
              ws = pd.read_pickle(path_ws)
              # print(path_ws, len(ws))
              if feat.proc.endswith("_raw"):
                snsr_data = np.array(ws)
              elif feat.proc.endswith("_raw_relevant"):
                snsr_data = relevant_part_of_the_spectrum(ws)
              elif feat.proc.endswith("_bands"):
                snsr_data = band_separated_features_from_spectrum(ws)
              elif feat.proc.endswith("_mean"):
                snsr_data = mean_across_all_spectrum(ws)
              else:
                raise Exception("Unknown proc " + feat)
              # Add this sensor making sure to take into account validity of both data sources.
              lst_arr_sensor_data.append(snsr_data)
              _, arr_reliable_src = map_data[feat.src_loc + "|" + feat.src_typ]
              _, arr_reliable_dest = map_data[feat.dest_loc + "|" + feat.dest_typ]
              arr_reliability += (arr_reliable_src + arr_reliable_dest)

      except Exception as e:

        if type(feat) == SingleFeature:
          str_status = "OK" if ((feat.loc + "|" + feat.typ) not in map_data) else "WHY???"
        else:
          str_status = "OK" if ( ((feat.src_loc + "|" + feat.src_typ) not in map_data) or
                                 ((feat.dest_loc + "|" + feat.dest_typ) not in map_data) ) else "WHY???"

        print("MISSING", exp_nm, usr, feat, str_status)
        print("EXCEPTION", e, type(e))
        import sys, os
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(exc_type, exc_tb.tb_lineno)
        continue
      # Build windows of only valid data.
      lst_win_sensors, y_win = get_only_valid_windows(lst_arr_sensor_data, arr_reliability, user_lbls,
                                                        win_size, win_step, strategy)
      # Record this group user's windows.
      map_groups[grp_id].append((lst_win_sensors, y_win, exp_nm + "_" + usr))
  return map_groups



def load_multi_person_windows_new(lst_features, lbl_type, win_size=50, win_step=10, strategy='all',
                                  lst_placement = ["Right_Wrist"], lst_type = ["Charge", "acc"]):
  # Load all experiments.
  # group_id -> Person_id -> [ (X_win, y_win), ...]
  map_results = defaultdict(lambda: defaultdict(list))

  map_users_in_exp = get_user_map()
  for exp_dir in glob.glob(RAW_DATA_DIR + "/*/"):
    exp_nm = exp_dir.split("/")[-2]
    # map P -> (Snsr_name -> Snsr_vals, Lbls)
    map_exp_data = get_exp_data(exp_nm, lbl_type, lst_placement, lst_type)

    lst_users = map_users_in_exp[exp_nm]
    grp_id = "_".join(sorted(lst_users))

    for usr_A, usr_B in itertools.combinations(["P1", "P2", "P3"], 2):
      map_data_A, user_lbls_A = map_exp_data[usr_A]
      map_data_B, user_lbls_B = map_exp_data[usr_B]

      min_size = min(len(user_lbls_A), len(user_lbls_B)) - 10
      import os
      lst_arr_sensor_data = []
      arr_reliability = np.zeros(len(user_lbls_B))
      arr_reliability = arr_reliability[0:min_size]
      user_lbls_A, user_lbls_B = user_lbls_A[0:min_size], user_lbls_B[0:min_size]
      try:
        for feat in lst_features:
          if type(feat) == PairSingleFeature:
            if feat.person == "P1":
              map_data = map_data_A
            elif feat.person == "P2":
              map_data = map_data_B
            snsr_data, arr_reliable = map_data[feat.loc + "|" + feat.typ]
            snsr_data, arr_reliable = snsr_data[0:min_size], arr_reliable[0:min_size]
            if feat.prep != 'raw':
              if feat.typ == 'acc' and feat.prep == 'norm':
                snsr_data = calculate_acc_to_norm(snsr_data)
              elif feat.typ == 'Charge' and feat.prep == 'diff':
                snsr_data = preprocess_capacitance(snsr_data)
              elif feat.typ == 'Charge' and feat.prep == 'minmax':
                snsr_data = capacitance_min_max(snsr_data)
              else:
                raise Exception("Unknown prep " + feat.prep, feat.typ)
            # Add this single sensor
            lst_arr_sensor_data.append(snsr_data)
            arr_reliability += arr_reliable

          elif type(feat) == PairMultiFeature:
            def translate(t, p):
              if p == "norm" and t == "acc":
                return "norm_acc"
              if p == "raw" and t == "Charge":
                return "Charge"
              print("------------------------->", t, p)
              return None

            src_nm = translate(feat.src_typ, feat.src_prep)
            dest_nm = translate(feat.dest_typ, feat.dest_prep)

            ext = None
            if feat.proc.startswith("coh"):
              ext = "_coh.pkl"
            elif feat.proc.startswith("corr"):
              ext = "_corr.pkl"
            elif feat.proc.startswith("angl"):
              ext = "_angl.pkl"

            usr_src = usr_A if feat.src_person == "P1" else usr_B
            usr_dest = usr_A if feat.dest_person == "P1" else usr_B
            path_ws = os.path.join(COH_SAVE_DIR,
                                   "|".join([usr_src, usr_dest, exp_nm, feat.src_loc, src_nm, feat.dest_loc, dest_nm])
                                   + ext)
            ws = pd.read_pickle(path_ws)
            if feat.proc.endswith("_raw"):
              snsr_data = np.array(ws)
            elif feat.proc.endswith("_raw_relevant"):
              snsr_data = relevant_part_of_the_spectrum(ws)
            elif feat.proc.endswith("_bands"):
              snsr_data = band_separated_features_from_spectrum(ws)
            elif feat.proc.endswith("_mean"):
              snsr_data = mean_across_all_spectrum(ws)
            else:
              raise Exception("Unknown proc " + feat)
            snsr_data = snsr_data[0:min_size]
            lst_arr_sensor_data.append(snsr_data)
            _, arr_reliable_src = map_exp_data[usr_src][0][feat.src_loc + "|" + feat.src_typ]
            _, arr_reliable_dest = map_exp_data[usr_dest][0][feat.dest_loc + "|" + feat.dest_typ]
            arr_reliable_src, arr_reliable_dest = arr_reliable_src[0:min_size], arr_reliable_dest[0:min_size]
            arr_reliability += (arr_reliable_src + arr_reliable_dest)
      except Exception as e:
        str_status = "????"
        if type(feat) == PairMultiFeature:
          str_status = "OK" if  ( (feat.src_loc + "|" + feat.src_typ) not in map_exp_data[usr_src][0] or \
              (feat.dest_loc + "|" + feat.dest_typ) not in map_exp_data[usr_dest][0] ) else "WHY???"
          if str_status == "WHY???":
            import traceback
            print(traceback.format_exc())


        elif type(feat) == PairSingleFeature:
          str_status = "OK" if ((feat.loc + "|" + feat.typ) not in map_data) else "WHY???"
        print( "MISSING", exp_nm, feat, str_status)
        print("EXCEPTION", e, type(e))
        import sys, os
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(exc_type, exc_tb.tb_lineno)
        continue

      # Add data.
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
      lst_win_sensors, y_win = get_only_valid_windows(lst_arr_sensor_data, arr_reliability, lbls_both,
                                                      win_size, win_step, strategy)
      # Record this group user's windows.
      if len(lst_win_sensors) > 0:
        map_results[grp_id][usr_A + "&" + usr_B].append((lst_win_sensors, y_win, exp_nm + "_" + usr_A + "&" + usr_B))
      else:
        print ("NO DATA", exp_nm, usr_A + "&" + usr_B)
  return map_results


PairSingleFeature = namedtuple('SingleFeature', [ 'person', 'prep', 'loc', 'typ'])
PairMultiFeature = namedtuple('MultiFeature', ['proc', 'src_person', 'src_prep', 'src_loc', 'src_typ',
                                             'dest_person', 'dest_prep', 'dest_loc', 'dest_typ'])


def gen_leave_pairs_of_one_group_out_new_version(lst_to_load, lbl_type, win_size=50, win_step=10, strategy='all'):
  """

    :param lst_to_load: list of features to load. Format for each feature is

      For features that do not include corr / coh, the format is

      <person><prep_type>Location|Type

      for example <P1><raw>Right_Wrist|Charge or <P2><norm>Right_Wrist|acc

      For features that rely on corr / coh, the format is

      <proc>(<person1><prep1>Location1|Type1,<person2><prep2>Location2|Type2)


      person ids should be either P1 or P2

    :param win_size: size of the windows in readings.
    :param win_step: step of the windows in readings.
    :param strategy: strategy for post processing sliding window labels. Use 'all' to return them all.
    :return: This yields: test_grp_id, X_train, y_train, lst_X_test, lst_y_test
    """
  regex_single = re.compile(r"<(?P<person>P[1-2])><(?P<prep>[_A-Za-z]*)>(?P<loc>[_A-Za-z]*)\|(?P<typ>[_A-Za-z]*)")

  regex_spec = re.compile(r"<(?P<feat>[_A-Za-z]*)>\(" \
                          r"<(?P<person1>P[1-2])><(?P<prep1>[A-Za-z]*)>(?P<loc1>[_A-Za-z]*)\|(?P<typ1>[_A-Za-z]*)\," \
                          r"<(?P<person2>P[1-2])><(?P<prep2>[A-Za-z]*)>(?P<loc2>[_A-Za-z]*)\|(?P<typ2>[_A-Za-z]*)" \
                          r"\)")
  pass
  set_placements, set_types = set(), set()
  lst_features = []
  for str_load_feat in lst_to_load:
    grps_single = regex_single.search(str_load_feat)
    grps_spec = regex_spec.search(str_load_feat)

    if grps_spec is not None:
      proc, pers1, prep1, loc1, typ1, pers2, prep2, loc2, typ2 = grps_spec.groups()
      feat = PairMultiFeature(proc, pers1, prep1, loc1, typ1, pers2, prep2, loc2, typ2)
      lst_features.append(feat)

      set_placements.add(loc1)
      set_types.add(typ1)
      set_placements.add(loc2)
      set_types.add(typ2)

    elif grps_single is not None:
      person, prep, loc, typ = grps_single.groups()
      feat = PairSingleFeature(person, prep, loc, typ)
      lst_features.append(feat)

      set_placements.add(loc)
      set_types.add(typ)
  lst_placement, lst_type = list(set_placements), list(set_types)
  map_group_pairs = load_multi_person_windows_new(lst_features, lbl_type, win_size=win_size, win_step=win_step, strategy=strategy,
                                              lst_placement=lst_placement, lst_type=lst_type)
  for test_grp_id in map_group_pairs.keys():
    lst_x_train, lst_y_train = [], []
    lst_x_test, lst_y_test, lst_exp_nm = [], [], []
    for grp_id, map_person_pairs in map_group_pairs.items():
      for pair_id, lst_pair_data in map_person_pairs.items():
        if grp_id == test_grp_id:
          lst_x_test.extend([vs[0] for vs in lst_pair_data])
          lst_y_test.extend([vs[1] for vs in lst_pair_data])
          lst_exp_nm.extend([vs[2] for vs in lst_pair_data])
        else:
          lst_x_train.extend([vs[0] for vs in lst_pair_data])
          lst_y_train.extend([vs[1] for vs in lst_pair_data])

    yield test_grp_id, lst_x_train, lst_y_train,  lst_x_test, lst_y_test, lst_exp_nm





SingleFeature = namedtuple('SingleFeature', ['prep', 'loc', 'typ'])
MultiFeature = namedtuple('MultiFeature', ['proc', 'src_prep', 'src_loc', 'src_typ',
                                             'dest_prep', 'dest_loc', 'dest_typ'])

def gen_leave_one_group_out_new_version(lst_to_load, lbl_type, win_size = 50 , win_step = 10, strategy = 'all'):
  """

  :param lst_to_load: list of features to load. Format for each feature is

    For features that do not include corr / coh, the format is

    <prep_type>Location|Type

    for example <raw>Right_Wrist|Charge or <norm>Right_Wrist|acc

    For features that rely on corr / coh, the format is

    <proc>(<prep1>Location1|Type1,<prep2>Location2|Type2)

  :param win_size: size of the windows in readings.
  :param win_step: step of the windows in readings.
  :param strategy: strategy for post processing sliding window labels. Use 'all' to return them all.
  :return: This yields: test_grp_id, X_train, y_train, lst_X_test, lst_y_test
  """

  regex_single = re.compile(r"<(?P<prep>[_A-Za-z]*)>(?P<loc>[_A-Za-z]*)\|(?P<typ>[_A-Za-z]*)")

  regex_spec =  re.compile( r"<(?P<feat>[_A-Za-z]*)>\(" \
                  r"<(?P<prep1>[A-Za-z]*)>(?P<loc1>[_A-Za-z]*)\|(?P<typ1>[_A-Za-z]*)\," \
                  r"<(?P<prep2>[A-Za-z]*)>(?P<loc2>[_A-Za-z]*)\|(?P<typ2>[_A-Za-z]*)" \
               r"\)" )

  set_placements, set_types = set(), set()
  lst_features = []
  for str_load_feat in lst_to_load:
    grps_single = regex_single.search(str_load_feat)
    grps_spec = regex_spec.search(str_load_feat)

    if grps_spec is not None:
      proc, prep1, loc1, typ1, prep2, loc2, typ2 = grps_spec.groups()
      feat = MultiFeature(proc, prep1, loc1, typ1, prep2, loc2, typ2)
      lst_features.append(feat)

      set_placements.add(loc1)
      set_types.add(typ1)
      set_placements.add(loc2)
      set_types.add(typ2)

    elif grps_single is not None:
      prep, loc, typ = grps_single.groups()
      feat = SingleFeature(prep, loc, typ)
      lst_features.append(feat)

      set_placements.add(loc)
      set_types.add(typ)



  lst_placement, lst_type = list(set_placements), list(set_types)
  map_groups = load_group_windows_new(lst_features, lbl_type, win_size=win_size, win_step=win_step, strategy=strategy,
                                  lst_placement=lst_placement, lst_type=lst_type)
  for test_grp_id in map_groups.keys():
    lst_x_train, lst_y_train = [], []
    for grp_id, lst_grp_data in map_groups.items():
      if grp_id == test_grp_id:
        lst_X_test, lst_y_test =  [vs[0] for vs in lst_grp_data], [vs[1] for vs in lst_grp_data]
        lst_exp_nms = [vs[2] for vs in lst_grp_data]
      else:
        lst_x_train.extend([vs[0] for vs in lst_grp_data])
        lst_y_train.extend([vs[1] for vs in lst_grp_data])
    yield test_grp_id, lst_x_train, lst_y_train, lst_X_test, lst_y_test, lst_exp_nms





if __name__ == '__main__':

  # gen_w_global(COH_SAVE_DIR)
  # exit(0)



  # get_exp_data("3", "paper_hard")
  # exit(0)


  """
    example <raw>Right_Wrist|Charge or <norm>Right_Wrist|acc

    For features that rely on corr / coh, the format is

    <proc>(<prep1>Location1|Type1,<prep2>Location2|Type2)
  """
  lst_features = ["<raw>Right_Wrist|Charge", "<diff>Right_Wrist|Charge", "<norm>Right_Wrist|acc",
                  "<corr_raw_relevant>(<norm>Right_Wrist|acc,<raw>Right_Wrist|Charge)"]
  for test_grp_id, lst_x_train, lst_y_train, lst_x_test, lst_y_test, lst_exp_nm in gen_leave_one_group_out_new_version(lst_features, "paper_hard"):
    print("----------------", "TEST", test_grp_id)
    for lst_x_day, lst_y_day in zip(lst_x_test, lst_y_test):
      print([len(x) for x in lst_x_day], len(lst_y_day))



  # gen_w_global(COH_SAVE_DIR)

  # w = load_multi_person_windows()
  # pass

  # gen_w_global(COH_SAVE_DIR)

  # run_coh_classic_test()

  # run_classic_test()

  # for test_grp_id, X_train, y_train, X_test, y_test in gen_leave_one_group_out():
  #   pass





