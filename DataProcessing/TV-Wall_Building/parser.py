import os
import pandas as pd
from time_converter import video_time_to_index


def read_files(data_folder):
    files = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f)) and len(f.split('_')) > 2]
    print(files)
    merged_data = None
    for f in files:
        data_type = f.split('.')[0].split("_")[2]
        position = f.split('.')[0].split("_")[1]

        if data_type == 'BNO055':
            names = ["imu_sn", "imu_t", "imu_sn2", "a_x", "a_y", "a_z", "g_x", "g_y", "g_z", "m_x", "m_y", "m_z", "roll", "pitch", "yaw"]
        elif data_type == 'Charge':
            names = ["c_t", "c_sn", "charge"]
        else:
            names = ["f_t", "f_sn", "frequency"]

        person = f.split('.')[0].split("_")[0]

        names = [person + "_" + position + "_" + n for n in names]
        #print(names)
        # print(f)
        data = pd.read_csv(os.path.join(data_folder, f), sep=' ', index_col=0, names=names)

        if merged_data is None:
            merged_data = data
        else:
            # print(data.index)
            merged_data = merged_data.join(data) # = pd.concat([merged_data, data], axis=1, join='inner').sort_index() #, join_axes=[merged_data.index])
        #print(merged_data.index)
    merged_data.dropna(inplace=True)
    return merged_data


def read_labels(label_file: str):
    sampling_rate = 10.0
    l = pd.read_csv(label_file, sep='\t')
    l.start = l.start.apply(video_time_to_index, args=(sampling_rate,))
    l.end = l.end.apply(video_time_to_index, args=(sampling_rate,))
    return l



if __name__ == '__main__':
    read_labels('exp1_labels.csv')
    exit()

    data_path = '/Users/hevesi/Downloads/Data_4/'
    d = read_files(data_path)
    d.to_csv('cap_data_1.csv')
    print(d)
    import matplotlib.pyplot as plt


    plt.plot(d["P2_Wrist_a_x"])
    plt.plot(d["P2_Wrist_a_y"])
    plt.plot(d["P2_Wrist_a_z"])
    plt.show()



