import matplotlib.pyplot as plt
import os
from parser import read_files, read_labels

if __name__ == '__main__':
    root = '/Users/hevesi/Downloads/Data_4/'
    d = read_files(root)
    l = read_labels('../exp1_labels.csv')
    print(d.columns.values)
    sample_rate = 10.0
    offset = 30
    for _, gt in l.iterrows():
        foldername = gt.activity + "_" + str(gt.start) + "-" + str(gt.end) + "_" + gt.person.replace(', ', '-')
        current_data =  d.loc[(gt.start -offset):(gt.end + offset)]

        output_dir = os.path.join('tmp', foldername)
        os.makedirs(output_dir, exist_ok=True)


        for p in ["P1", "P2", "P3"]:
            for loc in ["Wrist", "Calf"]:
                # plot acc
                fig = plt.figure(figsize=(12, 4))
                plt.plot(current_data.index / sample_rate, current_data['_'.join([p, loc, 'a_x'])], label="ax")
                plt.plot(current_data.index / sample_rate, current_data['_'.join([p, loc, 'a_y'])], label="ay")
                plt.plot(current_data.index / sample_rate, current_data['_'.join([p, loc, 'a_z'])], label="az")
                plt.axvspan(gt.start/sample_rate, gt.end/ sample_rate, facecolor='g', alpha=0.2)
                plt.title(p + ' accelerometer on ' + loc)
                plt.ylabel('acc [m/s^2]')
                plt.xlabel('time [s]')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, p + '_' + loc + '_accel.png'))
                plt.close(fig)


                # plot gyro
                fig = plt.figure(figsize=(12, 4))
                plt.plot(current_data.index / sample_rate, current_data['_'.join([p, loc, 'g_x'])], label="gx")
                plt.plot(current_data.index / sample_rate, current_data['_'.join([p, loc, 'g_y'])], label="gy")
                plt.plot(current_data.index / sample_rate, current_data['_'.join([p, loc, 'g_z'])], label="gz")
                plt.axvspan(gt.start/sample_rate, gt.end/ sample_rate, facecolor='g', alpha=0.2)
                plt.title(p + ' gyroscope on ' + loc)
                plt.ylabel('gyro values')
                plt.xlabel('time [s]')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, p + '_' + loc + '_gyro.png'))
                plt.close(fig)

                # plot charge
                fig = plt.figure(figsize=(12, 4))
                plt.plot(current_data.index / sample_rate, current_data['_'.join([p, loc, 'c'])], label="charge")
                plt.axvspan(gt.start/sample_rate, gt.end/ sample_rate, facecolor='g', alpha=0.2)
                plt.title(p + ' charge on ' + loc)
                plt.ylabel('charge values')
                plt.xlabel('time [s]')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, p + '_' + loc + '_charge.png'))
                plt.close(fig)

                # plot frequ
                try:
                    fig = plt.figure(figsize=(12, 4))
                    plt.plot(current_data.index / sample_rate, current_data['_'.join([p, loc, 'f'])], label="frequency")
                    plt.axvspan(gt.start/sample_rate, gt.end/ sample_rate, facecolor='g', alpha=0.2)
                    plt.title(p + ' frequency on ' + loc)
                    plt.ylabel('frequency values')
                    plt.xlabel('time [s]')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, p + '_' + loc + '_frequ.png'))
                    plt.close(fig)
                except Exception as e:
                    print(e)

