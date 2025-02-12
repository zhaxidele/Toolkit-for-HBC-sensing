import numpy as np
import os
from iteration_utilities import flatten
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from scipy.stats import mode

from dataloaders.dataloader_base import BASE_DATA

class RecGym_DATA(BASE_DATA):
    def __init__(self, root_path, window_size, overlap_size, transform=None):
        """
        root_path : Root directory of the data set
        window_size : Size of the window in seconds
        overlap_size : Size of the overlap in seconds
        transform : Optional transform to be applied on a sample
        """
        self.root_path = root_path
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.transform = transform
        self.used_cols = ["A_x", "A_y", "A_z", "G_x", "G_y", "G_z", "C_1", "Workout"]
        self.label_map = {
            "Adductor": 1, "ArmCurl": 2, "BenchPress": 3, "LegCurl": 4, "LegPress": 5, "Null": 6, 
            "Riding": 7, "RopeSkipping": 8, "Running": 9, "Squat": 10, "StairClimber": 11, "Walking": 12
        }
        self.labelToId = {v: k for v, k in self.label_map.items()}
        self.data_x, self.data_y = self.load_all_the_data()

    def load_all_the_data(self):
        print(" ----------------------- load all the data -------------------")
        df_all = pd.read_csv(os.path.join(self.root_path, "RecGym.csv"))
        df_all = df_all[self.used_cols]
        df_all.dropna(inplace=True)
        df_all["Workout"] = df_all["Workout"].map(self.labelToId)
        df_all = df_all.set_index('Workout')
        data_y = df_all.index.values
        data_x = df_all.values

        # Segment the data into windows
        window_size_samples = int(self.window_size * 20)  # 20Hz sampling rate
        overlap_size_samples = int(self.overlap_size * 20)
        step_size = window_size_samples - overlap_size_samples

        segmented_data_x = []
        segmented_data_y = []

        for start in range(0, len(data_x) - window_size_samples + 1, step_size):
            end = start + window_size_samples
            window_x = data_x[start:end]
            window_y = data_y[start:end]
            segmented_data_x.append(window_x)
            segmented_data_y.append(mode(window_y)[0])  # Most common label in the window

        self.data_x = np.array(segmented_data_x)
        self.data_y = np.array(segmented_data_y)

        print(self.data_x.shape)
        print(self.data_y.shape)
        return self.data_x, self.data_y

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        sample = self.data_x[idx]
        label = self.data_y[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

def load_data(root_path, batch_size, window_size, overlap_size):
    dataset = RecGym_DATA(root_path, window_size, overlap_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader


if __name__ == '__main__':
    root_path = "datasets"
    batch_size = 64
    window_size = 4  # Window size in seconds
    overlap_size = 2  # Overlap size in seconds
    dataloader = load_data(root_path, batch_size, window_size, overlap_size)

    for i, (samples, labels) in enumerate(dataloader):
        print(f"Batch {i+1}")
        print("Samples:", samples)
        print("Labels:", labels)
        if i == 0:  # Print only the first batch for brevity
            break