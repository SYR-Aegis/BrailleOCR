import cv2
import numpy as np

from torch.utils.data.dataset import Dataset


class TLGAN_Dataset(Dataset):
    def __init__(self, path_to_csv="./TLGAN.csv"):
        imgs = np.zeros((0, 3, 256, 256))
        labels = np.zeros((0))

        with open(path_to_csv, 'r') as f:
            pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class CRNN_Dataset(Dataset):
    def __init__(self):
        pass aaaa

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass