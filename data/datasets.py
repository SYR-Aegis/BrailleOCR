import cv2
import numpy as np
from PIL import Image
import os
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
    def __init__(self, path_to_csv="./CRNN.csv"):
        self.img_file=[]

        with open(path_to_csv, "r") as f:
            for line in f.readlines():
                self.img_file.append(line.strip().split(','))

    def __len__(self):
        return len(self.img_file)

    def __getitem__(self, idx):
        image = np.array(Image.open('./imgaes/',self.img_file[idx][0]))
        label = self.img_file[idx][1:]

        return image,label
