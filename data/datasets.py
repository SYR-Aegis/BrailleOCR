import cv2
import numpy as np
import os
import torch

from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt


class TLGAN_Dataset(Dataset):
    def __init__(self, path_to_csv="./TLGAN.csv", path_to_img="./images/TLGAN", path_to_GT="./images/gaussian_map"):
        imgs = []
        GTs = []

        with open(path_to_csv, 'r') as f:
            lines = f.read().split('\n')[:-1]

            for line in lines:
                line = line.split(',')
                img = cv2.imread(os.path.join(path_to_img, line[0]))
                GT = cv2.imread(os.path.join(path_to_GT, line[0]))

                imgs.append(img)
                GTs.append(GT)


        imgs = np.array(imgs).astype(np.int32)
        GTs = np.array(GTs).astype(np.int32)

        self.imgs = imgs
        self.GTs = GTs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = torch.tensor(self.imgs[idx]/np.max(self.imgs[idx]), dtype=torch.float32)
        gt = torch.tensor(self.GTs[idx]/np.max(self.GTs[idx]), dtype=torch.float32)

        return img, gt


class CRNN_Dataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
