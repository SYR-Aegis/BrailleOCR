import cv2
import numpy as np
import os
import torch

from torch.utils.data.dataset import Dataset


class TLGAN_Dataset(Dataset):
    def __init__(self, path_to_csv="./TLGAN.csv", path_to_img="./images/TLGAN", path_to_GT="./images/gaussian_map"):
        imgs = []
        GTs = []

        with open(path_to_csv, 'r') as f:
            lines = f.read().split('\n')[:-1]

            for line in lines:
                line = line.split(',')
                img = cv2.imread(os.path.join(path_to_img, line[0])).astype(np.float32)
                GT = cv2.imread(os.path.join(path_to_GT, line[0])).astype(np.float32)

                imgs.append(img)
                GTs.append(GT)


        imgs = np.array(imgs).astype(np.float32)
        GTs = np.array(GTs).astype(np.float32)

        self.imgs = imgs
        self.GTs = GTs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = torch.tensor(self.imgs[idx])
        gt = torch.tensor(self.GTs[idx])

        return img, gt


class CRNN_Dataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
