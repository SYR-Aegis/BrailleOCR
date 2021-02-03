import cv2
import numpy as np
from PIL import Image
import os
from torch.utils.data.dataset import Dataset
import json
import torch


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
    def __init__(self, path_to_csv = "./CRNN.csv", dict_dir="dict_file.txt", root = './images/CRNN', ):
        self.img_file=[]
        self.label = []
        self.root = root

        with open(path_to_csv, "r",encoding = "utf-8") as f:
            for line in f.readlines():
                self.img_file.append(line.strip().split(','))

        with open(dict_dir, "r",encoding = "utf-8") as fil:
            dict = fil.read()

        self.label_dict = json.loads(dict)

    def __len__(self):
        return len(self.img_file)

    def __getitem__(self, idx):
        image = np.array(Image.open(os.path.join(self.root,self.img_file[idx][0]))).astype(np.int32)
        image = torch.tensor(image/np.max(image),dtype=torch.float32)

        for key in self.img_file[idx][1:]:
            self.label.append(self.label_dict[key])

        return image, self.label
