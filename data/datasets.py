import cv2
import numpy as np
import os
import torch
from PIL import Image
import json

from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms


def channel_first(image):
    return np.transpose(image, (2, 0, 1))


def channel_last(image):
    return np.transpose(image, (1, 2, 0))


class TLGAN_Dataset(Dataset):
    def __init__(self, path_to_csv="./TLGAN.csv", path_to_img="./images/TLGAN", path_to_GT="./images/gaussian_map"):
        imgs = []
        GTs = []

        with open(path_to_csv, 'r') as f:
            lines = f.read().split('\n')[:-1]

            for line in lines:
                line = line.split(',')
                img = cv2.imread(os.path.join(path_to_img, line[0]))
                img = channel_first(img)

                GT = np.load(os.path.join(path_to_GT, line[0].split('.')[0]+".npy"))

                imgs.append(img)
                GTs.append(GT)

        imgs = np.array(imgs).astype(np.int32)
        GTs = np.array(GTs).astype(np.int32)

        self.imgs = imgs
        self.GTs = GTs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = torch.tensor(self.imgs[idx]/255, dtype=torch.float32)
        gt = torch.tensor(self.GTs[idx]/255, dtype=torch.float32)

        return img, gt


class CRNN_Dataset(Dataset):
    def __init__(self, path_to_csv="data/CRNN.csv", dict_dir="data/str2int.txt", root='data/images/CRNN', ):
        self.img_file = []
        self.root = root
        self.toTensor = transforms.ToTensor()

        with open(path_to_csv, "r", encoding="utf-8") as f:
            for line in f.readlines():
                self.img_file.append(line.strip().split(','))

        with open(dict_dir, "r", encoding="utf-8") as fil:
            dict_ = fil.read()

        self.label_dict = json.loads(dict_)

    def __len__(self):
        return len(self.img_file)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root, self.img_file[idx][0]))
        image = np.array(image).astype(np.float32)/255
        image = self.toTensor(image)

        label = []
        for key in self.img_file[idx][1:]:
            label.append(str(self.label_dict[key]))

        label = ",".join(label)
        return image, label
