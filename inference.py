import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import cv2
import numpy as np
import argparse

from copy import deepcopy

import torch
import torchvision.transforms as transforms

from crnn_tool import mapping_seq
from crnn_tool import get_seq2str
from models.crnn.crnn_model import CRNN
from models.tlgan.tlgan import Generator
from data.datasets import channel_first, channel_last


def make_axis(img):

    img = img.astype(np.uint8)
    contour2, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    axis_list = []
    for axis in contour2:
        if len(axis) > 10:
            axis = axis.squeeze()
            axis = axis.tolist()
            axis.sort()
            axis_list.append([axis[0], axis[-1]])

    return axis_list


def tlgan_infer(img, weight_path="./weights/generator.pt", img_col_size=60, img_row_size=256, left_margin=20):
    # img shape 256,256,3
    # image input for torch => channel first
    x = torch.tensor(channel_first(img))
    x = torch.unsqueeze(x, dim=0)

    netG = Generator()
    netG.load_state_dict(torch.load(weight_path, map_location="cpu"))

    x = netG(x)
    x = x.detach().numpy()[0]
    x = x > 0.2
    x = channel_last(x)

    plt.imshow(img)
    plt.show()

    axis = sorted(make_axis(x), key=lambda x : x[0][1])
    result = []
    for (min, max) in axis:
        # croppging
        tmp = img[min[1]-2:max[1]+20, min[0]:max[0], :]

        # check the amount of padding
        up_magin = (img_col_size - (max[1]-min[1]))//2 - 2
        down_magin = img_col_size - (max[1] - min[1]) - up_magin - 2
        left_magin = left_margin
        right_margin = img_row_size-left_magin-(max[0]-min[0])

        # append pad img
        tmp = np.pad(tmp, ((up_magin, down_magin), (left_magin, right_margin), (0, 0)), 'constant', constant_values=0)
        result.append(tmp)

    return result


def crnn_infer(axes, model_path="./weights/crnn.pth", visualize=False):
    fm.get_fontconfig_fonts()
    font_name = fm.FontProperties(fname="./data/fonts/H2GTRM.TTF").get_name()
    plt.rc('font', family=font_name)

    toTensor = transforms.ToTensor()
    labels = []

    if visualize:
        for img_num, img in enumerate(axes):
            vis_img = deepcopy(img)
            img = toTensor(img)
            img = img.view(1, *img.size())

            crnn = CRNN(64, 3, 1443, 256)
            crnn.load_state_dict(torch.load(model_path, map_location="cpu"))
            preds = crnn(img)
            predict = mapping_seq(preds)
            label = get_seq2str(predict[0])
            label = "".join(label)

            plt.imshow(vis_img)
            plt.title("predict label : " + "".join(label))
            plt.show()

            labels.append(label)

        return labels

    else:
        for img_num, img in enumerate(axes):
            img = toTensor(img)
            img = img.view(1, *img.size())

            crnn = CRNN(64, 3, 1443, 256)
            crnn.load_state_dict(torch.load(model_path, map_location="cpu"))
            preds = crnn(img)
            predict = mapping_seq(preds)
            label = get_seq2str(predict[0])
            label = "".join(label)

            labels.append(label)

        return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="inference script for the entire project")
    parser.add_argument("--tlgan_model_path", type=str, default="./weights/generator.pt")
    parser.add_argument("--crnn_model_path", type=str, default="./weights/crnn.pth")
    parser.add_argument("--path_to_img", type=str, default="./data/images/TLGAN/0.jpg")
    parser.add_argument("--visualize", type=bool, default=True)

    args = parser.parse_args()

    tlgan_model_path = args.tlgan_model_path
    crnn_model_path = args.crnn_model_path
    path_to_img = args.path_to_img
    vis = args.visualize

    img = cv2.imread(path_to_img).astype(np.float32)
    axes = tlgan_infer(img, weight_path=tlgan_model_path)
    result = crnn_infer(axes, model_path=crnn_model_path, visualize=vis)
