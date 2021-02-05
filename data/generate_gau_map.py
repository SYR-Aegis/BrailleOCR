import numpy as np
import argparse
import os
import torch
import csv
import math
import matplotlib.pyplot as plt
import cv2

def gaussian_map(pos_1,pos_2,shape=(3,256,256),raw = np.zeros((1,256,256))):
    y_width = pos_2[0]-pos_1[0]
    x_width = pos_2[1]-pos_1[1]
    y_mean = (pos_1[0] + pos_2[0])//2
    for i in range(pos_1[0],pos_2[0]):
        raw[:,i,pos_1[1]:pos_2[1]] = np.ones((1,1,x_width))*(1/(2*math.pi*y_width)) * math.exp((-10*(i-y_mean)**2)/(2* y_width**2))*100

    #plt.imshow(np.transpose(raw, (1, 2, 0)),cmap='gray')
    #plt.show()
    #raw = torch.Tensor(raw)
    return raw

def draw_map(pos):
    img = np.zeros((1,256,256))
    # background zero image

    for x_min,x_max,y_min,y_max in pos:
        img = gaussian_map((y_min,x_min),(y_max,x_max),shape=(3,256,256),raw =  img)
    # draw map in background

    # plt.imshow(np.transpose(img, (1, 2, 0)),cmap='gray')
    # plt.show()
    # testing

    return img



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert csv to image gaussian")
    parser.add_argument("--csv_file_path", type=str, default="./")
    parser.add_argument("--csv_file_name", type=str, default="TLGAN.csv")
    parser.add_argument("--map_save_path", type=str, default="./images/gaussian_map/")
    parser.add_argument("--n_text", type=int, default=3)

    args = parser.parse_args()
    n_text = args.n_text

    with open(os.path.join(args.csv_file_path, args.csv_file_name), 'r',encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            name = line[0]
            #name split
            line = line[1:]
            line = list(map(int,line))
            # csv str to int
            line = [line[i:i+4] for i in range(0, len(line), 4)]
            # split 4 x_min,x_max,y_min,y_max 
            img =draw_map(line)

            img = np.transpose(img, (1, 2, 0))
            # for channel last save mode    

            if not os.path.exists(args.map_save_path):
                os.mkdir(args.map_save_path)
            cv2.imwrite(os.path.join(args.map_save_path,name), img*255)
            # for un-regularized img


