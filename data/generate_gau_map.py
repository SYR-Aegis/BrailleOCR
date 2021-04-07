import numpy as np
import argparse
import os
import csv
import math
import glob


def gaussian_map(pos_1, pos_2, raw=np.zeros((1, 256, 256))):
    y_width = pos_2[0]-pos_1[0]
    x_width = pos_2[1]-pos_1[1]
    y_mean = (pos_1[0] + pos_2[0])//2
    for i in range(pos_1[0], pos_2[0]):
        raw[:, i, pos_1[1]:pos_2[1]] = np.ones((1, 1, x_width))*(1/(2*math.pi*y_width)) * math.exp((-10*(i-y_mean)**2)/(2 * y_width**2))*100

    return raw


def draw_map(pos):
    gau_map = np.zeros((1, 256, 256))
    # background zero image

    for x_min, x_max, y_min, y_max in pos:
        gau_map = gaussian_map((y_min, x_min), (y_max, x_max), raw=gau_map)
    # draw map in background

    return gau_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert csv to image gaussian")
    parser.add_argument("--csv_file_path", type=str, default="./")
    parser.add_argument("--csv_file_name", type=str, default="TLGAN.csv")
    parser.add_argument("--map_save_path", type=str, default="./images/gaussian_map/")

    args = parser.parse_args()

    if os.path.exists(args.map_save_path):
        for file in glob.glob(args.map_save_path+"/*"):
            os.remove(file)

    with open(os.path.join(args.csv_file_path, args.csv_file_name), 'r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            name = line[0]
            # name split
            line = line[1:]
            line = list(map(int, line))
            # csv str to int
            line = [line[i:i+4] for i in range(0, len(line), 4)]
            # split 4 x_min,x_max,y_min,y_max 
            img = draw_map(line)

            if not os.path.exists(args.map_save_path):
                os.mkdir(args.map_save_path)
            np.save(os.path.join(args.map_save_path, name.split('.')[0]+".npy"), img*255)
