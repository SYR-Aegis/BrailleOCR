from models.tlgan.tlgan import Generator
import data.datasets as datasets
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import cv2


def infer_model(img,weight_path = "./tlgan_model/generator.pt"):
    netG = Generator()
    netG.load_state_dict(torch.load(weight_path))

    x = netG(img)
    x = x.detach().numpy()[0]
    x = x> 0.2        
    x= np.transpose(x,(1,2,0))
    # plt.imshow(x,cmap='gray')
    # plt.show()

    
    axis = make_axis_cv(x)
    print(axis)
    return axis


def make_axis(img):
    img = np.squeeze(img)
    img = img.tolist()
    dp_list = [[False]* 256 for _ in range(256)]
    tmp_list = []
    flag = False
    
    def dfs(y,x):
        nonlocal dp_list
        nonlocal tmp_list

        dp_list[y][x] = True
        if img[y][x+1] == True:
            dfs(y,x+1)
            tmp_list[-1].append([y,x])
            dp_list[y][x] = True
        elif img[y+1][x] == True:
            dfs(y+1,x)
            tmp_list[-1].append([y,x])
        elif img[y][x-1] == True:
            dfs(y,x+1)
            tmp_list[-1].append([y,x])            
        else:
            return False


    margin = 3
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] == True and dp_list[i][j] == False:
                tmp_list.append([])
                dfs(i,j)
                tmp_list[-1].sort()
                if len(tmp_list[-1]):
                    min_point = tmp_list[-1][0]
                    max_point = tmp_list[-1][-1]
                    for alpha in range(min_point[0]-margin,max_point[0]+margin):
                        for beta in range(min_point[1]-margin,max_point[1]+margin):
                            dp_list[alpha][beta]= True 

    
    axis = []
    print(" -----------------")
    for i in tmp_list:
        if len(i)>30:
            i.sort()
            axis.append([i[0],i[-1]])
    
    return axis


def make_axis_cv(img):
    img = img.astype(np.uint8)
    contour2, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    axis_list = []
    for axis in contour2:
        if len(axis)>10:
            axis = axis.squeeze()
            axis = axis.tolist()
            axis.sort()
            axis_list.append([axis[0],axis[-1]])

    return axis_list


## test code

data = DataLoader(datasets.TLGAN_Dataset(path_to_csv="./data/TLGAN.csv", path_to_img="./data/images/TLGAN", path_to_GT="./data/images/gaussian_map"),
batch_size=1)
data = iter(data)
img = next(data)
img = next(data)[0]

infer_model(img)
