from models.tlgan.tlgan import Generator
import data.datasets as datasets
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import cv2
from tqdm import tqdm

def infer_model(img,weight_path = "./tlgan_model/generator.pt",img_col_size=60,img_row_size=256,left_margin=20):
    # img shape 256,256,3
    # image input for torch => channel first

    #fig = plt.figure()
    #ax1 = fig.add_subplot(1, 2, 1)
    #ax2 = fig.add_subplot(1, 2, 2)
    #ax1.imshow(img)
    #ax1.set_title("original")

    x = np.transpose(img,(2,0,1))
    x = torch.unsqueeze(x,dim=0)
    device = torch.device('cpu')
    netG = Generator()
    netG.load_state_dict(torch.load(weight_path, map_location=device))

    x = netG(x)
    x = x.detach().numpy()[0]
    x = x> 0.2        
    x= np.transpose(x,(1,2,0))
    # plt.imshow(x,cmap='gray')
    # plt.show()

    #ax2.imshow(x)
    #ax2.set_title("tlgan output")
    #plt.subplots_adjust(wspace=0.3)
    #plt.show()

    axis = sorted(make_axis_cv(x), key=lambda x : x[0][1])
    result = []
    for (min,max) in axis:
        tmp = img[min[1] -2 :max[1] + 2,min[0]:max[0],:]
        # croppging
        up_magin = (img_col_size - (max[1]-min[1]))//2 - 2
        down_magin = img_col_size - (max[1] - min[1]) - up_magin - 2
        left_magin = left_margin
        right_margin = img_row_size-left_magin-(max[0]-min[0])
        # margin check
        tmp = np.pad(tmp,((up_magin,down_magin),(left_magin,right_margin), (0, 0)),'constant', constant_values=0)
        result.append(tmp)
        # append pad img

        # plt.imshow(result[-1],cmap='gray')
        # plt.show()
    return result


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
def get_infer_image():
    data = DataLoader(datasets.TLGAN_Dataset(path_to_csv="./data/TLGAN.csv", path_to_img="./data/images/TLGAN", path_to_GT="./data/images/gaussian_map"),
    batch_size=1)
    data = iter(data)
    img = next(data)
    img = next(data)[0]
    img = img[0]
    img= np.transpose(img,(1,2,0))

    return infer_model(img)

def make_infer_image():
    excluse_list=[]
    excluse_count=0

    data = DataLoader(datasets.TLGAN_Dataset(path_to_csv="./data/TLGAN.csv", path_to_img="./data/images/TLGAN", path_to_GT="./data/images/gaussian_map"),
    batch_size=1)
    img_num=0
    for i, (xx, yy) in tqdm(enumerate(data), total=len(data)):
        img = xx[0]
        img= np.transpose(img,(1,2,0))
        source_images=infer_model(img)
        for num, image in enumerate(source_images):
            image=image*255
            if img_num in excluse_list:
                excluse_count+=1
                break
            cv2.imwrite("data/images/CRNN/"+str(img_num-excluse_count)+".jpg", image)
            img_num+=1

#make_infer_image()