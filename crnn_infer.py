from PIL import Image
import torchvision.transforms as transforms
import torch
from crnn_tool import mapping_seq
from crnn_tool import get_seq2str
from models.crnn.crnn_model import CRNN
from tlgan_infer import get_infer_image
import matplotlib.pyplot as plt
import cv2
import copy

def crnn_infer(image_path, model_path):
    toTensor = transforms.ToTensor()

    image = Image.open(image_path)
    image = toTensor(image)
    image = image.view(1, *image.size())

    device = torch.device('cpu')
    crnn=CRNN(64, 3, 1443, 256)
    crnn.load_state_dict(torch.load(model_path,  map_location=device))

    preds = crnn(image)
    predict = mapping_seq(preds)
    label= get_seq2str(predict[0])
    
    print("\n"+str(image_path) + ">>\n\npredicted sentence : " + "".join(label))

def tlgan_infer(model_path):
    toTensor = transforms.ToTensor()
    plt.rc('font', family='NanumGothic')

    source_images=get_infer_image()

    for img_num, image in enumerate(source_images):
        np_img = copy.deepcopy(image)

        image = toTensor(image)
        image = image.view(1, *image.size())

        device = torch.device('cpu')
        crnn=CRNN(64, 3, 1443, 256)
        crnn.load_state_dict(torch.load(model_path, map_location=device))

        preds = crnn(image)
        predict = mapping_seq(preds)
        label = get_seq2str(predict[0])
        label="".join(label)

        plt.imshow(np_img)
        plt.title("predict label : "+"".join(label))
        plt.show()

#crnn_infer("data/images/CRNN/117.jpg", "crnn_model/model_for_CRNNdata.pth")
tlgan_infer("crnn_model/model_for_TLGANdata.pth")

