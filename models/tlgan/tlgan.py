import sys
import numpy as np
import argparse
import os
import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim

import torchvision.datasets
import torchvision.transforms as transforms
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import datasets

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.img_1= nn.Sequential(
            nn.Conv2d(3,32,5,stride = 1,padding=2),
            nn.BatchNorm2d(32),
            nn.Tanh(),
        )
        self.img_2= nn.Sequential(
            nn.Conv2d(32,64,5,stride = 1,padding=2),
            nn.BatchNorm2d(64),
            nn.Tanh(),
        )
        self.img_3= nn.Sequential(
            nn.Conv2d(64,128,5,stride = 1,padding=2),
            nn.BatchNorm2d(128),
            nn.Tanh(),
        )
        self.img_4= nn.Sequential(
            nn.Conv2d(128,256,5,stride = 1,padding=2),
            nn.BatchNorm2d(256),
            nn.Tanh(),
        )
        self.downsample_1 = nn.Sequential(
            nn.Conv2d(256, 256, 5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.Tanh(),
        )

        self.downsample_2 = nn.Sequential(
            nn.Conv2d(256, 256, 5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.Tanh(),
        )

        self.downsample_3 = nn.Sequential(
            nn.Conv2d(256, 256, 5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.Tanh(),
        )

        self.img_5= nn.Sequential(
            nn.Conv2d(256,256,5,stride = 1,padding=2),
            nn.BatchNorm2d(256),
            nn.Tanh(),
        )
        self.img_6= nn.Sequential(
            nn.Conv2d(256,256,5,stride = 1,padding=2),
            nn.BatchNorm2d(256),
            nn.Tanh(),
        )

        self.img_7= nn.Sequential(
            nn.Conv2d(256,256,5,stride = 1,padding=2),
            nn.BatchNorm2d(256),
            nn.Tanh(),
        )


        self.upsample_1 = nn.Sequential(
            nn.ConvTranspose2d(256,128,2,stride=2),
            nn.BatchNorm2d(128),
            nn.Tanh(),
        )

        self.upsample_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.Tanh(),
        )


        self.upsample_3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.Tanh(),
        )

        self.postprocess = nn.Sequential(
            nn.Conv2d(32,1,5,padding=2),
            nn.ELU(),
        )


    def forward(self,x):
        x = self.img_1(x)
        x = self.img_2(x)
        x = self.img_3(x)
        x = self.img_4(x)
        x = self.downsample_1(x)
        x = self.downsample_2(x)
        x = self.downsample_3(x)
        x = self.img_5(x)
        x = self.img_6(x)
        x = self.img_7(x)
        x = self.upsample_1(x)
        x = self.upsample_2(x)
        x = self.upsample_3(x)
        x = self.postprocess(x)
        return x


class Discriminator(nn.Module):
    def __init__(self,img_size):
        self.img_size= img_size
        super(Discriminator,self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(1, 32, 5, 2, padding=2),
            nn.Tanh(),
            nn.Conv2d(32, 64, 5, 2, padding=2),
            nn.Tanh(),
            # 1/4
            nn.Conv2d(64, 64, 5, 1, padding=2),
            nn.Tanh(),
            nn.Conv2d(64, 128, 2, padding=2),
            nn.Tanh(),
            nn.Conv2d(128, 256, 2, padding=2),
            nn.Tanh(),
            #1/16
            nn.Conv2d(256, 256, 1, padding=2),
            nn.Tanh(),
            nn.Conv2d(256, 256, 2, padding=2),
            nn.Tanh(),
            nn.Conv2d(256, 128, 2, padding=2),
            nn.Tanh(),
            #1/64
        )
        #3,32,32 => 4*self.DIM,4,4

        self.post_process =nn.Sequential(
            nn.Linear(128*self.img_size//64,1),
            nn.Tanh(),
        )

    def forward(self,input):
        x = self.disc(input)
        x = x.view(-1,128*self.img_size//64)
        x = self.post_process(x)
        return x


class tlgan:
    def __init__(self,batch_size,save_dir,target_img = (3,256,256),is_cuda = True, save_path = "./tlgan_model/"):
        self.batch_size = batch_size
        self.target_img = target_img
        self.save_dir = save_dir
        self.is_cuda = is_cuda
        self.netG = Generator()
        self.netD = Discriminator(target_img[-1])
        self.save_path = save_path

        if self.is_cuda:
            self.netG.cuda()
            self.netD.cuda()
        print(self.netG,self.netD)

        self.compile()

    def gradient_penalty(self,real,fake):
        alpha = torch.rand(self.batch_size,1)
        target_size = 1*self.target_img[1]*self.target_img[2]
        alpha = alpha.expand(self.batch_size,target_size).view(self.batch_size,1,self.target_img[1], self.target_img[2])
        alpha = alpha.cuda() if self.is_cuda else alpha
        interpolates = alpha * real + (1-alpha) * fake

        interpolates = interpolates.cuda() if self.is_cuda else interpolates

        interpolates = autograd.Variable(interpolates,requires_grad=True)

        val_inter = self.netD(interpolates)

        grad = autograd.grad(val_inter,interpolates, grad_outputs=torch.ones(val_inter.size()).cuda() if self.is_cuda else torch.ones(
                                  val_inter.size()),create_graph=True,retain_graph=True,only_inputs=True)[0]

        grad = grad.view(grad.size(0),-1)

        grad = ((grad.norm(2,dim=1)-1)**2).mean() * 10

        return grad

    def compile(self):

        self.optimizer_disc = optim.Adam(self.netD.parameters(),lr = 1e-4,betas=(0.5,0.9))
        self.optimizer_gen = optim.Adam(self.netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

    def train_disc(self,img,real):

        self.netD.zero_grad()
        real = autograd.Variable(real)
        real = real.cuda() if self.is_cuda else real
        real_val = self.netD(real)
        real_val = real_val.mean()
        real_val.backward(-1 * torch.tensor(1, dtype=torch.float))

        img = autograd.Variable(img)
        img = img.cuda() if self.is_cuda else img
        fake = self.netG(img)
        fake = autograd.Variable(fake)
        fake = fake.cuda() if self.is_cuda else fake

        fake_val = self.netD(fake)
        fake_val = fake_val.mean()
        fake_val.backward(torch.tensor(1, dtype=torch.float))

        gp = self.gradient_penalty(real,fake)
        gp.backward()
        self.optimizer_disc.step()
        self.d_cost = real_val - fake_val + gp

    def train_gen(self,img,real,show= False):
        self.netG.zero_grad()
        img = autograd.Variable(img)
        img = img.cuda() if self.is_cuda else img
        fake = self.netG(img)

        real = autograd.Variable(real)
        real = real.cuda() if self.is_cuda else real

        criterion = torch.nn.MSELoss()
        l2_loss = criterion(fake,real)
        l2_loss.backward(retain_graph=True)

        fake_val = self.netD(fake)
        fake_val = fake_val.mean()
        fake_val.backward(-1*torch.tensor(1, dtype=torch.float))
        self.optimizer_gen.step()

        self.g_cost = fake_val
        if show == True:
            fake = fake.detach().cpu().numpy()
            show_fake= np.transpose(fake[0],(1,2,0))
            show_fake = np.squeeze(show_fake)
            plt.imshow(show_fake,cmap='gray')
            plt.show()

    def train(self,data,epochs):

        data_iter = iter(data)
        for epoch in range(epochs):
            try:
                img,real = next(data_iter)
                if img.shape[0] != self.batch_size:
                    img, real = next(data_iter)
            except StopIteration:
                data_iter = iter(data)
                img, real = next(data_iter)


            for p in self.netD.parameters():  # reset requires_grad
                p.requires_grad = True
            for p in self.netG.parameters():  # reset requires_grad
                p.requires_grad = False
            for _ in range(5):

                self.train_disc(img,real)

            for p in self.netD.parameters():  # reset requires_grad
                p.requires_grad = False
            for p in self.netG.parameters():  # reset requires_grad
                p.requires_grad = True

            if epoch%100 ==0 :
                self.train_gen(img,real,True)
                self.save_model()
            else:
                self.train_gen(img,real,False)

            print(epoch,self.d_cost,self.g_cost)

    def save_model(self):
        torch.save(self.netG.state_dict(),self.save_path + "generator.pt")
        torch.save(self.netD.state_dict(), self.save_path + "discriminator.pt")

    def load_model(self):
        self.netG.load_state_dict(torch.load(self.save_path + "generator.pt"))
        self.netD.load_state_dict(torch.load(self.save_path + "discriminator.pt"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tlgan")
    parser.add_argument("--model_path", type=str, default="./tlgan_model/")
    parser.add_argument("--data_path", type=str, default="../../data")
    parser.add_argument("--epoch", type=int , default=1000)
    parser.add_argument("--load_model", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=8)
    
    args = parser.parse_args()
    sys.path.append(args.data_path)
    data = DataLoader(datasets.TLGAN_Dataset(),batch_size=args.batch_size)
    tl_handler = tlgan(batch_size = args.batch_size,
                   save_dir="./data/",
                   target_img = (3,256,256),
                   is_cuda = True,
                    save_path=args.model_path)

    if os.path.exists(args.model_path) == False:
        os.makedirs(args.model_path)
    if args.load_model == True:
        tl_handler.load_model()

    tl_handler.train(data,args.epoch)



