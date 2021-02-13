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

import data.datasets as datasets
from models.tlgan.tlgan import Generator,Discriminator

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
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--epoch", type=int , default=1000)
    parser.add_argument("--load_model", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=8)
    
    args = parser.parse_args()
    sys.path.append(args.data_path)

    

    data = DataLoader(datasets.TLGAN_Dataset(path_to_csv="./data/TLGAN.csv", path_to_img="./data/images/TLGAN", path_to_GT="./data/images/gaussian_map"),
    batch_size=args.batch_size)
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



