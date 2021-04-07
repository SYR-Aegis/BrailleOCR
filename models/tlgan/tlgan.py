from torch import nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.img_1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.Tanh(),
        )
        self.img_2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.Tanh(),
        )
        self.img_3 = nn.Sequential(
            nn.Conv2d(64, 128, 5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.Tanh(),
        )
        self.img_4 = nn.Sequential(
            nn.Conv2d(128, 256, 5, stride=1, padding=2),
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

        self.img_5 = nn.Sequential(
            nn.Conv2d(256, 256, 5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.Tanh(),
        )
        self.img_6 = nn.Sequential(
            nn.Conv2d(256, 256, 5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.Tanh(),
        )

        self.img_7 = nn.Sequential(
            nn.Conv2d(256, 256, 5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.Tanh(),
        )

        self.upsample_1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
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
            nn.Conv2d(32, 1, 5, padding=2),
            nn.ELU(),
        )

    def forward(self, x):
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
    def __init__(self, img_size):
        self.img_size = img_size
        super(Discriminator, self).__init__()
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

        self.post_process = nn.Sequential(
            nn.Linear(128*self.img_size//64, 1),
            nn.Tanh(),
        )

    def forward(self, input):
        x = self.disc(input)
        x = x.view(-1, 128*self.img_size//64)
        x = self.post_process(x)
        return x
