import torch.nn.functional as F
import torch.nn as nn
from unet_parts import *


class encoder(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(encoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x5

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.decoder_illu = decoder_illu(n_channels=3, n_classes=3, bilinear=True)
        self.decoder_denoise = decoder_denoise()


class decoder_illu(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(decoder_illu, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.up1 = Up1(1024, 512 // factor, bilinear)
        self.up2 = Up1(512, 256 // factor, bilinear)
        self.up3 = Up1(256, 128 // factor, bilinear)
        self.up4 = Up1(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
    def forward(self, x):
        x1 = self.up1(x)
        x2 = self.up2(x1)
        x3 = self.up3(x2)
        x4 = self.up4(x3)
        logits = self.outc(x4)
        logits = torch.sigmoid(logits)
        return logits

class decoder_denoise(nn.Module):
    def __init__(self):
        super(decoder_denoise, self).__init__()
        self.E01 = nn.Conv2d(3, 32, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.Er01 = nn.ReLU()
        self.E02 = nn.Conv2d(32, 32, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.Er02 = nn.ReLU()
        self.E03 = nn.Conv2d(32, 32, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.Er03 = nn.ReLU()
        self.E04 = nn.Conv2d(32, 32, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.Er04 = nn.ReLU()
        self.E05 = nn.Conv2d(32, 3, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.Er05 = nn.Sigmoid()
    def forward(self, input):
        E_x01 = self.E01(input)
        E_r01 = self.Er01(E_x01)
        E_x02 = self.E02(E_r01)
        E_r02 = self.Er02(E_x02)
        E_x03 = self.E03(E_r02)
        E_r03 = self.Er03(E_x03)
        E_x04 = self.E04(E_r03)
        E_r04 = self.Er04(E_x04)
        E_x05 = self.E05(E_r04)
        noise = self.Er05(E_x05)
        return noise



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
