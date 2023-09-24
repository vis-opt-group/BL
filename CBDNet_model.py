import torch
import torch.nn as nn
import numpy as np


class CBDNet(nn.Module):
    def __init__(self):
        super(CBDNet, self).__init__()
        self.meta = {'mean': [0, 0, 0],
                     'std': [1, 1, 1],
                     'imageSize': [512, 512]}
        self.E01 = nn.Conv2d(3, 32, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.Er01 = nn.ReLU()
        self.E02 = nn.Conv2d(32, 32, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.Er02 = nn.ReLU()
        self.E03 = nn.Conv2d(32, 32, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.Er03 = nn.ReLU()
        self.E04 = nn.Conv2d(32, 32, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.Er04 = nn.ReLU()
        self.E05 = nn.Conv2d(32, 3, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.Er05 = nn.ReLU()
        self.DS01_layer00 = nn.Conv2d(6, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.DS01_relu00 = nn.ReLU()
        self.DS01_layer01 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.DS01_relu01 = nn.ReLU()
        self.DS01_layer02 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.DS01_relu02 = nn.ReLU()
        self.DS01_layer03 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.DS01_relu03 = nn.ReLU()
        self.DS02 = nn.Conv2d(64, 256, kernel_size=[2, 2], stride=(2, 2))
        self.DS02_layer00_cf = nn.Conv2d(256, 128, kernel_size=[1, 1], stride=(1, 1))
        self.DS02_layer00 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.DS02_relu00 = nn.ReLU()
        self.DS02_layer01 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.DS02_relu01 = nn.ReLU()
        self.DS02_layer02 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.DS02_relu02 = nn.ReLU()
        self.DS03 = nn.Conv2d(128, 512, kernel_size=[2, 2], stride=(2, 2))
        self.DS03_layer00_cf = nn.Conv2d(512, 256, kernel_size=[1, 1], stride=(1, 1))
        self.DS03_layer00 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.DS03_relu00 = nn.ReLU()
        self.DS03_layer01 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.DS03_relu01 = nn.ReLU()
        self.DS03_layer02 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.DS03_relu02 = nn.ReLU()
        self.UPS03_layer00 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.UPS03_relu00 = nn.ReLU()
        self.UPS03_layer01 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.UPS03_relu01 = nn.ReLU()
        self.UPS03_layer02 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.UPS03_relu02 = nn.ReLU()
        self.UPS03_layer03 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.UPS03_relu03 = nn.ReLU()
        self.USP02 = nn.ConvTranspose2d(512, 128, kernel_size=[2, 2], stride=(2, 2), bias=False)
        self.US02_layer00 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.US02_relu00 = nn.ReLU()
        self.US02_layer01 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.US02_relu01 = nn.ReLU()
        self.US02_layer02 = nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.US02_relu02 = nn.ReLU()
        self.USP01 = nn.ConvTranspose2d(256, 64, kernel_size=[2, 2], stride=(2, 2), bias=False)
        self.US01_layer00 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.US01_relu00 = nn.ReLU()
        self.US01_layer01 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.US01_relu01 = nn.ReLU()
        self.US01_layer02 = nn.Conv2d(64, 3, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
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
        predictionSigma = self.Er05(E_x05)
        input_all = torch.cat((input,predictionSigma), dim=1)
        DS01_x00 = self.DS01_layer00(input_all)
        DS01_r00 = self.DS01_relu00(DS01_x00)
        DS01_x01 = self.DS01_layer01(DS01_r00)
        DS01_r01 = self.DS01_relu01(DS01_x01)
        DS01_x02 = self.DS01_layer02(DS01_r01)
        DS01_r02 = self.DS01_relu02(DS01_x02)
        DS01_x03 = self.DS01_layer03(DS01_r02)
        DS01_r03 = self.DS01_relu03(DS01_x03)
        DS02_sx = self.DS02(DS01_r03)
        DS02_x00_cf = self.DS02_layer00_cf(DS02_sx)
        DS02_x00 = self.DS02_layer00(DS02_x00_cf)
        DS02_r00 = self.DS02_relu00(DS02_x00)
        DS02_x01 = self.DS02_layer01(DS02_r00)
        DS02_r01 = self.DS02_relu01(DS02_x01)
        DS02_x02 = self.DS02_layer02(DS02_r01)
        DS02_r02 = self.DS02_relu02(DS02_x02)
        DS03_sx = self.DS03(DS02_r02)
        DS03_x00_cf = self.DS03_layer00_cf(DS03_sx)
        DS03_x00 = self.DS03_layer00(DS03_x00_cf)
        DS03_r00 = self.DS03_relu00(DS03_x00)
        DS03_x01 = self.DS03_layer01(DS03_r00)
        DS03_r01 = self.DS03_relu01(DS03_x01)
        DS03_x02 = self.DS03_layer02(DS03_r01)
        DS03_r02 = self.DS03_relu02(DS03_x02)
        UPS03_x00 = self.UPS03_layer00(DS03_r02)
        UPS03_r00 = self.UPS03_relu00(UPS03_x00)
        UPS03_x01 = self.UPS03_layer01(UPS03_r00)
        UPS03_r01 = self.UPS03_relu01(UPS03_x01)
        UPS03_x02 = self.UPS03_layer02(UPS03_r01)
        UPS03_r02 = self.UPS03_relu02(UPS03_x02)
        UPS03_x03 = self.UPS03_layer03(UPS03_r02)
        UPS03_r03 = self.UPS03_relu03(UPS03_x03)
        USP02_x00 = self.USP02(UPS03_r03)
        US02_r00_sum = torch.add(USP02_x00, 1, DS02_r02)
        US02_x00 = self.US02_layer00(US02_r00_sum)
        US02_r00 = self.US02_relu00(US02_x00)
        US02_x01 = self.US02_layer01(US02_r00)
        US02_r01 = self.US02_relu01(US02_x01)
        US02_x02 = self.US02_layer02(US02_r01)
        US02_r02 = self.US02_relu02(US02_x02)
        USP01_x00 = self.USP01(US02_r02)
        US01_r00_sum = torch.add(USP01_x00, 1, DS01_r03)
        US01_x00 = self.US01_layer00(US01_r00_sum)
        US01_r00 = self.US01_relu00(US01_x00)
        US01_x01 = self.US01_layer01(US01_r00)
        US01_r01 = self.US01_relu01(US01_x01)
        US01_x02 = self.US01_layer02(US01_r01)
        prediction = torch.add(input, 1, US01_x02)
        return prediction


def CBDNet_pretrain(weights_path=None, **kwargs):
    """
    load imported model instance
    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = CBDNet()
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    # for v in model.parameters():
    #     v.require_grad = False
    return model
    

if __name__=='__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CBDNet_pretrain('model/CBDNet.pth').to(device)
    x = torch.randn((1,3,512,512)).to(device)
    y = model(x)
    print('Input  shape:', x.shape)
    print('Output shape:', y.shape)