import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse
from PIL import Image, ImageFilter


def pad_tensor(input):
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 16

    if width_org % divide != 0 or height_org % divide != 0:

      width_res = width_org % divide
      height_res = height_org % divide
      if width_res != 0:
        width_div = divide - width_res
        pad_left = int(width_div / 2)
        pad_right = int(width_div - pad_left)
      else:
        pad_left = 0
        pad_right = 0

      if height_res != 0:
        height_div = divide - height_res
        pad_top = int(height_div / 2)
        pad_bottom = int(height_div - pad_top)
      else:
        pad_top = 0
        pad_bottom = 0

      padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
      input = padding(input)
    else:
      pad_left = 0
      pad_right = 0
      pad_top = 0
      pad_bottom = 0

    height, width = input.data.shape[2], input.data.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom


def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
  height, width = input.shape[2], input.shape[3]
  return input[:, :, pad_top: height - pad_bottom, pad_left: width - pad_right]

class noisy_Dataset(Dataset):
    def __init__(self, img_path_head):
        self.img_path_head      = img_path_head
        self.imagenames         = np.sort([x for x in os.listdir(img_path_head)])
        self.numOfImages        = len(self.imagenames)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),    #to [-1, 1]
        ])
    def __len__(self):
        return self.numOfImages
    def __getitem__(self, idx):
        img = Image.open(self.img_path_head+self.imagenames[idx]).convert('RGB')
        c_img = self.transform(img.copy())
        img.close()
        x_pad = 0; y_pad = 0;
        if(c_img.size(1)%4!=0):
            x_pad = (4-c_img.size(1)%4)
        if(c_img.size(2)%4!=0):
            y_pad = (4-c_img.size(2)%4)
        c_img = F.pad(c_img[None], (0, x_pad, 0, y_pad), 'reflect').squeeze()
        return c_img, x_pad, y_pad

def vis_img(img):
    img = img.data.cpu().numpy()
    img = np.clip(img, 0, 1)
    img = np.moveaxis(img, 0, 2)
    return img

        
def denoise(t_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = CBDNet_pretrain('model/CBDNet.pth').to(device)
    model.eval()

    for param in list(model.parameters()):
        param.requires_grad=False

    target_img = Variable(t_data.type(torch.FloatTensor)).to(device)
    size = target_img.size()
    # print('size',size)
    target_img, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(target_img)
    restored_img = model(target_img)
    restored_img = pad_tensor_back(restored_img, pad_left, pad_right, pad_top, pad_bottom)
    restored_img = restored_img.cpu()
    t_data = t_data.cpu()

    # display = np.concatenate([vis_img(t_data[0]), vis_img(restored_img[0])], axis=1)

    diff = (abs(t_data[0] - restored_img[0]).sum()).data.numpy() / (size[3]*size[1]*size[2])

    return diff

