import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import networks
from unet_model import encoder
from unet_model import decoder
from Test_Patches import denoise
import numpy as np
from CBDNet_model import CBDNet_pretrain





class Network(nn.Module):
  def __init__(self, criterion, denoise_model):
    super(Network, self).__init__()
    self.encoder = encoder(n_channels=3, n_classes=3, bilinear=True)
    self.decoder = decoder()
    self.denoise_detection = denoise_model
    self._criterion = criterion

  def new(self):
    model_new = Network(self._criterion, self.denoise_detection).cuda()
    for x, y in zip(model_new.encoder.parameters(), self.encoder.parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    xe5 = self.encoder(input)
    illu = self.decoder.decoder_illu(xe5)
    enhanced = torch.div(input, illu)
    diff = denoise(enhanced, self.denoise_detection)
    noise = self.decoder.decoder_denoise(enhanced)
    if diff < 0.03:
        result = enhanced - 0 * noise
    else:
        result = enhanced - 1 * noise
    return result, illu

  def _loss(self, input, target, flag_lol):
    result = self(input)
    return self._criterion(result, target)


class LossFunction(nn.Module):
  def __init__(self):
    super(LossFunction, self).__init__()
    self.l2_loss = nn.MSELoss()
    self.vgg_loss = networks.PerceptualLoss()
    self.vgg_loss.cuda()
    self.vgg = networks.load_vgg16("./weights")
    self.vgg.eval()
    for param in self.vgg.parameters():
        param.requires_grad = False

  def forward(self, output, target):
    return self.l2_loss(output, target)
