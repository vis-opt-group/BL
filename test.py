import os
import sys
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn

from PIL import Image
from torch.autograd import Variable

from model_supervised import Network
from model_supervised import LossFunction

from multi_read_data import MemoryFriendlyLoader
from CBDNet_model import CBDNet_pretrain

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data_path', type=str, default=r'F:\Code\IJCV\Data_testset\MIT\testA')
parser.add_argument('--save_path', type=str, default=r'F:\Code\IJCV\Data_testset\MIT\新建文件夹')
parser.add_argument('--model', type=str, default='./weights/mit.pt')
parser.add_argument('--h', type=int, default=160, help='random seed')
parser.add_argument('--w', type=int, default=160, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
args = parser.parse_args()


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



def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)       
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)     
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)    
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = LossFunction().cuda()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")       
  denoise_model = CBDNet_pretrain('./weights/CBDNet.pth').to(device)
  denoise_model = CBDNet_pretrain().to(device)
  model = Network(criterion, denoise_model)
  state_dict = torch.load(args.model)
  model.load_state_dict(state_dict)
  model = model.cuda()

  # prepare DataLoader
  data_path = args.data_path
  test_low_data_names = data_path + '/*.png'
  test_high_data_names = data_path + '/*.png'
  TestDataset = MemoryFriendlyLoader(low_img_dir=test_low_data_names, high_img_dir=test_high_data_names,
                                     task='test', args=args)

  test_queue = torch.utils.data.DataLoader(
    TestDataset, batch_size=1,
    pin_memory=True, num_workers=0)

  # inference
  inference_dir = args.save_path
  infer(test_queue, model, inference_dir)

def infer(test_queue, model, out_dir):
  model.eval()
  with torch.no_grad():
      path = out_dir
      if not os.path.isdir(path):
        os.mkdir(path)
      for _, (input, target, image_name) in enumerate(test_queue):
        input = Variable(input, volatile=True).cuda()
        input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input)
        enhanced, illu = model(input)
        enhanced = pad_tensor_back(enhanced, pad_left, pad_right, pad_top, pad_bottom)
        
        image_name = image_name[0].split('.')[0]
        image_name = '%s.png' % (image_name)
        print('processing {}'.format(image_name))
        file_path = path + '/' + image_name
        save_images(enhanced, file_path)


def save_images(tensor, path):
  image_numpy = tensor[0].cpu().float().numpy()
  image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
  im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
  im.save(path, 'png')


if __name__ == '__main__':
  main()

