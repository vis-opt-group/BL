import numpy as np
import torch
import torch.utils.data
import random
from PIL import Image
from glob import glob
import torchvision.transforms as transforms

class MemoryFriendlyLoader(torch.utils.data.Dataset):
    def __init__(self, low_img_dir, high_img_dir, task, args):
        self.low_img_dir = low_img_dir
        self.high_img_dir = high_img_dir
        self.task = task
        self.h = args.h
        self.w = args.w
        self.train_low_data_names = glob(self.low_img_dir)
        self.train_low_data_names.sort()
        self.train_high_data_names = glob(self.high_img_dir)
        self.train_high_data_names.sort()

        self.count = len(self.train_high_data_names)

        transform_list = []
        transform_list += [transforms.ToTensor()]

        self.transform = transforms.Compose(transform_list)

    def load_images(self, file):
        im = Image.open(file)
        img = np.array(im, dtype="float32") / 255.0
        img_max = np.max(img)
        img_min = np.min(img)
        img_norm = np.float32((img - img_min) / np.maximum((img_max - img_min), 0.001))
        return img_norm

    def load_images_transform(self, file):
        im = Image.open(file).convert('RGB')
        #print(im.size)
        img_norm = self.transform(im).numpy()
        img_norm = np.transpose(img_norm, (1, 2, 0))
        #print(img_norm.shape)
        return img_norm

    def __getitem__(self, index):

        low = self.load_images_transform(self.train_low_data_names[index])
        high = self.load_images_transform(self.train_high_data_names[index])

        h = low.shape[0]
        w = low.shape[1]

        h_offset = random.randint(0, max(0, h - self.h - 1))
        w_offset = random.randint(0, max(0, w - self.w - 1))

        if self.task != 'test':
            low = low[h_offset:h_offset + self.h, w_offset:w_offset + self.w]
            high = high[h_offset:h_offset + self.h, w_offset:w_offset + self.w]

        low = np.asarray(low, dtype=np.float32)
        high = np.asarray(high, dtype=np.float32)
        low = np.transpose(low[:, :, :], (2, 0, 1))
        high = np.transpose(high[:, :, :], (2, 0, 1))

        if self.task == 'test':
            img_name = self.train_low_data_names[index].split('\\')[-1]
            #print(img_name)
            return torch.from_numpy(low), torch.from_numpy(high), img_name

        return torch.from_numpy(low), torch.from_numpy(high)

    def __len__(self):
        return self.count