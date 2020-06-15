from __future__ import division
import os
import sys
import time
import glob
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from torchvision.utils import save_image

import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image

from config_train import config

from utils.init_func import init_weight

from utils.darts_utils import create_exp_dir, save, plot_op, plot_path_width, objective_acc_lat

from util_gan.cyclegan import Generator
from util_gan.fid_score import compute_fid


class ImageDataset(Dataset):
    def __init__(self, dataset_dir, mode='train', transforms_=None):
        '''
        Construct a dataset with all images from a dir.

        dataset_dir: str. img folder path
        '''
        self.transform = transforms.Compose(transforms_)
        print(os.path.join(dataset_dir, mode))
        self.files = sorted(glob.glob(os.path.join(dataset_dir, mode, 'A') + '/*.jpg'))
        print('files:', len(self.files))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img = img.convert("RGB")
        item = self.transform(img)

        return item, os.path.basename(self.files[index % len(self.files)])

    def __len__(self):
        return len(self.files)


def main():
    teacher_model = Generator(3, 3)
    teacher_model.load_state_dict(torch.load(config.generator_A2B))
    teacher_model = torch.nn.DataParallel(teacher_model).cuda()
    teacher_model.eval()

    transforms_ = [ transforms.ToTensor(),
                     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    train_loader = DataLoader(ImageDataset(config.dataset_path, transforms_=transforms_), 
                        batch_size=1, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(ImageDataset(config.dataset_path, transforms_=transforms_, mode='test'), 
                        batch_size=1, shuffle=False, num_workers=config.num_workers)

    eval(train_loader, teacher_model, save_dir='./target_train')
    print('Test fid:', eval(test_loader, teacher_model, save_dir='./target_test', return_fid=True))


def eval(data_loader, model, save_dir=None, return_fid=False):
    model.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, (input, fname) in enumerate(data_loader):
        # Set model input
        real_A = Variable(input).cuda()
        fake_B = 0.5*(model(real_A).data + 1.0)

        if save_dir is not None:
            save_image(fake_B, os.path.join(save_dir, fname[0]))

    if return_fid:
        fid = compute_fid(save_dir, config.dataset_path + '/test/B')
        return fid


if __name__ == '__main__':
    main() 
