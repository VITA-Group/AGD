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

from config_eval import config
from datasets import ImageDataset

from utils.init_func import init_weight

from utils.darts_utils import create_exp_dir, save, plot_op, plot_path_width, objective_acc_lat
from model_eval import NAS_GAN_Eval

from util_gan.cyclegan import Generator
from util_gan.psnr import compute_psnr
from util_gan.lr import LambdaLR

from quantize import QConv2d, QConvTranspose2d, QuantMeasure
from thop import profile
from thop.count_hooks import count_convNd

from RRDBNet_arch import RRDBNet

import operations
import model_eval
operations.ENABLE_BN = config.ENABLE_BN
model_eval.ENABLE_TANH = config.ENABLE_TANH

def count_custom(m, x, y):
    m.total_ops += 0

custom_ops={QConv2d: count_convNd, QConvTranspose2d:count_convNd, QuantMeasure: count_custom, nn.InstanceNorm2d: count_custom}

def main():
    state = torch.load(os.path.join(config.load_path, 'arch.pt'))
    # Model #######################################
    model = NAS_GAN_Eval(state['alpha'], state['beta'], state['ratio'], num_cell=config.num_cell, op_per_cell=config.op_per_cell, 
                         width_mult_list=config.width_mult_list, quantize=config.quantize)

    if not config.real_measurement:
        flops, params = profile(model, inputs=(torch.randn(1, 3, 510, 350),), custom_ops=custom_ops)
        # flops = model.forward_flops(size=(3, 510, 350))
        print("params = %fMB, FLOPs = %fGB" % (params / 1e6, flops / 1e9))

    model = torch.nn.DataParallel(model).cuda()

    if config.ckpt:
        state_dict = torch.load(config.ckpt)
        model.load_state_dict(state_dict, strict=False)
    # else:
    #     features = [model.module.cells, model.module.conv_first, model.module.trunk_conv, model.module.upconv1, 
    #                 model.module.upconv2, model.module.HRconv, model.module.conv_last]
    #     init_weight(features, nn.init.kaiming_normal_, nn.BatchNorm2d, config.bn_eps, config.bn_momentum, mode='fan_in', nonlinearity='relu')

    # teacher_model = RRDBNet(3, 3, 64, 23, gc=32)
    # teacher_model.load_state_dict(torch.load(config.generator_A2B), strict=True)
    # teacher_model = torch.nn.DataParallel(teacher_model).cuda()
    # teacher_model.eval()

    # for param in teacher_model.parameters():
    #     param.require_grads = False

   
    # transforms_ = [ transforms.RandomCrop(config.image_height), 
    #                 transforms.RandomHorizontalFlip(), 
    #                 transforms.ToTensor()]
    # train_loader_model = DataLoader(ImageDataset(config.dataset_path, transforms_=transforms_, unaligned=True), 
    #                     batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    transforms_ = [ transforms.ToTensor()]
    test_loader = DataLoader(ImageDataset(config.dataset_path, transforms_=transforms_, mode='val'), 
                        batch_size=1, shuffle=False, num_workers=config.num_workers)

    with torch.no_grad():
        valid_psnr = infer(model, test_loader)

    print('PSNR:', valid_psnr)


def infer(model, test_loader):
    model.eval()

    for i, batch in enumerate(test_loader):
        real_A = batch['A'].cuda()
        fake_B = model(real_A).data.float().clamp_(0, 1)

        if not config.real_measurement:
            img_name = '%02d.png' % ((i+1) % 100)

            if i >= 99:
                img_name = 'a' + img_name

            save_image(fake_B, os.path.join('output/eval', img_name))

        # torch.cuda.empty_cache()

    if not config.real_measurement:
        psnr = compute_psnr('output/eval', config.dataset_path + '/val_hr')
    else:
        psnr = 0

    return psnr


if __name__ == '__main__':
    main() 
