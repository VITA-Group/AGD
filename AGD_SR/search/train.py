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

from config_train import config
from datasets import ImageDataset

from utils.init_func import init_weight

from utils.darts_utils import create_exp_dir, save, plot_op, plot_path_width, objective_acc_lat
from model_infer import NAS_GAN_Infer

from util_gan.cyclegan import Generator
from util_gan.psnr import compute_psnr
from util_gan.lr import LambdaLR

from quantize import QConv2d, QConvTranspose2d, QuantMeasure
from thop import profile
from thop.count_hooks import count_convNd

from RRDBNet_arch import RRDBNet

import operations
import model_infer
operations.ENABLE_BN = config.ENABLE_BN
model_infer.ENABLE_TANH = config.ENABLE_TANH

def count_custom(m, x, y):
    m.total_ops += 0

custom_ops={QConv2d: count_convNd, QConvTranspose2d:count_convNd, QuantMeasure: count_custom, nn.InstanceNorm2d: count_custom}

def main():
    config.save = 'ckpt/{}'.format(config.save)
    create_exp_dir(config.save, scripts_to_save=glob.glob('*.py')+glob.glob('*.sh'))
    logger = SummaryWriter(config.save)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info("args = %s", str(config))
    # preparation ################
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed = config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    state = torch.load(os.path.join(config.load_path, 'arch.pt'))
    # Model #######################################
    model = NAS_GAN_Infer(state['alpha'], state['beta'], state['ratio'], num_cell=config.num_cell, op_per_cell=config.op_per_cell, width_mult_list=config.width_mult_list, 
                          loss_weight=config.loss_weight, loss_func=config.loss_func, before_act=config.before_act, quantize=config.quantize)

    flops, params = profile(model, inputs=(torch.randn(1, 3, 510, 350),), custom_ops=custom_ops)
    flops = model.forward_flops(size=(3, 510, 350))
    logging.info("params = %fMB, FLOPs = %fGB", params / 1e6, flops / 1e9)

    model = torch.nn.DataParallel(model).cuda()

    if type(config.pretrain) == str:
        state_dict = torch.load(config.pretrain)
        model.load_state_dict(state_dict)
    # else:
    #     features = [model.module.cells, model.module.conv_first, model.module.trunk_conv, model.module.upconv1, 
    #                 model.module.upconv2, model.module.HRconv, model.module.conv_last]
    #     init_weight(features, nn.init.kaiming_normal_, nn.BatchNorm2d, config.bn_eps, config.bn_momentum, mode='fan_in', nonlinearity='relu')

    teacher_model = RRDBNet(3, 3, 64, 23, gc=32)
    teacher_model.load_state_dict(torch.load(config.generator_A2B), strict=True)
    teacher_model = torch.nn.DataParallel(teacher_model).cuda()
    teacher_model.eval()

    for param in teacher_model.parameters():
        param.require_grads = False

    # Optimizer ###################################
    base_lr = config.lr
    parameters = []
    parameters += list(model.module.cells.parameters())
    parameters += list(model.module.conv_first.parameters())
    parameters += list(model.module.trunk_conv.parameters())
    parameters += list(model.module.upconv1.parameters())
    parameters += list(model.module.upconv2.parameters())
    parameters += list(model.module.HRconv.parameters())
    parameters += list(model.module.conv_last.parameters())

    if config.opt == 'Adam':
        optimizer = torch.optim.Adam(
            parameters,
            lr=base_lr,
            betas=config.betas)
    elif config.opt == 'Sgd':
        optimizer = torch.optim.SGD(
            parameters,
            lr=base_lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay)
    else:
        logging.info("Wrong Optimizer Type.")
        sys.exit()

    # lr policy ##############################
    total_iteration = config.nepochs * config.niters_per_epoch

    if config.lr_schedule == 'linear':
        lr_policy = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(config.nepochs, 0, config.decay_epoch).step)
    elif config.lr_schedule == 'exponential':
        lr_policy = torch.optim.lr_scheduler.ExponentialLR(optimizer, config.lr_decay)
    elif config.lr_schedule == 'multistep':
        lr_policy = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)
    else:
        logging.info("Wrong Learning Rate Schedule Type.")
        sys.exit()


    # data loader ############################

    transforms_ = [ transforms.RandomCrop(config.image_height), 
                    transforms.RandomHorizontalFlip(), 
                    transforms.ToTensor()]
    train_loader_model = DataLoader(ImageDataset(config.dataset_path, transforms_=transforms_, unaligned=True), 
                        batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    transforms_ = [ transforms.ToTensor()]
    test_loader = DataLoader(ImageDataset(config.dataset_path, transforms_=transforms_, mode='val'), 
                        batch_size=1, shuffle=False, num_workers=config.num_workers)


    if config.eval_only:
        logging.info('Eval: psnr = %f', infer(0, model, test_loader, logger))
        sys.exit(0)

    tbar = tqdm(range(config.nepochs), ncols=80)
    for epoch in tbar:
        logging.info(config.save)
        logging.info("lr: " + str(optimizer.param_groups[0]['lr']))

        # training
        tbar.set_description("[Epoch %d/%d][train...]" % (epoch + 1, config.nepochs))
        train(train_loader_model, model, teacher_model, optimizer, lr_policy, logger, epoch)
        torch.cuda.empty_cache()
        lr_policy.step()

        # validation
        if epoch and not (epoch+1) % config.eval_epoch:
            tbar.set_description("[Epoch %d/%d][validation...]" % (epoch + 1, config.nepochs))
            
            with torch.no_grad():
                model.prun_mode = None

                valid_psnr = infer(epoch, model, test_loader, logger)

                logger.add_scalar('psnr/val', valid_psnr, epoch)
                logging.info("Epoch %d: valid_psnr %.3f"%(epoch, valid_psnr))
                
                logger.add_scalar('flops/val', flops, epoch)
                logging.info("Epoch %d: flops %.3f"%(epoch, flops))

            save(model, os.path.join(config.save, 'weights_%d.pt'%epoch))

    save(model, os.path.join(config.save, 'weights.pt'))



def train(train_loader_model, model, teacher_model, optimizer, lr_policy, logger, epoch):
    model.train()

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format, ncols=80)
    dataloader_model = iter(train_loader_model)


    for step in pbar:
        lr = optimizer.param_groups[0]['lr']

        optimizer.zero_grad()

        minibatch = dataloader_model.next()
        input = minibatch['A']
        input = input.cuda(non_blocking=True)
        target = teacher_model(input)

        loss = model.module._loss(input, target)
        logger.add_scalar('loss/train', loss, epoch*len(pbar)+step)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        pbar.set_description("[Step %d/%d]" % (step + 1, len(train_loader_model)))

    torch.cuda.empty_cache()
    del loss


def infer(epoch, model, test_loader, logger):
    model.eval()

    for i, batch in enumerate(test_loader):
        # Set model input
        real_A = Variable(batch['A']).cuda()
        fake_B = model(real_A).data.float().clamp_(0, 1)

        img_name = '08%02d_gen.png' % (i+1) if i < 99 else '0900_gen.png'
        save_image(fake_B, os.path.join('output/B_nasgan', img_name))

    psnr = compute_psnr('output/B_nasgan', config.dataset_path + '/val_hr')

    return psnr


if __name__ == '__main__':
    main() 
