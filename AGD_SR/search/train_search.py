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

import time

from tensorboardX import SummaryWriter

from torchvision.utils import save_image

import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image

from config_search import config
from datasets import ImageDataset

from utils.init_func import init_weight

from architect import Architect
from utils.darts_utils import create_exp_dir, save, plot_op, plot_path_width, objective_acc_lat
from model_search import NAS_GAN as Network
from model_infer import NAS_GAN_Infer

from util_gan.cyclegan import Generator
from util_gan.psnr import compute_psnr
from util_gan.lr import LambdaLR

from RRDBNet_arch import RRDBNet

import operations
import model_search
import model_infer
operations.ENABLE_BN = config.ENABLE_BN
model_search.ENABLE_TANH = model_infer.ENABLE_TANH = config.ENABLE_TANH


def main(pretrain=True):
    config.save = 'ckpt/{}'.format(config.save)
    create_exp_dir(config.save, scripts_to_save=glob.glob('*.py')+glob.glob('*.sh'))
    logger = SummaryWriter(config.save)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    assert type(pretrain) == bool or type(pretrain) == str
    update_arch = True
    if pretrain == True:
        update_arch = False
    logging.info("args = %s", str(config))
    # preparation ################
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed = config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Model #######################################
    model = Network(config.num_cell, config.op_per_cell, slimmable=config.slimmable, width_mult_list=config.width_mult_list, loss_weight=config.loss_weight, 
                    prun_modes=config.prun_modes, loss_func=config.loss_func, before_act=config.before_act, quantize=config.quantize)
    model = torch.nn.DataParallel(model).cuda()

    # print(model)

    teacher_model = RRDBNet(3, 3, 64, 23, gc=32)
    teacher_model.load_state_dict(torch.load(config.generator_A2B), strict=True)
    teacher_model = torch.nn.DataParallel(teacher_model).cuda()
    teacher_model.eval()

    for param in teacher_model.parameters():
        param.require_grads = False

    if type(pretrain) == str:
        partial = torch.load(pretrain + "/weights.pt")
        state = model.state_dict()
        pretrained_dict = {k: v for k, v in partial.items() if k in state and state[k].size() == partial[k].size()}
        state.update(pretrained_dict)
        model.load_state_dict(state)
    # else:
    #     features = [model.module.cells, model.module.conv_first, model.module.trunk_conv, model.module.upconv1, 
    #                 model.module.upconv2, model.module.HRconv, model.module.conv_last]
    #     init_weight(features, nn.init.kaiming_normal_, nn.BatchNorm2d, config.bn_eps, config.bn_momentum, mode='fan_in', nonlinearity='relu')

    architect = Architect(model, config)

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


    # data loader ###########################

    transforms_ = [ transforms.RandomCrop(config.image_height),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()]
    train_loader_model = DataLoader(ImageDataset(config.dataset_path, transforms_=transforms_, unaligned=True, portion=config.train_portion), 
                        batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    train_loader_arch = DataLoader(ImageDataset(config.dataset_path, transforms_=transforms_, unaligned=True, portion=config.train_portion-1), 
                        batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    transforms_ = [ transforms.ToTensor()]
    test_loader = DataLoader(ImageDataset(config.dataset_path, transforms_=transforms_, mode='val'), 
                        batch_size=1, shuffle=False, num_workers=config.num_workers)

    tbar = tqdm(range(config.nepochs), ncols=80)
    valid_psnr_history = []
    flops_history = []
    flops_supernet_history = []


    for epoch in tbar:
        logging.info(pretrain)
        logging.info(config.save)
        logging.info("lr: " + str(optimizer.param_groups[0]['lr']))

        logging.info("update arch: " + str(update_arch))

        # training
        tbar.set_description("[Epoch %d/%d][train...]" % (epoch + 1, config.nepochs))
        train(pretrain, train_loader_model, train_loader_arch, model, architect, teacher_model, optimizer, lr_policy, logger, epoch, update_arch=update_arch)
        torch.cuda.empty_cache()
        lr_policy.step()

        # validation
        if epoch and not (epoch+1) % config.eval_epoch:
            tbar.set_description("[Epoch %d/%d][validation...]" % (epoch + 1, config.nepochs))

            save(model, os.path.join(config.save, 'weights_%d.pt'%epoch))

            with torch.no_grad():
                if pretrain == True:
                    model.module.prun_mode = "min"
                    valid_psnr = infer(epoch, model, test_loader, logger)
                    logger.add_scalar('psnr/val_min', valid_psnr, epoch)
                    logging.info("Epoch %d: valid_psnr_min %.3f"%(epoch, valid_psnr))

                    if len(model.module._width_mult_list) > 1:
                        model.module.prun_mode = "max"
                        valid_psnr = infer(epoch, model, test_loader, logger)
                        logger.add_scalar('psnr/val_max', valid_psnr, epoch)
                        logging.info("Epoch %d: valid_psnr_max %.3f"%(epoch, valid_psnr))

                        model.module.prun_mode = "random"
                        valid_psnr = infer(epoch, model, test_loader, logger)
                        logger.add_scalar('psnr/val_random', valid_psnr, epoch)
                        logging.info("Epoch %d: valid_psnr_random %.3f"%(epoch, valid_psnr))

                else:
                    model.module.prun_mode = None

                    valid_psnr, flops = infer(epoch, model, test_loader, logger, finalize=True)

                    logger.add_scalar('psnr/val', valid_psnr, epoch)
                    logging.info("Epoch %d: valid_psnr %.3f"%(epoch, valid_psnr))
                    
                    logger.add_scalar('flops/val', flops, epoch)
                    logging.info("Epoch %d: flops %.3f"%(epoch, flops))

                    valid_psnr_history.append(valid_psnr)
                    flops_history.append(flops)
                    
                    if update_arch:
                        flops_supernet_history.append(architect.flops_supernet)

                if update_arch:
                    state = {}
                    state['alpha'] = getattr(model.module, 'alpha')
                    state['beta'] = getattr(model.module, 'beta')
                    state['ratio'] = getattr(model.module, 'ratio')
                    state["psnr"] = valid_psnr
                    state["flops"] = flops

                    torch.save(state, os.path.join(config.save, "arch_%d.pt"%(epoch)))

                    if config.flops_weight > 0:
                        if flops < config.flops_min:
                            architect.flops_weight /= 2
                        elif flops > config.flops_max:
                            architect.flops_weight *= 2
                        logger.add_scalar("arch/flops_weight", architect.flops_weight, epoch+1)
                        logging.info("arch_flops_weight = " + str(architect.flops_weight))

    save(model, os.path.join(config.save, 'weights.pt'))
    
    if update_arch:
        torch.save(state, os.path.join(config.save, "arch.pt"))


def train(pretrain, train_loader_model, train_loader_arch, model, architect, teacher_model, optimizer, lr_policy, logger, epoch, update_arch=True):
    model.train()

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format, ncols=80)
    dataloader_model = iter(train_loader_model)
    dataloader_arch = iter(train_loader_arch)

    for step in pbar:
        minibatch = dataloader_model.next()

        # end = time.time()

        input = minibatch['A']
        input = input.cuda(non_blocking=True)
        target = teacher_model(input)

        # time_data = time.time() - end
        # end = time.time()

        if update_arch:
            pbar.set_description("[Step %d/%d]" % (step + 1, len(train_loader_arch)))

            minibatch = dataloader_arch.next()
            input_search = minibatch['A']
            input_search = input_search.cuda(non_blocking=True)
            target_search = teacher_model(input_search)

            loss_arch = architect.step(input, target, input_search, target_search)
            if (step+1) % 10 == 0:
                logger.add_scalar('loss_arch/train', loss_arch, epoch*len(pbar)+step)
                logger.add_scalar('arch/flops_supernet', architect.flops_supernet, epoch*len(pbar)+step)

        # print(model.module.alpha[1])
        # print(model.module.ratio[1])

        loss = model.module._loss(input, target, pretrain)

        # time_fw = time.time() - end
        # end = time.time()

        optimizer.zero_grad()
        logger.add_scalar('loss/train', loss, epoch*len(pbar)+step)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        # time_bw = time.time() - end
        # end = time.time()

        # print("[Step %d/%d]" % (step + 1, len(train_loader_model)), 'Loss:', loss, 'Time Data:', time_data, 'Time Forward:', time_fw, 'Time Backward:', time_bw)

        pbar.set_description("[Step %d/%d]" % (step + 1, len(train_loader_model)))

    torch.cuda.empty_cache()
    del loss
    if update_arch: del loss_arch


def infer(epoch, model, test_loader, logger, finalize=False):
    model.eval()

    for i, batch in enumerate(test_loader):
        # Set model input
        real_A = Variable(batch['A']).cuda()
        fake_B = model(real_A).data.float().clamp_(0, 1)
        
        img_name = '08%02d_gen.png' % (i+1) if i < 99 else '0900_gen.png'
        save_image(fake_B, os.path.join('output/B_nasgan', img_name))

    psnr = compute_psnr('output/B_nasgan', config.dataset_path + '/val_hr')

    if finalize:
        model_infer = NAS_GAN_Infer(getattr(model.module, 'alpha'), getattr(model.module, 'beta'), getattr(model.module, 'ratio'), num_cell=config.num_cell, op_per_cell=config.op_per_cell, 
                                    width_mult_list=config.width_mult_list, loss_weight=config.loss_weight, loss_func=config.loss_func, before_act=config.before_act, quantize=config.quantize)
        flops = model_infer.forward_flops((3, 510, 350))
        return psnr, flops

    else:
        return psnr


if __name__ == '__main__':
    main(pretrain=config.pretrain) 
