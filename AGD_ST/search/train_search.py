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
from datasets import ImageDataset, PairedImageDataset

from utils.init_func import init_weight

from architect import Architect
from utils.darts_utils import create_exp_dir, save, plot_op, plot_path_width, objective_acc_lat
from model_search import NAS_GAN as Network
from model_infer import NAS_GAN_Infer

from util_gan.cyclegan import Generator
from util_gan.fid_score import compute_fid
from util_gan.lr import LambdaLR

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
    model = Network(config.layers, slimmable=config.slimmable, width_mult_list=config.width_mult_list, width_mult_list_sh=config.width_mult_list_sh, 
                    loss_weight=config.loss_weight, prun_modes=config.prun_modes, quantize=config.quantize)
    model = torch.nn.DataParallel(model).cuda()

    # print(model)

    # teacher_model = Generator(3, 3)
    # teacher_model.load_state_dict(torch.load(config.generator_A2B))
    # teacher_model = torch.nn.DataParallel(teacher_model).cuda()

    # for param in teacher_model.parameters():
    #     param.require_grads = False

    if type(pretrain) == str:
        partial = torch.load(pretrain + "/weights.pt")
        state = model.state_dict()
        pretrained_dict = {k: v for k, v in partial.items() if k in state and state[k].size() == partial[k].size()}
        state.update(pretrained_dict)
        model.load_state_dict(state)
    # else:
    #     features = [model.module.stem, model.module.cells, model.module.header]
    #     init_weight(features, nn.init.kaiming_normal_, nn.InstanceNorm2d, config.bn_eps, config.bn_momentum, mode='fan_in', nonlinearity='relu')

    architect = Architect(model, config)

    # Optimizer ###################################
    base_lr = config.lr
    parameters = []
    parameters += list(model.module.stem.parameters())
    parameters += list(model.module.cells.parameters())
    parameters += list(model.module.header.parameters())

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

    transforms_ = [ 
                    # transforms.Resize(int(config.image_height*1.12), Image.BICUBIC), 
                    # transforms.RandomCrop(config.image_height), 
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

    # train_loader_model = DataLoader(ImageDataset(config.dataset_path, transforms_=transforms_, unaligned=True, portion=config.train_portion), 
    #                     batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    # train_loader_arch = DataLoader(ImageDataset(config.dataset_path, transforms_=transforms_, unaligned=True, portion=config.train_portion-1), 
    #                     batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    train_loader_model = DataLoader(PairedImageDataset(config.dataset_path, config.target_path, transforms_=transforms_, portion=config.train_portion), 
                        batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    train_loader_arch = DataLoader(PairedImageDataset(config.dataset_path, config.target_path, transforms_=transforms_, portion=config.train_portion-1), 
                        batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    transforms_ = [ transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    test_loader = DataLoader(ImageDataset(config.dataset_path, transforms_=transforms_, mode='test'), 
                        batch_size=1, shuffle=False, num_workers=config.num_workers)


    tbar = tqdm(range(config.nepochs), ncols=80)
    valid_fid_history = []
    flops_history = []
    flops_supernet_history = []

    best_fid = 1000
    best_epoch = 0

    for epoch in tbar:
        logging.info(pretrain)
        logging.info(config.save)
        logging.info("lr: " + str(optimizer.param_groups[0]['lr']))

        logging.info("update arch: " + str(update_arch))

        # training
        tbar.set_description("[Epoch %d/%d][train...]" % (epoch + 1, config.nepochs))
        train(pretrain, train_loader_model, train_loader_arch, model, architect, optimizer, lr_policy, logger, epoch, update_arch=update_arch)
        torch.cuda.empty_cache()
        lr_policy.step()

        # validation
        if epoch and not (epoch+1) % config.eval_epoch:
            tbar.set_description("[Epoch %d/%d][validation...]" % (epoch + 1, config.nepochs))

            save(model, os.path.join(config.save, 'weights_%d.pt'%epoch))

            with torch.no_grad():
                if pretrain == True:
                    model.module.prun_mode = "min"
                    valid_fid = infer(epoch, model, test_loader, logger)
                    logger.add_scalar('fid/val_min', valid_fid, epoch)
                    logging.info("Epoch %d: valid_fid_min %.3f"%(epoch, valid_fid))

                    if len(model.module._width_mult_list) > 1:
                        model.module.prun_mode = "max"
                        valid_fid = infer(epoch, model, test_loader, logger)
                        logger.add_scalar('fid/val_max', valid_fid, epoch)
                        logging.info("Epoch %d: valid_fid_max %.3f"%(epoch, valid_fid))

                        model.module.prun_mode = "random"
                        valid_fid = infer(epoch, model, test_loader, logger)
                        logger.add_scalar('fid/val_random', valid_fid, epoch)
                        logging.info("Epoch %d: valid_fid_random %.3f"%(epoch, valid_fid))

                else:
                    model.module.prun_mode = None

                    valid_fid, flops = infer(epoch, model, test_loader, logger, finalize=True)

                    logger.add_scalar('fid/val', valid_fid, epoch)
                    logging.info("Epoch %d: valid_fid %.3f"%(epoch, valid_fid))
                    
                    logger.add_scalar('flops/val', flops, epoch)
                    logging.info("Epoch %d: flops %.3f"%(epoch, flops))

                    valid_fid_history.append(valid_fid)
                    flops_history.append(flops)
                    
                    if update_arch:
                        flops_supernet_history.append(architect.flops_supernet)

                if valid_fid < best_fid:
                    best_fid = valid_fid
                    best_epoch = epoch
                logging.info("Best fid:%.3f, Best epoch:%d"%(best_fid, best_epoch))

                if update_arch:
                    state = {}
                    state['alpha'] = getattr(model.module, 'alpha')
                    state['beta'] = getattr(model.module, 'beta')
                    state['ratio'] = getattr(model.module, 'ratio')
                    state['beta_sh'] = getattr(model.module, 'beta_sh')
                    state['ratio_sh'] = getattr(model.module, 'ratio_sh')
                    state["fid"] = valid_fid
                    state["flops"] = flops

                    torch.save(state, os.path.join(config.save, "arch_%d_%f.pt"%(epoch, flops)))

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


def train(pretrain, train_loader_model, train_loader_arch, model, architect, optimizer, lr_policy, logger, epoch, update_arch=True):
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
        target = minibatch['B']
        target = target.cuda(non_blocking=True)

        # time_data = time.time() - end
        # end = time.time()

        if update_arch:
            pbar.set_description("[Step %d/%d]" % (step + 1, len(train_loader_arch)))

            minibatch = dataloader_arch.next()
            input_search = minibatch['A']
            input_search = input_search.cuda(non_blocking=True)
            target_search = minibatch['B']
            target_search = target_search.cuda(non_blocking=True)

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

    outdir = 'output/gen_epoch_%d' % (epoch)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for i, batch in enumerate(test_loader):
        # Set model input
        real_A = Variable(batch['A']).cuda()
        fake_B = 0.5*(model(real_A).data + 1.0)
        
        save_image(fake_B, os.path.join(outdir, '%04d.png' % (i+1)))

    fid = compute_fid(outdir, config.dataset_path + '/test/B')

    if finalize:
        model_infer = NAS_GAN_Infer(getattr(model.module, 'alpha'), getattr(model.module, 'beta'), getattr(model.module, 'ratio'), getattr(model.module, 'beta_sh'), getattr(model.module, 'ratio_sh'), 
            layers=config.layers, width_mult_list=config.width_mult_list, width_mult_list_sh=config.width_mult_list_sh, loss_weight=config.loss_weight, quantize=config.quantize)
        flops = model_infer.forward_flops((3, 256, 256))
        return fid, flops

    else:
        return fid


if __name__ == '__main__':
    main(pretrain=config.pretrain) 
