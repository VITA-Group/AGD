# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict

C = edict()
config = C
cfg = C

C.seed = 12345

"""please config ROOT_dir and user when u first using"""
C.repo_name = 'AGD_ST'
C.abs_dir = osp.realpath(".")
C.this_dir = C.abs_dir.split(osp.sep)[-1]
C.root_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]
C.log_dir = osp.abspath(osp.join(C.root_dir, 'log', C.this_dir))
C.log_dir_link = osp.join(C.abs_dir, 'log')
C.snapshot_dir = osp.abspath(osp.join(C.log_dir, "snapshot"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

"""Data Dir and Weight Dir"""

C.dataset = 'horse2zebra'

if C.dataset == 'horse2zebra':
    C.dataset_path = "/home/yf22/dataset/horse2zebra"
    C.target_path = '/home/yf22/dataset/horse2zebra/target_train'
    C.num_train_imgs = 1067
    C.num_eval_imgs = 120
elif C.dataset == 'zebra2horse':
    C.dataset_path = "/home/yf22/dataset/zebra2horse"
    C.target_path = '/home/yf22/dataset/zebra2horse/target_train'
    C.num_train_imgs = 1334
    C.num_eval_imgs = 140
elif C.dataset == 'summer2winter':
    C.dataset_path = "/home/yf22/dataset/summer2winter"
    C.target_path = '/home/yf22/dataset/summer2winter/target_train'
    C.num_train_imgs = 1231
    C.num_eval_imgs = 309
elif C.dataset == 'winter2summer':
    C.dataset_path = "/home/yf22/dataset/winter2summer"
    C.target_path = '/home/yf22/dataset/winter2summer/target_train'
    C.num_train_imgs = 962
    C.num_eval_imgs = 238
else:
    print("No such dataset.")
    sys.exit()

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(osp.join(C.root_dir, 'furnace'))


C.num_workers = 4

C.layers = 9

C.width_mult_list = [4./12, 6./12, 8./12, 10./12, 1.]

C.width_mult_list_sh = [4/12, 6./12, 8./12, 10./12, 1.]

C.batch_size = 1
C.niters_per_epoch = C.num_train_imgs // C.batch_size
C.image_height = 256 # this size is after down_sampling
C.image_width = 256

C.quantize = False

C.generator_A2B = 'horse2zebra/netG_A2B_epoch_199.pth'

C.load_path = 'ckpt/search'

C.real_measurement = False

C.ckpt = 'ckpt/finetune/weights.pt'
