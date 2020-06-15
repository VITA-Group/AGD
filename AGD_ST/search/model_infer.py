import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
# from utils.darts_utils import drop_path, compute_speed, compute_speed_tensorrt
from pdb import set_trace as bp
import numpy as np
from thop import profile
from matplotlib import pyplot as plt
from util_gan.vgg_feature import VGGFeature
from thop import profile


def make_divisible(v, divisor=8, min_value=3):
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/\
    0344c5503ee55e24f0de7f37336a6e08f10976fd/\
    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# https://github.com/YongfeiYan/Gumbel_Softmax_VAE/blob/master/gumbel_softmax_vae.py
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=1, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    
    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


class MixedOp(nn.Module):
    def __init__(self, C_in, C_out, op_idx, quantize, stride=1):
        super(MixedOp, self).__init__()
        self._op = OPS[PRIMITIVES[op_idx]](C_in, C_out, stride, slimmable=False, width_mult_list=[1.])
        self.quantize = quantize

    def forward(self, x):
        return self._op(x, quantize=self.quantize)

    def forward_latency(self, size):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        latency, size_out = self._op.forward_latency(size)
        return latency, size_out

    def forward_flops(self, size):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        flops, size_out = self._op.forward_flops(size, quantize=self.quantize)

        return flops, size_out


class SingleOp(nn.Module):
    def __init__(self, op, C_in, C_out, kernel_size=3 , stride=1, quantize=True):
        super(SingleOp, self).__init__()
        self._op = op(C_in, C_out, kernel_size=kernel_size, stride=stride, slimmable=False, width_mult_list=[1.])
        self.quantize = quantize

    def forward(self, x):
        result = self._op(x, quantize=self.quantize)

        return result

    def forward_flops(self, size):
        flops, size_out = self._op.forward_flops(size, quantize=self.quantize)

        return flops, size_out



class NAS_GAN_Infer(nn.Module):
    def __init__(self, alpha, beta, ratio, beta_sh, ratio_sh, layers=16, width_mult_list=[1.,], width_mult_list_sh=[1.,], loss_weight = [1e0, 1e5, 1e0, 1e-7], quantize=True):
        super(NAS_GAN_Infer, self).__init__()
        assert layers >= 3
        self._layers = layers
        self._width_mult_list = width_mult_list
        self._width_mult_list_sh = width_mult_list_sh
        self._flops = 0
        self._params = 0

        self.len_stem = 3
        self.len_header = 3
        self.len_beta_sh = self.len_stem + self.len_header
        self.len_ratio_sh = self.len_stem + self.len_header - 1

        self.base_weight = loss_weight[0]
        self.style_weight = loss_weight[1]
        self.content_weight = loss_weight[2]
        self.tv_weight = loss_weight[3]

        op_idx_list = F.softmax(alpha, dim=-1).argmax(-1)

        if quantize == 'search':
            quantize_list = F.softmax(beta, dim=-1).argmax(-1) == 1
            quantize_list_sh = F.softmax(beta_sh, dim=-1).argmax(-1) == 1
        elif quantize:
            quantize_list = [True for _ in range(layers)]
            quantize_list_sh = [True for _ in range(beta_sh.size(0))]        
        else:
            quantize_list = [False for _ in range(layers)]
            quantize_list_sh = [False for _ in range(beta_sh.size(0))]

        ratio_list = F.softmax(ratio, dim=-1).argmax(-1)
        ratio_list_sh = F.softmax(ratio_sh, dim=-1).argmax(-1)

        self.vgg = torch.nn.DataParallel(VGGFeature()).cuda()

        # Construct Stem
        self.stem = nn.ModuleList()
        self.stem.append(SingleOp(ConvNorm, 3, make_divisible(64*width_mult_list_sh[ratio_list_sh[0]]), 7, quantize=quantize_list_sh[0]))

        in_features = 64
        out_features = in_features*2

        for i in range(2):
            self.stem.append(SingleOp(ConvNorm, make_divisible(in_features*width_mult_list_sh[ratio_list_sh[i]]), make_divisible(out_features*width_mult_list_sh[ratio_list_sh[i+1]]), 3, stride=2, quantize=quantize_list_sh[1+i]))
            in_features = out_features
            out_features = in_features*2

        # Construct Blocks
        self.cells = nn.ModuleList()
        for i in range(layers):
            if i == 0:
                self.cells.append(MixedOp(make_divisible(in_features * width_mult_list_sh[ratio_list_sh[self.len_stem-1]]), make_divisible(in_features * width_mult_list[ratio_list[i]]), op_idx_list[i], quantize_list[i]))
            else:
                self.cells.append(MixedOp(make_divisible(in_features * width_mult_list[ratio_list[i-1]]), make_divisible(in_features * width_mult_list[ratio_list[i]]), op_idx_list[i], quantize_list[i]))

        # Construct Header
        self.header = nn.ModuleList()

        out_features = in_features//2
        
        self.header.append(SingleOp(ConvTranspose2dNorm, make_divisible(in_features*width_mult_list[ratio_list[self._layers-1]]), make_divisible(out_features*width_mult_list_sh[ratio_list_sh[self.len_stem]]), 3, stride=2, quantize=quantize_list_sh[self.len_stem]))
        
        in_features = out_features
        out_features = in_features//2
        
        self.header.append(SingleOp(ConvTranspose2dNorm, make_divisible(in_features*width_mult_list_sh[ratio_list_sh[self.len_stem]]), make_divisible(out_features*width_mult_list_sh[ratio_list_sh[self.len_stem+1]]), 3, stride=2, quantize=quantize_list_sh[self.len_stem+1]))

        self.header.append(SingleOp(Conv, make_divisible(64*width_mult_list_sh[ratio_list_sh[self.len_stem+1]]), 3, 7, quantize=quantize_list_sh[self.len_stem+2]))
                        
        self.tanh = nn.Tanh()


    def forward(self, input):

        out = input
        for i, module in enumerate(self.stem):
            out = module(out)

        for i, cell in enumerate(self.cells):
            out = cell(out)

        for i, module in enumerate(self.header):
            out = module(out)

        out = self.tanh(out)

        return out
        ###################################
    
    def forward_flops(self, size):
        flops_total = []

        for i, module in enumerate(self.stem):
            flops, size = module.forward_flops(size) 
            flops_total.append(flops)

        for i, cell in enumerate(self.cells):
            flops, size = cell.forward_flops(size)
            flops_total.append(flops)

        for i, module in enumerate(self.header):
            flops, size = module.forward_flops(size) 
            flops_total.append(flops)

        return sum(flops_total)

        ###################################

    def gram(self, x):
        (bs, ch, h, w) = x.size()
        f = x.view(bs, ch, w*h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (ch * h * w)
        return G


    def _criterion(self, y_hat, x):
        base_loss = self.base_weight * nn.L1Loss()(y_hat, x)

        y_c_features = self.vgg(x)
        y_hat_features = self.vgg(y_hat)

        y_hat_gram = [self.gram(fmap) for fmap in y_hat_features]
        x_gram = [self.gram(fmap) for fmap in y_c_features]

        style_loss = 0
        for j in range(4):
            style_loss += self.style_weight * nn.functional.mse_loss(y_hat_gram[j], x_gram[j])

        recon = y_c_features[1]      
        recon_hat = y_hat_features[1]
        content_loss = self.content_weight * nn.L1Loss()(recon_hat, recon)

        diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
        tv_loss = self.tv_weight*(diff_i + diff_j)

        total_loss = base_loss + style_loss + content_loss + tv_loss

        return total_loss


    def _loss(self, input, target):
        logit = self(input)
        loss = self._criterion(logit, target)

        return loss
