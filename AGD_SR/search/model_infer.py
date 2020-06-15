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

ENABLE_TANH = False

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


class Cell(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, op_idx_list, quantize_list, ratio_list, nf=64, op_per_cell=5, width_mult_list=[1.]):
        super(Cell, self).__init__()

        self.nf = nf
        self.op_per_cell = op_per_cell
        self._width_mult_list = width_mult_list

        self.ops = nn.ModuleList()

        for i in range(op_per_cell):
            if i == 0:
                self.ops.append(MixedOp(self.nf, make_divisible(self.nf * width_mult_list[ratio_list[i]]), op_idx_list[i], quantize_list[i]))
            elif i == op_per_cell - 1:
                self.ops.append(MixedOp(make_divisible(self.nf * width_mult_list[ratio_list[i-1]]), self.nf, op_idx_list[i], quantize_list[i]))
            else:
                self.ops.append(MixedOp(make_divisible(self.nf * width_mult_list[ratio_list[i-1]]), make_divisible(self.nf * width_mult_list[ratio_list[i]]), op_idx_list[i], quantize_list[i]))


    def forward(self, x):
        out = x
        for op in self.ops:
            out = op(out)
        return out*0.2 + x


    def forward_flops(self, size):
        flops_total = []

        for i, op in enumerate(self.ops):
            flops, size = op.forward_flops(size)
            flops_total.append(flops)

        return sum(flops_total), size




class NAS_GAN_Infer(nn.Module):
    def __init__(self, alpha, beta, ratio, num_cell=5, op_per_cell=5, width_mult_list=[1.,], loss_weight = [1e0, 1e5, 1e0, 1e-7], 
                    loss_func='MSE', before_act=True, quantize=False):

        super(NAS_GAN_Infer, self).__init__()

        self.num_cell = num_cell
        self.op_per_cell = op_per_cell

        self._layers = num_cell * op_per_cell

        self._width_mult_list = width_mult_list
        self._flops = 0
        self._params = 0

        self.base_weight = loss_weight[0]
        self.style_weight = loss_weight[1]
        self.content_weight = loss_weight[2]
        self.tv_weight = loss_weight[3]

        op_idx_list = F.softmax(alpha, dim=-1).argmax(-1)

        if quantize == 'search':
            quantize_list = F.softmax(beta, dim=-1).argmax(-1) == 1
        elif quantize:
            quantize_list = [ [True for m in range(op_per_cell)] for n in range(num_cell)]      
        else:
            quantize_list = [ [False for m in range(op_per_cell)] for n in range(num_cell)]  

        ratio_list = F.softmax(ratio, dim=-1).argmax(-1)

        self.vgg = torch.nn.DataParallel(VGGFeature(before_act=before_act)).cuda()

        self.nf = 64

        self.conv_first = Conv(3, self.nf, 3, 1, 1, bias=True)
        
        self.cells = nn.ModuleList()
        for i in range(num_cell):
            self.cells.append(Cell(op_idx_list[i], quantize_list[i], ratio_list[i], nf=self.nf, op_per_cell=op_per_cell, width_mult_list=width_mult_list))

        self.trunk_conv = Conv(self.nf, self.nf, 3, 1, 1, bias=True)
        self.upconv1 = Conv(self.nf, self.nf, 3, 1, 1, bias=True)
        self.upconv2 = Conv(self.nf, self.nf, 3, 1, 1, bias=True)
        self.HRconv = Conv(self.nf, self.nf, 3, 1, 1, bias=True)
        self.conv_last = Conv(self.nf, 3, 3, 1, 1, bias=True)
                        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

        self.tanh = nn.Tanh()

        self.loss_func = nn.MSELoss() if loss_func == 'MSE' else nn.L1Loss()


    def forward(self, input):
        out = orig = self.conv_first(input)

        for i, cell in enumerate(self.cells):
            out = cell(out)

        out = self.trunk_conv(out)

        out = out + orig

        out = self.lrelu(self.upconv1(F.interpolate(out, scale_factor=2, mode='nearest')))
        out = self.lrelu(self.upconv2(F.interpolate(out, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(out)))

        if ENABLE_TANH:
            out = (self.tanh(out)+1)/2

        return out
        ###################################
    
    def forward_flops(self, size):
        flops_total = []

        flops, size = self.conv_first.forward_flops(size)
        flops_total.append(flops) 

        for i, cell in enumerate(self.cells):
            flops, size = cell.forward_flops(size)
            flops_total.append(flops)

        flops, size = self.trunk_conv.forward_flops(size)
        flops_total.append(flops)

        size = (size[0], size[1]*2, size[2]*2)
        flops, size = self.upconv1.forward_flops(size)
        flops_total.append(flops)

        size = (size[0], size[1]*2, size[2]*2)
        flops, size = self.upconv2.forward_flops(size)
        flops_total.append(flops)

        flops, size = self.HRconv.forward_flops(size)
        flops_total.append(flops)

        flops, size = self.conv_last.forward_flops(size)
        flops_total.append(flops)

        return sum(flops_total)

        ###################################

    def gram(self, x):
        (bs, ch, h, w) = x.size()
        f = x.view(bs, ch, w*h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (ch * h * w)
        return G


    # def _criterion(self, y_hat, x):
    #     base_loss = self.base_weight * nn.L1Loss()(y_hat, x)

    #     y_c_features = self.vgg(x)
    #     y_hat_features = self.vgg(y_hat)

    #     y_hat_gram = [self.gram(fmap) for fmap in y_hat_features]
    #     x_gram = [self.gram(fmap) for fmap in y_c_features]

    #     style_loss = 0
    #     for j in range(4):
    #         style_loss += self.style_weight * nn.functional.mse_loss(y_hat_gram[j], x_gram[j])

    #     recon = y_c_features[1]
    #     recon_hat = y_hat_features[1]
    #     content_loss = self.content_weight * nn.L1Loss()(recon_hat, recon)

    #     diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
    #     diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
    #     tv_loss = self.tv_weight * (diff_i + diff_j)

    #     total_loss = base_loss + style_loss + content_loss + tv_loss

    #     # print(style_loss.data, content_loss.data, tv_loss.data)

    #     return total_loss

    def _criterion(self, y_hat, x):
        base_loss = self.base_weight * self.loss_func(y_hat, x)

        y_c_features = self.vgg(x)
        y_hat_features = self.vgg(y_hat)

        content_loss = self.content_weight * self.loss_func(y_c_features, y_hat_features)

        diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
        tv_loss = self.tv_weight * (diff_i + diff_j)

        total_loss = base_loss + content_loss + tv_loss

        return total_loss


    def _loss(self, input, target):
        logit = self(input)
        loss = self._criterion(logit, target)

        return loss
