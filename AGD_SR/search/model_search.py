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


class SingleOp(nn.Module):
    def __init__(self, op, C_in, C_out, kernel_size=3 , stride=1, slimmable=True, width_mult_list=[1.], quantize=True):
        super(SingleOp, self).__init__()
        self._op = op(C_in, C_out, kernel_size=kernel_size, stride=stride, slimmable=slimmable, width_mult_list=width_mult_list)
        self._width_mult_list = width_mult_list
        self.quantize = quantize
        self.slimmable = slimmable

    def set_prun_ratio(self, ratio):
        self._op.set_ratio(ratio)

    def forward(self, x, beta, ratio):
        if isinstance(ratio[0], torch.Tensor):
            ratio0 = self._width_mult_list[ratio[0].argmax()]
            r_score0 = ratio[0][ratio[0].argmax()]
        else:
            ratio0 = ratio[0]
            r_score0 = 1.
        if isinstance(ratio[1], torch.Tensor):
            ratio1 = self._width_mult_list[ratio[1].argmax()]
            r_score1 = ratio[1][ratio[1].argmax()]
        else:
            ratio1 = ratio[1]
            r_score1 = 1.

        if self.slimmable:
            self.set_prun_ratio((ratio0, ratio1))

        if self.quantize == 'search':
            result = (beta[0]*self._op(x, quantize=False) + beta[1]*self._op(x, quantize=True)) * r_score0 * r_score1
        elif self.quantize:
            result = self._op(x, quantize=True) * r_score0 * r_score1
        else:
            result = self._op(x, quantize=False) * r_score0 * r_score1

        return result

    def forward_flops(self, size, beta, ratio):
        if isinstance(ratio[0], torch.Tensor):
            ratio0 = self._width_mult_list[ratio[0].argmax()]
            r_score0 = ratio[0][ratio[0].argmax()]
        else:
            ratio0 = ratio[0]
            r_score0 = 1.
        if isinstance(ratio[1], torch.Tensor):
            ratio1 = self._width_mult_list[ratio[1].argmax()]
            r_score1 = ratio[1][ratio[1].argmax()]
        else:
            ratio1 = ratio[1]
            r_score1 = 1.

        if self.slimmable:
            self.set_prun_ratio((ratio0, ratio1))

        if self.quantize == 'search':
            flops_full, size_out = self._op.forward_flops(size, quantize=False)
            flops_quant, _ = op.forward_flops(size, quantize=True)
            flops = beta[0] * flops_full + beta[1] * flops_quant
        elif self.quantize:
            flops, size_out = op.forward_flops(size, quantize=True)
        else:
            flops, size_out = self._op.forward_flops(size, quantize=False)

        flops = flops * r_score0 * r_score1

        return flops, size_out


class MixedOp(nn.Module):

    def __init__(self, C_in, C_out, stride=1, slimmable=True, width_mult_list=[1.], quantize=True):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self._width_mult_list = width_mult_list
        self.quantize = quantize
        self.slimmable = slimmable

        for primitive in PRIMITIVES:
            op = OPS[primitive](C_in, C_out, stride, slimmable=slimmable, width_mult_list=width_mult_list)
            self._ops.append(op)

    def set_prun_ratio(self, ratio):
        for op in self._ops:
            op.set_ratio(ratio)

    def forward(self, x, alpha, beta, ratio):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0
        if isinstance(ratio[0], torch.Tensor):
            ratio0 = self._width_mult_list[ratio[0].argmax()]
            r_score0 = ratio[0][ratio[0].argmax()]
        else:
            ratio0 = ratio[0]
            r_score0 = 1.
        if isinstance(ratio[1], torch.Tensor):
            ratio1 = self._width_mult_list[ratio[1].argmax()]
            r_score1 = ratio[1][ratio[1].argmax()]
        else:
            ratio1 = ratio[1]
            r_score1 = 1.

        if self.slimmable:
            self.set_prun_ratio((ratio0, ratio1))

        for w, op in zip(alpha, self._ops):
            if self.quantize == 'search':
                result = result + (beta[0]*op(x, quantize=False) + beta[1]*op(x, quantize=True)) * w * r_score0 * r_score1
            elif self.quantize:
                result = result + op(x, quantize=True) * w * r_score0 * r_score1
            else:
                result = result + op(x, quantize=False) * w * r_score0 * r_score1
            # print(type(op), result.shape)
        return result


    def forward_latency(self, size, alpha, ratio):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0
        if isinstance(ratio[0], torch.Tensor):
            ratio0 = self._width_mult_list[ratio[0].argmax()]
            r_score0 = ratio[0][ratio[0].argmax()]
        else:
            ratio0 = ratio[0]
            r_score0 = 1.
        if isinstance(ratio[1], torch.Tensor):
            ratio1 = self._width_mult_list[ratio[1].argmax()]
            r_score1 = ratio[1][ratio[1].argmax()]
        else:
            ratio1 = ratio[1]
            r_score1 = 1.

        if self.slimmable:
            self.set_prun_ratio((ratio0, ratio1))

        for w, op in zip(alpha, self._ops):
            latency, size_out = op.forward_latency(size)
            result = result + latency * w * r_score0 * r_score1
        return result, size_out


    def forward_flops(self, size, alpha, beta, ratio):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0
        if isinstance(ratio[0], torch.Tensor):
            ratio0 = self._width_mult_list[ratio[0].argmax()]
            r_score0 = ratio[0][ratio[0].argmax()]
        else:
            ratio0 = ratio[0]
            r_score0 = 1.
        if isinstance(ratio[1], torch.Tensor):
            ratio1 = self._width_mult_list[ratio[1].argmax()]
            r_score1 = ratio[1][ratio[1].argmax()]
        else:
            ratio1 = ratio[1]
            r_score1 = 1.

        if self.slimmable:
            self.set_prun_ratio((ratio0, ratio1))

        for w, op in zip(alpha, self._ops):
            if self.quantize == 'search':
                flops_full, size_out = op.forward_flops(size, quantize=False)
                flops_quant, _ = op.forward_flops(size, quantize=True)
                flops = (beta[0] * flops_full + beta[1] * flops_quant)

            elif self.quantize:
                flops, size_out = op.forward_flops(size, quantize=True)

            else:
                flops, size_out = op.forward_flops(size, quantize=False)

            result = result + flops * w * r_score0 * r_score1

        return result, size_out


class Cell(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf=64, op_per_cell=5, slimmable=True, width_mult_list=[1.], quantize=False):
        super(Cell, self).__init__()

        self.nf = nf
        self.op_per_cell = op_per_cell
        self.slimmable = slimmable
        self._width_mult_list = width_mult_list
        self.quantize = quantize

        self.ops = nn.ModuleList()

        for _ in range(op_per_cell):
            self.ops.append(MixedOp(self.nf, self.nf, slimmable=slimmable, width_mult_list=width_mult_list, quantize=quantize))

    def forward(self, x, alpha, beta, ratio):
        out = x

        for i, op in enumerate(self.ops):
            if i == 0:
                out = op(out, alpha[i], beta[i], [1, ratio[i]])
            elif i == self.op_per_cell - 1:
                out = op(out, alpha[i], beta[i], [ratio[i-1], 1])
            else:
                out = op(out, alpha[i], beta[i], [ratio[i-1], ratio[i]])

        return out*0.2 + x


    def forward_flops(self, size, alpha, beta, ratio):
        flops_total = []

        for i, op in enumerate(self.ops):
            if i == 0:
                flops, size = op.forward_flops(size, alpha[i], beta[i], [1, ratio[i]])
                flops_total.append(flops)
            elif i == self.op_per_cell - 1:
                flops, size = op.forward_flops(size, alpha[i], beta[i], [ratio[i-1], 1])
                flops_total.append(flops)               
            else:
                flops, size = op.forward_flops(size, alpha[i], beta[i], [ratio[i-1], ratio[i]])
                flops_total.append(flops)

        return sum(flops_total), size


class NAS_GAN(nn.Module):
    def __init__(self, num_cell=5, op_per_cell=5, slimmable=True, width_mult_list=[1.,], loss_weight = [1e0, 1e5, 1e0, 1e-7], 
                prun_modes='arch_ratio', loss_func='MSE', before_act=True, quantize=False):

        super(NAS_GAN, self).__init__()
        
        self.num_cell = num_cell
        self.op_per_cell = op_per_cell

        self._layers = num_cell * op_per_cell

        self._width_mult_list = width_mult_list
        self._prun_modes = prun_modes
        self.prun_mode = None # prun_mode is higher priority than _prun_modes
        self._flops = 0
        self._params = 0

        self.base_weight = loss_weight[0]
        self.style_weight = loss_weight[1]
        self.content_weight = loss_weight[2]
        self.tv_weight = loss_weight[3]

        self.vgg = torch.nn.DataParallel(VGGFeature(before_act=before_act)).cuda()

        self.quantize = quantize
        self.slimmable = slimmable

        self.nf = 64

        self.conv_first = Conv(3, self.nf, 3, 1, 1, bias=True)
        
        self.cells = nn.ModuleList()
        for i in range(self.num_cell):
            cell = Cell(self.nf, op_per_cell=op_per_cell, slimmable=slimmable, width_mult_list=width_mult_list, quantize=quantize)
            self.cells.append(cell)

        self.trunk_conv = Conv(self.nf, self.nf, 3, 1, 1, bias=True)
        self.upconv1 = Conv(self.nf, self.nf, 3, 1, 1, bias=True)
        self.upconv2 = Conv(self.nf, self.nf, 3, 1, 1, bias=True)
        self.HRconv = Conv(self.nf, self.nf, 3, 1, 1, bias=True)
        self.conv_last = Conv(self.nf, 3, 3, 1, 1, bias=True)
                        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

        self.tanh = nn.Tanh()

        self.loss_func = nn.MSELoss() if loss_func == 'MSE' else nn.L1Loss()

        self._arch_params = self._build_arch_parameters()
        self._reset_arch_parameters()


    def sample_prun_ratio(self, mode="arch_ratio"):
        '''
        mode: "min"|"max"|"random"|"arch_ratio"(default)
        '''
        assert mode in ["min", "max", "random", "arch_ratio"]
        if mode == "arch_ratio":
            ratio = self._arch_params["ratio"]
            ratio_sampled = []
            for cell_id in range(self.num_cell):
                ratio_cell = []
                for op_id in range(self.op_per_cell-1):
                    ratio_cell.append(gumbel_softmax(F.log_softmax(ratio[cell_id][op_id], dim=-1), hard=True))
                ratio_sampled.append(ratio_cell)

            return ratio_sampled

        elif mode == "min":
            ratio_sampled = []
            for cell_id in range(self.num_cell):
                ratio_cell = []
                for op_id in range(self.op_per_cell-1):
                    ratio_cell.append(self._width_mult_list[0])
                ratio_sampled.append(ratio_cell)

            return ratio_sampled

        elif mode == "max":
            ratio_sampled = []
            for cell_id in range(self.num_cell):
                ratio_cell = []
                for op_id in range(self.op_per_cell-1):
                    ratio_cell.append(self._width_mult_list[-1])
                ratio_sampled.append(ratio_cell)

            return ratio_sampled

        elif mode == "random":
            ratio_sampled = []

            for cell_id in range(self.num_cell):
                ratio_cell = []
                for op_id in range(self.op_per_cell-1):
                    ratio_cell.append(np.random.choice(self._width_mult_list))
                ratio_sampled.append(ratio_cell)
        
            return ratio_sampled


    def forward(self, input):
        alpha = F.softmax(getattr(self, "alpha"), dim=-1)
        beta = F.softmax(getattr(self, "beta"), dim=-1)

        if self.prun_mode is not None:
            ratio = self.sample_prun_ratio(mode=self.prun_mode)
        else:
            ratio = self.sample_prun_ratio(mode=self._prun_modes)

        out = orig = self.conv_first(input)

        for i, cell in enumerate(self.cells):
            out = cell(out, alpha[i], beta[i], ratio[i])

        out = self.trunk_conv(out)

        out = out + orig

        out = self.lrelu(self.upconv1(F.interpolate(out, scale_factor=2, mode='nearest')))
        out = self.lrelu(self.upconv2(F.interpolate(out, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(out)))

        if ENABLE_TANH:
            out = (self.tanh(out)+1)/2

        return out

    
    def forward_flops(self, size, alpha=True, beta=True, ratio=True):
        if alpha:
            alpha = F.softmax(getattr(self, "alpha"), dim=-1)
        else:
            alpha = torch.ones_like(getattr(self, 'alpha')).cuda() * 1./len(PRIMITIVES)

        if beta:
            beta = F.softmax(getattr(self, "beta"), dim=-1)
        else:
            beta = torch.ones_like(getattr(self, 'beta')).cuda() * 1./2

        if ratio:
            if self.prun_mode is not None:
                ratio = self.sample_prun_ratio(mode=self.prun_mode)
            else:
                ratio = self.sample_prun_ratio(mode=self._prun_modes)
        else:
            ratio = self.sample_prun_ratio(mode='max')

        flops_total = []

        flops, size = self.conv_first.forward_flops(size)
        flops_total.append(flops) 
        
        for i, cell in enumerate(self.cells):
            flops, size = cell.forward_flops(size, alpha[i], beta[i], ratio[i])
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


    def _loss(self, input, target, pretrain=False):
        loss = 0
        if pretrain is not True:
            # "random width": sampled by gambel softmax
            self.prun_mode = None
            logit = self(input)
            loss = loss + self._criterion(logit, target)

        if len(self._width_mult_list) > 1:
            self.prun_mode = "max"

            logit = self(input)
            loss = loss + self._criterion(logit, target)

            self.prun_mode = "min"
            logit = self(input)
            loss = loss + self._criterion(logit, target)

            if pretrain == True:
                self.prun_mode = "random"
                logit = self(input)
                loss = loss + self._criterion(logit, target)

                self.prun_mode = "random"
                logit = self(input)
                loss = loss + self._criterion(logit, target)

        elif pretrain == True and len(self._width_mult_list) == 1:
            self.prun_mode = "max"
            logit = self(input)
            loss = loss + self._criterion(logit, target)

        return loss


    def _build_arch_parameters(self):
        num_ops = len(PRIMITIVES)

        setattr(self, 'alpha', nn.Parameter(Variable(1e-3*torch.ones(self.num_cell, self.op_per_cell, num_ops).cuda(), requires_grad=True)))
        setattr(self, 'beta', nn.Parameter(Variable(1e-3*torch.ones(self.num_cell, self.op_per_cell, 2).cuda(), requires_grad=True)))

        if self._prun_modes == 'arch_ratio':
            # prunning ratio
            num_widths = len(self._width_mult_list)
        else:
            num_widths = 1

        setattr(self, 'ratio', nn.Parameter(Variable(1e-3*torch.ones(self.num_cell, self.op_per_cell-1, num_widths).cuda(), requires_grad=True)))

        return {"alpha": self.alpha, "beta": self.beta, "ratio": self.ratio}


    def _reset_arch_parameters(self):
        num_ops = len(PRIMITIVES)
        if self._prun_modes == 'arch_ratio':
            # prunning ratio
            num_widths = len(self._width_mult_list)
        else:
            num_widths = 1

        getattr(self, "alpha").data = Variable(1e-3*torch.ones(self.num_cell, self.op_per_cell, num_ops).cuda(), requires_grad=True)
        getattr(self, "beta").data = Variable(1e-3*torch.ones(self.num_cell, self.op_per_cell, 2).cuda(), requires_grad=True)
        getattr(self, "ratio").data = Variable(1e-3*torch.ones(self.num_cell, self.op_per_cell-1, num_widths).cuda(), requires_grad=True)

        
