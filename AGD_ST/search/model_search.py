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
    def __init__(self, op, C_in, C_out, kernel_size=3 , stride=1, slimmable=True, width_mult_list=[1.], quantize=False, width_mult_list_left=None):
        super(SingleOp, self).__init__()
        self._op = op(C_in, C_out, kernel_size=kernel_size, stride=stride, slimmable=slimmable, width_mult_list=width_mult_list)
        self._width_mult_list = width_mult_list
        self.quantize = quantize
        self.slimmable = slimmable

        self._width_mult_list_left = width_mult_list_left if width_mult_list_left is not None else width_mult_list

    def set_prun_ratio(self, ratio):
        self._op.set_ratio(ratio)

    def forward(self, x, beta, ratio):
        if isinstance(ratio[0], torch.Tensor):
            ratio0 = self._width_mult_list_left[ratio[0].argmax()]
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
            ratio0 = self._width_mult_list_left[ratio[0].argmax()]
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
            flops_quant, _ = self._op.forward_flops(size, quantize=True)
            flops = beta[0] * flops_full + beta[1] * flops_quant
        elif self.quantize:
            flops, size_out = self._op.forward_flops(size, quantize=True)
        else:
            flops, size_out = self._op.forward_flops(size, quantize=False)

        flops = flops * r_score0 * r_score1

        return flops, size_out


class MixedOp(nn.Module):

    def __init__(self, C_in, C_out, stride=1, slimmable=True, width_mult_list=[1.], quantize=False, width_mult_list_left=None):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self._width_mult_list = width_mult_list
        self.quantize = quantize
        self.slimmable = slimmable
        
        self._width_mult_list_left = width_mult_list_left if width_mult_list_left is not None else width_mult_list

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
            ratio0 = self._width_mult_list_left[ratio[0].argmax()]
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
            ratio0 = self._width_mult_list_left[ratio[0].argmax()]
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
            ratio0 = self._width_mult_list_left[ratio[0].argmax()]
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


class NAS_GAN(nn.Module):
    def __init__(self, layers=16, slimmable=True, width_mult_list=[1.,], width_mult_list_sh=[1.,], loss_weight = [1e0, 1e5, 1e0, 1e-7], prun_modes='arch_ratio', quantize=False):
        super(NAS_GAN, self).__init__()
        assert layers >= 3
        self._layers = layers
        self._width_mult_list = width_mult_list
        self._width_mult_list_sh = width_mult_list_sh
        self._prun_modes = prun_modes
        self.prun_mode = None # prun_mode is higher priority than _prun_modes
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

        self.vgg = torch.nn.DataParallel(VGGFeature()).cuda()

        self.quantize = quantize
        self.slimmable = slimmable

        # self.stem = [nn.ReflectionPad2d(3), ConvNorm(3, 64, 7, slimmable=False)]
        self.stem = nn.ModuleList()
        self.stem.append(SingleOp(ConvNorm, 3, 64, 7, slimmable=slimmable, width_mult_list=width_mult_list_sh, quantize=quantize))

        in_features = 64
        out_features = in_features*2

        for _ in range(2):
            self.stem.append(SingleOp(ConvNorm, in_features, out_features, 3, stride=2, slimmable=slimmable, width_mult_list=width_mult_list_sh, quantize=quantize))
            in_features = out_features
            out_features = in_features*2

        self.cells = nn.ModuleList()
        for i in range(layers):
            if i == 0:
                op = MixedOp(in_features, in_features, slimmable=slimmable, width_mult_list=width_mult_list, quantize=quantize, width_mult_list_left=width_mult_list_sh)
            else:
                op = MixedOp(in_features, in_features, slimmable=slimmable, width_mult_list=width_mult_list, quantize=quantize)
            self.cells.append(op)

        self.header = nn.ModuleList()

        out_features = in_features//2

        self.header.append(SingleOp(ConvTranspose2dNorm, in_features, out_features, 3, stride=2, slimmable=slimmable, 
                                    width_mult_list=width_mult_list_sh, quantize=quantize, width_mult_list_left=width_mult_list))
        
        in_features = out_features
        out_features = in_features//2

        self.header.append(SingleOp(ConvTranspose2dNorm, in_features, out_features, 3, stride=2, slimmable=slimmable, 
                                    width_mult_list=width_mult_list_sh, quantize=quantize))

        self.header.append(SingleOp(Conv, 64, 3, 7, slimmable=slimmable, width_mult_list=width_mult_list_sh, quantize=quantize))

        self.tanh = nn.Tanh()
             
        # contains arch_params names: {"alpha": alpha, "beta": beta, "ratio": ratio}
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
            for layer in range(self._layers):
                ratio_sampled.append(gumbel_softmax(F.log_softmax(ratio[layer], dim=-1), hard=True))

            ratio_sh = self._arch_params["ratio_sh"]
            ratio_sampled_sh = []
            for layer in range(self.len_ratio_sh):
                ratio_sampled_sh.append(gumbel_softmax(F.log_softmax(ratio_sh[layer], dim=-1), hard=True))

            return ratio_sampled, ratio_sampled_sh

        elif mode == "min":
            ratio_sampled = []
            for layer in range(self._layers):
                ratio_sampled.append(self._width_mult_list[0])

            ratio_sampled_sh = []
            for layer in range(self.len_ratio_sh):
                ratio_sampled_sh.append(self._width_mult_list_sh[0])

            return ratio_sampled, ratio_sampled_sh

        elif mode == "max":
            ratio_sampled = []
            for layer in range(self._layers):
                ratio_sampled.append(self._width_mult_list[-1])

            ratio_sampled_sh = []
            for layer in range(self.len_ratio_sh):
                ratio_sampled_sh.append(self._width_mult_list_sh[-1])

            return ratio_sampled, ratio_sampled_sh

        elif mode == "random":
            ratio_sampled = []
            for layer in range(self._layers):
                ratio_sampled.append(np.random.choice(self._width_mult_list))
            
            ratio_sampled_sh = []
            for layer in range(self.len_ratio_sh):
                ratio_sampled_sh.append(np.random.choice(self._width_mult_list_sh))

            return ratio_sampled, ratio_sampled_sh


    def forward(self, input):
        alpha = F.softmax(getattr(self, "alpha"), dim=-1)
        beta = F.softmax(getattr(self, "beta"), dim=-1)

        beta_sh = F.softmax(getattr(self, "beta_sh"), dim=-1)

        if self.prun_mode is not None:
            ratio, ratio_sh = self.sample_prun_ratio(mode=self.prun_mode)
        else:
            ratio, ratio_sh = self.sample_prun_ratio(mode=self._prun_modes)

        # print('alpha:', alpha, 'beta:', beta, 'ratio:', ratio)

        out = input
        for i, module in enumerate(self.stem):
            if i == 0:
                out = module(out, beta_sh[i], [1, ratio_sh[i]])
            else:
                out = module(out, beta_sh[i], [ratio_sh[i-1], ratio_sh[i]])


        for i, cell in enumerate(self.cells):
            if i == 0:
                out = cell(out, alpha[i], beta[i], [ratio_sh[self.len_stem-1], ratio[i]])
            else:
                out = cell(out, alpha[i], beta[i], [ratio[i-1], ratio[i]])


        for i, module in enumerate(self.header):
            if i == 0:
                out = module(out, beta_sh[self.len_stem+i], [ratio[self._layers-1], ratio_sh[self.len_stem+i]])
            elif i == self.len_header-1:
                out = module(out, beta_sh[self.len_stem+i], [ratio_sh[self.len_stem+i-1], 1])
            else:
                out = module(out, beta_sh[self.len_stem+i], [ratio_sh[self.len_stem+i-1], ratio_sh[self.len_stem+i]])

        out = self.tanh(out)

        return out
        ###################################
    
    def forward_flops(self, size, alpha=True, beta=True, ratio=True):
        if alpha:
            alpha = F.softmax(getattr(self, "alpha"), dim=-1)
        else:
            alpha = torch.ones_like(getattr(self, 'alpha')).cuda() * 1./len(PRIMITIVES)

        if beta:
            beta = F.softmax(getattr(self, "beta"), dim=-1)
            beta_sh = F.softmax(getattr(self, "beta_sh"), dim=-1)
        else:
            beta = torch.ones_like(getattr(self, 'beta')).cuda() * 1./2
            beta_sh = torch.ones_like(getattr(self, 'beta_sh')).cuda() * 1./2

        if ratio:
            if self.prun_mode is not None:
                ratio, ratio_sh = self.sample_prun_ratio(mode=self.prun_mode)
            else:
                ratio, ratio_sh = self.sample_prun_ratio(mode=self._prun_modes)
        else:
            ratio, ratio_sh = self.sample_prun_ratio(mode='max')

        flops_total = []

        for i, module in enumerate(self.stem):
            if i == 0:
                flops, size = module.forward_flops(size, beta_sh[i], [1, ratio_sh[i]])  
                flops_total.append(flops)          
            else:
                flops, size = module.forward_flops(size, beta_sh[i], [ratio_sh[i-1], ratio_sh[i]])
                flops_total.append(flops)

        for i, cell in enumerate(self.cells):
            if i == 0:
                flops, size = cell.forward_flops(size, alpha[i], beta[i], [ratio_sh[self.len_stem-1], ratio[i]])
                flops_total.append(flops)
            else:
                flops, size = cell.forward_flops(size, alpha[i], beta[i], [ratio[i-1], ratio[i]])
                flops_total.append(flops)

        for i, module in enumerate(self.header):
            if i == 0:
                flops, size = module.forward_flops(size, beta_sh[self.len_stem+i], [ratio[self._layers-1], ratio_sh[self.len_stem+i]])
                flops_total.append(flops)
            elif i == self.len_header-1:
                flops, size = module.forward_flops(size, beta_sh[self.len_stem+i], [ratio_sh[self.len_stem+i-1], 1])
                flops_total.append(flops)
            else:
                flops, size = module.forward_flops(size, beta_sh[self.len_stem+i], [ratio_sh[self.len_stem+i-1], ratio_sh[self.len_stem+i]])
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
        tv_loss = self.tv_weight * (diff_i + diff_j)

        total_loss = base_loss + style_loss + content_loss + tv_loss

        # print(style_loss.data, content_loss.data, tv_loss.data)

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

        setattr(self, 'alpha', nn.Parameter(Variable(1e-3*torch.ones(self._layers, num_ops).cuda(), requires_grad=True)))
        setattr(self, 'beta', nn.Parameter(Variable(1e-3*torch.ones(self._layers, 2).cuda(), requires_grad=True)))

        if self._prun_modes == 'arch_ratio':
            # prunning ratio
            num_widths = len(self._width_mult_list)
            num_widths_sh = len(self._width_mult_list_sh)
        else:
            num_widths = 1
            num_widths_sh = 1

        setattr(self, 'ratio', nn.Parameter(Variable(1e-3*torch.ones(self._layers, num_widths).cuda(), requires_grad=True)))

        setattr(self, 'beta_sh', nn.Parameter(Variable(1e-3*torch.ones(self.len_beta_sh, 2).cuda(), requires_grad=True)))
        setattr(self, 'ratio_sh', nn.Parameter(Variable(1e-3*torch.ones(self.len_ratio_sh, num_widths_sh).cuda(), requires_grad=True)))

        return {"alpha": self.alpha, "beta": self.beta, "ratio": self.ratio, "beta_sh": self.beta, "ratio_sh": self.ratio_sh}


    def _reset_arch_parameters(self):
        num_ops = len(PRIMITIVES)
        if self._prun_modes == 'arch_ratio':
            # prunning ratio
            num_widths = len(self._width_mult_list)
            num_widths_sh = len(self._width_mult_list_sh)
        else:
            num_widths = 1
            num_widths_sh = 1

        getattr(self, "alpha").data = Variable(1e-3*torch.ones(self._layers, num_ops).cuda(), requires_grad=True)
        getattr(self, "beta").data = Variable(1e-3*torch.ones(self._layers, 2).cuda(), requires_grad=True)
        getattr(self, "ratio").data = Variable(1e-3*torch.ones(self._layers, num_widths).cuda(), requires_grad=True)

        getattr(self, "beta_sh").data = Variable(1e-3*torch.ones(self.len_beta_sh, 2).cuda(), requires_grad=True)
        getattr(self, "ratio_sh").data = Variable(1e-3*torch.ones(self.len_ratio_sh, num_widths_sh).cuda(), requires_grad=True)

        
