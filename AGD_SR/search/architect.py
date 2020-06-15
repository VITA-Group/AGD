import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchcontrib
import numpy as np
from pdb import set_trace as bp
from thop import profile
from operations import *
from genotypes import PRIMITIVES


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, args, distill=False):
        # self.network_momentum = args.momentum
        # self.network_weight_decay = args.weight_decay
        self.model = model
        self._args = args
        self.quantize = args.quantize
        self._distill = distill
        self._kl = nn.KLDivLoss().cuda()

        self.optimizer = torch.optim.Adam(list(self.model.module._arch_params.values()), lr=args.arch_learning_rate, betas=(0.5, 0.999))#, weight_decay=args.arch_weight_decay)
        
        self.flops_weight = args.flops_weight

        print("architect initialized!")


    def step(self, input_train, target_train, input_valid, target_valid, eta=None, network_optimizer=None, unrolled=False):
        self.optimizer.zero_grad()
        if unrolled:
            loss = self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else:
            loss, loss_flops = self._backward_step(input_valid, target_valid)

        # loss.backward(retain_graph=True)

        # # bp()
        # if loss_flops != 0: 
        #     loss_flops.backward(retain_graph=True)

        if loss_flops != 0:
            loss += loss_flops

        loss.backward()

        self.optimizer.step()

        return loss


    def _backward_step(self, input_valid, target_valid):
        loss = self.model.module._loss(input_valid, target_valid)
        self.model.module.prun_mode = None

        flops = 0

        if len(self.model.module._width_mult_list) == 1:
            if self.quantize == 'search':
                # r0 = 1/2; r1 = 1/2
                r0 = self._args.alpha_weight
                r1 = self._args.beta_weight
            else:
                r0 = 1
            
            flops = flops + r0 * self.model.module.forward_flops((3, 510, 350), alpha=True, beta=False, ratio=False)
            
            if self.quantize == 'search':
                flops = flops + r1 * self.model.module.forward_flops((3, 510, 350), alpha=False, beta=True, ratio=False)

        else:
            if self.quantize == 'search':
                # r0 = 1/3; r1 = 1/3; r2 = 1/3
                r0 = self._args.alpha_weight
                r1 = self._args.beta_weight
                r2 = self._args.ratio_weight
            else:
                # r0 = 1/2; r2 = 1/2
                r0 = self._args.alpha_weight
                r2 = self._args.ratio_weight


            flops = flops + r0 * self.model.module.forward_flops((3, 510, 350), alpha=True, beta=False, ratio=False)

            if self.quantize == 'search':
                flops = flops + r1 * self.model.module.forward_flops((3, 510, 350), alpha=False, beta=True, ratio=False)

            flops = flops + r2 * self.model.module.forward_flops((3, 510, 350), alpha=False, beta=False, ratio=True)

        self.flops_supernet = flops
        loss_flops = self.flops_weight * flops

        # print(flops, loss_flops, loss)
        return loss, loss_flops


