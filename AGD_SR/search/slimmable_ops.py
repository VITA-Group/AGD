import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as bp
from quantize import QConv2d, QConvTranspose2d, calculate_qparams, Quantize, QuantMeasure


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


class USConv2d(QConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, width_mult_list=[1.], num_bits=8, num_bits_weight=8):
        super(USConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias, num_bits=num_bits, num_bits_weight=num_bits_weight)

        self.in_channels_max = in_channels
        self.out_channels_max = out_channels
        self.width_mult_list = width_mult_list
        self.ratio = (1., 1.)
    
    def set_ratio(self, ratio):
        self.ratio = ratio

    def forward(self, input, quantize=False):
        assert self.ratio[0] in self.width_mult_list, str(self.ratio[0]) + " in? " + str(self.width_mult_list)
        self.in_channels = make_divisible(self.in_channels_max * self.ratio[0])
        assert self.ratio[1] in self.width_mult_list, str(self.ratio[1]) + " in? " + str(self.width_mult_list)
        self.out_channels = make_divisible(self.out_channels_max * self.ratio[1])

        weight = self.weight[:self.out_channels, :self.in_channels, :, :]

        if self.groups != 1:
            self.groups = self.out_channels

        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias

        if quantize:
            if not hasattr(self, 'quantize_input'):
                self.quantize_input = QuantMeasure(self.num_bits, shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1))

            qinput = self.quantize_input(input)
            weight_qparams = calculate_qparams(
                weight, num_bits=self.num_bits_weight, flatten_dims=(1, -1), reduce_dim=None)
            qweight = Quantize(weight, qparams=weight_qparams)

            if self.bias is not None:
                qbias = Quantize(
                    bias, num_bits=self.num_bits_weight + self.num_bits,
                    flatten_dims=(0, -1))
            else:
                qbias = None
            
            output = F.conv2d(qinput, qweight, qbias, self.stride,
                                  self.padding, self.dilation, self.groups)

        else:
            output = F.conv2d(input, weight, bias, self.stride,
                                  self.padding, self.dilation, self.groups)
        return output


class USConvTranspose2d(QConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1, dilation=1, groups=1, bias=True, width_mult_list=[1.], num_bits=8, num_bits_weight=8):
        super(USConvTranspose2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation,
            groups=groups, bias=bias, num_bits=num_bits, num_bits_weight=num_bits_weight)

        self.in_channels_max = in_channels
        self.out_channels_max = out_channels
        self.width_mult_list = width_mult_list
        self.ratio = (1., 1.)
    
    def set_ratio(self, ratio):
        self.ratio = ratio

    def forward(self, input, quantize=False):
        assert self.ratio[0] in self.width_mult_list, str(self.ratio[0]) + " in? " + str(self.width_mult_list)
        self.in_channels = make_divisible(self.in_channels_max * self.ratio[0])
        assert self.ratio[1] in self.width_mult_list, str(self.ratio[1]) + " in? " + str(self.width_mult_list)
        self.out_channels = make_divisible(self.out_channels_max * self.ratio[1])

        weight = self.weight[ :self.in_channels, :self.out_channels, :, :]

        if self.groups != 1:
            self.groups = self.out_channels

        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias

        if quantize:
            if not hasattr(self, 'quantize_input'):
                self.quantize_input = QuantMeasure(self.num_bits, shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1))
                
            qinput = self.quantize_input(input)
            weight_qparams = calculate_qparams(
                weight, num_bits=self.num_bits_weight, flatten_dims=(1, -1), reduce_dim=None)
            qweight = Quantize(weight, qparams=weight_qparams)

            if self.bias is not None:
                qbias = Quantize(
                    bias, num_bits=self.num_bits_weight + self.num_bits,
                    flatten_dims=(0, -1))
            else:
                qbias = None
            
            output = F.conv_transpose2d(qinput, qweight, qbias, self.stride,
                                        self.padding, self.output_padding, self.groups, self.dilation)

        else:
            output = F.conv_transpose2d(input, weight, bias, self.stride,
                                        self.padding, self.output_padding, self.groups, self.dilation)
        return output


class USBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, width_mult_list=[1.]):
        super(USBatchNorm2d, self).__init__(
            num_features, affine=False, track_running_stats=False)
        self.num_features_max = num_features
        self.width_mult_list = width_mult_list
        # for tracking performance during training
        self.bn = nn.ModuleList(
            [ nn.BatchNorm2d(i) for i in [ make_divisible(self.num_features_max * width_mult) for width_mult in width_mult_list ] ]
        )
        self.ratio = 1.
    
    def set_ratio(self, ratio):
        self.ratio = ratio

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        assert self.ratio in self.width_mult_list
        idx = self.width_mult_list.index(self.ratio)
        y = self.bn[idx](input)

        return y
