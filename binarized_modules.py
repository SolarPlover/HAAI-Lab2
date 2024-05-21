import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd.function import Function, InplaceFunction

import numpy as np


class Binarize(InplaceFunction):

    def forward(ctx, input, quant_mode='det', allow_scale=False, inplace=False):
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        scale = output.abs().max() if allow_scale else 1

        if quant_mode == 'det':
            return output.div(scale).sign().mul(scale)
        else:
            return (output.div(scale).add_(1).div_(2).add_(torch.rand(output.size()).add(-0.5)).clamp_(0, 1).round().
                    mul_(2).add_(-1).mul(scale))

    def backward(ctx, grad_output):
        # STE
        grad_input = grad_output
        return grad_input, None, None, None


class Quantize(InplaceFunction):
    def forward(ctx, input, quant_mode='det', numBits=4, inplace=False):
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        scale = (2 ** numBits - 1) / (output.max() - output.min())
        output = output.mul(scale).clamp(-2 ** (numBits - 1) + 1, 2 ** (numBits - 1))
        if quant_mode == 'det':
            output = output.round().div(scale)
        else:
            output = output.round().add(torch.rand(output.size()).add(-0.5)).div(scale)
        return output

    def backward(grad_output):
        # STE
        grad_input = grad_output
        return grad_input, None, None


def binarized(input, quant_mode='det'):
    return Binarize.apply(input, quant_mode)


def quantize(input, quant_mode, numBits):
    return Quantize.apply(input, quant_mode, numBits)


class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        if input.size(1) != 784:
            input_b = binarized(input)
        else:
            input_b = input
        weight_b = binarized(self.weight)
        out = nn.functional.linear(input_b, weight_b)
        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out 