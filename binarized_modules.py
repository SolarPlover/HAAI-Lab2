import torch.nn as nn
from torch.autograd.function import InplaceFunction

class Binarize(InplaceFunction):

    def forward(ctx, input, allow_scale=False, inplace=False):
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        scale = output.abs().max() if allow_scale else 1
        return output.div(scale).sign().mul(scale)

    def backward(ctx, grad_output):
        # STE
        grad_input = grad_output
        return grad_input, None, None, None

def binarized(input):
    return Binarize.apply(input)

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