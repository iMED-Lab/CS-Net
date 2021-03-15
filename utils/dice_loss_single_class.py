import torch
from torch.autograd import Function, Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        # target = _make_one_hot(target, 2)
        self.save_for_backward(input, target)
        eps = 0.0001
        # dot是返回两个矩阵的点集
        # inter,uniun:两个值的大小分别是10506.6,164867.2
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps
        # print("inter,uniun:",self.inter,self.union)

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        # 这里没有打印出来，难道没有执行到这里吗
        # print("grad_input, grad_target:",grad_input, grad_target)

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    # print("size of input, target:", input.shape, target.shape)

    for i, c in enumerate(zip(input, target)):
        # c[0],c[1]的大小都是原图大小torch.Size([1, 576, 544])
        # print("size of c0 c1:", c[0].shape,c[1].shape)
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


def dice_coeff_loss(input, target):
    return 1 - dice_coeff(input, target)
