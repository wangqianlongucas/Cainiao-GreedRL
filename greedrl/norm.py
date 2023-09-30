import torch

from torch import nn


class Norm1D(nn.Module):

    def __init__(self, dim, ntype='batch', affine=False):
        super(Norm1D, self).__init__()
        clazz_dict = {'batch': nn.BatchNorm1d, 'instance': nn.InstanceNorm1d}
        self.nn_norm = clazz_dict[ntype](dim, eps=1e-10, affine=affine)

    def forward(self, x):
        return self.nn_norm(x.permute(0, 2, 1)).permute(0, 2, 1)


class Norm2D(nn.Module):

    def __init__(self, dim, ntype='batch', affine=False):
        super(Norm2D, self).__init__()
        clazz_dict = {'batch': nn.BatchNorm2d, 'instance': nn.InstanceNorm2d}
        self.nn_norm = clazz_dict[ntype](dim, eps=1e-10, affine=affine)

    def forward(self, x):
        return self.nn_norm(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
