from torch import nn

from .utils import get_act
from .norm import Norm1D, Norm2D


class Dense(nn.Module):

    def __init__(self, input_dim, output_dim, bias=True, norm1d='none', norm2d='none', act='none'):
        super(Dense, self).__init__()
        assert norm1d == 'none' or norm2d == 'none', "one of [norm1d, norm2d] must be none"

        if norm1d != 'none':
            self.nn_norm = Norm1D(input_dim, norm1d)
        elif norm2d != 'none':
            self.nn_norm = Norm2D(input_dim, norm2d)
        else:
            self.nn_norm = None

        self.nn_act = get_act(act)
        self.nn_linear = nn.Linear(input_dim, output_dim, bias)

    def weight(self):
        return self.nn_linear.weight

    def forward(self, x):
        if self.nn_norm is not None:
            x = self.nn_norm(x)
        x = self.nn_act(x)
        x = self.nn_linear(x)
        return x
