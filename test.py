import paddle
import torch
import numpy as np


def compare_linear():
    pl = paddle.nn.Linear(in_features=8, out_features=4)
    tl = torch.nn.Linear(in_features=8, out_features=4)

    print(pl.weight.shape, tl.weight.shape)

    x1 = torch.ones((4, 8))
    x1 = tl(x1)

    weights = tl.weight.detach().numpy().T
    bias = tl.bias.detach().numpy().T
    pl.weight.set_value(weights)
    pl.bias.set_value(bias)
    x2 = paddle.ones((4, 8))
    x2 = pl(x2)

    x1 = x1.detach().numpy()
    x2 = x2.detach().numpy()
    print(np.linalg.norm(x1 - x2))


if __name__ == '__main__':
    compare_linear()