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


def compare_gather():
    shape = (5, 1, 5)
    data = np.random.randint(0, 100, shape)

    t1 = paddle.to_tensor(data)
    t2 = torch.from_numpy(data)
    print(t1)
    print(t2)

    # 0, 1, 2, 3, 4
    # 4, 3, 2, 1, 0
    dim1 = paddle.to_tensor([4, 3, 2])
    dim2 = torch.tensor([4, 3, 2]).repeat(5, 1, 1)
    print(dim1)
    print(dim2)

    out1 = paddle.gather(t1, dim1, 2)
    out2 = torch.gather(t2, 2, dim2)

    print(out1)
    print(out2)


if __name__ == '__main__':
    compare_gather()