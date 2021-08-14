import numpy as np
import paddle
import torch

from pdparams.common.model import Model as PaddleModel
from pth.common.model import Model as TorchModel


if __name__ == '__main__':
    print(PaddleModel)

    print(TorchModel)

    print('done!')