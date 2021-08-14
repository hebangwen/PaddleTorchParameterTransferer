import numpy as np
import paddle
import torch

from pdparams.common.model import get_model as get_paddle_model
from pth.common.model import get_model as get_torch_model


def compare(l1, l2):
    if len(l1) != len(l2):
        return False

    for i, j in zip(l1, l2):
        if i != j:
            return False

    return True


if __name__ == '__main__':
    paddle_model = get_paddle_model('train', 21)
    torch_model = get_torch_model('train', 21)

    torch_weight = torch.load('weights/snapshot_20.pth.tar', map_location=torch.device('cpu'))['network']
    # paddle实现的网络没有`module`变量保存, 所以需要修改键值对才能传送
    paddle_weight = paddle_model.state_dict()
    # assert len(torch_weight) == len(paddle_weight), '权重参数应该相等才能实现替换'
    print(len(torch_weight), len(paddle_weight))

    # https://zhuanlan.zhihu.com/p/188744602
    # 报错: `torch param <module.backbone_net.resnet.bn1.num_batches_tracked> not exist in paddle model` 这个是一个单独的用来设置动量的参数
    # paddle的Linear层的参数设置与torch的Linear层的参数设置不一样
    count = 0
    fc_weight_names = ['pose_net.root_fc.0.weight', 'pose_net.root_fc.2.weight', 'pose_net.hand_fc.0.weight', 'pose_net.hand_fc.2.weight']
    for name, params in torch_weight.items():
        temp = name.replace('module.', '')

        if temp in paddle_weight:
            torch_params = params.detach().numpy()
            paddle_params = paddle_weight[temp]
            if compare(params.shape, paddle_params.shape):
                paddle_weight[temp] = paddle.to_tensor(torch_params)
                count += 1
            elif temp in fc_weight_names:
                paddle_weight[temp] = paddle.to_tensor(torch_params.T)
                count += 1
            else:
                print(f'torch param <{name}> dose not match paddle param <{temp}>')
                print(f'torch shape: {torch_params.shape}, paddle shape: {paddle_params.shape}')

        elif 'running_mean' in name:
            torch_params = params.detach().numpy()
            paddle_params = paddle_weight[temp[:-12] + '_mean']
            if compare(params.shape, paddle_params.shape):
                paddle_weight[temp[:-12] + '_mean'] = paddle.to_tensor(torch_params)
                count += 1
            else:
                print(f'torch param <{name}> dose not match paddle param <{temp[:-12] + "_mean"}>')
                print(f'torch shape: {torch_params.shape}, paddle shape: {paddle_params.shape}')

        elif 'running_var' in name:
            torch_params = params.detach().numpy()
            paddle_params = paddle_weight[temp[:-11] + '_variance']
            if compare(params.shape, paddle_params.shape):
                paddle_weight[temp[:-11] + '_variance'] = paddle.to_tensor(torch_params)
                count += 1
            else:
                print(f'torch param <{name}> dose not match paddle param <{temp[:-11] + "_variance"}>')
                print(f'torch shape: {torch_params.shape}, paddle shape: {paddle_params.shape}')

        else:
            print(f'torch param <{name}> not exist in paddle model')

    print(count)
    paddle_model.load_dict(paddle_weight)
    paddle.save(paddle_weight, 'weights/interhand.pdparams')

    print('done!')