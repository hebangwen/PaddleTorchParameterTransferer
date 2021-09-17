from paddle.vision.models.resnet import resnet50 as paddle_resnet
from torchvision.models.resnet import resnet50 as torch_resnet
import numpy as np
import paddle
import torch


def compare(l1, l2):
    if len(l1) != len(l2):
        return False

    for i, j in zip(l1, l2):
        if i != j:
            return False

    return True


if __name__ == "__main__":
    torch_model = torch_resnet(pretrained=True)
    paddle_model = paddle_resnet(pretrained=False)

    paddle_weight = paddle_model.state_dict()
    torch_weight = torch_model.state_dict()
    print(len(torch_weight), len(paddle_weight))

    count = 0
    fc_weight_names = ['fc.weight']
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

    print("convert finished")
    paddle_model.load_dict(paddle_weight)

    for k in paddle_weight:
        if 'bn' in k:
            print(k)

    torch_model.eval()
    paddle_model.eval()

    inputs = torch.randn((1, 3, 224, 224)).detach().cpu().numpy()
    torch_output = torch_model(torch.tensor(inputs)).detach().cpu().numpy()
    paddle_output = paddle_model(paddle.to_tensor(inputs)).detach().cpu().numpy()
    dist = np.linalg.norm(torch_output - paddle_output)
    print(dist)
