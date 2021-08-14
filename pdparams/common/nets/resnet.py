import paddle
import paddle.nn as nn
from paddle.vision.models.resnet import BasicBlock,BottleneckBlock,model_urls
import paddle.utils.download as download

class ResNetBackbone(nn.Layer):

    def __init__(self, resnet_type):

        resnet_spec = {18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'),
                       34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'),
                       50: (BottleneckBlock, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'),
                       101: (BottleneckBlock, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'),
                       152: (BottleneckBlock, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152')}
        block, layers, channels, name = resnet_spec[resnet_type]

        self.name = name
        self.inplanes = 64
        super(ResNetBackbone, self).__init__()
        self.conv1 = nn.Conv2D(3, 64, kernel_size=7, stride=2, padding=3,
                               bias_attr=False)  # RGB
        self.bn1 = nn.BatchNorm2D(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                m.weight=m.create_parameter(m.weight.shape,m._param_attr,m.weight.dtype,nn.initializer.Normal(mean=0, std=0.001))
                #(nn.initializer.Normal(mean=0, std=0.001))
            elif isinstance(m, nn.BatchNorm2D):
                m.weight=m.create_parameter(m.weight.shape,m._weight_attr,m.weight.dtype,nn.initializer.Constant(value=1))
                m.bias=m.create_parameter(attr=m._bias_attr,shape=m.bias.shape,default_initializer=nn.initializer.Constant(value=0),is_bias=True)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def init_weights(self):

        org_resnet_path = download.get_weights_path_from_url(model_urls[self.name][0])
        # drop orginal resnet fc layer, add 'None' in case of no fc layer, that will raise error
        org_resnet=paddle.load(org_resnet_path)
        org_resnet.pop('fc.weight', None)
        org_resnet.pop('fc.bias', None)

        self.set_state_dict(org_resnet)
        print("Initialize resnet from model zoo")
