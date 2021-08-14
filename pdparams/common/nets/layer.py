import paddle.nn as nn
import paddle.nn.functional as F


def make_linear_layers(feat_dims, relu_final=True):
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i + 1]))
        if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2 and relu_final):
            layers.append(nn.ReLU())

    return nn.Sequential(*layers)


def make_conv_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(
            nn.Conv2D(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i + 1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
            ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2 and bnrelu_final):
            layers.append(nn.BatchNorm2D(feat_dims[i + 1]))
            layers.append(nn.ReLU())

    return nn.Sequential(*layers)


def make_deconv_layers(feat_dims, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(
            nn.Conv2DTranspose(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i + 1],
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias_attr=False))

        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2 and bnrelu_final):
            layers.append(nn.BatchNorm2D(feat_dims[i + 1]))
            layers.append(nn.ReLU())

    return nn.Sequential(*layers)


class Interpolate(nn.Layer):

    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x


def make_upsample_layers(feat_dims, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(
            Interpolate(2, 'bilinear'))
        layers.append(
            nn.Conv2D(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i + 1],
                kernel_size=3,
                stride=1,
                padding=1
            ))

        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2 and bnrelu_final):
            layers.append(nn.BatchNorm2D(feat_dims[i + 1]))
            layers.append(nn.ReLU())

    return nn.Sequential(*layers)


class ResBlock(nn.Layer):

    def __init__(self, in_feat, out_feat):
        super(ResBlock, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        self.conv = make_conv_layers([in_feat, out_feat, out_feat], bnrelu_final=False)
        self.bn = nn.BatchNorm2D(out_feat)
        if self.in_feat != self.out_feat:
            self.shortcut_conv = nn.Conv2D(in_feat, out_feat, kernel_size=1, stride=1, padding=0)
            self.shortcut_bn = nn.BatchNorm2D(out_feat)

    def forward(self, input):
        x = self.bn(self.conv(input))
        if self.in_feat != self.out_feat:
            x = F.relu(x + self.shortcut_bn(self.shortcut_conv(input)))
        else:
            x = F.relu(x + input)
        return x


def make_conv3d_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(
            nn.Conv3D(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i + 1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
            ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2 and bnrelu_final):
            layers.append(nn.BatchNorm3D(feat_dims[i + 1]))
            layers.append(nn.ReLU())

    return nn.Sequential(*layers)


def make_deconv3d_layers(feat_dims, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(
            nn.Conv3DTranspose(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i + 1],
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias_attr=False))

        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2 and bnrelu_final):
            layers.append(nn.BatchNorm3D(feat_dims[i + 1]))
            layers.append(nn.ReLU())

    return nn.Sequential(*layers)
