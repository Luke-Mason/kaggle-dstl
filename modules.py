import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from HyperParams import HyperParams

def conv3x3(in_, out):
    """
    3x3 convolution with padding
    """
    return nn.Conv2d(in_, out, 3, padding=1)


def concat(xs):
    """
    Concatenate a list of tensors along the channel dimension
    :param xs: List of tensors
    :return: Concatenated tensor
    """
    return torch.cat(xs, 1)

class UNetModule(nn.Module):
    """
    A single module of the UNet
    """
    def __init__(self, hps: HyperParams, in_: int, out: int):
        super().__init__()
        self.conv1 = conv3x3(in_, out)
        self.conv2 = conv3x3(out, out)
        self.bn = hps.bn
        self.activation = getattr(F, hps.activation)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(out)
            self.bn2 = nn.BatchNorm2d(out)

    def forward(self, x):
        x = self.conv1(x)
        if self.bn:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        if self.bn:
            x = self.bn2(x)
        x = self.activation(x)
        return x


class DenseLayer(nn.Module):
    """
    A dense layer
    """
    def __init__(self, in_, out, *, dropout, bn):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_) if bn else None
        self.activation = nn.ReLU(inplace=True)
        self.conv = conv3x3(in_, out)
        self.dropout = nn.Dropout2d(p=dropout) if dropout else None

    def forward(self, x):
        x = self.activation(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class DenseBlock(nn.Module):
    """
    A dense block with n_layers dense layers and k output channels
    """
    def __init__(self, in_, k, n_layers, dropout, bn):
        super().__init__()
        self.out = k * n_layers
        layer_in = in_
        self.layers = []
        for i in range(n_layers):
            layer = DenseLayer(layer_in, k, dropout=dropout, bn=bn)
            self.layers.append(layer)
            setattr(self, 'layer_{}'.format(i), layer)
            layer_in += k

    def forward(self, x):
        inputs = [x]
        outputs = []
        for i, layer in enumerate(self.layers[:-1]):
            outputs.append(layer(inputs[i]))
            inputs.append(concat([outputs[i], inputs[i]]))
        return torch.cat([self.layers[-1](inputs[-1])] + outputs, 1)


class DenseUNetModule(DenseBlock):
    """
    A dense block with 4 dense layers and k output channels
    """
    def __init__(self, hps: HyperParams, in_: int, out: int):
        n_layers = 4
        super().__init__(in_, out // n_layers, n_layers,
                         dropout=hps.dropout, bn=hps.bn)


class DownBlock(nn.Module):
    """
    A down block with a 1x1 convolution, a ReLU, a 3x3 convolution, a dropout
    """
    def __init__(self, in_, out, scale, *, dropout, bn):
        super().__init__()
        self.in_ = in_
        self.bn = nn.BatchNorm2d(in_) if bn else None
        self.activation = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_, out, 1)
        self.dropout = nn.Dropout2d(p=dropout) if dropout else None
        self.pool = nn.MaxPool2d(scale, scale)

    def forward(self, x):
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.pool(x)
        return x


class UpBlock(nn.Module):
    """
    An up block with a 1x1 convolution, a ReLU, a 3x3 convolution, a dropout
    """
    def __init__(self, in_, out, scale):
        super().__init__()
        self.up_conv = nn.Conv2d(in_, out, 1)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=scale)

    def forward(self, x):
        return self.upsample(self.up_conv(x))


class Conv3BN(nn.Module):
    """
    A single convolutional layer with batch normalization and activation
    """
    def __init__(self, hps: HyperParams, in_: int, out: int):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.bn = nn.BatchNorm2d(out) if hps.bn else None
        self.activation = getattr(F, hps.activation)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x, inplace=True)
        return x


class UNet3lModule(nn.Module):
    """
    A single module of the UNet that has 3 convolutional layers
    """
    def __init__(self, hps: HyperParams, in_: int, out: int):
        super().__init__()
        self.l1 = Conv3BN(hps, in_, out)
        self.l2 = Conv3BN(hps, out, out)
        self.l3 = Conv3BN(hps, out, out)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x


class UNet2Module(nn.Module):
    """
    A single module of the UNet that has 2 convolutional layers
    """
    def __init__(self, hps: HyperParams, in_: int, out: int):
        super().__init__()
        self.l1 = Conv3BN(hps, in_, out)
        self.l2 = Conv3BN(hps, out, out)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x

class BasicConv2d(nn.Module):
    """
    A single convolutional layer with batch normalization and activation
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class InceptionModule(nn.Module):
    """
    An inception module
    """
    def __init__(self, hps: HyperParams, in_: int, out: int):
        super().__init__()
        out_1 = out * 3 // 8
        out_2 = out * 2 // 8
        self.conv1x1 = BasicConv2d(in_, out_1, kernel_size=1)
        self.conv3x3_pre = BasicConv2d(in_, in_ // 2, kernel_size=1)
        self.conv3x3 = BasicConv2d(in_ // 2, out_1, kernel_size=3, padding=1)
        self.conv5x5_pre = BasicConv2d(in_, in_ // 4, kernel_size=1)
        self.conv5x5 = BasicConv2d(in_ // 4, out_2, kernel_size=5, padding=2)
        assert hps.bn
        assert hps.activation == 'relu'

    def forward(self, x):
        return torch.cat([
            self.conv1x1(x),
            self.conv3x3(self.conv3x3_pre(x)),
            self.conv5x5(self.conv5x5_pre(x)),
        ], 1)


class Inception2Module(nn.Module):
    """
    An inception module with 2 convolutional layers
    """
    def __init__(self, hps: HyperParams, in_: int, out: int):
        super().__init__()
        self.l1 = InceptionModule(hps, in_, out)
        self.l2 = InceptionModule(hps, out, out)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x

