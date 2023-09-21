from functools import partial
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from HyperParams import HyperParams
from modules import UNetModule, conv3x3, concat, DenseUNetModule, DenseBlock, \
    UpBlock, DownBlock, BasicConv2d, InceptionModule, Inception2Module, Conv3BN, \
    UNet2Module, UNet3lModule


class BaseNet(nn.Module):
    """
    Base class for all networks in this file (except for the old one) that
    sets up the dropout and global step counter
    """
    def __init__(self, hps: HyperParams):
        super().__init__()
        self.hps = hps
        if hps.dropout:
            self.dropout2d = nn.Dropout2d(p=hps.dropout)
        else:
            self.dropout2d = lambda x: x
        self.register_buffer('global_step', torch.IntTensor(1).zero_())


class MiniNet(BaseNet):
    """
    A small network for testing purposes.
    """
    def __init__(self, hps):
        super().__init__(hps)
        self.conv1 = nn.Conv2d(hps.n_channels, 4, 1)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, hps.n_classes, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        b = self.hps.patch_overlap_size
        return F.sigmoid(x[:, :, b:-b, b:-b])


class OldNet(BaseNet):
    """
    A network similar to the one used in the original paper (but with batch
    normalization) that uses 3x3 convolutions and 2x2 max pooling.
    """
    def __init__(self, hps):
        super().__init__(hps)
        self.conv1 = nn.Conv2d(hps.n_channels, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv3 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv4 = nn.Conv2d(64, hps.n_classes, 7, padding=3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        b = self.hps.patch_overlap_size
        return F.sigmoid(x[:, :, b:-b, b:-b])


class SmallNet(BaseNet):
    """
    A small network for testing purposes.
    """
    def __init__(self, hps):
        super().__init__(hps)
        self.conv1 = nn.Conv2d(hps.n_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, hps.n_classes, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        b = self.hps.patch_overlap_size
        return F.sigmoid(x[:, :, b:-b, b:-b])


# UNet:
# http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png


class SmallUNet(BaseNet):
    """
    A small UNet for testing purposes.
    """
    def __init__(self, hps):
        super().__init__(hps)
        self.conv1 = nn.Conv2d(hps.n_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv7 = nn.Conv2d(32, hps.n_classes, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x1 = self.pool(x)
        x1 = F.relu(self.conv3(x1))
        x1 = F.relu(self.conv4(x1))
        x1 = F.relu(self.conv5(x1))
        x1 = self.upsample(x1)
        x = torch.cat([x, x1], 1)
        x = F.relu(self.conv6(x))
        x = self.conv7(x)
        b = self.hps.patch_overlap_size
        return F.sigmoid(x[:, :, b:-b, b:-b])




class UNet(BaseNet):
    """
    A UNet with 5 scales and 3x3 convolutions. The number/size of filters in
    each scale is determined by the `filter_factors` attribute multiplied by the
    `filters_base` attribute.
    """
    module = UNetModule
    filter_factors = [1, 2, 4, 8, 16]

    def __init__(self, hps: HyperParams):
        super().__init__(hps)
        self.pool = nn.MaxPool2d(2, 2)
        self.pool_top = nn.MaxPool2d(hps.top_scale, hps.top_scale)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsample_top = nn.UpsamplingNearest2d(scale_factor=hps.top_scale)
        filter_sizes = [hps.filters_base * s for s in self.filter_factors]
        self.down, self.up = [], []
        for i, nf in enumerate(filter_sizes):
            low_nf = hps.n_channels if i == 0 else filter_sizes[i - 1]
            self.down.append(self.module(hps, low_nf, nf))
            setattr(self, 'down_{}'.format(i), self.down[-1])
            if i != 0:
                self.up.append(self.module(hps, low_nf + nf, low_nf))
                setattr(self, 'conv_up_{}'.format(i), self.up[-1])
        self.conv_final = nn.Conv2d(filter_sizes[0], hps.n_classes, 1)

    def forward(self, x):
        xs = []
        for i, down in enumerate(self.down):
            if i == 0:
                x_in = x
            elif i == 1:
                x_in = self.pool_top(xs[-1])
            else:
                x_in = self.pool(xs[-1])
            x_out = down(x_in)
            x_out = self.dropout2d(x_out)
            xs.append(x_out)

        x_out = xs[-1]
        for i, (x_skip, up) in reversed(list(enumerate(zip(xs[:-1], self.up)))):
            upsample = self.upsample_top if i == 0 else self.upsample
            x_out = up(torch.cat([upsample(x_out), x_skip], 1))
            x_out = self.dropout2d(x_out)

        x_out = self.conv_final(x_out)
        b = self.hps.patch_overlap_size
        return F.sigmoid(x_out[:, :, b:-b, b:-b])

class UNet3l(UNet):
    """
    A UNet with 3 convolutional layers for testing purposes
    """
    module = UNet3lModule


class UNet2(BaseNet):
    """
    A second UNet
    """
    def __init__(self, hps):
        super().__init__(hps)
        b = hps.filters_base
        self.filters = [b * 2, b * 2, b * 4, b * 8, b * 16]
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.down, self.down_pool, self.mid, self.up = [[] for _ in range(4)]
        for i, nf in enumerate(self.filters):
            low_nf = hps.n_channels if i == 0 else self.filters[i - 1]
            self.down_pool.append(
                nn.Conv2d(low_nf, low_nf, 3, padding=1, stride=2))
            setattr(self, 'down_pool_{}'.format(i), self.down_pool[-1])
            self.down.append(UNet2Module(hps, low_nf, nf))
            setattr(self, 'down_{}'.format(i), self.down[-1])
            if i != 0:
                self.mid.append(Conv3BN(hps, low_nf, low_nf))
                setattr(self, 'mid_{}'.format(i), self.mid[-1])
                self.up.append(UNet2Module(hps, low_nf + nf, low_nf))
                setattr(self, 'up_{}'.format(i), self.up[-1])
        self.conv_final = nn.Conv2d(self.filters[0], hps.n_classes, 1)

    def forward(self, x):
        xs = []
        for i, (down, down_pool) in enumerate(zip(self.down, self.down_pool)):
            x_out = down(down_pool(xs[-1]) if xs else x)
            xs.append(x_out)

        x_out = xs[-1]
        for x_skip, up, mid in reversed(list(zip(xs[:-1], self.up, self.mid))):
            x_out = up(torch.cat([self.upsample(x_out), mid(x_skip)], 1))

        x_out = self.conv_final(x_out)
        b = self.hps.patch_overlap_size
        return F.sigmoid(x_out[:, :, b:-b, b:-b])


class InceptionUNet(UNet):
    """
    A UNet with inception modules
    """
    module = InceptionModule


class Inception2UNet(UNet):
    """
    A UNet with inception modules v2
    """
    module = Inception2Module


class SimpleSegNet(BaseNet):
    """
    A simple SegNet
    """
    def __init__(self, hps):
        super().__init__(hps)
        s = hps.filters_base
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.input_conv = BasicConv2d(hps.n_channels, s, 1)
        self.enc_1 = BasicConv2d(s * 1, s * 2, 3, padding=1)
        self.enc_2 = BasicConv2d(s * 2, s * 4, 3, padding=1)
        self.enc_3 = BasicConv2d(s * 4, s * 8, 3, padding=1)
        self.enc_4 = BasicConv2d(s * 8, s * 8, 3, padding=1)
        # https://github.com/pradyu1993/segnet - decoder lacks relu (???)
        self.dec_4 = BasicConv2d(s * 8, s * 8, 3, padding=1)
        self.dec_3 = BasicConv2d(s * 8, s * 4, 3, padding=1)
        self.dec_2 = BasicConv2d(s * 4, s * 2, 3, padding=1)
        self.dec_1 = BasicConv2d(s * 2, s * 1, 3, padding=1)
        self.conv_final = nn.Conv2d(s, hps.n_classes, 1)

    def forward(self, x):
        # Input
        x = self.input_conv(x)
        # Encoder
        x = self.enc_1(x)
        x = self.pool(x)
        x = self.enc_2(x)
        x = self.pool(x)
        x = self.enc_3(x)
        x = self.pool(x)
        x = self.enc_4(x)
        # Decoder
        x = self.dec_4(x)
        x = self.upsample(x)
        x = self.dec_3(x)
        x = self.upsample(x)
        x = self.dec_2(x)
        x = self.upsample(x)
        x = self.dec_1(x)
        # Output
        x = self.conv_final(x)
        b = self.hps.patch_overlap_size
        return F.sigmoid(x[:, :, b:-b, b:-b])


class DenseUNet(UNet):
    """
    A UNet with dense blocks
    """
    module = DenseUNetModule


class DenseNet(BaseNet):
    """
    DenseNet implementation based on https://arxiv.org/pdf/1611.09326v2.pdf
    """
    def __init__(self, hps):
        super().__init__(hps)
        k = hps.filters_base
        block_layers = [3, 5, 7, 5, 3]
        block_in = [n * k for n in [3, 8, 16, 8, 4]]
        scale_factors = [4, 2]
        dense = partial(DenseBlock, dropout=hps.dropout, bn=hps.bn)
        self.input_conv = nn.Conv2d(hps.n_channels, block_in[0], 3, padding=1)
        self.blocks = []
        self.scales = []
        self.n_layers = len(block_layers) // 2
        for i, (in_, l) in enumerate(zip(block_in, block_layers)):
            if i < self.n_layers:
                block = dense(in_, k, l)
                scale = DownBlock(block.out + in_, block_in[i + 1],
                                  scale_factors[i],
                                  dropout=hps.dropout, bn=hps.bn)
            elif i == self.n_layers:
                block = dense(in_, k, l)
                scale = None
            else:
                block = dense(in_ + self.scales[2 * self.n_layers - i].in_,
                              k, l)
                scale = UpBlock(self.blocks[-1].out, in_,
                                scale_factors[2 * self.n_layers - i])
            setattr(self, 'block_{}'.format(i), block)
            setattr(self, 'scale_{}'.format(i), scale)
            self.blocks.append(block)
            self.scales.append(scale)
        self.output_conv = nn.Conv2d(self.blocks[-1].out, hps.n_classes, 1)

    def forward(self, x):
        # Input
        x = self.input_conv(x)
        # Network
        skips = []
        for i, (block, scale) in enumerate(zip(self.blocks, self.scales)):
            if i < self.n_layers:
                x = concat([block(x), x])
                skips.append(x)
                x = scale(x)
            elif i == self.n_layers:
                x = block(x)
            else:
                x = block(concat([scale(x), skips[2 * self.n_layers - i]]))
        # Output
        x = self.output_conv(x)
        b = self.hps.patch_overlap_size
        return F.sigmoid(x[:, :, b:-b, b:-b])
