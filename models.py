from functools import partial

import torch
from torch import nn

from torchinfo import summary



class ResNet50m(nn.Module):
    def __init__(self):
        super().__init__()

    
    def forward(self, x):
        return x

    
    def load_weights(self, path):
        print("Loaded weights successfully")



"""
helper modules for ResNet
"""

#conv3x3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same', stride=1, bias=False)
#conv3x3 = partial(nn.Conv2d, kernel_size=3, bias=False, padding='same') #padding not supported for strided convolutions

class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.padding = ((self.kernel_size[0] // 2), (self.kernel_size[1] // 2))

conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)

def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()

        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation

        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()


    def forward(self, x):
        residual = x
        if self.apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x) + residual

        x = self.activate(x)
        return x

    @property
    def apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels:int, out_channels:int, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)

        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.expanded_channels, kernel_size=1, stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels) if self.apply_shortcut else None
        )

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def apply_shortcut(self):
        return self.in_channels != self.expanded_channels


def conv_bn(in_channels:int, out_channels:int, conv, *args, **kwargs):
    return nn.Sequential(
        conv(in_channels, out_channels, *args, **kwargs),
        nn.BatchNorm2d(out_channels)
    )

class ResNetBasicBlock(ResNetResidualBlock): 
    expansion = 1

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)

        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False)
        )


class ResNetBottleNeckBlock(ResNetResidualBlock):
     expansion=4

     def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)

        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1)
        )

class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()

        downsampling = 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(
            block(in_channels, out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, out_channels, downsampling=1, *args, **kwargs) for _ in range(n-1)]
        )

    def forward(self, x):
        return self.blocks(x)


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], depths=[2,2,2,2], 
                 activation='relu', block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()

        self.block_sizes = blocks_sizes

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.block_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.block_sizes[0]),
            activation_func(activation),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))

        self.blocks = nn.ModuleList([
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=depths[0], activation=activation, block=block, *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion, out_channels, n=n, activation=activation, block=block, *args, **kwargs)
            for (in_channels, out_channels), n in zip(self.in_out_block_sizes, depths[1:])]
        ])

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x

class ResNetDecoder(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        return self.decoder(x)

class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes, *args, **kwargs):
        super().__init__()

        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResNetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, num_classes)

    def forward(self, x):
        return self.decoder( self.encoder(x))


#resnet50 = ResNet(3, 1000, block=ResNetBottleNeckBlock, depths=[3, 4, 6, 3])

#summary(resnet50)