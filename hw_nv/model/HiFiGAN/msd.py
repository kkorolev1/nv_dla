import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from hw_nv.base.base_model import BaseModel


class MSDSub(nn.Module):
    def __init__(self,
                 factor: int,
                 kernel_sizes: List[int],
                 strides: List[int],
                 groups: List[int],
                 channels: List[int]):
        super().__init__()
        self.factor = factor

        if factor == 1:
            self.pooling = nn.Identity()
            norm_module = nn.utils.spectral_norm
        else:
            self.pooling = nn.AvgPool1d(kernel_size=4, stride=self.factor)
            norm_module = nn.utils.weight_norm

        # Adding first input channel
        channels = [1] + channels

        layers = []

        for i in range(len(kernel_sizes)):
            layers.append(
                nn.Sequential(
                    norm_module(
                        nn.Conv1d(
                            in_channels=channels[i],
                            out_channels=channels[i + 1],
                            kernel_size=kernel_sizes[i],
                            stride=strides[i],
                            groups=groups[i],
                            padding=(kernel_sizes[i] - 1) // 2
                        )
                    ),
                    nn.LeakyReLU()
                )
            )

        layers.append(
            norm_module(
                nn.Conv1d(
                    in_channels=channels[-1],
                    out_channels=1,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
        )

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        features_from_layers = []
        x = self.pooling(x)
        for layer in self.layers:
            x = layer(x)
            features_from_layers.append(x)
        return x, features_from_layers[:-1]


class MSD(BaseModel):
    def __init__(self,
                 factors: List[int],
                 kernel_sizes: List[int],
                 strides: List[int],
                 groups: List[int],
                 channels: List[int]):
        super().__init__()
        self.discriminators = nn.ModuleList([
            MSDSub(
                factor=factor,
                kernel_sizes=kernel_sizes,
                strides=strides,
                groups=groups,
                channels=channels
            )
            for factor in factors
        ])

    def forward(self, x):
        disc_outputs = []
        disc_features = []
        for disc in self.discriminators:
            output, features_list = disc(x)
            disc_outputs.append(output)
            disc_features.append(features_list)
        return disc_outputs, disc_features
    

# kernel_sizes = [15, 41, 41, 41, 41, 5]
# strides = [1, 4, 4, 4, 4, 1]
# groups = [1, 4, 16, 64, 256, 1]
# channels = [16, 64, 256, 1024, 1024, 1024]
# msd = MSDSub(factor=2, kernel_sizes=kernel_sizes, strides=strides, groups=groups, channels=channels)
# x = torch.ones((1, 1, 2000))
# y, fm = msd(x)
# print(y.shape, len(fm))

# msd = MSD(factors=[1, 2, 4], kernel_sizes=kernel_sizes, strides=strides, groups=groups, channels=channels)
# x = torch.ones((1, 1, 2000))
# y, fm = msd(x)
# print(len(y), len(fm))