import torch
import torch.nn as nn
from typing import List


class ResBlock(nn.Module):
    def __init__(self,
                 num_channels: int,
                 kernel_size: int,
                 dilations: List[List[int]]):
        super().__init__()
        self.layers = self._create_layers(num_channels, kernel_size, dilations)
    
    def _create_layers(self, num_channels, kernel_size, dilations):
        layers = []
        for i in range(len(dilations)):
            sub_layers = []
            for j in range(len(dilations[0])):
                sub_layers.append(
                    nn.Sequential(
                        nn.LeakyReLU(),
                        nn.utils.weight_norm(
                            nn.Conv1d(
                                in_channels=num_channels,
                                out_channels=num_channels,
                                kernel_size=kernel_size,
                                dilation=dilations[i][j],
                                padding="same"
                            )
                        )
                    )
                )
            sub_layers = nn.ModuleList(sub_layers)
            layers.append(sub_layers)
        return nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            residual = x
            for sub_layer in layer:
                x = sub_layer(x)
            x = x + residual
        return x
        

class MRF(nn.Module):
    def __init__(self,
                 num_channels: int,
                 kernel_sizes: List[int],
                 dilations: List[List[List[int]]]):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResBlock(
                num_channels=num_channels,
                kernel_size=kernel_sizes[i],
                dilations=dilations[i]
            )
            for i in range(len(kernel_sizes))
        ])

    def forward(self, x):
        out = self.res_blocks[0](x)
        for block in self.res_blocks[1:]:
            out += block(x)
        return out
