import torch
import torch.nn as nn
from typing import List

from hw_nv.base.base_model import BaseModel
from hw_nv.model.HiFiGAN.mrf import MRF


class GeneratorBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 conv_trans_kernel_size: int,
                 mrf_kernel_sizes: List[int],
                 mrf_dilations: List[List[List[int]]]):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.LeakyReLU(),
            nn.utils.weight_norm(
                nn.ConvTranspose1d(
                    in_channels=in_channels,
                    out_channels=in_channels // 2,
                    kernel_size=conv_trans_kernel_size,
                    stride=conv_trans_kernel_size // 2,
                    padding=(conv_trans_kernel_size - conv_trans_kernel_size // 2) // 2
                )
            ),
            MRF(
                num_channels=in_channels // 2,
                kernel_sizes=mrf_kernel_sizes,
                dilations=mrf_dilations
            )
        )

    def forward(self, x):
        return self.sequential(x)


class Generator(BaseModel):
    def __init__(self,
                 in_channels: int,
                 hid_dim: int,
                 conv_trans_kernel_sizes: List[int],
                 mrf_kernel_sizes: List[int],
                 mrf_dilations: List[List[List[int]]]):
        super().__init__()
        self.head = nn.utils.weight_norm(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=hid_dim,
                kernel_size=7,
                dilation=1,
                padding="same"
            )
        )
        self.blocks = nn.ModuleList([
            GeneratorBlock(
                in_channels=hid_dim // (2 ** i),
                conv_trans_kernel_size=conv_trans_kernel_sizes[i],
                mrf_kernel_sizes=mrf_kernel_sizes,
                mrf_dilations=mrf_dilations
            )
            for i in range(len(conv_trans_kernel_sizes))
        ])
        tail_in_channels = hid_dim // (2 ** len(conv_trans_kernel_sizes))
        self.tail = nn.Sequential(
            nn.LeakyReLU(),
            nn.utils.weight_norm(
                nn.Conv1d(
                    in_channels=tail_in_channels,
                    out_channels=1,
                    kernel_size=7,
                    padding="same"
                )
            ),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.head(x)
        for block in self.blocks:
            x = block(x)
        x = self.tail(x)
        return x
    
    def remove_normalization(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose1d):
                nn.utils.remove_weight_norm(module)