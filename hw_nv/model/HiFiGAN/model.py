import torch
import torch.nn as nn
from typing import List, Dict

from hw_nv.base.base_model import BaseModel
from hw_nv.model.HiFiGAN.generator import Generator
from hw_nv.model.HiFiGAN.mpd import MPD
from hw_nv.model.HiFiGAN.msd import MSD


class HiFiGAN(BaseModel):
    def __init__(self,
                 generator_config: Dict,
                 mpd_config: Dict,
                 msd_config: Dict):
        super().__init__()
        self.generator = Generator(**generator_config)
        self.mpd = MPD(**mpd_config)
        self.msd = MSD(**msd_config)