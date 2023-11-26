import torch
from torch import nn
import torch.nn.functional as F


class MelSpectrogramLoss(nn.Module):
    def __init__(self, mel_spectrogram_multiplier):
        super().__init__()
        self.mel_spectrogram_multiplier = mel_spectrogram_multiplier

    def forward(self, gt_spectrogram, pred_spectrogram):
        return self.mel_spectrogram_multiplier * F.l1_loss(pred_spectrogram, gt_spectrogram)
