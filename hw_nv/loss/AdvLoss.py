import torch
from torch import nn
import torch.nn.functional as F


class DiscriminatorAdvLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gt_outputs, pred_outputs):
        total_loss = 0.0
        for gt_output, pred_output in zip(gt_outputs, pred_outputs):
            gt_loss = torch.mean((gt_output - 1) ** 2)
            pred_loss = torch.mean(pred_output ** 2)
            total_loss += gt_loss + pred_loss
        return total_loss

        
class GeneratorAdvLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_outputs):
        total_loss = 0.0
        for pred_output in pred_outputs:
            pred_loss = torch.mean((pred_output - 1) ** 2)
            total_loss += pred_loss
        return total_loss
