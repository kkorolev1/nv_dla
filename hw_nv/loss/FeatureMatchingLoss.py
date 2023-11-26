import torch
from torch import nn
import torch.nn.functional as F


class FeatureMatchingLoss(nn.Module):
    def __init__(self, feature_matching_multiplier):
        super().__init__()
        self.feature_matching_multiplier = feature_matching_multiplier

    def forward(self, gt_features_list, pred_features_list):
        total_loss = 0.0
        # First cycle over different discriminators
        # Second cycle over features from different levels
        for disc_gt_features, disc_pred_features in zip(gt_features_list, pred_features_list):
            for gt_features, pred_features in zip(disc_gt_features, disc_pred_features):
                total_loss += F.l1_loss(pred_features, gt_features)
        return self.feature_matching_multiplier * total_loss
