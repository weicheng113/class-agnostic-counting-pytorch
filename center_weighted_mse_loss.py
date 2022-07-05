import torch
from torch import nn


class CenterWeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, preds, targets):
        if isinstance(preds, dict):
            preds = preds['logits']

        preds = preds.cpu()
        targets = targets.cpu()

        mask = torch.as_tensor(torch.randn_like(targets) > 0.8, dtype=torch.float)
        radius = 8
        size_map = [*preds.shape[-2:]]
        center = torch.zeros_like(preds)

        center[:, :,
        size_map[0] // 2 - radius: size_map[0] // 2 + radius,
        size_map[1] // 2 - radius: size_map[1] // 2 + radius] = 1
        # Center has weight of 10. pdf > 0.8 has at least of weight 1. All elsewhere will be ignored.
        center = mask + 10. * center

        loss = torch.mul(center, (preds - targets) ** 2).squeeze()
        return loss.mean()
