import torch
from torch import nn
from torch.nn import functional as F


class ELBOLoss(nn.Module):
    def __init__(self, reduction="sum", beta=1.0):
        super(ELBOLoss, self).__init__()
        self.reduction = reduction
        self.beta = beta

    def forward(self, y_pred, y_true, mu, log_var):
        mse = F.mse_loss(y_pred, y_true, reduction=self.reduction)
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = mse + self.beta * kl_div
        return loss
