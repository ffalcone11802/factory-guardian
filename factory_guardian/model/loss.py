import torch
from torch import nn, Tensor
from torch.nn import functional as F


class ELBOLoss(nn.Module):
    """
    The class ELBOLoss computes the Evidence Lower Bound (ELBO) loss
    for Variational Autoencoders (VAEs).

    This class combines Mean Squared Error (MSE) for reconstruction accuracy with a
    Kullback-Leibler divergence (KL-divergence) term to enforce the latent space structure.

    Args:
        reduction (str): Specifies the reduction to apply to the output. Defaults to "sum".
        beta (float): Weighting factor for the KL-divergence term. Defaults to 1.0.
    """

    def __init__(
        self,
        reduction: str = "sum",
        beta: float = 1.0
    ):
        super(ELBOLoss, self).__init__()

        self.reduction = reduction
        self.beta = beta

    def forward(
        self,
        y_recon: Tensor,
        y_true: Tensor,
        mu: Tensor,
        log_var: Tensor
    ) -> Tensor:
        """
        Perform a forward pass through the loss.

        Args:
            y_recon (Tensor): Model output values.
            y_true (Tensor): Model input values.
            mu (Tensor): Mean of the latent variable distribution.
            log_var (Tensor): Log variance of the latent variable distribution.

        Returns:
            Tensor: Total loss computed from the MSE loss and scaled KL divergence.
        """
        # MSE
        mse = F.mse_loss(y_recon, y_true, reduction=self.reduction)

        # KL divergence
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        loss = mse + self.beta * kl_div
        return loss
