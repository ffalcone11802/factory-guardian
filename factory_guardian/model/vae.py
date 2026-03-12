from typing import Tuple
import torch
from torch import nn, Tensor

from factory_guardian.model.layers import DSConvBlock, UpConvBlock


class LiteVAE(nn.Module):
    """
    The LiteVAE class implements a lightweight Variational Autoencoder (VAE)
    for image processing tasks.

    This class defines an encoder-decoder architecture for encoding input data
    into a latent space and decoding it back to reconstruct the input. It uses depthwise
    separable convolutions in the encoder and upsampling blocks in the decoder for
    lightweight processing. The latent space representation is parameterized by a mean
    and log variance using fully connected layers.

    Args:
        num_channels (int, optional): Number of input channels. Defaults to 3.
        latent_dim (int, optional): Dimensionality of the latent space. Defaults to 128.
    """

    def __init__(
        self,
        num_channels: int = 3,
        latent_dim: int = 128
    ):
        super(LiteVAE, self).__init__()

        # Base multiplier for the number of channels
        base = 32

        # Height and width of the input image after encoding
        self.min_h_w = 4

        # Encoder
        self.encoder = nn.Sequential(
            DSConvBlock(num_channels, base),   # 256 -> 128
            DSConvBlock(base, base * 2),       # 128 -> 64
            DSConvBlock(base * 2, base * 4),   # 64 -> 32
            DSConvBlock(base * 4, base * 8),   # 32 -> 16
            DSConvBlock(base * 8, base * 8),   # 16 -> 8
            DSConvBlock(base * 8, base * 8),   # 8 -> 4
        )

        # Calculate the size of the encoder output
        self.feature_dim = base * 8 * self.min_h_w * self.min_h_w

        # Latent space parameters
        self.fc_mu = nn.Linear(self.feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_dim, latent_dim)

        # Decoder input layer
        self.fc_dec = nn.Linear(latent_dim, self.feature_dim)

        # Decoder
        self.decoder = nn.Sequential(
            UpConvBlock(base * 8, base * 8),   # 4 -> 8
            UpConvBlock(base * 8, base * 8),   # 8 -> 16
            UpConvBlock(base * 8, base * 4),   # 16 -> 32
            UpConvBlock(base * 4, base * 2),   # 32 -> 64
            UpConvBlock(base * 2, base),       # 64 -> 128
            UpConvBlock(base, num_channels, norm=False, activation="sigmoid"),   # 128 -> 256
        )

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode input data into mean and log variance for a latent representation.

        Process input data using the encoder, flatten the result, and compute
        the mean and log variance for the latent representation of the data.

        Args:
            x (Tensor): Input data to encode.

        Returns:
            Tuple[Tensor, Tensor]: Mean and log variance of the latent representation.
        """
        x = self.encoder(x)
        x = torch.flatten(x, 1)

        # Bottleneck
        mu, logvar =  self.fc_mu(x), self.fc_logvar(x)

        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Perform the reparameterization trick to sample from a Gaussian distribution.

        Sample from a Gaussian distribution parameterized by the provided mean (`mu`)
        and log variance (`logvar`) using the reparameterization trick during training.
        Otherwise, it directly returns the mean.

        Args:
            mu (Tensor): Mean of the Gaussian distribution.
            logvar (Tensor): Log variance of the Gaussian distribution.

        Returns:
            Tensor: Sampled value during training or the mean (`mu`) otherwise.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode the given latent vector into an output tensor.

        Args:
            z (Tensor): Latent vector of shape (batch_size, latent_dim).

        Returns:
            Tensor: Output tensor of decoded data.
        """
        x = self.fc_dec(z)
        x = x.view(z.size(0), -1, self.min_h_w, self.min_h_w)
        output = self.decoder(x)
        return output

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Perform a forward pass through the network.

        Encode the input into latent space representations, reparameterize
        to generate a latent vector, and decode it back to the output space.

        Args:
            x (Tensor): Input data to the network.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple containing the decoded output,
                mean of the latent representation, and log variance of the latent representation.
        """
        # Encode
        mu, logvar = self.encode(x)

        # Sample from the latent space
        z = self.reparameterize(mu, logvar)

        # Decode
        output = self.decode(z)

        return output, mu, logvar
