import torch
from torch import nn
from torch.nn import functional as F

from layers import ConvBlock, UpConvBlock, ResBlock


class Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()
        # self.init_conv = nn.Conv2d(3, 32, 3, padding=1)
        # self.block1 = ResBlock(32)
        # self.down1 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        # self.block2 = ResBlock(64)
        # self.down2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        # self.block3 = ResBlock(128)
        # self.down3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        # self.block4 = ResBlock(256)
        #
        # self.activation = nn.LeakyReLU(0.2, inplace=True)
        ngf = 16
        self.conv1 = ConvBlock(in_channels, ngf)    # 128 x 128 -> 64 x 64
        self.cv1 = ConvBlock(ngf, ngf, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBlock(ngf, ngf * 2)        # 64 x 64 -> 32 x 32
        self.cv2 = ConvBlock(ngf * 2, ngf * 2, kernel_size=1, stride=1, padding=0)
        self.conv3 = ConvBlock(ngf * 2, ngf * 4)    # 32 x 32 -> 16 x 16
        self.cv3 = ConvBlock(ngf * 4, ngf * 4, kernel_size=1, stride=1, padding=0)
        # self.conv4 = ConvBlock(ngf * 4, ngf * 8)    # 16 x 16 -> 8 x 8

    def forward(self, x):
        x = self.conv1(x)
        s1 = self.cv1(x)
        x = self.conv2(s1)
        s2 = self.cv2(x)
        x = self.conv3(s2)
        x = self.cv3(x)
        # x = self.conv4(x)
        return x
        # s1 = self.activation(self.init_conv(x))
        # s1 = self.block1(s1)
        # x = self.activation(self.down1(s1))
        # s2 = self.block2(x)
        # x = self.activation(self.down2(s2))
        # x = self.block3(x)
        # x = self.activation(self.down3(x))
        # x = self.block4(x)
        # return x, (s2,)


class Decoder(nn.Module):
    def __init__(self, out_channels=3):
        super(Decoder, self).__init__()
        # self.block1 = ResBlock(latent_channels)
        # self.up1 = nn.ConvTranspose2d(latent_channels, 128, 4, stride=2, padding=1)
        # self.block2 = ResBlock(128)
        # self.up2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        # self.block3 = ResBlock(64)
        # self.up3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        # self.block4 = ResBlock(32)
        # self.out_conv = nn.Conv2d(32, 3, 3, padding=1)
        ngf = 16
        # self.up_conv1 = UpConvBlock(ngf * 8, ngf * 4)     # 8 x 8 -> 16 x 16
        self.up_conv2 = UpConvBlock(ngf * 4, ngf * 2)     # 16 x 16 -> 32 x 32
        self.up_cv2 = UpConvBlock(ngf * 2, ngf * 2, kernel_size=1, stride=1, padding=0)
        self.up_conv3 = UpConvBlock(ngf * 2, ngf)         # 32 x 32 -> 64 x 64
        self.up_cv3 = UpConvBlock(ngf, ngf, kernel_size=1, stride=1, padding=0)
        self.up_conv4 = UpConvBlock(ngf, out_channels)    # 64 x 64 -> 128 x 128
        self.up_cv4 = UpConvBlock(out_channels, out_channels, activation="sigmoid", kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # s2 = skips[0]
        # x = self.block1(z)
        # x = self.up1(x)
        # x = self.block2(x)
        # x = self.up2(x)
        # # skip connection solo qui
        # # x = x + s2
        # x = self.block3(x)
        # x = self.up3(x)
        # x = self.block4(x)
        # x = nn.Sigmoid()(self.out_conv(x))
        # return x
        # x = self.up_conv1(x)
        x = self.up_conv2(x)
        x = self.up_cv2(x)
        x = self.up_conv3(x)
        x = self.up_cv3(x)
        x = self.up_conv4(x)
        x = self.up_cv4(x)
        return x


class LiteVAE(nn.Module):
    def __init__(self, channels, latent_dim=64):
        super(LiteVAE, self).__init__()

        # Encoder
        self.encoder = Encoder(channels)

        # Compute the size of the encoder output
        self.enc_output_dim = 64 * 16 * 16

        # Latent space parameters
        # self.fc_mu = nn.Linear(256, latent_dim)
        # self.fc_log_var = nn.Linear(256, latent_dim)
        self.fc_mu = nn.Linear(self.enc_output_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.enc_output_dim, latent_dim)

        # Decoder input layer
        self.decoder_input = nn.Linear(latent_dim, self.enc_output_dim)

        # Decoder
        self.decoder = Decoder(channels)

        self.latent_dim = latent_dim

    def encode(self, x):
        # Encode input to get mean and log variance of latent distribution
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        # Reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        # Decode from latent space
        x = self.decoder_input(z)
        x = x.view(x.size(0), 64, 16, 16)  # Reshape to match encoder output
        output = self.decoder(x)
        return output

    def forward(self, x):
        # Full forward pass
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        decoded = self.decode(z)
        return decoded, mu, log_var
