import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(8, out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LiteVAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()

        ngf = 16

        # -------- Encoder --------
        self.enc1 = ConvBlock(3, ngf)          # 128 → 64
        self.enc2 = ConvBlock(ngf, ngf * 2)    # 64 → 32
        self.enc3 = ConvBlock(ngf * 2, ngf * 4)# 32 → 16
        self.enc4 = ConvBlock(ngf * 4, ngf * 8)# 16 → 8
        self.enc5 = ConvBlock(ngf * 8, ngf * 8)# 8 → 4
        # self.enc6 = ConvBlock(ngf * 8, ngf * 8)

        # Compression layer (important!)
        # self.compress = nn.Conv2d(ngf * 8, ngf * 4, 1)

        self.enc_output_dim = (ngf * 8) * 8 * 8

        # Latent space
        self.fc_mu = nn.Linear(self.enc_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_output_dim, latent_dim)

        # Decoder input
        self.decoder_input = nn.Linear(latent_dim, self.enc_output_dim)

        # -------- Decoder --------
        # self.dec0 = UpBlock(ngf * 8, ngf * 8)
        self.dec1 = UpBlock(ngf * 8, ngf * 4)  # 4 → 8
        self.dec2 = UpBlock(ngf * 4, ngf * 2)  # 8 → 16
        self.dec3 = UpBlock(ngf * 2, ngf)      # 16 → 32
        self.dec4 = UpBlock(ngf, ngf)          # 32 → 64
        self.final = nn.ConvTranspose2d(ngf, 3, 4, stride=2, padding=1)

        self.latent_dim = latent_dim

    def encode(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        # x = self.enc6(x)

        # x = self.compress(x)

        x = torch.flatten(x, start_dim=1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(z.size(0), -1, 8, 8)

        # x = self.dec0(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)

        x = torch.sigmoid(self.final(x))
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
# import torch
# from torch import nn
# from torch.nn import functional as F
#
# from layers import ConvBlock, UpConvBlock
#
#
# class LiteVAE(nn.Module):
#     def __init__(self, latent_dim=128):
#         super(LiteVAE, self).__init__()
#         ngf = 16
#
#         # Encoder
#         self.enc1 = ConvBlock(3, ngf, norm=False)  # 64×64
#         self.enc2 = ConvBlock(ngf, ngf * 2, norm=False)  # 32×32
#         self.enc3 = ConvBlock(ngf * 2, ngf * 4, norm=False)  # 16×16
#         self.enc4 = ConvBlock(ngf * 4, ngf * 8, norm=False) # 8×8
#         self.enc5 = ConvBlock(ngf * 8, ngf * 8, norm=False) # 4×4
#         # self.enc6 = ConvBlock(ngf * 8, ngf * 8, norm=False)  # 2×2
#         # self.enc7 = ConvBlock(ngf * 8, ngf * 8, norm=False)  # 1×1
#
#         # Calculate the size of the encoder output
#         self.enc_output_dim = 128 * 4 * 4
#
#         # Latent space parameters
#         self.fc_mu = nn.Linear(self.enc_output_dim, latent_dim)
#         self.fc_log_var = nn.Linear(self.enc_output_dim, latent_dim)
#
#         # Decoder input layer
#         self.decoder_input = nn.Linear(latent_dim, self.enc_output_dim)
#
#         # Decoder
#         # self.dec1 = UpConvBlock(ngf * 8, ngf * 8, norm=False)  # 2×2
#         # self.dec2 = UpConvBlock(ngf * 8, ngf * 8, norm=False)  # 4×4
#         self.dec3 = UpConvBlock(ngf * 8, ngf * 8, norm=False)  # 8×8
#         self.dec4 = UpConvBlock(ngf * 8, ngf * 4, norm=False)  # 16×16
#         self.dec5 = UpConvBlock(ngf * 4, ngf * 2, norm=False)  # 32×32
#         self.dec6 = UpConvBlock(ngf * 2, ngf, norm=False)  # 64×64
#         self.dec7 = UpConvBlock(ngf, 3, norm=False, activation="sigmoid") # 128×128
#
#         self.latent_dim = latent_dim
#
#     def forward(self, x):
#         # Encode input to get mean and log variance of latent distribution
#         e1 = self.enc1(x)
#         e2 = self.enc2(e1)
#         e3 = self.enc3(e2)
#         e4 = self.enc4(e3)
#         e5 = self.enc5(e4)
#         # e6 = self.enc6(e5)
#         # e7 = self.enc7(e6)
#
#         x = e5.view(x.size(0), -1)  # Flatten
#
#         mu = self.fc_mu(x)
#         log_var = self.fc_log_var(x)
#
#         z = self.reparameterize(mu, log_var)
#
#         # Decode from latent space
#         x = self.decoder_input(z)
#         x = x.view(x.size(0), 128, 4, 4)  # Reshape to match encoder output
#
#         # d1 = self.dec1(x)
#         # d1 = torch.cat([d1, e6], dim=1)
#         # d2 = self.dec2(d1)
#         # d2 = torch.cat([d2, e5], dim=1)
#         x = x + e5
#         d3 = self.dec3(x)
#         # d3 = torch.cat([d3, e4], dim=1)
#         # d3 = d3 + e4
#         d4 = self.dec4(d3)
#         d5 = self.dec5(d4)
#         # d5 = torch.cat([d5, e2], dim=1)
#         d6 = self.dec6(d5)
#         # d6 = torch.cat([d6, e1], dim=1)
#         output = self.dec7(d6)
#
#         return output, mu, log_var
#
#     def reparameterize(self, mu, log_var):
#         # Reparameterization trick
#         std = torch.exp(0.5 * log_var)
#         eps = torch.randn_like(std)
#         z = mu + eps * std
#         return z
#
#     def decode(self, z):
#         pass
#
#     def forward_(self, x):
#         # Full forward pass
#         mu, log_var = self.encode(x)
#         z = self.reparameterize(mu, log_var)
#         decoded = self.decode(z)
#         return decoded, mu, log_var
#
#
# # class ResBlock(nn.Module):
# #     def __init__(self, in_ch, out_ch):
# #         super().__init__()
# #         self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
# #         self.bn1   = nn.BatchNorm2d(out_ch)
# #         self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
# #         self.bn2   = nn.BatchNorm2d(out_ch)
# #
# #         if in_ch != out_ch:
# #             self.skip = nn.Conv2d(in_ch, out_ch, 1)
# #         else:
# #             self.skip = nn.Identity()
# #
# #     def forward(self, x):
# #         residual = self.skip(x)
# #         x = F.relu(self.bn1(self.conv1(x)))
# #         x = self.bn2(self.conv2(x))
# #         return F.relu(x + residual)
# #
# # class Encoder(nn.Module):
# #     def __init__(self, latent_dim=128):
# #         super().__init__()
# #
# #         self.block1 = ResBlock(3, 32)
# #         self.down1  = nn.Conv2d(32, 64, 4, stride=2, padding=1)
# #
# #         self.block2 = ResBlock(64, 64)
# #         self.down2  = nn.Conv2d(64, 128, 4, stride=2, padding=1)
# #
# #         self.block3 = ResBlock(128, 128)
# #
# #         self.mu     = nn.Conv2d(128, latent_dim, 1)
# #         self.logvar = nn.Conv2d(128, latent_dim, 1)
# #
# #     def forward(self, x):
# #         s1 = self.block1(x)
# #         x  = self.down1(s1)
# #
# #         s2 = self.block2(x)
# #         x  = self.down2(s2)
# #
# #         x  = self.block3(x)
# #
# #         return self.mu(x), self.logvar(x), (s1, s2)
# #
# # class Decoder(nn.Module):
# #     def __init__(self, latent_dim=128):
# #         super().__init__()
# #
# #         self.block1 = ResBlock(latent_dim, 128)
# #         self.up1    = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
# #
# #         self.block2 = ResBlock(64, 64)
# #         self.up2    = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
# #
# #         self.block3 = ResBlock(32, 32)
# #
# #         self.out    = nn.Conv2d(32, 3, 3, padding=1)
# #
# #     def forward(self, z, skips):
# #         s1, s2 = skips
# #
# #         x = self.block1(z)
# #         x = self.up1(x)
# #
# #         # skip connection (somma, non concat)
# #         x = x + s2
# #         x = self.block2(x)
# #         x = self.up2(x)
# #
# #         x = x + s1
# #         x = self.block3(x)
# #
# #         return torch.sigmoid(self.out(x))
# #
# # class LiteVAE(nn.Module):
# #     def __init__(self, latent_dim=128):
# #         super().__init__()
# #         self.encoder = Encoder(latent_dim)
# #         self.decoder = Decoder(latent_dim)
# #
# #     def reparameterize(self, mu, logvar):
# #         std = torch.exp(0.5 * logvar)
# #         eps = torch.randn_like(std)
# #         return mu + eps * std
# #
# #     def forward(self, x):
# #         mu, logvar, skips = self.encoder(x)
# #         z = self.reparameterize(mu, logvar)
# #         x_hat = self.decoder(z, skips)
# #         return x_hat, mu, logvar
