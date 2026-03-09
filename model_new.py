import torch
from torch import nn
from layers import DSConvBlock, UpConvBlock


# class DSConv(nn.Module):
#     def __init__(self, in_c, out_c, stride=2):
#         super().__init__()
#         self.depthwise = nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False)
#         self.pointwise = nn.Conv2d(in_c, out_c, 1, bias=False)
#         self.bn = nn.BatchNorm2d(out_c)
#         self.act = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         x = self.bn(x)
#         return self.act(x)


class LiteVAE(nn.Module):
    def __init__(self, img_size=256, z_dim=128):
        super().__init__()

        base = 32

        # -------- Encoder --------
        self.encoder = nn.Sequential(
            DSConvBlock(3, base),          # 256 → 128
            DSConvBlock(base, base*2),     # 128 → 64
            DSConvBlock(base*2, base*4),   # 64 → 32
            DSConvBlock(base*4, base*8),   # 32 → 16
            DSConvBlock(base*8, base*8),   # 16 → 8
            DSConvBlock(base*8, base*8),   # 8 → 4
        )

        self.feature_dim = base*8*4*4

        self.fc_mu = nn.Linear(self.feature_dim, z_dim)
        self.fc_logvar = nn.Linear(self.feature_dim, z_dim)

        self.fc_dec = nn.Linear(z_dim, self.feature_dim)

        # -------- Decoder --------
        self.decoder = nn.Sequential(
            UpConvBlock(base*8, base*8),
            UpConvBlock(base*8, base*8),
            UpConvBlock(base*8, base*4),
            UpConvBlock(base*4, base*2),
            UpConvBlock(base*2, base),
            UpConvBlock(base, 3, norm=False, activation="sigmoid"),
        )

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z):
        x = self.fc_dec(z)
        x = x.view(z.size(0), -1, 4, 4)
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
