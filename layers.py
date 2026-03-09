from torch import nn
from torch.nn import functional as F


class DSConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
        norm=True,
        activation="relu"
    ):
        super(DSConvBlock, self).__init__()

        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels) if norm else None

        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation '{activation}'")

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x


class UpConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        norm=True,
        activation="relu"
    ):
        super(UpConvBlock, self).__init__()

        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.bn = nn.BatchNorm2d(out_channels) if norm else None

        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation '{activation}'")

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x


# class ConvBlock(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         kernel_size=4,
#         stride=2,
#         padding=1,
#         norm=True,
#         activation="leaky"
#     ):
#         super(ConvBlock, self).__init__()
#
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not norm)
#         self.bn = nn.BatchNorm2d(out_channels, affine=False) if norm else None
#
#         if activation == "relu":
#             self.activation = nn.ReLU(inplace=True)
#         elif activation == "leaky":
#             self.activation = nn.LeakyReLU(0.2, inplace=True)
#         else:
#             raise ValueError(f"Unsupported activation '{activation}'")
#
#     def forward(self, x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         x = self.activation(x)
#         return x
#
#
# class UpConvBlock(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         kernel_size=4,
#         stride=2,
#         padding=1,
#         norm=True,
#         activation="leaky"
#     ):
#         super(UpConvBlock, self).__init__()
#
#         self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=not norm)
#         self.bn = nn.BatchNorm2d(out_channels, affine=False) if norm else None
#
#         if activation == "relu":
#             self.activation = nn.ReLU(inplace=True)
#         elif activation == "leaky":
#             self.activation = nn.LeakyReLU(0.2, inplace=True)
#         elif activation == "sigmoid":
#             self.activation = nn.Sigmoid()
#         else:
#             raise ValueError(f"Unsupported activation '{activation}'")
#
#     def forward(self, x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         x = self.activation(x)
#         return x
#
#
# class ResBlock(nn.Module):
#     def __init__(self, ch):
#         super().__init__()
#         self.dw1 = nn.Conv2d(ch, ch, 3, padding=1, groups=ch)
#         self.pw1 = nn.Conv2d(ch, ch, 1)
#         self.bn1 = nn.BatchNorm2d(ch)
#
#         # self.dw2 = nn.Conv2d(ch, ch, 3, padding=1, groups=ch)
#         # self.pw2 = nn.Conv2d(ch, ch, 1)
#         # self.bn2 = nn.BatchNorm2d(ch)
#
#         self.activation = nn.LeakyReLU(0.2, inplace=True)
#
#     def forward(self, x):
#         h = self.activation(self.bn1(self.pw1(self.dw1(x))))
#         # h = self.bn2(self.pw2(self.dw2(h)))
#         return self.activation(h + x)
#
#     #     self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
#     #     self.bn1 = nn.BatchNorm2d(ch)
#     #     self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
#     #     self.bn2 = nn.BatchNorm2d(ch)
#     #
#     # def forward(self, x):
#     #     h = F.relu(self.bn1(self.conv1(x)))
#     #     h = self.bn2(self.conv2(h))
#     #     return F.relu(h + x)
