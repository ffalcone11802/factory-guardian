from torch import nn, Tensor


class DSConvBlock(nn.Module):
    """
    The DSConvBlock class defines a Depthwise Separable Convolutional Block.

    This class implements a depthwise separable convolution, which is a type of
    convolution known for being computationally efficient. The block includes
    depthwise and pointwise convolutions, optional batch normalization, and an
    optional activation function.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        stride (int, optional): Stride of the convolutional kernel. Defaults to 2.
        padding (int, optional): Padding added to the input. Defaults to 1.
        norm (bool, optional): Whether to apply batch normalization. Defaults to True.
        activation (str, optional): Activation function to be used. Defaults to "relu".
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        norm: bool = True,
        activation: str = "relu"
    ):
        super(DSConvBlock, self).__init__()

        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False
        )
        # Pointwise convolution
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, bias=False
        )
        # Batch normalization
        self.bn = nn.BatchNorm2d(out_channels) if norm else None

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation '{activation}'")

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform a forward pass through the block.

        Args:
            x (Tensor): Input tensor to be processed.

        Returns:
            Tensor: Output tensor after applying the depthwise, pointwise convolutions,
                optional batch normalization, and activation function.
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x


class UpConvBlock(nn.Module):
    """
    The UpSampleBlock class defines a block for performing transposed convolution
    followed by optional normalization and activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 4.
        stride (int, optional): Stride of the convolutional kernel. Defaults to 2.
        padding (int, optional): Padding added to the input. Defaults to 1.
        norm (bool, optional): Whether to apply batch normalization. Defaults to True.
        activation (str, optional): Activation function to be used. Defaults to "relu".
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        norm: int = True,
        activation: str = "relu"
    ):
        super(UpConvBlock, self).__init__()

        # Transposed convolution
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        # Batch normalization
        self.bn = nn.BatchNorm2d(out_channels) if norm else None

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation '{activation}'")

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform a forward pass through the block.

        Args:
            x (Tensor): Input tensor to be processed.

        Returns:
            Tensor: Output tensor after applying the transposed convolution,
                and optional batch normalization and activation function.
        """
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x
