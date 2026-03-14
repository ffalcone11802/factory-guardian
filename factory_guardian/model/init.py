from typing import Callable
import torch.nn as nn
from torch.nn.init import xavier_uniform_, kaiming_normal_, normal_, constant_

_NORM_LAYERS = (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)
_WEIGHT_LAYERS = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)


def _init_weights(
    m: nn.Module,
    weight_init: Callable,
    gain: float = 0.02,
    mean: float = 0.0
):
    """
    Initialize weights for layers in a neural network module.

    Support different initialization methods for weights based on the type of
    the layer and its parameters.

    Args:
        m (nn.Module): Module to initialize weights for.
        weight_init (Callable): Initialization function to apply.
        gain (float): Scaling factor for Xavier initialization or standard deviation
            for normal initialization. Defaults to 0.02.
        mean (float): Mean value for normal initialization. Defaults to 0.0.
    """
    if isinstance(m, _WEIGHT_LAYERS):
        # If the layer is a convolutional layer or linear layer,
        # apply the specified initialization function
        if weight_init is xavier_uniform_:
            weight_init(m.weight, gain=gain)
        elif weight_init is normal_:
            weight_init(m.weight, mean=mean, std=gain)
        else:
            weight_init(m.weight, nonlinearity="relu")

        # If the layer has a bias term, initialize it to zero
        if m.bias is not None:
            constant_(m.bias, 0.0)

        return

    # If the layer is a normalization layer, apply the normal initialization
    if isinstance(m, _NORM_LAYERS) and getattr(m, "affine", False):
        normal_(m.weight, mean=1.0, std=gain)
        constant_(m.bias, 0.0)


def xavier_uniform_init(m: nn.Module, gain: float = 0.02):
    """
    Initialize weights of a neural network module using Xavier uniform distribution.

    Args:
        m (nn.Module): Module whose weights are to be initialized.
        gain (float): Scaling factor for the Xavier uniform distribution. Defaults to 0.02.
    """
    _init_weights(m, xavier_uniform_, gain=gain)


def normal_init(m: nn.Module, mean: float = 0.0, std: float = 0.02):
    """
    Initialize the weights of a given module using a normal distribution.

    Args:
        m (nn.Module): Module whose weights are to be initialized.
        mean (float): Mean value of the normal distribution. Defaults to 0.0.
        std (float): Standard deviation of the normal distribution. Defaults to 0.02.
    """
    _init_weights(m, normal_, mean=mean, gain=std)


def kaiming_normal_init(m: nn.Module):
    """
    Initialize the weights of a module using Kaiming normal initialization.

    Args:
        m (nn.Module): Module whose weights are to be initialized.
    """
    _init_weights(m, kaiming_normal_)


def get_init_function(init_type: str = "xavier"):
    """
    Return the initialization function corresponding to the specified initialization type.

    Args:
        init_type (str): Initialization type. Defaults to "xavier".

    Returns:
        Callable: The corresponding initialization function.

    Raises:
        NotImplementedError: If the specified initialization type is not implemented.
    """
    init_map = {
        "xavier": xavier_uniform_init,
        "normal": normal_init,
        "kaiming": kaiming_normal_init,
    }

    try:
        return init_map[init_type]
    except KeyError:
        raise NotImplementedError(f"Initialization method '{init_type}' is not implemented")
