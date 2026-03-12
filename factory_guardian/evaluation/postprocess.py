from typing import Tuple
import torch
from torch import Tensor
from torch.nn import functional as F


def post_process(
    inputs: Tensor,
    outputs: Tensor,
    kernel_size: int = 7,
    sigma: float = 2.0
) -> Tuple[Tensor, Tensor]:
    """
    Perform post-processing to compute anomaly map and anomaly score.

    Generate an anomaly map based on the provided input and output tensors,
    using the specified kernel size and sigma for its computation.
    Derive an anomaly score from the anomaly map.

    Args:
        inputs (Tensor): Model input values.
        outputs (Tensor): Model output values.
        kernel_size (int, optional): Size of the Gaussian kernel. Defaults to 7.
        sigma (float, optional): Standard deviation of the Gaussian kernel. Defaults to 2.0.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing the computed anomaly map and anomaly score.
    """
    # Compute anomaly map
    anom_map = anomaly_map(inputs, outputs, kernel_size, sigma)

    # Compute anomaly score
    anom_score = anomaly_score(anom_map)

    return anom_map, anom_score


def anomaly_map(
    inputs: Tensor,
    outputs: Tensor,
    kernel_size: int = 7,
    sigma: float = 2.0
) -> Tensor:
    """
    Compute the anomaly map based on the difference between inputs and outputs
    and apply Gaussian smoothing.

    Args:
        inputs (Tensor): Model input values.
        outputs (Tensor): Model output values.
        kernel_size (int, optional): Size of the Gaussian kernel. Defaults to 7.
        sigma (float, optional): Standard deviation of the Gaussian kernel. Defaults to 2.0.

    Returns:
        Tensor: The computed smoothed anomaly map.
    """
    # L2 distance between inputs and outputs
    anom_map = torch.sum((inputs - outputs) ** 2, dim=1, keepdim=True)

    # Gaussian smoothing
    anom_map = gaussian_smoothing(anom_map, kernel_size, sigma)

    return anom_map


def anomaly_score(anom_map: Tensor) -> Tensor:
    """
    Compute the anomaly score by calculating the mean of the top-k values
    in a flattened anomaly map.

    Args:
        anom_map (Tensor): Tensor containing the anomaly map. The first dimension
            represents the batch size, while the remaining dimensions represent
            the spatial dimensions of the map.

    Returns:
        Tensor: Tensor of anomaly scores, where each element corresponds to
            the score for a single input in the batch.
    """
    flat = anom_map.view(anom_map.size(0), -1)

    # Pick the top-k values based on the percentage of pixels
    k = max(1, int(0.01 * flat.size(1)))
    topk = torch.topk(flat, k, dim=1)[0]

    # Compute the mean of the top-k values
    score = topk.mean(dim=1)

    return score


def gaussian_smoothing(
    error_map: Tensor,
    kernel_size: int = 7,
    sigma: float = 2.0
) -> Tensor:
    """
    Apply Gaussian smoothing to a 4D tensor.

    Perform convolutional smoothing on the input tensor using a Gaussian kernel.
    Support multi-channel input by treating each channel independently.

    Args:
        error_map (Tensor): Input tensor of shape (N, C, H, W), where N is the batch size,
            C is the number of channels, H is the height, and W is the width.
        kernel_size (int): Size of the kernel (assumed to be square). Defaults to 7.
        sigma (float): Standard deviation of the Gaussian distribution. Defaults to 2.0.

    Returns:
        Tensor: Smoothed tensor of the same shape as the input.
    """
    channels = error_map.shape[1]

    # Create the Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma, channels)
    kernel = kernel.to(error_map.device)

    # Apply the Gaussian kernel to the input tensor
    smoothed = F.conv2d(
        error_map,
        kernel,
        padding=kernel_size // 2,
        groups=channels
    )

    return smoothed


def gaussian_kernel(
    kernel_size: int = 7,
    sigma: float = 2.0,
    channels: int = 1
) -> Tensor:
    """
    Generate a Gaussian kernel for image processing or convolutional operations.

    The kernel is created based on the specified size, standard deviation, and the
    number of channels, and is normalized so that the sum of all elements equals 1.

    Args:
        kernel_size (int): Size of the kernel (assumed to be square). Defaults to 7.
        sigma (float): Standard deviation of the Gaussian distribution. Defaults to 2.0.
        channels (int): Number of channels for the kernel. Defaults to 1.

    Returns:
        Tensor: A tensor representing the Gaussian kernel with shape
            (channels, 1, kernel_size, kernel_size).
    """
    # Create a 2D grid of coordinates
    coords = torch.arange(kernel_size).float() - kernel_size // 2
    grid = coords.repeat(kernel_size).view(kernel_size, kernel_size)
    x_grid = grid
    y_grid = grid.t()

    # Calculate the Gaussian kernel
    kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    # Normalize the kernel
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)

    return kernel
