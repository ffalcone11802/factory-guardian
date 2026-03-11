import torch
from torch.nn import functional as F


def post_process(inputs, outputs, kernel_size=7, sigma=2.0):
    anom_map = anomaly_map(inputs, outputs, kernel_size, sigma)
    anom_score = anomaly_score(anom_map)
    return anom_map, anom_score


def anomaly_map(inputs, outputs, kernel_size, sigma):
    anom_map = torch.sum((inputs - outputs) ** 2, dim=1, keepdim=True)
    anom_map = gaussian_smoothing(anom_map, kernel_size, sigma)
    return anom_map


def anomaly_score(anom_map):
    flat = anom_map.view(anom_map.size(0), -1)
    k = max(1, int(0.01 * flat.size(1)))
    topk = torch.topk(flat, k, dim=1)[0]
    score = topk.mean(dim=1)
    return score


def gaussian_smoothing(error_map, kernel_size=7, sigma=2.0):
    channels = error_map.shape[1]
    kernel = gaussian_kernel(kernel_size, sigma, channels)
    kernel = kernel.to(error_map.device)

    smoothed = F.conv2d(
        error_map,
        kernel,
        padding=kernel_size // 2,
        groups=channels
    )
    return smoothed


def gaussian_kernel(kernel_size=7, sigma=2.0, channels=1):
    coords = torch.arange(kernel_size).float() - kernel_size // 2
    grid = coords.repeat(kernel_size).view(kernel_size, kernel_size)
    x_grid = grid
    y_grid = grid.t()

    kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel
