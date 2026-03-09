import torch
from torch import nn
from torch.nn import functional as F


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


def apply_gaussian_smoothing(error_map, kernel_size=7, sigma=2.0):
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


def gaussian(window_size, sigma):
    """Create a 1D Gaussian kernel."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    return g


def create_window(window_size, channel, device):
    """
    Create a 2D Gaussian window to compute SSIM.
    window_size: e.g. 11
    channel: number of channels (3 for RGB)
    """
    _1D_window = gaussian(window_size, sigma=1.5).to(device).unsqueeze(1)
    _2D_window = _1D_window @ _1D_window.t()  # outer product → (window_size × window_size)
    window = _2D_window.unsqueeze(0).unsqueeze(0)  # shape (1,1,ws,ws)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    return window


class SSIMLoss(nn.Module):
    """
    Differentiable SSIM loss. Computes (1 - mean(SSIM map)) between two images.
    Assumes inputs are in [0,1].
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3  # assume 3 channels for RGB
        self.register_buffer("C1", torch.tensor((0.01 * 255) ** 2))
        self.register_buffer("C2", torch.tensor((0.03 * 255) ** 2))
        # The window will be created on first forward
        self.register_buffer("window", create_window(window_size, self.channel, device="cpu"))

    def _ssim_map(self, img1, img2, window, size_average=True):
        """
        Compute an SSIM map between img1 and img2 using a given window.
        img1, img2: [B, C, H, W], values in [0,1]
        window: [C,1,ws,ws]
        """
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        C1, C2 = self.C1, self.C2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map

    def forward(self, img_gen, img_ref):
        """
        img_gen, img_ref: [B,3,H,W] in range [-1,1] or [0,1]. We'll assume [-1,1], so map to [0,1].
        Returns: 1 - mean(SSIM map over all pixels & channels).
        """
        # Convert from [-1,1] to [0,1]
        gen = img_gen #+ 1.0) / 2.0
        ref = img_ref #+ 1.0) / 2.0

        # Ensure window is on same device
        window = self.window.to(gen.device)
        # Compute SSIM map
        if gen.shape[1] == self.channel:
            ssim_map = self._ssim_map(gen, ref, window, self.size_average)
        else:
            # If somehow not 3 channels, rebuild window for correct num channels
            channel = gen.shape[1]
            window = create_window(self.window_size, channel, gen.device)
            ssim_map = self._ssim_map(gen, ref, window, self.size_average)

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            # Return a per-batch-item SSIM loss
            return 1 - ssim_map.view(ssim_map.size(0), -1).mean(1)
