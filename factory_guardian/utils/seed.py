def set_numpy_seed(seed: int = 42):
    """
    Set the NumPy random seed for reproducibility.

    Args:
        seed (int): Integer to set as the random seed for NumPy. Defaults to 42.
    """
    import numpy as np
    np.random.seed(seed)


def set_random_seed(seed: int = 42):
    """
    Set the random seed for the random module for reproducibility.

    Args:
        seed (int): Integer to set as the random seed for the random module. Defaults to 42.
    """
    import random
    random.seed(seed)


def set_torch_seed(seed: int = 42):
    """
    Set the PyTorch random seed for reproducibility.

    Configures PyTorch to use the specified seed for all random number
    generation mechanisms (both CPU and GPU). Ensures deterministic
    operations by adjusting CUDA backend settings.

    Args:
        seed (int): Integer to set as the random seed for PyTorch. Defaults to 42.
    """
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_seed(seed: int = 42):
    """
    Set the random seed for numpy, random, and torch modules to ensure
    reproducibility across evaluations.

    Args:
        seed (int): Integer to set as the random seed for all modules. Defaults to 42.
    """
    set_numpy_seed(seed)
    set_random_seed(seed)
    set_torch_seed(seed)
