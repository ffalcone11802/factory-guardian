from typing import Union, Generator
from os import PathLike
import shutil
from pathlib import Path


def check_folder(
    path: Union[str, PathLike[str]],
    replace: bool = False
) -> Path:
    """
    Ensure the provided folder exists, optionally replacing it if it exists.

    If the specified folder exists and the `replace` argument is set to True,
    the folder is removed and recreated. If it does not exist, it is created
    with all necessary parent folders.

    Args:
        path (Union[str, PathLike[str]]): Path to the target folder.
        replace (bool): Whether to replace the folder if it exists. Defaults to False.

    Returns:
        Path: Absolute resolved path to the ensured folder.
    """
    _path = Path(path)

    if _path.exists():
        if replace:
            shutil.rmtree(_path, ignore_errors=True)
            _path.mkdir(parents=True, exist_ok=True)
    else:
        _path.mkdir(parents=True, exist_ok=True)

    return _path.resolve()


def list_folders(
    path: Union[str, PathLike[str]]
) -> Generator[Path, None, None]:
    """
    List folders in the given path.

    Iterate through the contents of the specified path and yields folders found within it.

    Args:
        path (Union[str, PathLike[str]]): Path to the folder to list.

    Returns:
        Generator[Path, None, None]: A generator yielding Path objects for each folder found.
    """
    _dirs = [d for d in Path(path).iterdir()]
    for d in _dirs:
        if d.is_dir():
            yield d


def path_joiner(
    *args: Union[str, PathLike[str]]
) -> Path:
    """
    Join multiple path components and return a single path.

    Args:
        *args (Union[str, PathLike[str]]): Path components to join.

    Returns:
        Path: The joined path.
    """
    return Path(*args)


# Fixed folders

_RESULTS_FOLDER = "results"

PLOTS_FOLDER = path_joiner(_RESULTS_FOLDER, "plots").resolve()
CHECKPOINTS_FOLDER = path_joiner(_RESULTS_FOLDER, "checkpoints").resolve()
PARAMS_FOLDER = path_joiner(_RESULTS_FOLDER, "params").resolve()
WEIGHTS_FOLDER = path_joiner(_RESULTS_FOLDER, "weights").resolve()

ONNX_FOLDER = Path("onnx_inference").resolve()
