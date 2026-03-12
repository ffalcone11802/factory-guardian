from typing import Union, Generator
from os import PathLike

import shutil
from pathlib import Path


def check_dir(
    path: Union[str, PathLike[str]],
    replace: bool = False
) -> Path:
    _path = Path(path)

    if _path.exists():
        if replace:
            shutil.rmtree(_path, ignore_errors=True)
            _path.mkdir(parents=True, exist_ok=True)
    else:
        _path.mkdir(parents=True, exist_ok=True)

    return _path.resolve()


def list_dir(
    path: Union[str, PathLike[str]]
) -> Generator[Path, None, None]:
    _dirs = [d for d in Path(path).iterdir()]
    for d in _dirs:
        if d.is_dir():
            yield d


def path_joiner(
    *args: Union[str, PathLike[str]]
) -> Path:
    return Path(*args)


_RESULTS_FOLDER = "results"

PLOTS_FOLDER = path_joiner(_RESULTS_FOLDER, "plots").resolve()
CHECKPOINTS_FOLDER = path_joiner(_RESULTS_FOLDER, "checkpoints").resolve()
PARAMS_FOLDER = path_joiner(_RESULTS_FOLDER, "params").resolve()
WEIGHTS_FOLDER = path_joiner(_RESULTS_FOLDER, "weights").resolve()
ONNX_FOLDER = Path("onnx_inference").resolve()
