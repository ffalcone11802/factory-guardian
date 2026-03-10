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


def parent_dir(
    path: Union[str, PathLike[str]]
) -> str:
    return str(Path(path).parent)


def working_dir() -> str:
    return str(Path.cwd())


def check_path(
    path: Union[str, PathLike[str]]
) -> bool:
    return Path(path).exists()


def path_joiner(
    *args: Union[str, PathLike[str]]
) -> Path:
    return Path(*args)
