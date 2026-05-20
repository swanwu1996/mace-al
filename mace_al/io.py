from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ase import Atoms
from ase.io import read, write


def read_atoms(path: str | Path) -> list[Atoms]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    images = read(str(path), ":")
    return images if isinstance(images, list) else [images]


def write_atoms(path: str | Path, images: Iterable[Atoms], append: bool = False) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = list(images)
    if not images:
        path.touch(exist_ok=True)
        return 0
    write(str(path), images, format="extxyz", append=append and path.exists())
    return len(images)


def append_atoms(path: str | Path, images: Iterable[Atoms]) -> int:
    return write_atoms(path, images, append=True)

