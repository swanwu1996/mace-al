from __future__ import annotations

from pathlib import Path

import numpy as np
from ase.io import read, write


DEFAULT_SOURCE = "/home/qhwu/BaFeF4/9-mace/1-mace_tryrun/bafeF4_train_stage1.extxyz"


def init_demo(root: str | Path, source: str = DEFAULT_SOURCE) -> None:
    root = Path(root).resolve()
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Missing demo source dataset: {source_path}")

    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    images = read(str(source_path), ":")
    if not isinstance(images, list):
        images = [images]
    if len(images) < 4:
        raise ValueError(f"Demo source needs at least 4 structures: {source_path}")

    train = images[:-2]
    test = images[-2:]
    seeds = []
    rng = np.random.default_rng(2026)
    for atoms in images[:4]:
        seed = atoms.copy()
        seed.calc = None
        for key in ["REF_forces", "forces"]:
            if key in seed.arrays:
                del seed.arrays[key]
        for key in ["REF_energy", "energy", "REF_stress", "stress"]:
            seed.info.pop(key, None)
        seed.positions[:] = seed.positions + rng.normal(0.0, 0.01, size=seed.positions.shape)
        seed.info["config_type"] = "demo_seed"
        seeds.append(seed)

    write(str(data_dir / "train.extxyz"), train, format="extxyz")
    write(str(data_dir / "test.extxyz"), test, format="extxyz")
    write(str(data_dir / "seeds.extxyz"), seeds, format="extxyz")
    print(f"Wrote {len(train)} train, {len(test)} test, {len(seeds)} seed structures to {data_dir}")


def init_representative_demo(
    root: str | Path,
    source: str = DEFAULT_SOURCE,
    train_count: int = 20,
    test_count: int = 5,
    seed_count: int = 8,
) -> None:
    root = Path(root).resolve()
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Missing representative source dataset: {source_path}")
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    images = read(str(source_path), ":")
    if not isinstance(images, list):
        images = [images]
    if len(images) < train_count + test_count + seed_count:
        raise ValueError(f"Need at least {train_count + test_count + seed_count} structures in {source_path}")

    train = images[:train_count]
    test = images[train_count : train_count + test_count]
    seed_src = images[train_count + test_count : train_count + test_count + seed_count]
    rng = np.random.default_rng(2027)
    seeds = []
    for atoms in seed_src:
        seed = atoms.copy()
        seed.calc = None
        for key in ["REF_forces", "forces"]:
            if key in seed.arrays:
                del seed.arrays[key]
        for key in ["REF_energy", "energy", "REF_stress", "stress"]:
            seed.info.pop(key, None)
        seed.positions[:] = seed.positions + rng.normal(0.0, 0.015, size=seed.positions.shape)
        seed.info["config_type"] = "representative_seed"
        seeds.append(seed)

    write(str(data_dir / "train.extxyz"), train, format="extxyz")
    write(str(data_dir / "test.extxyz"), test, format="extxyz")
    write(str(data_dir / "seeds.extxyz"), seeds, format="extxyz")
    print(f"Wrote representative demo: {len(train)} train, {len(test)} test, {len(seeds)} seeds to {data_dir}")
