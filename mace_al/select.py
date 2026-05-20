from __future__ import annotations

import numpy as np

from .io import read_atoms, write_atoms
from .paths import Layout


def farthest_point_sampling(points: np.ndarray, n_samples: int, min_dist: float) -> list[int]:
    if len(points) == 0 or n_samples <= 0:
        return []
    sampled = [int(np.argmax(points[:, 0]))]
    min_distances = np.linalg.norm(points - points[sampled[0]], axis=1)
    while len(sampled) < n_samples:
        idx = int(np.argmax(min_distances))
        if min_distances[idx] < min_dist:
            break
        sampled.append(idx)
        new_distances = np.linalg.norm(points - points[idx], axis=1)
        min_distances = np.minimum(min_distances, new_distances)
    return sampled


def run_select(cfg: dict, layout: Layout) -> str:
    candidates = read_atoms(layout.stage("explore") / "candidates.xyz")
    if not candidates:
        out = layout.stage("select") / "selected.xyz"
        write_atoms(out, [])
        return str(out)

    points = np.array(
        [
            [
                float(a.info.get("mace_max_force_std", 0.0)),
                float(a.info.get("mace_mean_force_std", 0.0)),
                float(a.info.get("mace_energy_std_per_atom", 0.0)),
            ]
            for a in candidates
        ]
    )
    idx = farthest_point_sampling(
        points,
        int(cfg["select"]["max_selected"]),
        float(cfg["select"]["min_distance"]),
    )
    selected = [candidates[i] for i in idx]
    out = layout.stage("select") / "selected.xyz"
    write_atoms(out, selected)
    return str(out)

