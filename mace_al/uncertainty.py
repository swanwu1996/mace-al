from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms


@dataclass(frozen=True)
class Uncertainty:
    energy_std_per_atom: float
    max_force_std: float
    mean_force_std: float


def committee_uncertainty(atoms: Atoms, calculators) -> Uncertainty:
    energies = []
    forces = []
    for calc in calculators:
        image = atoms.copy()
        image.calc = calc
        energies.append(float(image.get_potential_energy()))
        forces.append(np.asarray(image.get_forces(), dtype=float))
    force_arr = np.stack(forces, axis=0)
    force_std_atom = np.linalg.norm(force_arr.std(axis=0), axis=1)
    return Uncertainty(
        energy_std_per_atom=float(np.std(energies) / max(len(atoms), 1)),
        max_force_std=float(np.max(force_std_atom)),
        mean_force_std=float(np.mean(force_std_atom)),
    )


def in_trust_region(u: Uncertainty, explore_cfg: dict) -> bool:
    return (
        float(explore_cfg["force_std_low"]) <= u.max_force_std <= float(explore_cfg["force_std_high"])
        and u.energy_std_per_atom <= float(explore_cfg["energy_std_per_atom_high"])
    )

