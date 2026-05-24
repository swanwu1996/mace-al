from __future__ import annotations

import glob
import gc
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from .io import append_atoms, read_atoms, write_atoms
from .paths import Layout, format_generation
from .uncertainty import committee_uncertainty, in_trust_region


def mace_calculator(model: Path, cfg: dict):
    mace_cfg = cfg["mace"]
    cuda_visible_devices = mace_cfg.get("cuda_visible_devices")
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    from mace.calculators import MACECalculator

    kwargs = {}
    if "head" in mace_cfg:
        kwargs["head"] = mace_cfg["head"]
    return MACECalculator(
        model_paths=str(model),
        device=mace_cfg.get("device", "cuda"),
        default_dtype=mace_cfg.get("default_dtype", "float64"),
        **kwargs,
    )


def committee_models(cfg: dict, layout: Layout) -> list[Path]:
    pattern = layout.rel(format_generation(cfg["explore"]["model_glob"], layout.generation))
    models = sorted(Path(p) for p in glob.glob(str(pattern)))
    if models:
        return models
    fallback = layout.rel(cfg["explore"]["fallback_model"])
    return [fallback] if fallback.exists() else []


def release_mace_resources(calculators: list) -> None:
    calculators.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _explore_one_task(args: tuple) -> tuple[list, list[str]]:
    cfg, layout_root, generation, seed_i, seed_atoms, temp, gpu_id = args
    cfg = deepcopy(cfg)
    if gpu_id is not None:
        cfg.setdefault("mace", {})["cuda_visible_devices"] = str(gpu_id)

    layout = Layout.from_config(cfg, root=layout_root, generation=generation)
    explore_cfg = cfg["explore"]
    models = committee_models(cfg, layout)
    calculators = [mace_calculator(model, cfg) for model in models]
    prod_calc = calculators[0]
    out_dir = layout.stage("explore")
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    rows = []

    try:
        atoms = seed_atoms.copy()
        atoms.calc = prod_calc
        MaxwellBoltzmannDistribution(atoms, temperature_K=float(temp), force_temp=True)
        dyn = Langevin(
            atoms,
            timestep=float(explore_cfg["timestep_fs"]) * units.fs,
            temperature_K=float(temp),
            friction=1.0 / (float(explore_cfg["friction_fs"]) * units.fs),
        )
        source = f"seed{seed_i:04d}_T{int(temp)}"
        traj_file = out_dir / f"trajectory_{source}.xyz"
        last_selected = -10**9

        for step in range(0, int(explore_cfg["md_steps"]) + 1):
            if step > 0:
                dyn.run(1)
            if step % int(explore_cfg["sample_interval"]) != 0:
                continue
            frame = atoms.copy()
            u = committee_uncertainty(frame, calculators)
            chosen = in_trust_region(u, explore_cfg) and (
                step - last_selected >= int(explore_cfg["min_frame_gap"])
            )
            frame.info.update(
                {
                    "al_generation": layout.generation,
                    "al_source": source,
                    "md_step": step,
                    "temperature_K": float(temp),
                    "mace_energy_std_per_atom": u.energy_std_per_atom,
                    "mace_max_force_std": u.max_force_std,
                    "mace_mean_force_std": u.mean_force_std,
                }
            )
            append_atoms(traj_file, [frame])
            if chosen:
                selected.append(frame)
                last_selected = step
            rows.append(
                f"{source},{temp},{step},{u.energy_std_per_atom:.8g},"
                f"{u.max_force_std:.8g},{u.mean_force_std:.8g},{int(chosen)}"
            )
    finally:
        release_mace_resources(calculators)

    return selected, rows


def run_explore(cfg: dict, layout: Layout) -> Path:
    seeds = read_atoms(layout.rel(cfg["project"]["seed_file"]))
    if not seeds:
        raise FileNotFoundError(f"No seed structures: {layout.rel(cfg['project']['seed_file'])}")

    models = committee_models(cfg, layout)
    if len(models) < 2:
        raise RuntimeError(
            "Exploration needs at least two MACE models in the committee. "
            f"Run `maceal mace --generation {layout.generation}` first."
        )

    explore_cfg = cfg["explore"]
    out_dir = layout.stage("explore")
    selected = []
    rows = ["source,temp_K,step,energy_std_per_atom,max_force_std,mean_force_std,selected"]
    gpus = [str(gpu) for gpu in explore_cfg.get("gpus", [])]
    workers = int(explore_cfg.get("parallel_workers", len(gpus) if gpus else 1))

    if workers > 1 and gpus:
        tasks = []
        for seed_i, seed_atoms in enumerate(seeds):
            for temp_i, temp in enumerate(explore_cfg["temperature"]):
                gpu_id = gpus[len(tasks) % len(gpus)]
                tasks.append((cfg, str(layout.root), layout.generation, seed_i, seed_atoms, temp, gpu_id))
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            futures = [executor.submit(_explore_one_task, task) for task in tasks]
            for future in as_completed(futures):
                task_selected, task_rows = future.result()
                selected.extend(task_selected)
                rows.extend(task_rows)
    else:
        calculators = [mace_calculator(model, cfg) for model in models]
        prod_calc = calculators[0]
        try:
            for seed_i, seed_atoms in enumerate(seeds):
                for temp in explore_cfg["temperature"]:
                    atoms = seed_atoms.copy()
                    atoms.calc = prod_calc
                    MaxwellBoltzmannDistribution(atoms, temperature_K=float(temp), force_temp=True)
                    dyn = Langevin(
                        atoms,
                        timestep=float(explore_cfg["timestep_fs"]) * units.fs,
                        temperature_K=float(temp),
                        friction=1.0 / (float(explore_cfg["friction_fs"]) * units.fs),
                    )
                    source = f"seed{seed_i:04d}_T{int(temp)}"
                    traj_file = out_dir / f"trajectory_{source}.xyz"
                    last_selected = -10**9

                    for step in range(0, int(explore_cfg["md_steps"]) + 1):
                        if step > 0:
                            dyn.run(1)
                        if step % int(explore_cfg["sample_interval"]) != 0:
                            continue
                        frame = atoms.copy()
                        u = committee_uncertainty(frame, calculators)
                        chosen = in_trust_region(u, explore_cfg) and (
                            step - last_selected >= int(explore_cfg["min_frame_gap"])
                        )
                        frame.info.update(
                            {
                                "al_generation": layout.generation,
                                "al_source": source,
                                "md_step": step,
                                "temperature_K": float(temp),
                                "mace_energy_std_per_atom": u.energy_std_per_atom,
                                "mace_max_force_std": u.max_force_std,
                                "mace_mean_force_std": u.mean_force_std,
                            }
                        )
                        append_atoms(traj_file, [frame])
                        if chosen:
                            selected.append(frame)
                            last_selected = step
                        rows.append(
                            f"{source},{temp},{step},{u.energy_std_per_atom:.8g},"
                            f"{u.max_force_std:.8g},{u.mean_force_std:.8g},{int(chosen)}"
                        )
        finally:
            release_mace_resources(calculators)

    selected = selected[: int(explore_cfg["max_candidates"])]
    write_atoms(out_dir / "candidates.xyz", selected)
    (out_dir / "uncertainty.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return out_dir / "candidates.xyz"
