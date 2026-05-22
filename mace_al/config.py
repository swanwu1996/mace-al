from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    if suffix in {".yaml", ".yml"}:
        import yaml

        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    if suffix == ".toml":
        return load_toml(path)
    raise ValueError(f"Unsupported config format: {path}")


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_run_config(param_path: str | Path, machine_path: str | Path) -> tuple[dict[str, Any], Path]:
    param_path = Path(param_path).resolve()
    machine_path = Path(machine_path).resolve()
    root = param_path.parent
    cfg = deep_update(default_config(), load_config(param_path))
    machine = load_config(machine_path)
    cfg["machine"] = machine
    for section in ["mace", "explore", "dft", "select"]:
        if isinstance(machine.get(section), dict):
            cfg[section] = deep_update(cfg.get(section, {}), machine[section])
    return cfg, root


def load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib

        with path.open("rb") as f:
            return tomllib.load(f)
    except ModuleNotFoundError:
        try:
            import tomli

            with path.open("rb") as f:
                return tomli.load(f)
        except ModuleNotFoundError:
            return parse_simple_toml(path)


def parse_simple_toml(path: Path) -> dict[str, Any]:
    data: dict[str, dict[str, Any]] = {}
    section: dict[str, Any] | None = None
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = data.setdefault(line[1:-1], {})
            continue
        if section is None or "=" not in line:
            raise ValueError(f"Cannot parse config line: {raw}")
        key, value = [part.strip() for part in line.split("=", 1)]
        value = value.replace("true", "True").replace("false", "False")
        section[key] = ast.literal_eval(value)
    return data


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def default_config() -> dict[str, Any]:
    return {
        "version": "0.1.0",
        "generation": 0,
        "work_path": "./cache",
        "project": {
            "name": "mace_active_learning",
            "seed_file": "./data/seeds.extxyz",
            "train_file": "./data/train.extxyz",
            "test_file": "./data/test.extxyz",
            "foundation_model": "/home/qhwu/mace_mod/mace-omat-0-medium.model",
            "foundation_models": [
                "/home/qhwu/mace_mod/mace-mh-1.model",
                "/home/qhwu/mace_mod/mace-mpa-0-medium.model",
                "/home/qhwu/mace_mod/mace-omat-0-medium.model",
            ],
        },
        "mace": {
            "device": "cuda",
            "default_dtype": "float64",
            "env_script": "module load anaconda && conda activate mace",
            "seeds": [1, 2, 3, 4],
            "max_num_epochs": 50,
            "batch_size": 2,
            "valid_batch_size": 2,
            "lr": 1e-4,
            "energy_key": "REF_energy",
            "forces_key": "REF_forces",
            "stress_key": "REF_stress",
            "energy_weight": 1.0,
            "forces_weight": 10.0,
            "stress_weight": 1.0,
            "E0s": "average",
        },
        "explore": {
            "model_glob": "cache/Generation-{generation}/mace/committee/*.model",
            "fallback_model": "/home/qhwu/mace_mod/mace-omat-0-medium.model",
            "md_steps": 2000,
            "timestep_fs": 0.5,
            "temperature": [300, 600, 900],
            "sample_interval": 20,
            "friction_fs": 100.0,
            "force_std_low": 0.08,
            "force_std_high": 0.35,
            "energy_std_per_atom_high": 0.02,
            "max_candidates": 200,
            "min_frame_gap": 10,
        },
        "select": {
            "max_selected": 50,
            "min_distance": 0.01,
            "descriptor": "mace_uncertainty",
        },
        "dft": {
            "job": 10,
            "template_dir": "./templates/vasp",
            "potcar_root": "/home/qhwu/PBE64",
            "potcar_map": {},
            "use_genque": False,
            "genque_choice": "2",
            "submit_script": "./templates/vasp/submit_vasp.sh",
            "submit_command": "sbatch {submit_script}",
            "done_file": "vasprun.xml",
            "poll_seconds": 300,
            "max_wait_seconds": 604800,
            "force_limit": 20.0,
            "require_ediff_reached": True,
        },
        "run": {
            "max_generations": 3,
            "auto_init_committee": True,
            "submit_dft": True,
            "run_dft_direct": False,
            "wait_dft": True,
            "train_next_generation": True,
            "stop_after_train_next": False,
            "final_generation": "last",
            "final_model_count": 4,
        },
    }


def default_machine() -> dict[str, Any]:
    return {
        "mace": {
            "env_script": "module load anaconda && conda activate mace",
            "device": "cuda",
        },
        "explore": {
            "device": "cuda",
        },
        "dft": {
            "use_genque": False,
            "genque_choice": "2",
            "submit_script": "./templates/vasp/submit_vasp.sh",
            "submit_command": "sbatch {submit_script}",
            "done_file": "vasprun.xml",
            "poll_seconds": 300,
            "max_wait_seconds": 604800,
            "potcar_root": "/home/qhwu/PBE64",
        },
    }
