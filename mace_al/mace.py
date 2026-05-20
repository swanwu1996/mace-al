from __future__ import annotations

import subprocess
import shutil
from pathlib import Path

from .paths import Layout


def foundation_models(cfg: dict, layout: Layout) -> list[Path]:
    project = cfg["project"]
    values = project.get("foundation_models") or [project["foundation_model"]]
    models = [layout.rel(value) for value in values]
    missing = [str(model) for model in models if not model.exists()]
    if missing:
        raise FileNotFoundError("Missing foundation model(s): " + ", ".join(missing))
    return models


def init_committee_from_foundations(cfg: dict, layout: Layout, copy: bool = False) -> Path:
    out_dir = layout.stage("mace") / "committee"
    out_dir.mkdir(parents=True, exist_ok=True)
    for model in foundation_models(cfg, layout):
        dst = out_dir / model.name
        if dst.exists():
            continue
        if copy:
            shutil.copy2(model, dst)
        else:
            dst.symlink_to(model)
    return out_dir


def train_committee(cfg: dict, layout: Layout) -> Path:
    mace_cfg = cfg["mace"]
    project = cfg["project"]
    out_dir = layout.stage("mace") / "committee"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_file = layout.rel(project["train_file"])
    if not train_file.exists():
        raise FileNotFoundError(f"Missing train file: {train_file}")
    e0s = mace_cfg.get("E0s")
    for foundation in foundation_models(cfg, layout):
        foundation_tag = foundation.stem.replace(".", "_").replace("-", "_")
        for seed in mace_cfg["seeds"]:
            name = f"{project['name']}_g{layout.generation}_{foundation_tag}_s{seed}"
            cmd = [
                "mace_run_train",
                f"--name={name}",
                f"--foundation_model={foundation}",
                "--multiheads_finetuning=False",
                f"--train_file={train_file}",
                "--valid_fraction=0.1",
                f"--energy_key={mace_cfg['energy_key']}",
                f"--forces_key={mace_cfg['forces_key']}",
                f"--stress_key={mace_cfg['stress_key']}",
                f"--energy_weight={mace_cfg['energy_weight']}",
                f"--forces_weight={mace_cfg['forces_weight']}",
                f"--stress_weight={mace_cfg['stress_weight']}",
                f"--lr={mace_cfg['lr']}",
                f"--batch_size={mace_cfg['batch_size']}",
                f"--valid_batch_size={mace_cfg['valid_batch_size']}",
                f"--max_num_epochs={mace_cfg['max_num_epochs']}",
                "--ema",
                "--ema_decay=0.99",
                f"--default_dtype={mace_cfg['default_dtype']}",
                f"--device={mace_cfg['device']}",
                f"--seed={seed}",
                f"--model_dir={out_dir}",
                f"--checkpoints_dir={out_dir / 'checkpoints'}",
                f"--results_dir={out_dir / 'results'}",
                f"--log_dir={out_dir / 'logs'}",
            ]
            if e0s is not None:
                cmd.insert(18, f"--E0s={e0s}")
            subprocess.run(cmd, cwd=layout.root, check=True)
    return out_dir
