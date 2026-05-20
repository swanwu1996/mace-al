from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .io import read_atoms
from .paths import Layout


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_selection(layout: Layout) -> Path | None:
    explore_dir = layout.stage("explore")
    csv_path = explore_dir / "uncertainty.csv"
    if not csv_path.exists():
        return None

    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        return None

    force = np.array([float(r["max_force_std"]) for r in rows])
    energy = np.array([float(r["energy_std_per_atom"]) for r in rows])
    selected = np.array([int(r["selected"]) for r in rows], dtype=bool)
    temp = np.array([float(r["temp_K"]) for r in rows])

    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=160)
    sc = ax.scatter(force[~selected], energy[~selected], c=temp[~selected], s=26, alpha=0.65, cmap="viridis", label="sampled")
    if selected.any():
        ax.scatter(force[selected], energy[selected], facecolors="none", edgecolors="crimson", s=90, linewidths=1.8, label="selected")
    ax.set_xlabel("max force committee std (eV/A)")
    ax.set_ylabel("energy committee std / atom (eV)")
    ax.set_title(f"Generation {layout.generation} selection map")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("temperature (K)")
    out = layout.stage("select") / "selection_map.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def _read_metric_file(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def plot_training(layout: Layout) -> list[Path]:
    results_dir = layout.stage("mace") / "committee" / "results"
    files = sorted(results_dir.glob("*_train.txt"))
    if not files:
        return []

    plt = _plt()
    outputs = []
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), dpi=160)
    for path in files:
        rows = _read_metric_file(path)
        eval_rows = [r for r in rows if r.get("mode") == "eval" and r.get("epoch") is not None]
        opt_rows = [r for r in rows if r.get("mode") == "opt" and r.get("epoch") is not None]
        label = path.name.replace("_train.txt", "")
        if eval_rows:
            x = [r["epoch"] for r in eval_rows]
            e = [1000.0 * float(r.get("rmse_e_per_atom", np.nan)) for r in eval_rows]
            f = [1000.0 * float(r.get("rmse_f", np.nan)) for r in eval_rows]
            axes[0].plot(x, e, marker="o", label=label)
            axes[1].plot(x, f, marker="o", label=label)
        elif opt_rows:
            x = [r["epoch"] for r in opt_rows]
            loss = [float(r.get("loss", np.nan)) for r in opt_rows]
            axes[0].plot(x, loss, marker="o", label=label)

    axes[0].set_title(f"Generation {layout.generation} energy validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("RMSE energy (meV/atom)")
    axes[0].grid(True, alpha=0.25)
    axes[1].set_title(f"Generation {layout.generation} force validation")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("RMSE force (meV/A)")
    axes[1].grid(True, alpha=0.25)
    for ax in axes:
        ax.legend(fontsize=7, loc="best")
    out = layout.stage("mace") / "committee" / "training_metrics.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    outputs.append(out)
    return outputs


def plot_generation_summary(cfg: dict, root: str | Path) -> Path | None:
    root = Path(root).resolve()
    work = root / cfg.get("work_path", "./cache")
    if not work.exists():
        return None

    generations = []
    candidates = []
    selected = []
    labeled = []
    for gen_path in sorted(work.glob("Generation-*")):
        try:
            gen = int(gen_path.name.split("-", 1)[1])
        except Exception:
            continue
        layout = Layout.from_config(cfg, root=root, generation=gen)
        generations.append(gen)
        candidates.append(len(read_atoms(layout.stage("explore") / "candidates.xyz")))
        selected.append(len(read_atoms(layout.stage("select") / "selected.xyz")))
        labeled.append(len(read_atoms(layout.stage("dft") / "learn_calculated.xyz")))

    if not generations:
        return None

    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=160)
    x = np.arange(len(generations))
    width = 0.25
    ax.bar(x - width, candidates, width, label="candidates")
    ax.bar(x, selected, width, label="selected")
    ax.bar(x + width, labeled, width, label="DFT labeled")
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in generations])
    ax.set_xlabel("generation")
    ax.set_ylabel("structures")
    ax.set_title("Active-learning generation summary")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    out_dir = work / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "generation_summary.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_all(cfg: dict, root: str | Path) -> list[Path]:
    root = Path(root).resolve()
    work = root / cfg.get("work_path", "./cache")
    outputs: list[Path] = []
    for gen_path in sorted(work.glob("Generation-*")) if work.exists() else []:
        try:
            gen = int(gen_path.name.split("-", 1)[1])
        except Exception:
            continue
        layout = Layout.from_config(cfg, root=root, generation=gen)
        sel = plot_selection(layout)
        if sel:
            outputs.append(sel)
        outputs.extend(plot_training(layout))
    summary = plot_generation_summary(cfg, root)
    if summary:
        outputs.append(summary)
    return outputs

