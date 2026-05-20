from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from .dft import collect_vasp, prepare_vasp, run_vasp_direct, submit_vasp, wait_vasp
from .explore import run_explore
from .io import read_atoms
from .mace import init_committee_from_foundations, train_committee
from .paths import Layout
from .plotting import plot_generation_summary, plot_selection, plot_training
from .select import run_select


STAGES = ["committee", "explore", "select", "prepare_dft", "submit_dft", "collect_dft", "train_next"]


def marker(layout: Layout, stage: str) -> Path:
    return layout.gen_path / f".{stage}.done"


def is_done(layout: Layout, stage: str) -> bool:
    return marker(layout, stage).exists()


def mark_done(layout: Layout, stage: str, payload: dict | None = None) -> None:
    layout.gen_path.mkdir(parents=True, exist_ok=True)
    data = {"stage": stage, "generation": layout.generation, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    if payload:
        data.update(payload)
    marker(layout, stage).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def stage_log(layout: Layout, message: str) -> None:
    print(f"[Generation-{layout.generation}] {message}", flush=True)


def has_committee(layout: Layout) -> bool:
    return any((layout.stage("mace") / "committee").glob("*.model"))


def run_automatic(cfg: dict, root: str | Path, start: int = 0, stop: int | None = None) -> None:
    run_cfg = cfg.get("run", {})
    max_generations = int(stop if stop is not None else run_cfg.get("max_generations", 1))
    root = Path(root).resolve()
    completed = True

    for generation in range(start, max_generations):
        layout = Layout.from_config(cfg, root=root, generation=generation)

        if generation == 0 and run_cfg.get("auto_init_committee", True) and not has_committee(layout):
            stage_log(layout, "initializing committee from foundation models")
            init_committee_from_foundations(cfg, layout)
            mark_done(layout, "committee", {"mode": "foundation"})

        if not has_committee(layout):
            stage_log(layout, "training committee")
            train_committee(cfg, layout)
            mark_done(layout, "committee", {"mode": "finetune"})

        if not is_done(layout, "explore"):
            stage_log(layout, "running ASE/MACE exploration")
            candidates = run_explore(cfg, layout)
            mark_done(layout, "explore", {"candidates": len(read_atoms(candidates))})

        if not is_done(layout, "select"):
            stage_log(layout, "selecting DFT candidates")
            selected = run_select(cfg, layout)
            plot_selection(layout)
            mark_done(layout, "select", {"selected": len(read_atoms(selected))})

        selected_count = len(read_atoms(layout.stage("select") / "selected.xyz"))
        if selected_count == 0:
            stage_log(layout, "no selected structures; stopping active learning")
            break

        if not is_done(layout, "prepare_dft"):
            stage_log(layout, "preparing VASP jobs")
            prepare_vasp(cfg, layout)
            mark_done(layout, "prepare_dft", {"jobs": selected_count})

        if cfg.get("run", {}).get("run_dft_direct", False) and not is_done(layout, "run_dft_direct"):
            stage_log(layout, "running VASP directly")
            run_vasp_direct(cfg, layout)
            mark_done(layout, "run_dft_direct")

        if cfg.get("run", {}).get("submit_dft", True) and not is_done(layout, "submit_dft"):
            stage_log(layout, "submitting VASP jobs")
            submit_vasp(cfg, layout)
            mark_done(layout, "submit_dft")

        if cfg.get("run", {}).get("wait_dft", True) or cfg.get("run", {}).get("run_dft_direct", False):
            stage_log(layout, "waiting for VASP jobs")
            wait_vasp(cfg, layout)
            if not is_done(layout, "collect_dft"):
                stage_log(layout, "collecting VASP labels")
                labeled = collect_vasp(cfg, layout)
                mark_done(layout, "collect_dft", {"labeled": len(read_atoms(labeled))})

            labeled_count = len(read_atoms(layout.stage("dft") / "learn_calculated.xyz"))
            if labeled_count == 0:
                stage_log(layout, "no labeled structures; stopping active learning")
                break

            if cfg.get("run", {}).get("train_next_generation", True) and generation + 1 < max_generations:
                next_layout = Layout.from_config(cfg, root=root, generation=generation + 1)
                if not is_done(layout, "train_next") and not has_committee(next_layout):
                    stage_log(layout, f"training Generation-{generation + 1} committee")
                    train_committee(cfg, next_layout)
                    plot_training(next_layout)
                    mark_done(layout, "train_next", {"next_generation": generation + 1})
                if cfg.get("run", {}).get("stop_after_train_next", False):
                    break
        else:
            completed = False
            if cfg.get("run", {}).get("submit_dft", True):
                stage_log(layout, "submitted DFT jobs; rerun after they finish to continue")
            else:
                stage_log(layout, "DFT submission disabled; stopping after input preparation")
            break

    if completed:
        plot_generation_summary(cfg, root)
        export_final_models(cfg, root)


def export_final_models(cfg: dict, root: str | Path) -> Path:
    root = Path(root).resolve()
    work = root / cfg.get("work_path", "./cache")
    generations = []
    for path in work.glob("Generation-*"):
        try:
            generations.append(int(path.name.split("-", 1)[1]))
        except Exception:
            continue
    if not generations:
        return root / "final_models"

    final_generation = max(generations)
    committee = Layout.from_config(cfg, root=root, generation=final_generation).stage("mace") / "committee"
    all_models = sorted(committee.glob("*.model"))
    preferred = [model for model in all_models if not model.name.endswith("_compiled.model")]
    compiled = [model for model in all_models if model.name.endswith("_compiled.model")]
    models = preferred or compiled
    out = root / "final_models"
    out.mkdir(parents=True, exist_ok=True)
    count = int(cfg.get("run", {}).get("final_model_count", len(models)))
    exported = []
    for model in models[:count]:
        dst = out / model.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if model.is_symlink():
            dst.symlink_to(model.resolve())
        else:
            shutil.copy2(model, dst)
        exported.append(str(dst.relative_to(root)))

    manifest = {
        "final_generation": final_generation,
        "exported_models": exported,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(exported)} final model(s) to {out}", flush=True)
    return out
