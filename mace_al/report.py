from __future__ import annotations

from pathlib import Path

from .io import read_atoms
from .paths import Layout
from .plotting import plot_all


def report(cfg: dict, root: str | Path, verbose: bool = False) -> None:
    root = Path(root).resolve()
    work = root / cfg.get("work_path", "./cache")
    if not work.exists():
        print(f"No cache directory: {work}")
        return
    print("generation,candidates,selected,dft_jobs,dft_labeled,dft_failed")
    for gen_path in sorted(work.glob("Generation-*")):
        try:
            generation = int(gen_path.name.split("-", 1)[1])
        except Exception:
            continue
        layout = Layout.from_config(cfg, root=root, generation=generation)
        candidates = len(read_atoms(layout.stage("explore") / "candidates.xyz"))
        selected = len(read_atoms(layout.stage("select") / "selected.xyz"))
        dft_dir = layout.stage("dft")
        jobs = sorted(p for p in dft_dir.glob("cand_*") if p.is_dir())
        labeled = len(read_atoms(dft_dir / "learn_calculated.xyz"))
        failed = len(jobs) - labeled
        print(f"{generation},{candidates},{selected},{len(jobs)},{labeled},{failed}")
        if verbose:
            for job in jobs:
                status = "done" if (job / str(cfg["dft"]["done_file"])).exists() else "pending"
                print(f"  {job.relative_to(root)} {status}")
    outputs = plot_all(cfg, root)
    if outputs:
        print("plots")
        for out in outputs:
            print(f"  {out.relative_to(root)}")
