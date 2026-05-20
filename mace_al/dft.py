from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from ase.io import read, write

from .io import append_atoms, read_atoms, write_atoms
from .paths import Layout


def species_order(atoms) -> list[str]:
    seen = []
    for symbol in atoms.get_chemical_symbols():
        if symbol not in seen:
            seen.append(symbol)
    return seen


def potcar_dir_for_symbol(symbol: str, dft_cfg: dict) -> str:
    return dft_cfg.get("potcar_map", {}).get(symbol, symbol)


def write_potcar(job_dir: Path, atoms, dft_cfg: dict) -> None:
    root = dft_cfg.get("potcar_root")
    if not root:
        return
    root_path = Path(root)
    out = job_dir / "POTCAR"
    spec_lines = []
    with out.open("wb") as fout:
        for symbol in species_order(atoms):
            potcar_name = potcar_dir_for_symbol(symbol, dft_cfg)
            potcar = root_path / potcar_name / "POTCAR"
            if not potcar.exists():
                raise FileNotFoundError(f"Missing POTCAR for {symbol}: {potcar}")
            spec_lines.append(f"{symbol}: {potcar}\n")
            with potcar.open("rb") as fin:
                shutil.copyfileobj(fin, fout)
    (job_dir / "POTCAR.spec").write_text("".join(spec_lines), encoding="utf-8")


def prepare_vasp(cfg: dict, layout: Layout) -> Path:
    selected = read_atoms(layout.stage("select") / "selected.xyz")
    if not selected:
        raise FileNotFoundError(f"No selected structures in {layout.stage('select') / 'selected.xyz'}")
    dft_cfg = cfg["dft"]
    template_dir = layout.rel(dft_cfg["template_dir"])
    if not template_dir.exists():
        raise FileNotFoundError(f"Missing VASP template dir: {template_dir}")

    out_root = layout.stage("dft")
    for i, atoms in enumerate(selected):
        job_dir = out_root / f"cand_{i:05d}"
        job_dir.mkdir(parents=True, exist_ok=True)
        for src in template_dir.iterdir():
            if src.is_file() and src.name != "POSCAR":
                shutil.copy2(src, job_dir / src.name)
        write(str(job_dir / "POSCAR"), atoms, format="vasp", direct=True, vasp5=True)
        write_potcar(job_dir, atoms, dft_cfg)
        if dft_cfg.get("use_genque", False):
            subprocess.run(
                ["genque"],
                cwd=job_dir,
                input=f"{dft_cfg.get('genque_choice', '2')}\n",
                text=True,
                check=True,
            )
    write_atoms(out_root / "learn_add.xyz", selected)
    return out_root


def submit_vasp(cfg: dict, layout: Layout) -> None:
    dft_cfg = cfg["dft"]
    command = str(dft_cfg["submit_command"]).split()
    for job_dir in sorted(p for p in layout.stage("dft").glob("cand_*") if p.is_dir()):
        marker = job_dir / ".submitted"
        if marker.exists():
            continue
        subprocess.run(command, cwd=job_dir, check=True)
        marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")


def run_vasp_direct(cfg: dict, layout: Layout) -> None:
    dft_cfg = cfg["dft"]
    command = dft_cfg.get("direct_command")
    if not command:
        raise ValueError("dft.direct_command is required for direct VASP execution")
    for job_dir in sorted(p for p in layout.stage("dft").glob("cand_*") if p.is_dir()):
        marker = job_dir / ".direct_done"
        if marker.exists():
            continue
        subprocess.run(command, cwd=job_dir, shell=True, executable="/bin/bash", check=True)
        marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")


def wait_vasp(cfg: dict, layout: Layout) -> None:
    dft_cfg = cfg["dft"]
    deadline = time.time() + int(dft_cfg["max_wait_seconds"])
    done_file = str(dft_cfg["done_file"])
    while time.time() < deadline:
        jobs = sorted(p for p in layout.stage("dft").glob("cand_*") if p.is_dir())
        pending = [p for p in jobs if not (p / done_file).exists()]
        if not pending:
            return
        print(f"Waiting for {len(pending)} VASP jobs...", flush=True)
        time.sleep(int(dft_cfg["poll_seconds"]))
    raise TimeoutError("VASP jobs did not finish before timeout")


def collect_vasp(cfg: dict, layout: Layout) -> Path:
    labeled = []
    force_limit = cfg["dft"].get("force_limit")
    removed = []
    for job_dir in sorted(p for p in layout.stage("dft").glob("cand_*") if p.is_dir()):
        try:
            if (job_dir / "vasprun.xml").exists():
                atoms = read(str(job_dir / "vasprun.xml"), index=-1)
            elif (job_dir / "OUTCAR").exists():
                atoms = read(str(job_dir / "OUTCAR"), index=-1)
            else:
                continue
            atoms.info["REF_energy"] = float(atoms.get_potential_energy())
            atoms.arrays["REF_forces"] = np.asarray(atoms.get_forces(), dtype=float)
            try:
                atoms.info["REF_stress"] = atoms.get_stress(voigt=True)
            except Exception:
                pass
            atoms.info["al_generation"] = layout.generation
            atoms.info["al_dft_dir"] = str(job_dir.relative_to(layout.root))
            if force_limit is not None and np.abs(atoms.arrays["REF_forces"]).max() > float(force_limit):
                removed.append(atoms)
            else:
                labeled.append(atoms)
        except Exception as exc:
            print(f"Skip failed VASP parse {job_dir}: {exc}", flush=True)
    out = layout.stage("dft") / "learn_calculated.xyz"
    write_atoms(out, labeled)
    if removed:
        write_atoms(layout.stage("dft") / "removed_by_force.xyz", removed)
    append_atoms(layout.rel(cfg["project"]["train_file"]), labeled)
    return out
