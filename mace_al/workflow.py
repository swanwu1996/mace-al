from __future__ import annotations

from pathlib import Path

from .dft import collect_vasp, prepare_vasp, submit_vasp, wait_vasp
from .explore import run_explore
from .mace import train_committee
from .paths import Layout
from .select import run_select


def run_generation(cfg: dict, root: str | Path, generation: int, submit: bool, wait: bool) -> None:
    layout = Layout.from_config(cfg, root=root, generation=generation)
    print(f"=== Generation-{generation}: explore ===", flush=True)
    run_explore(cfg, layout)
    print(f"=== Generation-{generation}: select ===", flush=True)
    run_select(cfg, layout)
    print(f"=== Generation-{generation}: prepare VASP ===", flush=True)
    prepare_vasp(cfg, layout)
    if submit:
        print(f"=== Generation-{generation}: submit VASP ===", flush=True)
        submit_vasp(cfg, layout)
    if wait:
        wait_vasp(cfg, layout)
        print(f"=== Generation-{generation}: collect labels ===", flush=True)
        collect_vasp(cfg, layout)
        next_layout = Layout.from_config(cfg, root=root, generation=generation + 1)
        print(f"=== Generation-{generation + 1}: train MACE committee ===", flush=True)
        train_committee(cfg, next_layout)


def run_loop(cfg: dict, root: str | Path, start: int, stop: int, submit: bool, wait: bool) -> None:
    for generation in range(start, stop):
        run_generation(cfg, root, generation, submit, wait)

