from __future__ import annotations

import json
import re
from pathlib import Path


GENERATION_RE = re.compile(r"\bal_generation=([0-9]+)\b")


def _iter_extxyz_blocks(path: Path):
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        try:
            atom_count = int(lines[i].strip())
        except ValueError as exc:
            raise ValueError(f"Bad extxyz atom-count line {i + 1} in {path}: {lines[i]!r}") from exc
        block = lines[i : i + atom_count + 2]
        if len(block) != atom_count + 2:
            raise ValueError(f"Truncated extxyz frame at line {i + 1} in {path}")
        yield block
        i += atom_count + 2


def rebuild_generation_traces(cfg: dict, root: str | Path) -> list[dict]:
    """Rebuild per-generation labeled data traces from the cumulative train file.

    This intentionally preserves only labeled structures. Raw VASP directories,
    exploration trajectories, and model checkpoints are provenance data and must
    be kept separately if needed.
    """

    root = Path(root).resolve()
    train_file = Path(cfg["project"]["train_file"])
    if not train_file.is_absolute():
        train_file = root / train_file
    work = Path(cfg.get("work_path", "./cache"))
    if not work.is_absolute():
        work = root / work
    if not train_file.exists() or train_file.stat().st_size == 0:
        return []

    by_generation: dict[int, list[list[str]]] = {}
    for block in _iter_extxyz_blocks(train_file):
        match = GENERATION_RE.search(block[1])
        if match is None:
            continue
        generation = int(match.group(1))
        by_generation.setdefault(generation, []).append(block)

    summary = []
    for generation in sorted(by_generation):
        gen_dir = work / f"Generation-{generation}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        trace_file = gen_dir / "recovered_labels.extxyz"
        with trace_file.open("w", encoding="utf-8") as handle:
            for block in by_generation[generation]:
                handle.write("\n".join(block))
                handle.write("\n")
        readme = gen_dir / "README_recovered.txt"
        if not readme.exists():
            readme.write_text(
                "This directory contains labeled structures reconstructed from the cumulative training file.\n"
                "It is a lightweight provenance trace, not a replacement for raw VASP outputs, exploration "
                "trajectories, or training checkpoints.\n",
                encoding="utf-8",
            )
        summary.append(
            {
                "generation": generation,
                "recovered_labeled_structures": len(by_generation[generation]),
                "file": str(trace_file.relative_to(root)),
            }
        )

    work.mkdir(parents=True, exist_ok=True)
    (work / "recovered_generation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (work / "recovered_generation_summary.tsv").write_text(
        "generation\trecovered_labeled_structures\tfile\n"
        + "".join(
            f"{row['generation']}\t{row['recovered_labeled_structures']}\t{row['file']}\n"
            for row in summary
        ),
        encoding="utf-8",
    )
    return summary
