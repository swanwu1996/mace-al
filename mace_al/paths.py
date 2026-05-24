from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Layout:
    root: Path
    work_path: Path
    generation: int

    @classmethod
    def from_config(cls, cfg: dict, root: str | Path | None = None, generation: int | None = None) -> "Layout":
        root_path = Path(root or ".").resolve()
        work = Path(cfg.get("work_path", "./cache"))
        if not work.is_absolute():
            work = root_path / work
        gen = int(cfg.get("generation", 0) if generation is None else generation)
        return cls(root=root_path, work_path=work, generation=gen)

    @property
    def gen_path(self) -> Path:
        return self.work_path / f"Generation-{self.generation}"

    def stage_path(self, name: str) -> Path:
        return self.gen_path / name

    def stage(self, name: str) -> Path:
        path = self.stage_path(name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def rel(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path


def format_generation(value: str, generation: int) -> str:
    return value.format(generation=generation, iter=generation)
