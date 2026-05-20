# MACE-AL

MACE-AL is a DP-GEN-style active-learning workflow for MACE foundation models.
It runs ASE/MACE exploration, selects uncertain configurations, labels them
with VASP, fine-tunes MACE committee models, and exports production models.

中文说明见 [docs/manual_zh.md](docs/manual_zh.md).

English manual: [docs/manual_en.md](docs/manual_en.md).

## Quick Start

```bash
git clone https://github.com/qhwu/mace-al.git
cd mace-al

# Optional demo data
bash scripts/run_mace_al.sh init-demo

# Full DP-GEN-style run
bash scripts/run_mace_al.sh run param.json machine.json

# Report
bash scripts/run_mace_al.sh run-report param.json machine.json -v
```

The main interface separates scientific parameters and machine parameters:

- `param.json`: MACE models, training settings, exploration, trust levels, selection, DFT settings.
- `machine.json`: module/conda environment, CUDA device, VASP command, POTCAR root, queue behavior.

Final usable models are exported to:

```text
final_models/
```

## Features

- Starts from one or more pretrained MACE foundation models.
- Committee uncertainty selection with lower/upper trust levels.
- ASE-based MD exploration.
- VASP labeling with automatic POTCAR assembly from a local pseudopotential tree.
- Restartable generation state machine using `.done` markers.
- Direct VASP execution or Slurm submission.
- DP-GEN-like `run PARAM MACHINE` and `run-report PARAM MACHINE` CLI.
- Automatic plots for selection maps, training metrics, and generation summaries.

## Package

```bash
pip install -e .
maceal run param.json machine.json
```

On this cluster, the wrapper loads the MACE conda environment:

```bash
bash scripts/run_mace_al.sh run param.json machine.json
```

Representative BaFeF4 demo:

```bash
bash scripts/run_mace_al.sh --root test_representative init-representative-demo
bash scripts/run_mace_al.sh run test_representative/param.json test_representative/machine.json
bash scripts/run_mace_al.sh run-report test_representative/param.json test_representative/machine.json -v
```
