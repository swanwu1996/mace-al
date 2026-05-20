# MACE-AL

MACE-AL is a DP-GEN-style active-learning workflow for MACE foundation models.
It runs ASE/MACE exploration, selects uncertain configurations, labels them
with VASP, fine-tunes MACE committee models, and exports production models.

中文说明见 [docs/manual_zh.md](docs/manual_zh.md).

English manual: [docs/manual_en.md](docs/manual_en.md).

## Quick Start

```bash
git clone https://github.com/swanwu1996/mace-al.git
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

Representative BaFeF4 full loop with direct VASP and next-generation model export:

```bash
bash scripts/run_mace_al.sh --root test_representative_full init-representative-demo --train-count 20 --test-count 4 --seed-count 2
bash scripts/run_mace_al.sh run test_representative_full/param.json test_representative_full/machine.json
bash scripts/run_mace_al.sh run-report test_representative_full/param.json test_representative_full/machine.json -v
```

MACE-AL checks VASP electronic convergence before collecting labels. Jobs with
messages such as `EDIFF was not reached` or `electronic self-consistency was not
achieved` are written to `failed_jobs.txt` and skipped.

Production HfO2 validation demo:

```bash
bash scripts/run_mace_al.sh run test_hfo2_production/param.json test_hfo2_production/machine.json
bash scripts/run_mace_al.sh run-report test_hfo2_production/param.json test_hfo2_production/machine.json -v
```

The HfO2 demo uses a 12-atom Pca21 seed, two pretrained MACE foundation models,
16 explored candidates, 8 selected VASP labels, and strict VASP settings
(`ENCUT=600`, `EDIFF=1E-6`, `NELM=240`, `ALGO=All`, `2x2x2` k-mesh). The
validated local run produced 8 converged labels, 0 failed DFT jobs, diagnostic
plots, and two Generation-1 exported models.
