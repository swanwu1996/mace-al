# MACE-AL User Manual

## 1. Overview

MACE-AL is an automatic active-learning workflow for fine-tuning MACE foundation models.
It follows the same high-level idea as DP-GEN:

```text
initial committee
-> model-driven exploration
-> uncertainty-based selection
-> DFT labeling
-> dataset update
-> committee fine-tuning
-> repeat
-> final model export
```

The production command is:

```bash
maceal run param.json machine.json
```

or, on this cluster:

```bash
bash scripts/run_mace_al.sh run param.json machine.json
```

## 2. Files

```text
param.json       Scientific/workflow parameters
machine.json     Machine, queue, module, and VASP parameters
mace_al/         Python package
scripts/         Cluster entrypoints
templates/vasp/  VASP input templates
docs/            Manuals
```

Runtime outputs are written to:

```text
cache/Generation-N/
final_models/
```

## 3. `param.json`

`param.json` controls the scientific workflow.

Important sections:

- `project`: input structures, train/test files, and MACE foundation models.
- `mace`: fine-tuning settings.
- `explore`: ASE-MD and uncertainty thresholds.
- `select`: maximum number of selected structures.
- `dft`: VASP template path and POTCAR element mapping.
- `run`: number of generations and automation behavior.

Example:

```json
{
  "project": {
    "seed_file": "./data/seeds.extxyz",
    "train_file": "./data/train.extxyz",
    "foundation_models": [
      "/home/qhwu/mace_mod/mace-mpa-0-medium.model",
      "/home/qhwu/mace_mod/mace-omat-0-medium.model"
    ]
  },
  "run": {
    "max_generations": 3,
    "submit_dft": true,
    "wait_dft": true,
    "train_next_generation": true
  }
}
```

## 4. `machine.json`

`machine.json` contains machine-specific settings. Keep it separate from `param.json`
so the same scientific workflow can run on different clusters.

Example:

```json
{
  "mace": {
    "env_script": "module load anaconda && conda activate mace",
    "device": "cuda"
  },
  "explore": {
    "device": "cuda"
  },
  "dft": {
    "potcar_root": "/home/qhwu/PBE64",
    "use_genque": true,
    "genque_choice": "2",
    "submit_command": "sbatch 2-vasp_gpu.sh",
    "done_file": "vasprun.xml"
  }
}
```

For a tiny local smoke test, direct VASP execution is supported:

```json
{
  "dft": {
    "direct_command": "module purge && module load vasp && vasp_std > vasp.out 2>&1"
  },
  "run": {
    "run_dft_direct": true,
    "submit_dft": false
  }
}
```

## 5. Data Format

Training data should be ASE `extxyz` with:

- `REF_energy` in `atoms.info`
- `REF_forces` in `atoms.arrays`
- optional `REF_stress` in `atoms.info`

Seed structures do not need labels.

## 6. Running

Initialize demo data:

```bash
bash scripts/run_mace_al.sh init-demo
```

Run active learning:

```bash
bash scripts/run_mace_al.sh run param.json machine.json
```

Report status:

```bash
bash scripts/run_mace_al.sh run-report param.json machine.json -v
```

Export final models again:

```bash
bash scripts/run_mace_al.sh export param.json machine.json
```

## 7. Restart Behavior

Every completed stage writes a marker:

```text
cache/Generation-N/.explore.done
cache/Generation-N/.select.done
cache/Generation-N/.prepare_dft.done
cache/Generation-N/.collect_dft.done
```

If a run stops after VASP submission, rerun the same command after jobs finish:

```bash
bash scripts/run_mace_al.sh run param.json machine.json
```

The workflow resumes from the first unfinished stage.

## 8. Output Models

At the end of a completed run, models are exported to:

```text
final_models/
```

`manifest.json` records the final generation and exported files.

## 9. Plots

MACE-AL automatically generates diagnostic plots:

- `cache/Generation-N/select/selection_map.png`: committee uncertainty map; selected structures are highlighted.
- `cache/Generation-N/mace/committee/training_metrics.png`: validation energy/force metrics from MACE training logs.
- `cache/plots/generation_summary.png`: candidates, selected structures, and DFT labels by generation.

You can regenerate plots at any time:

```bash
bash scripts/run_mace_al.sh run-report param.json machine.json -v
```

## 10. Smoke and Representative Tests

Small development tests are included:

```bash
bash scripts/run_mace_al.sh run test_smoke/param.json test_smoke/machine.json
bash scripts/run_mace_al.sh run test_vasp/param.json test_vasp/machine.json
bash scripts/run_mace_al.sh run test_full/param.json test_full/machine.json
```

`test_full` performs a complete mini-loop: exploration, VASP labeling, MACE fine-tuning,
and final model export.

`test_representative` is a more meaningful BaFeF4 active-learning demo. It uses multiple
seed structures and temperatures so the selection plot is informative:

```bash
bash scripts/run_mace_al.sh --root test_representative init-representative-demo
bash scripts/run_mace_al.sh run test_representative/param.json test_representative/machine.json
```

By default it stops after VASP input preparation to avoid expensive DFT jobs. Set
`submit_dft=true` or `run_dft_direct=true` if you want to label those structures.

`test_representative_full` is the full local BaFeF4 loop used to validate the workflow:

```bash
bash scripts/run_mace_al.sh --root test_representative_full init-representative-demo --train-count 20 --test-count 4 --seed-count 2
bash scripts/run_mace_al.sh run test_representative_full/param.json test_representative_full/machine.json
bash scripts/run_mace_al.sh run-report test_representative_full/param.json test_representative_full/machine.json -v
```

It performs MACE/CUDA exploration, direct VASP labeling, label collection, next-generation
MACE fine-tuning, plot generation, and final model export.
