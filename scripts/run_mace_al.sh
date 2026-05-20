#!/bin/bash
set -euo pipefail

module load anaconda
conda activate mace

cd "$(dirname "$0")/.."
python -m mace_al.cli "$@"
