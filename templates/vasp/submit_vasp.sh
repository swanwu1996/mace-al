#!/bin/bash
#SBATCH -J mace_al_vasp
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

module purge
module load vasp-gpu

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}

mpirun -np "${SLURM_NTASKS:-1}" vasp_std > vasp.out 2>&1
