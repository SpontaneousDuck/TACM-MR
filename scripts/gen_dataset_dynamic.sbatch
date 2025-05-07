#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --job-name=gen_tacm
#SBATCH --mem=100GB
#SBATCH --ntasks=1
#SBATCH --output=%j.out
#SBATCH --error=%j.out
#SBATCH --open-mode=append
#SBATCH --signal=SIGUSR1@90
#SBATCH --requeue
#SBATCH --partition=usace

eval "$(conda shell.bash hook)"
conda activate rf_amr

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# export PL_FAULT_TOLERANT_TRAINING=1

srun --unbuffered --export=ALL python3 gen_dataset_dynamic.py "$@"