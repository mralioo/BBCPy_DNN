#!/bin/bash
#SBATCH --job-name=cpu_job
#SBATCH --partition=cpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=20
#SBATCH --output=logs/job-%j.out



apptainer run --nv ./hydra_ida/bbcpy_lightning.sif python src/baseline_train.py experiment=baseline_riemann
