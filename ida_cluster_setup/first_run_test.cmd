#!/bin/bash
#SBATCH --job-name=riemann-s1-cpu-job
#SBATCH --partition=cpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=20
#SBATCH --output=logs/job-%j.out
#SBATCH --output=jobs_outputs/%x_%j.o
#SBATCH --error=jobs_outputs/%x_%j.e
#SBATCH --mail-user=ali.alouane@outlook.de

echo "I am a job task with ID $JOB_ID"
cd /home/bbci/data/teaching/ALI_MA/bbcpy_AutoML/
apptainer run -B ./S1.sqfs:/input-data:image-src=/ --nv ./bbcpy_lightning.sif python src/baseline_train.py experiment=baseline_riemann
