#!/bin/bash
#SBATCH --job-name=riemann-s1-cpu-job
#SBATCH --partition=cpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=20
#SBATCH --output=logs/job-%j.out
#SBATCH --output=jobs_outputs/%x_%j.o
#SBATCH --error=jobs_outputs/%x_%j.e
#SBATCH --mail-user=ali.alouane@outlook.de

echo "I am a job with ID $SLURM_JOB_ID"

# 1. copy the squashed dataset to the nodes /tmp
cp /home/ali_alouane/MA_BCI/squashfs_smr_data/squashfs_example.sqfs /tmp/
# 2. go to the project directory
cd /home/ali_alouane/MA_BCI/
# 3. bind the squashed dataset to your apptainer environment and run your script with apptainer
apptainer run -B /tmp/S1.sqfs:/input-data:image-src=/ ./env_images/bbcpy_lightning.sif python bbcpy_autoML/src/baseline_train.py experiment=baseline_riemann
