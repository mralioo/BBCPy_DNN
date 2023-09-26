#!/bin/bash
#SBATCH --job-name=riemann-tangent-S1
#SBATCH --partition=cpu-2h
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=8
#SBATCH --output=../jobs_outputs/riemann-tangent-S1/%x_%j.o
#SBATCH --error=../jobs_outputs/riemann-tangent-S1/%x_%j.e
#SBATCH --mail-user=mr.ali.alouane@gmail.com

echo "I am a job with ID $SLURM_JOB_ID"
echo "current working directory is $(pwd)"

# 1. copy the squashed dataset to the nodes /tmp
# 1. copy the squashed dataset to the nodes /tmp
cp ./../squashfs_smr_data/S1.sqfs /tmp/

# 3. bind the squashed dataset to your apptainer environment and run your script with apptainer
apptainer run -B /tmp/S1.sqfs:/input-data:image-src=/ ./../env_images/bbcpy_lightning_v3.sif python ./src/baseline_train.py +experiment=0_riemann_tangent
