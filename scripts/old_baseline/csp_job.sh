#!/bin/bash
#SBATCH --job-name=csp-S9
#SBATCH --partition=cpu-2h
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=8
#SBATCH --output=../jobs_outputs/csp-S9/%x_%j.o
#SBATCH --mail-user=mr.ali.alouane@gmail.com

echo "I am a job with ID $SLURM_JOB_ID"
echo "current working directory is $(pwd)"

# 1. copy the squashed dataset to the nodes /tmp
cp ./../squashfs_smr_data/S9_1234567891011.sqfs /tmp/

# 3. bind the squashed dataset to your apptainer environment and run your script with apptainer
apptainer run -B /tmp/S9_1234567891011.sqfs:/input-data:image-src=/ ./../env_images/bbcpy_lightning_v3.sif python ./src/baseline_train.py +experiment=0_csp
