#!/bin/bash
#SBATCH --job-name=my_gpu_job
#SBATCH --partition=gpu-test
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%j.out
#SBATCH --output=../jobs_outputs/%x_%j.o
#SBATCH --error=../jobs_outputs/%x_%j.e

echo "I am a job with ID $SLURM_JOB_ID"
echo "current working directory is $(pwd)"

# 1. copy the squashed dataset to the nodes /tmp
cp ./../squashfs_smr_data/Subj_1_test.sqfs /tmp/

# 3. bind the squashed dataset to your apptainer environment and run your script with apptainer
apptainer run -B /tmp/Subj_1_test.sqfs:/input-data:image-src=/ ./../env_images/bbcpy_lightning.sif python ./src/baseline_train.py +de>
