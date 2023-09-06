#!/bin/bash
#SBATCH --job-name=eegnet-S1
#SBATCH --partition=cpu-2h
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=8
#SBATCH --output=../jobs_outputs/eegnet-S1/%x_%j.o
#SBATCH --error=../jobs_outputs/eegnet-S1/%x_%j.e

echo "I am a job with ID $SLURM_JOB_ID"
echo "current working directory is $(pwd)"

# 1. copy the squashed dataset to the nodes /tmp
cp ./../squashfs_smr_data/S1.sqfs /tmp/

# 3. bind the squashed dataset to your apptainer environment and run your script with apptainer
apptainer run -B /tmp/S1.sqfs:/input-data:image-src=/ ./../env_images/bbcpy_lightning_v4.sif python ./src/dnn_train.py +experiment=eegnet.yaml
