#!/bin/bash

#SBATCH --job-name=eegnet-S1-10
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=../jobs_outputs/eegnet-S1-10/%x_%j.o
#SBATCH --error=../jobs_outputs/eegnet-S1-10/%x_%j.e

echo "I am a job with ID $SLURM_JOB_ID"
echo "current working directory is $(pwd)"

# Define data you want to run
SUBJECT=("S1" "S2" "S3" "S4" "S5" "S6" "S7" "S8" "S9" "S10")

# Loop over the data
for S in "${SUBJECT[@]}"; do

    echo "Processing data $S"
    # 1. copy the squashed dataset to the nodes /tmp
    cp ./../squashfs_smr_data/${S}.sqfs /tmp/
    # 3. bind the squashed dataset to your apptainer environment and run your script with apptainer
    apptainer run -B /tmp/${S}.sqfs:/input-data:image-src=/ ./../env_images/bbcpy_lightning_v5.sif python ./src/baseline_train.py +experiment=eegnet +data.subject_sessions_dict="{$S: "all"}" logger.mlflow.experiment_name="${S}-all-RL"

done
