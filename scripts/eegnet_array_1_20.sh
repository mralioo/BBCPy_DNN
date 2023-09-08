#!/bin/bash

#SBATCH --job-name=S1-S20-eegnet
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=../jobs_outputs/eegnet-S%a/%x_%j.o
#SBATCH --error=../jobs_outputs/eegnet-S%a/%x_%j.e
#SBATCH --mail-user=mr.ali.alouane@gmail.com
#SBATCH --array=1-20

echo "I am a job with ID $SLURM_JOB_ID for subject S$SLURM_ARRAY_TASK_ID"
echo "current working directory is $(pwd)"

SUBJECT="S$SLURM_ARRAY_TASK_ID"

# 1. copy the squashed dataset to the nodes /tmp
cp ./../squashfs_smr_data/${SUBJECT}.sqfs /tmp/

echo "Processing data for subject $SUBJECT"

# 3. bind the squashed dataset to your apptainer environment and run your script with apptainer
apptainer run -B /tmp/${SUBJECT}.sqfs:/input-data:image-src=/ ./../env_images/bbcpy_lightning_v5.sif python ./src/dnn_train.py experiment=eegnet +data.subject_sessions_dict="{$SUBJECT: "all"}" logger.mlflow.run_name="${SUBJECT}-all-RL"


