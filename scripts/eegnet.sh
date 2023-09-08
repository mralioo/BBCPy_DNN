#!/bin/bash
#SBATCH --job-name=eegnet-S1
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=16
#SBATCH --output=../jobs_outputs/eegnet-S1/%x_%j.o
#SBATCH --error=../jobs_outputs/eegnet-S1/%x_%j.e
#SBATCH --mail-user=mr.ali.alouane@gmail.com

echo "I am a job with ID $SLURM_JOB_ID"
echo "current working directory is $(pwd)"

# 1. copy the squashed dataset to the nodes /tmp
cp ./../squashfs_smr_data/S1.sqfs /tmp/

# 3. bind the squashed dataset to your apptainer environment and run your script with apptainer
apptainer run -B /tmp/S1.sqfs:/input-data:image-src=/ ./../env_images/bbcpy_lightning_v5.sif python ./src/dnn_train.py +experiment=eegnet.yaml +data.subject_sessions_dict="{"S1": "all"}" logger.mlflow.experiment_name="eegnet-S1-all"
