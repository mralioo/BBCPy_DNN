#!/bin/bash
#SBATCH --job-name=debug-hpo-eegnet
#SBATCH --partition=gpu-test
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=10G   # Request 100 GB of RAM
#SBATCH --output=../jobs_outputs/debug-hpo-eegnet/%x_%j.o
#SBATCH --error=../jobs_outputs/debug-hpo-eegnet/%x_%j.e
#SBATCH --array=0-2   # Creates 100 jobs. Adjust this number based on how many trials you want to run.

echo "I am a job with ID $SLURM_JOB_ID"
echo "current working directory is $(pwd)"

# 1. copy the squashed dataset to the nodes /tmp
cp ./../squashfs_smr_data/S5.sqfs /tmp/
#export CUDA_VISIBLE_DEVICES=0
# 3. bind the squashed dataset to your apptainer environment and run your script with apptainer
apptainer run --nv -B /tmp/S5.sqfs:/input-data:image-src=/ ./../env_images/bbcpy_lightning_v5.sif python ./src/dnn_hpo_train.py experiment=eegnet +data.subject_sessions_dict='{S5: "all"}' hparams_search=eegnet_optuna.yaml logger.mlflow.run_name="eegnet-hpo-best-pvc"