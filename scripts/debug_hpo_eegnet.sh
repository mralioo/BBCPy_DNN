#!/bin/bash
#SBATCH --job-name=debug-hpo-eegnet
#SBATCH --partition=gpu-5h
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G        # Some buffer above 30GB
#SBATCH --ntasks-per-node=1   # One main task that runs the trial and manages CV
#SBATCH --cpus-per-task=5     # Assuming you want to run each CV fold in parallel
#SBATCH --output=../jobs_outputs/debug-hpo-eegnet/%x_%j.o
#SBATCH --error=../jobs_outputs/debug-hpo-eegnet/%x_%j.e

echo "I am a job with ID $SLURM_JOB_ID"
echo "current working directory is $(pwd)"

# 1. copy the squashed dataset to the nodes /tmp
cp ./../squashfs_smr_data/hpo_best_pvc_debug.sqfs /tmp/
#export CUDA_VISIBLE_DEVICES=0
# 3. bind the squashed dataset to your apptainer environment and run your script with apptainer
apptainer run --nv -B /tmp/hpo_best_pvc_debug.sqfs:/input-data:image-src=/ ./../env_images/bbcpy_lightning_v5.sif python ./src/dnn_hpo_train.py experiment=eegnet +data.subject_sessions_dict='{S5: "all", S9:"all"}' hparams_search=eegnet_optuna.yaml logger.mlflow.run_name="eegnet-hpo-best-pvc"
