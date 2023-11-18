#!/bin/bash
#SBATCH --job-name=hpo-LR-Transformer
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --mem=400G        # Some buffer above 30GB
#SBATCH --ntasks-per-node=1   # One main task that runs the trial and manages CV
#SBATCH --cpus-per-task=6 # Assuming you want to run each CV fold in parallel
#SBATCH --output=../jobs_outputs/Transformer-hpo-LR/%x_%j.o
#SBATCH --error=../jobs_outputs/Transformer-hpo-LR/%x_%j.e

echo "I am a job with ID $SLURM_JOB_ID"
echo "current working directory is $(pwd)"

# 1. copy the squashed dataset to the nodes /tmp
cp ./../squashfs_smr_data/S5.sqfs /tmp/

# 3. bind the squashed dataset to your apptainer environment and run your script with apptainer
apptainer run --nv -B /tmp/S5.sqfs:/input-data:image-src=/ ./../env_images/bbcpy_env2.sif python ./src/dnn_train.py experiment=3_transformer_LR +data.subject_sessions_dict='{S5: "all"}' hparams_search=3_transformer_optuna logger.mlflow.experiment_name="C1-LR" logger.mlflow.run_name="LR-C1-S5"
