#!/bin/bash
#SBATCH --job-name=debug-tsception--hpo-LR
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --mem=200G        # Some buffer above 30GB
#SBATCH --ntasks-per-node=1   # One main task that runs the trial and manages CV
#SBATCH --cpus-per-task=3 # Assuming you want to run each CV fold in parallel
#SBATCH --output=../jobs_outputs/debug-tsception--hpo-LR/%x_%j.o
#SBATCH --error=../jobs_outputs/debug-tsception--hpo-LR/%x_%j.e

echo "I am a job with ID $SLURM_JOB_ID"
echo "current working directory is $(pwd)"

# 1. copy the squashed dataset to the nodes /tmp
cp ./../squashfs_smr_data/hpo_best_pvc_debug.sqfs /tmp/

# 3. bind the squashed dataset to your apptainer environment and run your script with apptainer
apptainer run --nv -B /tmp/hpo_best_pvc_debug.sqfs:/input-data:image-src=/ ./../env_images/bbcpy_en.sif python ./src/dnn_hpo_train.py experiment=2_Tsception_LR +data.subject_sessions_dict='{S5: "all"}' hparams_search=2_tsception_optuna logger.mlflow.run_name="C1-m-LR"