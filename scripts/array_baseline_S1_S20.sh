#!/bin/bash

#SBATCH --job-name=S1-S20-baseline
#SBATCH --partition=cpu-2d
#SBATCH --gpus-per-node=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --output=../jobs_outputs/S%a/%x_%j.o
#SBATCH --error=../jobs_outputs/S%a/%x_%j.e
#SBATCH --mail-user=mr.ali.alouane@gmail.com
#SBATCH --array=1-5

echo "I am a job with ID $SLURM_JOB_ID for subject S$SLURM_ARRAY_TASK_ID"
echo "current working directory is $(pwd)"

SUBJECT="S$SLURM_ARRAY_TASK_ID"

# 1. copy the squashed dataset to the nodes /tmp
cp ./../squashfs_smr_data/${SUBJECT}.sqfs /tmp/

# Define experiments you want to run
experiments=("csp" "riemann" "riemann_tangent")

# Loop over the experiments
for experiment in "${experiments[@]}"; do

    echo "Processing experiment $experiment for subject $SUBJECT"

    # 3. bind the squashed dataset to your apptainer environment and run your script with apptainer
    apptainer run -B /tmp/${SUBJECT}.sqfs:/input-data:image-src=/ ./../env_images/bbcpy_en.sif python ./src/baseline_train.py +experiment="0_${experiment}" +hparams_search="hpo_${experiment}"  +data.subject_sessions_dict="{$SUBJECT: "all"}" logger.mlflow.experiment_name="${experiment}-RL" logger.mlflow.run_name="${SUBJECT}-RL"

done
