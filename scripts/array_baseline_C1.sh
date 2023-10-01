#!/bin/bash

#SBATCH --job-name=baseline_C1
#SBATCH --partition=cpu-2d
#SBATCH --gpus-per-node=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --output=../jobs_outputs/baseline_C1/S%a_%x_%j.o
#SBATCH --error=../jobs_outputs/baseline_C1/S%a_%x_%j.e
#SBATCH --mail-user=mr.ali.alouane@gmail.com
#SBATCH --array=0-4  # This should range from 0 to (number of subjects - 1)

# Declare a list of specific subjects
SUBJECTS=("S57" "S39" "S9" "S30" "S52" "S51" "S49")  # You can add or remove subjects as needed

# Use the SLURM_ARRAY_TASK_ID as an index to get the subject
SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

echo "I am a job with ID $SLURM_JOB_ID for subject $SUBJECT"
echo "current working directory is $(pwd)"

# 1. copy the squashed dataset to the nodes /tmp
cp ./../squashfs_smr_data/${SUBJECT}.sqfs /tmp/

# Define experiments you want to run
experiments=("csp" "riemann" "riemann_tangent")

# Loop over the experiments
for experiment in "${experiments[@]}"; do

    echo "Processing experiment $experiment for subject $SUBJECT"
    echo "************************************************************"
    echo "************************************************************"
    echo "************************************************************"
    echo "************************************************************"

    # 3. bind the squashed dataset to your apptainer environment and run your script with apptainer
    apptainer run -B /tmp/${SUBJECT}.sqfs:/input-data:image-src=/ ./../env_images/bbcpy_en.sif python ./src/baseline_train.py experiment="0_${experiment}" hparams_search="hpo_${experiment}"  +data.subject_sessions_dict="{$SUBJECT: "all"}" logger.mlflow.experiment_name="${experiment}-RL" logger.mlflow.run_name="${SUBJECT}-RL"

done
