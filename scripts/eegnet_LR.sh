#!/bin/bash
#SBATCH --job-name=eegnet-LR
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --mem=100GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=../jobs_outputs/eegnet-LR/%x_%j.o
#SBATCH --error=../jobs_outputs/eegnet-LR/%x_%j.e

# List of subjects
SUBJECTS=("S57" "S39" "S30" "S52" "S51" "S49" "S36")

# Loop through each subject
for SUBJECT in "${SUBJECTS[@]}"; do

    echo "I am a job for subject $SUBJECT"
    echo "current working directory is $(pwd)"

    # ... (rest of your script remains unchanged, but ensure to change the run_name in the apptainer command)

    apptainer run -B /tmp/${SUBJECT}.sqfs:/input-data:image-src=/ ./../env_images/bbcpy_en.sif python ./src/baseline_train.py experiment=1_eegnet_LR +data.subject_sessions_dict="{$SUBJECT: "all"}" logger.mlflow.experiment_name="EEGNET-LR" logger.mlflow.run_name="${SUBJECT}-LR"

done