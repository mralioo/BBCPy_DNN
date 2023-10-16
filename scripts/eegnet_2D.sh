#!/bin/bash
#SBATCH --job-name=eegnet-2d
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --mem=100GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=../jobs_outputs/eegnet-2D/%x_%j.o
#SBATCH --error=../jobs_outputs/eegnet-2D/%x_%j.e

# List of subjects
SUBJECTS=("S57" "S39" "S30" "S52" "S51" "S49" "S36")

# Loop through each subject
for SUBJECT in "${SUBJECTS[@]}"; do

    echo "I am a job for subject $SUBJECT"
    echo "current working directory is $(pwd)"

    # 1. copy the squashed dataset to the nodes /tmp
    cp ./../squashfs_smr_data/${SUBJECT}.sqfs /tmp/

    # ... (rest of your script remains unchanged, but ensure to change the run_name in the apptainer command)

    apptainer run -B /tmp/${SUBJECT}.sqfs:/input-data:image-src=/ ./../env_images/bbcpy_env.sif python ./src/dnn_train.py experiment=1_eegnet_2D +data.subject_sessions_dict="{$SUBJECT: "all"}" logger.mlflow.experiment_name="EEGNET-2D" logger.mlflow.run_name="${SUBJECT}-2D-EEGNET"

done
