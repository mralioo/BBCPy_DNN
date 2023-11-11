#!/bin/bash
#SBATCH --job-name=transformer-LR
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --mem=50GB
#SBATCH --ntasks-per-node=1   # One main task that runs the trial and manages CV
#SBATCH --cpus-per-task=6 # Assuming you want to run each CV fold in parallel
#SBATCH --output=../jobs_outputs/transformer-LR/%x_%j.o
#SBATCH --error=../jobs_outputs/transformer-LR/%x_%j.e

# List of subjects
CATEGORY="LR-C1"
SUBJECTS=( "S5" "S60" "S57" "S39" "S9" "S49" "S30" "S51" "S52" "S8" "S36" "S20" "S50" "S4" "S38" "S2" "S43" "S28" )

# Loop through each subject
for SUBJECT in "${SUBJECTS[@]}"; do

    echo "I am a job for subject $SUBJECT"
    echo "current working directory is $(pwd)"

    # 1. copy the squashed dataset to the nodes /tmp
    cp ./../squashfs_smr_data/${SUBJECT}.sqfs /tmp/
    # ... (rest of your script remains unchanged, but ensure to change the run_name in the apptainer command)

    apptainer run --nv -B /tmp/${SUBJECT}.sqfs:/input-data:image-src=/ ./../env_images/bbcpy_env.sif python ./src/dnn_train.py experiment=3_transformer_LR +data.subject_sessions_dict="{$SUBJECT: "all"}" logger.mlflow.experiment_name="${CATEGORY}" logger.mlflow.run_name="${SUBJECT}-transformer"

done