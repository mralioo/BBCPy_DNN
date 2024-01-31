#!/bin/bash
#SBATCH --job-name=C3-2D-E
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --mem=50GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=../jobs_outputs/C3-eegnet-2D/%x_%j.o
#SBATCH --error=../jobs_outputs/C3-eegnet-2D/%x_%j.e

# List of subjects
CATEGORY="2D-C3"
SUBJECTS=( "S13" "S7" "S43" "S19" "S15" "S33" "S12" "S24" "S3" "S40" "S55" "S34" "S10" "S58" "S16" "S27" "S18" "S48" "S31" "S47" "S6" "S21" "S56" "S22" "S62")
# Loop through each subject
for SUBJECT in "${SUBJECTS[@]}"; do

    echo "I am a job for subject $SUBJECT"
    echo "current working directory is $(pwd)"

    # 1. copy the squashed dataset to the nodes /tmp
    cp ./../squashfs_smr_data/${SUBJECT}.sqfs /tmp/

    # ... (rest of your script remains unchanged, but ensure to change the run_name in the apptainer command)

    apptainer run --nv -B /tmp/${SUBJECT}.sqfs:/input-data:image-src=/ ./../env_images/bbcpy_env.sif python ./src/dnn_train.py experiment=1_eegnet_2D +data.subject_sessions_dict="{$SUBJECT: "all"}" logger.mlflow.experiment_name="${CATEGORY}" logger.mlflow.run_name="${SUBJECT}-EEGNET"

done
