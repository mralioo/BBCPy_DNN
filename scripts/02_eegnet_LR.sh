#!/bin/bash
#SBATCH --job-name=C2-LR-E
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --mem=50GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=../jobs_outputs/C2-eegnet-LR/%x_%j.o
#SBATCH --error=../jobs_outputs/C2-eegnet-LR/%x_%j.e

# List of subjects
CATEGORY="LR-C2"
SUBJECTS=( "S29" "S26" "S23" "S19" "S53" "S41" "S35" "S61" "S45" "S14" "S15" "S11" "S25" "S1" )

# Loop through each subject
for SUBJECT in "${SUBJECTS[@]}"; do

    echo "I am a job for subject $SUBJECT"
    echo "current working directory is $(pwd)"

    # 1. copy the squashed dataset to the nodes /tmp
    cp ./../squashfs_smr_data/${SUBJECT}.sqfs /tmp/

    # ... (rest of your script remains unchanged, but ensure to change the run_name in the apptainer command)

    apptainer run --nv -B /tmp/${SUBJECT}.sqfs:/input-data:image-src=/ ./../env_images/bbcpy_env.sif python ./src/dnn_train.py experiment=1_eegnet_LR +data.subject_sessions_dict="{$SUBJECT: "all"}" logger.mlflow.experiment_name="${CATEGORY}" logger.mlflow.run_name="${SUBJECT}-EEGNET"

done
