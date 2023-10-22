#!/bin/bash
#SBATCH --job-name=info-subjects
#SBATCH --partition=cpu-2h
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=8
#SBATCH --output=../jobs_outputs/info-subjects/%x_%j.o
#SBATCH --error=../jobs_outputs/info-subjects/%x_%j.e

# Collect subject names from the folder
SUBJECTS=("S40" "S31" "S32" "S33" "S35" "S34" "S38")

# Loop through each subject
for SUBJECT in "${SUBJECTS[@]}"; do

  echo "I am a job with ID $SLURM_JOB_ID"
  echo "current working directory is $(pwd)"

  # 1. copy the squashed dataset to the nodes /tmp
    cp ./../squashfs_smr_data/${SUBJECT}.sqfs /tmp/

  # 3. bind the squashed dataset to your apptainer environment and run your script with apptainer
  apptainer run -B /tmp/${SUBJECT}.sqfs:/input-data:image-src=/ ./../env_images/bbcpy_env_1.sif python ./src/subject_info.py --subject_name "${SUBJECT}"

done

