#!/bin/bash
#SBATCH --job-name=info-subjects
#SBATCH --partition=cpu-2h
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=8
#SBATCH --output=../jobs_outputs/info-subjects/%x_%j.o
#SBATCH --error=../jobs_outputs/info-subjects/%x_%j.e

# Define the folder containing subject files
SUBJECT_FOLDER="./../squashfs_smr_data"

# Collect subject names from the folder
SUBJECTS=()
for SUBJECT_FILE in "$SUBJECT_FOLDER"/S*.sqfs; do
  if [[ -f "$SUBJECT_FILE" ]]; then
    SUBJECTS+=("$(basename "$SUBJECT_FILE")")
  fi
done

# Loop through each subject
for SUBJECT in "${SUBJECTS[@]}"; do

  # Extract subject name without extension
  SUBJECT_NAME="${SUBJECT%.*}"

  echo "I am a job with ID $SLURM_JOB_ID"
  echo "current working directory is $(pwd)"

  # 1. copy the squashed dataset to the nodes /tmp
  cp "$SUBJECT_FOLDER/$SUBJECT" /tmp/

  # 3. bind the squashed dataset to your apptainer environment and run your script with apptainer
  apptainer run -B /tmp/"$SUBJECT":/input-data:image-src=/ ./../env_images/bbcpy_env_1.sif python ./src/subject_info.py --subject_name "$SUBJECT_NAME"

done

