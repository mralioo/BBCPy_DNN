#!/bin/bash
#SBATCH --job-name=C1-LR-csp
#SBATCH --partition=cpu-5h
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=8
#SBATCH --output=../jobs_outputs/C1-csp-LR/%x_%j.o
#SBATCH --error=../jobs_outputs/C1-csp-LR/%x_%j.e

# List of subjects
CATEGORY="LR-C1"
SUBJECTS=( "S5" "S60" "S57" "S39" "S9" "S49" "S30" "S51" "S52" "S8" "S36" "S20" "S50" "S4" "S38" "S2" "S43" "S28" )

# Loop through each subject
for SUBJECT in "${SUBJECTS[@]}"; do

  echo "I am a job with ID $SLURM_JOB_ID"
  echo "current working directory is $(pwd)"

  # 1. copy the squashed dataset to the nodes /tmp
    cp ./../squashfs_smr_data/${SUBJECT}.sqfs /tmp/

  # 3. bind the squashed dataset to your apptainer environment and run your script with apptainer
  apptainer run -B /tmp/${SUBJECT}.sqfs:/input-data:image-src=/ ./../env_images/bbcpy_env.sif python ./src/baseline_train.py experiment=0_csp_LR +data.subject_sessions_dict="{$SUBJECT: "all"}" logger.mlflow.experiment_name="${CATEGORY}" logger.mlflow.run_name="${SUBJECT}-CSP"

done