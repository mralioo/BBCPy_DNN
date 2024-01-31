#!/bin/bash
#SBATCH --job-name=C2-2D-csp
#SBATCH --partition=cpu-5h
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=8
#SBATCH --output=../jobs_outputs/C2-csp-2D/%x_%j.o
#SBATCH --error=../jobs_outputs/C2-csp-2D/%x_%j.e

# List of subjects
CATEGORY="2D-C2"
SUBJECTS=( "S53" "S61" "S14" "S35" "S54" "S41" "S45" "S50" "S11" "S42" "S25" "S17" "S32" )

# Loop through each subject
for SUBJECT in "${SUBJECTS[@]}"; do

  echo "I am a job with ID $SLURM_JOB_ID"
  echo "current working directory is $(pwd)"

  # 1. copy the squashed dataset to the nodes /tmp
    cp ./../squashfs_smr_data/${SUBJECT}.sqfs /tmp/

  # 3. bind the squashed dataset to your apptainer environment and run your script with apptainer
  apptainer run -B /tmp/${SUBJECT}.sqfs:/input-data:image-src=/ ./../env_images/bbcpy_env.sif python ./src/baseline_train.py experiment=0_csp_2D +data.subject_sessions_dict="{$SUBJECT: "all"}" logger.mlflow.experiment_name="${CATEGORY}" logger.mlflow.run_name="${SUBJECT}-CSP"

done