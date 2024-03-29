#!/bin/bash
#SBATCH --job-name=C2-LR-R
#SBATCH --partition=cpu-5h
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=8
#SBATCH --output=../jobs_outputs/C2-riemann-LR/%x_%j.o
#SBATCH --error=../jobs_outputs/C2-riemann-LR/%x_%j.e

# List of subjects
CATEGORY="LR-C2"
SUBJECTS=( "S29" "S26" "S23" "S19" "S53" "S41" "S35" "S61" "S45" "S14" "S15" "S11" "S25" "S1" )

# Loop through each subject
for SUBJECT in "${SUBJECTS[@]}"; do

  echo "I am a job with ID $SLURM_JOB_ID"
  echo "current working directory is $(pwd)"

  # 1. copy the squashed dataset to the nodes /tmp
    cp ./../squashfs_smr_data/${SUBJECT}.sqfs /tmp/

  # 3. bind the squashed dataset to your apptainer environment and run your script with apptainer
  apptainer run -B /tmp/${SUBJECT}.sqfs:/input-data:image-src=/ ./../env_images/bbcpy_env.sif python ./src/baseline_train.py experiment=0_riemann_LR +data.subject_sessions_dict="{$SUBJECT: "all"}" logger.mlflow.experiment_name="${CATEGORY}" logger.mlflow.run_name="${SUBJECT}-RIEMANN"

done
