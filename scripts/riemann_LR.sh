#!/bin/bash
#SBATCH --job-name=riemann-LR
#SBATCH --partition=cpu-2h
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=8
#SBATCH --output=../jobs_outputs/riemann-LR/%x_%j.o
#SBATCH --error=../jobs_outputs/riemann-LR/%x_%j.e

# List of subjects
SUBJECTS=("S57" "S39" "S30" "S52" "S51" "S49" "S36")

# Loop through each subject
for SUBJECT in "${SUBJECTS[@]}"; do

  echo "I am a job with ID $SLURM_JOB_ID"
  echo "current working directory is $(pwd)"

  # 1. copy the squashed dataset to the nodes /tmp
    cp ./../squashfs_smr_data/${SUBJECT}.sqfs /tmp/

  # 3. bind the squashed dataset to your apptainer environment and run your script with apptainer
  apptainer run -B /tmp/${SUBJECT}.sqfs:/input-data:image-src=/ ./../env_images/bbcpy_env_1.sif python ./src/baseline_train.py experiment=0_riemann_LR +data.subject_sessions_dict="{$SUBJECT: "all"}" logger.mlflow.experiment_name="RIEMANN-LR" logger.mlflow.run_name="${SUBJECT}-LR-RIEMANN"

done
