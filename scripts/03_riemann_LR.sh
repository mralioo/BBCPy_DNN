#!/bin/bash
#SBATCH --job-name=riemann-LR
#SBATCH --partition=cpu-5h
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=8
#SBATCH --output=../jobs_outputs/riemann-LR/%x_%j.o
#SBATCH --error=../jobs_outputs/riemann-LR/%x_%j.e

# List of subjects
#SUBJECTS=("S57" "S39" "S30" "S52" "S51" "S49" "S36")
#SUBJECTS=( "S52" "S51" "S38" "S39" "S57")
CATEGORY="LR-C3"
SUBJECTS=( "S13" "S10" "S54" "S32" "S7" "S58" "S56" "S48" "S62" "S34" "S12" "S33" "S55" "S22" "S47" "S16" "S27" "S18" "S42" "S31" "S21" "S24" "S17" "S6" "S40" "S3")


# Loop through each subject
for SUBJECT in "${SUBJECTS[@]}"; do

  echo "I am a job with ID $SLURM_JOB_ID"
  echo "current working directory is $(pwd)"

  # 1. copy the squashed dataset to the nodes /tmp
    cp ./../squashfs_smr_data/${SUBJECT}.sqfs /tmp/

  # 3. bind the squashed dataset to your apptainer environment and run your script with apptainer
  apptainer run -B /tmp/${SUBJECT}.sqfs:/input-data:image-src=/ ./../env_images/bbcpy_env.sif python ./src/baseline_train.py experiment=0_riemann_LR +data.subject_sessions_dict="{$SUBJECT: "all"}" logger.mlflow.experiment_name="${CATEGORY}" logger.mlflow.run_name="${SUBJECT}-RIEMANN"

done
