#!/bin/bash
#SBATCH --job-name=C1-2D-TR
#SBATCH --partition=cpu-5h
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=8
#SBATCH --output=../jobs_outputs/C1-riemann-tangent-2D/%x_%j.o
#SBATCH --error=../jobs_outputs/C1-riemann-tangent-2D/%x_%j.e

# List of subjects
CATEGORY="2D-C1"
SUBJECTS=( "S52" "S51" "S38" "S39" "S57" "S9" "S49" "S30" "S36" "S20" "S60" "S4" "S23" "S28" "S5" "S8" "S2" "S29" "S26" "S1" )


# Loop through each subject
for SUBJECT in "${SUBJECTS[@]}"; do

  echo "I am a job with ID $SLURM_JOB_ID"
  echo "current working directory is $(pwd)"

  # 1. copy the squashed dataset to the nodes /tmp
    cp ./../squashfs_smr_data/${SUBJECT}.sqfs /tmp/

  # 3. bind the squashed dataset to your apptainer environment and run your script with apptainer
  apptainer run -B /tmp/${SUBJECT}.sqfs:/input-data:image-src=/ ./../env_images/bbcpy_env.sif python ./src/baseline_train.py experiment=0_riemann_tangent_2D +data.subject_sessions_dict="{$SUBJECT: "all"}" logger.mlflow.experiment_name="${CATEGORY}" logger.mlflow.run_name="${SUBJECT}-TRIEMANN"

done