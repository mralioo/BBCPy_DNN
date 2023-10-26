#!/bin/bash
#SBATCH --job-name=csp-hpo-LR
#SBATCH --partition=cpu-2h
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=8
#SBATCH --output=logs/job-%j.out
#SBATCH --output=../jobs_outputs/csp-S9/%x_%j.o
#SBATCH --error=../jobs_outputs/csp-S9/%x_%j.e

echo "I am a job with ID $SLURM_JOB_ID"
echo "current working directory is $(pwd)"

SUBJECTS=( "S5" "S60" "S57" )

# Loop through each subject
for SUBJECT in "${SUBJECTS[@]}"; do

  echo "I am a job with ID $SLURM_JOB_ID"
  echo "current working directory is $(pwd)"

  # 1. copy the squashed dataset to the nodes /tmp
    cp ./../squashfs_smr_data/${SUBJECT}.sqfs /tmp/

  # 3. bind the squashed dataset to your apptainer environment and run your script with apptainer
  apptainer run -B /tmp/${SUBJECT}.sqfs:/input-data:image-src=/ ./../env_images/bbcpy_env.sif python ./src/baseline_train.py experiment=0_csp_LR hparams_search=hpo_csp_excllev +data.subject_sessions_dict="{$SUBJECT: "all"}" logger.mlflow.experiment_name="HPO-CSP-LR" logger.mlflow.run_name="${SUBJECT}-LR-CSP"

done