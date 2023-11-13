#!/bin/bash
#SBATCH --job-name=riemann-hpo-LR
#SBATCH --partition=cpu-2h
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=8
#SBATCH --output=logs/job-%j.out
#SBATCH --output=../jobs_outputs/riemann-hpo-LR/%x_%j.o
#SBATCH --error=../jobs_outputs/riemann-hpo-LR/%x_%j.e


SUBJECTS=( "S52" "S51" "S38" "S39" "S57")

# Loop through each subject
for SUBJECT in "${SUBJECTS[@]}"; do

  echo "I am a job with ID $SLURM_JOB_ID"
  echo "current working directory is $(pwd)"

  # 1. copy the squashed dataset to the nodes /tmp
    cp ./../squashfs_smr_data/${SUBJECT}.sqfs /tmp/

  # 3. bind the squashed dataset to your apptainer environment and run your script with apptainer
  apptainer run -B /tmp/${SUBJECT}.sqfs:/input-data:image-src=/ ./../env_images/bbcpy_env.sif python ./src/baseline_train.py tune=True experiment=0_riemann_LR hparams_search=hpo_riemann +data.subject_sessions_dict="{$SUBJECT: "all"}" logger.mlflow.experiment_name="HPO-RIEMANN-LR" logger.mlflow.run_name="${SUBJECT}-LR-RIEMANN" paths.results_dir="tune_results"

done