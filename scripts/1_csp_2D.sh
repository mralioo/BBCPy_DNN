#!/bin/bash
#SBATCH --job-name=csp-2D
#SBATCH --partition=cpu-5h
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=8
#SBATCH --output=../jobs_outputs/csp-2D/%x_%j.o
#SBATCH --error=../jobs_outputs/csp-2D/%x_%j.e

# List of subjects
#  Category 1 : 'S52', 'S51', 'S38', 'S39', 'S57', 'S9', 'S49', 'S30', 'S36', 'S20', 'S60', 'S4', 'S23', 'S28', 'S5', 'S8', 'S2', 'S29', 'S26', 'S1'
CATEGORY="2D-C1"
SUBJECTS=( "S52" "S51" "S38" "S39" "S57" "S9" "S49" "S30" "S36" "S20" "S60" "S4" "S23" "S28" "S5" "S8" "S2" "S29" "S26" "S1" )
#SUBJECTS=( "S52" "S51" "S38" "S39" "S57")
# category 2 : 'S53', 'S61', 'S14', 'S35', 'S54', 'S41', 'S45', 'S50', 'S11', 'S42', 'S25', 'S17', 'S32'
#SUBJECTS=( "S53" "S61" "S14" "S35" "S54" "S41" "S45" "S50" "S11" "S42" "S25" "S17" "S32" )
# category 3 : 'S13', 'S7', 'S43', 'S19', 'S15', 'S33', 'S12', 'S24', 'S3', 'S40', 'S55', 'S34', 'S10', 'S58', 'S16', 'S27', 'S18', 'S48', 'S31', 'S47', 'S6', 'S21', 'S56', 'S22', 'S62'
#SUBJECTS=( "S13" "S7" "S43" "S19" "S15" "S33" "S12" "S24" "S3" "S40" "S55" "S34" "S10" "S58" "S16" "S27" "S18" "S48" "S31" "S47" "S6" "S21" "S56" "S22" "S62")

# Loop through each subject
for SUBJECT in "${SUBJECTS[@]}"; do

  echo "I am a job with ID $SLURM_JOB_ID"
  echo "current working directory is $(pwd)"

  # 1. copy the squashed dataset to the nodes /tmp
    cp ./../squashfs_smr_data/${SUBJECT}.sqfs /tmp/

  # 3. bind the squashed dataset to your apptainer environment and run your script with apptainer
  apptainer run -B /tmp/${SUBJECT}.sqfs:/input-data:image-src=/ ./../env_images/bbcpy_env.sif python ./src/baseline_train.py experiment=0_csp_2D +data.subject_sessions_dict="{$SUBJECT: "all"}" logger.mlflow.experiment_name="${CATEGORY}" logger.mlflow.run_name="${SUBJECT}-CSP"

done