#!/bin/bash
#SBATCH --job-name=debug-hpo-eegnet-2D
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --mem=200G      # Some buffer above 30GB
#SBATCH --ntasks-per-node=1   # One main task that runs the trial and manages CV
#SBATCH --cpus-per-task=3 # Assuming you want to run each CV fold in parallel
#SBATCH --output=../jobs_outputs/debug-hpo-eegnet-2D/%x_%j.o
#SBATCH --error=../jobs_outputs/debug-hpo-eegnet-2D/%x_%j.e

# Define Python script arguments and parameters
PYTHON_SCRIPT="./src/dnn_hpo_train.py"
EXPERIMENT_ARG=1_eegnet_2D
DATA_ARG="{S5: 'all', S9: 'all'}"   # Corrected this line
HPARAMS_ARG=1_eegnet_optuna
LOGGER_ARG="C1-m-2D"

# Define paths
SQUASHFS_PATH="./../squashfs_smr_data/hpo_best_pvc_debug.sqfs"
ENV_IMAGE_PATH="./../env_images/bbcpy_en.sif"
TMP_PATH="/tmp/hpo_best_pvc_debug.sqfs"

# Echo job details
echo "I am a job with ID $SLURM_JOB_ID"
echo "current working directory is $(pwd)"

# Copy the squashed dataset to the node's /tmp
cp $SQUASHFS_PATH $TMP_PATH

# Run the script with apptainer, binding the squashed dataset
apptainer run --nv -B $TMP_PATH:/input-data:image-src=/ $ENV_IMAGE_PATH python $PYTHON_SCRIPT experiment=$EXPERIMENT_ARG +data.subject_sessions_dict="$DATA_ARG" hparams_search=$HPARAMS_ARG logger.mlflow.run_name=$LOGGER_ARG
