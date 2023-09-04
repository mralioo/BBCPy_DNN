#!/bin/bash

#SBATCH --job-name=S19-baseline
#SBATCH --partition=cpu-2h
#SBATCH --gpus-per-node=0
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --output=../jobs_outputs/S19/%x_%j.o
#SBATCH --error=../jobs_outputs/S19/%x_%j.e
#SBATCH --mail-user=mr.ali.alouane@gmail.com

echo "I am a job with ID $SLURM_JOB_ID"
echo "current working directory is $(pwd)"

# 1. copy the squashed dataset to the nodes /tmp
cp ./../squashfs_smr_data/S19.sqfs /tmp/

# Define experiments you want to run
experiments=("0_csp" "0_mbcsp" "0_riemann" "0_riemann_tangent")

# Loop over the experiments
for experiment in "${experiments[@]}"; do

    echo "Processing experiment $experiment"

    # 3. bind the squashed dataset to your apptainer environment and run your script with apptainer
    apptainer run -B /tmp/S19.sqfs:/input-data:image-src=/ ./../env_images/bbcpy_lightning_v3.sif python ./src/baseline_train.py +experiment=${experiment} +data.subject_sessions_dict="{S19: "all"}" logger.mlflow.experiment_name="A-RL-S19-all"

done
