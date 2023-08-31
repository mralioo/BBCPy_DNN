#!/bin/bash

#SBATCH --job-name=S1-baseline
#SBATCH --partition=cpu-2h
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=8
#SBATCH --output=../jobs_outputs/S1/%x_%j.o
#SBATCH --mail-user=mr.ali.alouane@gmail.com

echo "I am a job with ID $SLURM_JOB_ID"
echo "current working directory is $(pwd)"

# 1. copy the squashed dataset to the nodes /tmp
cp ./../squashfs_smr_data/S1.sqfs /tmp/

# Define experiments you want to run
experiments=("0_csp" "0_mbcsp" "0_riemann" "0_riemann_tangent")

# Loop over the experiments
for experiment in "${experiments[@]}"; do

    echo "Processing experiment $experiment"

    # 3. bind the squashed dataset to your apptainer environment and run your script with apptainer
    apptainer run -B "/tmp/S1.sqfs:/input-data:image-src=/" "./../env_images/bbcpy_lightning_v3.sif" python "./src/baseline_train.py +experiment=${experiment}"

done
