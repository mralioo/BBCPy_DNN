#!/bin/bash
#SBATCH --job-name=my_cpu_job
#SBATCH --partition=cpu-test
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%j.out

echo "I am a job with ID $SLURM_JOB_ID"
echo "current working directory is $(pwd)"

# 1. copy the squashed dataset to the nodes /tmp
cp ./squashfs_smr_data/squashfs_example.sqfs /tmp/

# 3. bind the squashed dataset to your apptainer environment and run your script with apptainer
apptainer run -B /tmp/S1.sqfs:/input-data:image-src=/ ./env_images/bbcpy_lightning.sif python ./bbcpy_autoML/src/baseline_train.py experiment=baseline_riemann debug=default_baseline
