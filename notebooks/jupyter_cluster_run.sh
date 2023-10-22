#!/bin/bash
#SBATCH --job-name=cpu_job
#SBATCH --partition=cpu-2h
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/test-jupyter-with-data-%j.out

# 1. copy the squashed dataset to the nodes /tmp
cp /home/space/datasets-sqfs/squashfs_example.sqfs /tmp/

# 2. bind the squashed dataset to your apptainer environment and run your script with apptainer
apptainer run -B /tmp/squashfs_example.sqfs:/input-data:image-src=/ ./python_container.sif jupyter notebook --ip 0.0.0.0 --no-browser
