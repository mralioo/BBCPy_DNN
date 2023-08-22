#!/bin/bash
#SBATCH --job-name=my_gpu_job
#SBATCH --partition=gpu-test
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%j.out

apptainer run /opt/apps/pytorch-2.0.1-gpu.sif python example_script.py
