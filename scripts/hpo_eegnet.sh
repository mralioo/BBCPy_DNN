#!/bin/bash
#SBATCH --job-name=eegnet-hpo-best-pvc
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --ntasks-per-node=20
#SBATCH --output=../jobs_outputs/eegnet-hpo-best-pvc/%x_%j.o
#SBATCH --error=../jobs_outputs/eegnet-hpo-best-pvc/%x_%j.e


echo "I am a job with ID $SLURM_JOB_ID"
echo "current working directory is $(pwd)"

# 1. copy the squashed dataset to the nodes /tmp
cp ./../squashfs_smr_data/hpo_best_pvc.sqfs /tmp/

# 3. bind the squashed dataset to your apptainer environment and run your script with apptainer
apptainer run --nv -B /tmp/hpo_best_pvc.sqfs:/input-data:image-src=/ ./../env_images/bbcpy_lightning_v5.sif python ./src/dnn_hpo_train.py experiment=eegnet +data.subject_sessions_dict='{S5: "all", S9:"all" ,S20:"all", S2:"all", S19:"all", S14:"all"}' hparams_search=eegnet_optuna.yaml logger.mlflow.run_name="eegnet-hpo-best-pvc"
