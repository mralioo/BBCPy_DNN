#!/bin/bash

#$ -binding linear:4 # request 4 cpus (8 with Hyperthreading) (some recommend 4 per GPU)
#$ -N ex       # set consistent base name for output and error file (allows for easy deletion alias)
#$ -q all.q    # don't fill the qlogin queue (can some add why and when to use?)
#$ -cwd        # change working directory (to current)
#$ -V          # provide environment variables
##$ -t 1-100    # start 100 instances: from 1 to 100

##$ -M ali.alouane@outlook.de     # (debugging) send mail to address...
##$ -m ea                            # ... at job end and abort

##$ -t 1-100    # start 100 instances: from 1 to 100
##$ -o jobs_outputs/$JOB_NAME/$JOB_ID-$TASK_ID.o
##$ -e jobs_outputs/$JOB_NAME/$JOB_ID-$TASK_ID.e

#$ -o jobs_outputs/$JOB_NAME/$JOB_ID.o
#$ -e jobs_outputs/$JOB_NAME/$JOB_ID.e

# if you also want to request a GPU, add the following line to the above block:
#$ -l cuda=1   # request one GPU


# enter the virtual environment
#source ~/.bashrc
#conda ~/miniconda/bin/activate bbcpy_env
#echo "conda environment $CONDA_ENV_NAME is activated"
#conda init bash

eval "$(conda shell.bash hook)"
conda activate bbcpy_env

echo "I am a job task with ID $JOB_ID"



cd ~/Desktop/bci_proj/BCI-PJ2021SS
python -m bbcpy.train.train_model_test_pipeline -f bbcpy/params/test.yml --batch_size=32 # here you perform any commands to start your program

# This will run the script in for loop, we need to specify the parameter values
#for batch in $(seq 16 32 64); do
#    python -m bbcpy.train.train_model_test_pipeline -f bbcpy/params/test.yml --batch_size=$batch
#done
