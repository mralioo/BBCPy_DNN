#!/bin/bash

#$ -binding linear:4 # request 4 cpus (8 with Hyperthreading) (some recommend 4 per GPU)
#$ -N ex       # set consistent base name for output and error file (allows for easy deletion alias)
#$ -q all.q    # don't fill the qlogin queue (can some add why and when to use?)
#$ -cwd        # change working directory (to current)
#$ -l h_vmem=1G # allocate 1GB of virtual memory
#$ -V          # provide environment variables
##$ -t 1-100    # start 100 instances: from 1 to 100

#$ -M ali.alouane@outlook.de     # (debugging) send mail to address...
#$ -m ea                            # ... at job end and abort

#$ -o jobs_outputs/$JOB_NAME/$JOB_ID-$TASK_ID.o
#$ -e jobs_outputs/$JOB_NAME/$JOB_ID-$TASK_ID.e


#$ -l mem_free=500M
#echo "Only runs on nodes which have more than 500 megabytes of free memory"


echo  "I am a job task with ID $JOB_ID" STARTED on $(date)

python src/baseline_train.py paths=cluster experiment=baseline_riemann logger.mlflow.run_name=subject1

# display resource consumption
qstat -j $JOB_ID | awk 'NR==1,/^scheduling info:/'

echo FINISHED on $(date)
