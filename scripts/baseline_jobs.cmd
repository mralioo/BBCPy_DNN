
#$ -m ea                            # ... at job end and abort

##$ -t 1-100    # start 100 instances: from 1 to 100
##$ -o jobs_outputs/$JOB_NAME/$JOB_ID-$TASK_ID.o
##$ -e jobs_outputs/$JOB_NAME/$JOB_ID-$TASK_ID.e

#$ -o jobs_outputs/$JOB_NAME/$JOB_ID.o
#$ -e jobs_outputs/$JOB_NAME/$JOB_ID.e
#-l h_vmem=16G
##$ -l mem_free=20G
#echo "Only runs on nodes which have more than 500 megabytes of free memory"

# if you also want to request a GPU, add the following line to the above block:
#$ -l cuda=1  # request one GPU

echo "I am a job task with ID $JOB_ID"

python src/baseline_train.py experiment=baseline_experiment_alpha_band_csp