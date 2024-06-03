#!/bin/bash

JOB1=./BlokusPentobi/only_simulate_batch_job.sh
JOB2=./BlokusPentobi/only_fit_batch_job.sh

prev_jobid=""

for i in {1..2}; do
    if [ "$i" != 1 ]; then
        jobid=$(sbatch --dependency=afterok:$prev_jobid --parsable $JOB1)
    else
        jobid=$(sbatch --parsable $JOB1)
    fi
    echo "Submitted job $jobid"
    prev_jobid=$jobid
    jobid=$(sbatch --dependency=afterok:$prev_jobid --parsable $JOB2)
    prev_jobid=$jobid
done