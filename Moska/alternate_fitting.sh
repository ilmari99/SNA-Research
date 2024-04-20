#!/bin/bash

JOB1=./Moska/only_simulate_batch_job.sh
JOB2=./Moska/only_fit_batch_job.sh

prev_jobid=""

for i in {1..10}; do
    if [ "$i" -eq 1 ]; then
    fi
        jobid=$(sbatch --dependency=afterok:$prev_jobid --parsable $JOB1)
    else
        jobid=$(sbatch --parsable $JOB1)
    prev_jobid=$jobid
    jobid=$(sbatch --dependency=afterok:$prev_jobid --parsable $JOB2)
    prev_jobid=$jobid
done
