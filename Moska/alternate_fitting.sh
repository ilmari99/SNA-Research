#!/bin/bash

JOB1=./Moska/only_simulate_batch_job.sh
JOB2=./Moska/only_fit_batch_job.sh

prev_jobid=""

for i in {1..10}; do
    jobid=$(sbatch --dependency=afterok:$prev_jobid --parsable $JOB1)
    prev_jobid=$jobid
    jobid=$(sbatch --dependency=afterok:$prev_jobid --parsable $JOB2)
    prev_jobid=$jobid
done
