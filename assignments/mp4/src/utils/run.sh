#!/bin/bash
# This script runs two batch jobs to run cs598 mp4


# MODIFY THESE
declare training_file="main.py"
declare walltime="06:00:00"
declare jobname="cs598-mp4"
declare netid="yournetid"
declare -a directory=("~/scratch/part-1/src/" "~/scratch/part-2/src/")

for job in "${directory[@]}"
do
    python gen_pbs.py $training_file $walltime $jobname $netid $directory $job > run.pbs
    echo "Submitting $job"
    qsub run.pbs -A bauh
done
