#!/bin/bash
#PBS -l nodes=1:ppn=16:xk
#PBS -N ResNet101-image-ranking-accuracy
#PBS -l walltime=6:00:00
#PBS -e $PBS_JOBNAME.$PBS_JOBID.err
#PBS -o $PBS_JOBNAME.$PBS_JOBID.out
#PBS -M zna2@illinois.edu
cd ~/scratch/image-ranking/src/
. /opt/modules/default/init/bash
module load bwpy
module load cudatoolkit
aprun -n 1 -N 1 python accuracy.py
