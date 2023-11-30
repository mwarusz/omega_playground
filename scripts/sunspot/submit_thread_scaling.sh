#!/bin/bash

#PBS -A CSC249ADSE15_CNDA
#PBS -N omega-thread-scaling
#PBS -l nodes=1:ppn=208
#PBS -q workq
#PBS -l walltime=01:00:00
#PBS -o job-omega-thread-scaling-$PBS_JOBID.log

cd $PBS_O_WORKDIR

outdir=${1:-"sunspot_thread_scaling"}
script_dir=$PBS_O_HOME/omega_playground/scripts/

threads="1 2 4 8 16 24 32 40 48 56"
nvertlevels=64
ntracers=1
ncellsx_base=50
nsteps_base=512

ncellsx=$ncellsx_base
nsteps=$nsteps_base

for l in {1..5}
do
  echo "$ncellsx $nsteps"
  $script_dir/run_thread_scaling.sh $ncellsx $nvertlevels $ntracers $nsteps "$threads" $outdir 
  ncellsx=$((2*$ncellsx))
  nsteps=$(($nsteps/4))
done
