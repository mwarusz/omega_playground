#!/bin/bash

#SBATCH --account=CLI115
#SBATCH --job-name=omega-thread-scaling
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --time=01:00:00
#SBATCH --output=job-omega-thread-scaling-%j.log

outdir=${1:-"frontier_thread_scaling"}
script_path=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
script_dir=$(dirname $script_path)

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
  $script_dir/../run_thread_scaling.sh $ncellsx $nvertlevels $ntracers $nsteps "$threads" $outdir 
  ncellsx=$((2*$ncellsx))
  nsteps=$(($nsteps/4))
done
