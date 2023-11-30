#!/bin/bash

#SBATCH --job-name=omega-thread-scaling
#SBATCH --nodes=1
#SBATCH --ntasks-per-socket=48
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --partition=all
#SBATCH --time=02:00:00
#SBATCH --output=job-omega-thread-scaling-%j.log

outdir=${1:-"frontier_thread_scaling"}
script_path=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
script_dir=$(dirname $script_path)

outdir="blake_results"
mkdir -p $outdir && cd $outdir

threads="1 2 4 8 16 24 32 40 48 56 64 72 80 88 96"
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
