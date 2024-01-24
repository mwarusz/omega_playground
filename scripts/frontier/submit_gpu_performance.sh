#!/bin/bash

#SBATCH --account=CLI115
#SBATCH --job-name=omega-gpu-performance
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --time=01:00:00
#SBATCH --output=job-omega-gpu-performance-%j.log

script_path=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
script_dir=$(dirname $script_path)

outfile=${1:-"frontier_gpu_performance.txt"}

nvertlevels=64
ntracers=1
ncellsx_base=50
nsteps_base=512

ncellsx=$ncellsx_base
nsteps=$nsteps_base

for l in {1..5}
do
  printf "%-5d" $ncellsx >> $outfile
  ./benchmark/benchmark $ncellsx $nvertlevels $ntracers $nsteps 2>> $outfile 
  ncellsx=$((2*$ncellsx))
  nsteps=$(($nsteps/4))
done
