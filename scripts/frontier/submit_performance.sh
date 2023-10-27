#!/bin/bash

#SBATCH --account=CLI115
#SBATCH --job-name=omega-performance
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --time=01:00:00
#SBATCH --output=job-omega-performance-%j.log

script_path=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
script_dir=$(dirname $script_path)

outfile=${1:-"frontier_performance"}

export OMP_NUM_THREADS=56
export OMP_PLACES="threads"
export OMP_PROC_BIND="spread"

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
