#!/bin/bash

#SBATCH --job-name=omega-pvc-performance
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=PV
#SBATCH --time=01:00:00
#SBATCH --output=job-omega-pvc-performance-%j.log

outfile=${1:-"blake-pvc-performance.txt"}
script_path=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
script_dir=$(dirname $script_path)

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
