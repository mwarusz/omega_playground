#!/bin/bash

#SBATCH --job-name=omega-thread-scaling
#SBATCH --nodes=1
#SBATCH --ntasks-per-socket=48
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --partition=all
#SBATCH --time=02:00:00
#SBATCH --output=job-omega-thread-scaling-%j.log

script_path=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
script_dir=$(dirname $script_path)

outdir="blake_results"
mkdir -p $outdir && cd $outdir

source $script_dir/../../machines/blake-cpu.env

threads="1 2 4 8 16 24 32 40 48 56 64 72 80 88 96"
nvertlevels=64
ntracers=1

$script_dir/../run_thread_scaling.sh 50  $nvertlevels $ntracers  256 "$threads"
$script_dir/../run_thread_scaling.sh 100 $nvertlevels $ntracers  128 "$threads"
$script_dir/../run_thread_scaling.sh 200 $nvertlevels $ntracers  32  "$threads"
$script_dir/../run_thread_scaling.sh 400 $nvertlevels $ntracers  8   "$threads"
$script_dir/../run_thread_scaling.sh 800 $nvertlevels $ntracers  2   "$threads"
