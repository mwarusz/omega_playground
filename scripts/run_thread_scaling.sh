#/usr/bin/env bash

NCELLSX=${1:-128}
NLEVELS=${2:-64}
NTRACERS=${3:-1}
NSTEPS=${4:-10}
NTHREADS=${5:-"1 2 4"}
OUTDIR=${6:-"thread_scaling"}

mkdir -p $OUTDIR

export OMP_PLACES="threads"
export OMP_PROC_BIND="spread"

for nt in ${NTHREADS}
do
  export OMP_NUM_THREADS=${nt}
  OUTFILE="scaling_${NCELLSX}_${NLEVELS}_${NTRACERS}_${NSTEPS}.txt"
  printf "%-4d" "$nt" >> "${OUTDIR}/${OUTFILE}"
  ./benchmark/benchmark ${NCELLSX} ${NLEVELS} ${NTRACERS} ${NSTEPS} 2>> "${OUTDIR}/${OUTFILE}"
done
