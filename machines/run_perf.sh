#!/usr/bin/bash

arch=cpu
comp=gcc

dirname=workmac-$arch-$comp
mkdir $dirname

opts=("split" "fused" "fused_accum"
      "split_osimd" "fused_osimd" "fused_accum_osimd"
      "split_ksimd" "fused_ksimd" "fused_accum_ksimd"
      "noif_split" "noif_fused" "noif_fused_accum"
      "noif_split_osimd" "noif_fused_osimd" "noif_fused_accum_osimd"
      "noif_split_ksimd" "noif_fused_ksimd" "noif_fused_accum_ksimd")

for opt in ${opts[@]}; do
  exec=./exp/steady_zonal_omega_${opt}
  ln -sf ~/omega_test_meshes/icos-cvt-hi/icos1920-sorted.nc perf_mesh.nc
  ${exec}  > $dirname/log_${arch}_1920_${opt}.txt 2>&1
  ln -sf ~/omega_test_meshes/icos-cvt-hi/icos960-sorted.nc perf_mesh.nc
  ${exec}  > $dirname/log_${arch}_960_${opt}.txt 2>&1
  ln -sf ~/omega_test_meshes/icos-cvt-hi/icos480-sorted.nc perf_mesh.nc
  ${exec}  > $dirname/log_${arch}_480_${opt}.txt 2>&1
  ln -sf ~/omega_test_meshes/icos-cvt-hi/icos240-sorted.nc perf_mesh.nc
  ${exec}  > $dirname/log_${arch}_240_${opt}.txt 2>&1
done
