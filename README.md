# Building

Example of gpu build on Frontier. First look in `machines/frontier.cmake` and change `Kokkos_DIR` to your Kokkos installation. Then
do:
```
mkdir build
cd build
source ../machines/frontier-gpu.env
cmake .. -C ../machines/frontier-gpu.cmake
make -j8
```

To build on a different machine have a look at existing configurations in the `machines` directory, and write your own if necessary.

# Testing
To run the tests do
```
make test
```

# Benchmarking

After the code is build, the executable `benchmark/benchmark` in the build directory can be used for benchmarking. It takes four optional arguments
```
./benchmark/benchmark [number_of_cells_in_x=64] [number_of_vertical_levels=64] [number_of_tracers=1] [number_of_timesteps=10]
```
The number of cells in y is assumed to be the same as in x. Hence, the total number of cells is `number_of_cells_in_x^2`.
