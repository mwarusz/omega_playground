# Quickstart

```
mdkir build
cd build

# example build for CPU
cmake .. -DYAKL_CXX_FLAGS="-O2"

# example build for CUDA (V100)
cmake .. -DYAKL_ARCH=CUDA -DYAKL_CUDA_FLAGS="-O2 -arch=sm_70"

make
make test
```
