#/usr/bin/env bash

script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
machine=${1:-local}

machinefile=$script_dir/machines/${machine}.env 

if [ -e $machinefile ]
then
  source $machinefile
else
  echo "unknown machine = $machine"
  return
fi

# use cmake --fresh if available
case $(cmake --version | awk '/cmake version/ {print $3}') in
  [012].*|3.?|3.?.*|3.2[0123].*)
  # too old, need to clean the cache manually
  CMAKE_FRESH=""
  rm -f CMakeCache.txt
  ;;
*)
  CMAKE_FRESH="--fresh"
  ;;
esac

# empty YAKL_ARCH expects YAKL_CXX_FLAGS
if [ "${YAKL_ARCH}" == "" ]
then
  YAKL_FLAGS_ARCH="CXX"
else
  YAKL_FLAGS_ARCH=${YAKL_ARCH}
fi

cmake "$CMAKE_FRESH" $script_dir -DYAKL_ARCH=${YAKL_ARCH} -DYAKL_"${YAKL_FLAGS_ARCH}"_FLAGS="${COMPILE_FLAGS}" -DOMEGA_LINK_FLAGS="${LINK_FLAGS}" -DOMEGA_VECTOR_LENGTH=${VECTOR_LENGTH} -DOMEGA_USE_CXX_SIMD=${USE_CXX_SIMD}
