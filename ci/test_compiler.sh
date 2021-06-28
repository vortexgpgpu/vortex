#!/bin/bash

# exit when any command fails
set -e

# rebuild runtime
make -C runtime clean
make -C runtime

# clear POCL cache
rm -rf ~/.cache/pocl

# rebuild runtime tests
make -C tests/runtime clean
make -C tests/runtime
make -C tests/runtime run

# rebuild native kernels
make -C tests/regression clean-all
make -C tests/regression
make -C tests/regression run

# rebuild opencl kernels
make -C tests/opencl clean-all
make -C tests/opencl
make -C tests/opencl run