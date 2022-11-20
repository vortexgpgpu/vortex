#!/bin/bash

# exit when any command fails
set -e

# clear POCL cache
rm -rf ~/.cache/pocl

# force rebuild test kernels
make -C tests clean-all

# ensure build
make -s

# run tests
make -C tests/kernel run-simx
make -C tests/regression run-simx
make -C tests/opencl run-simx