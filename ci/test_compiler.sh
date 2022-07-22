#!/bin/bash

# exit when any command fails
set -e

# ensure build
make -s

# clear POCL cache
rm -rf ~/.cache/pocl

# rebuild runtime library
make -C runtime clean
make -C runtime

# rebuild kernel library
make -C kernel clean
make -C kernel

# rebuild kernel tests
make -C tests/kernel clean
make -C tests/kernel

# rebuild regression tests
make -C tests/regression clean-all
make -C tests/regression

# rebuild opencl tests
make -C tests/opencl clean-all
make -C tests/opencl