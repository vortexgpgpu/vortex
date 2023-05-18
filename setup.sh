#!/bin/bash

export RISCV_TOOLCHAIN_PATH=/opt/riscv-gnu-toolchain
export VERILATOR_ROOT=/opt/verilator
export PATH=$VERILATOR_ROOT/bin:$PATH

# sudo ci/toolchain_install.sh -all

make -s
