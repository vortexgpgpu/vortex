#!/bin/bash

REPOSITORY=https://github.com/vortexgpgpu/vortex-toolchain-prebuilt/raw/master

# RISCV-GNU_TOOLCHAIN

for x in {a..o} 
do
    wget $REPOSITORY/riscv-gnu-toolchain/ubuntu/bionic/riscv-gnu-toolchain.tar.bz2.parta$x
done

cat riscv-gnu-toolchain.tar.bz2.parta* > riscv-gnu-toolchain.tar.bz2
tar -xvf riscv-gnu-toolchain.tar.bz2
rm -f riscv-gnu-toolchain.tar.bz2*

# VERILATOR

wget $REPOSITORY/verilator/ubuntu/bionic/verilator.tar.bz2
tar -xvf verilator.tar.bz2
rm -f verilator.tar.bz2