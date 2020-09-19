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
sudo cp opt/riscv-gnu-toolchain /opt/
rm -rf riscv-gnu-toolchain

# LLVM

for x in {a..f} 
do
    wget $REPOSITORY/llvm-riscv/ubuntu/bionic/llvm-riscv.tar.bz2.parta$x
done
tar -xvf llvm-riscv.tar.bz2
rm -f llvm-riscv.tar.bz2
sudo cp opt/llvm-riscv /opt/
rm -rf llvm-riscv

# POCL

wget $REPOSITORY/pocl/ubuntu/bionic/pocl.tar.bz2
tar -xvf pocl.bz2
rm -f pocl.bz2
sudo cp opt/pocl /opt/
rm -rf pocl

# VERILATOR

wget $REPOSITORY/verilator/ubuntu/bionic/verilator.tar.bz2
tar -xvf verilator.tar.bz2
rm -f verilator.tar.bz2
sudo cp opt/verilator /opt/
rm -rf verilator