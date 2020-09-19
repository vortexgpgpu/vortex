#!/bin/bash

REPOSITORY=https://github.com/vortexgpgpu/vortex-toolchain-prebuilt/raw/master

riscv()
{
    for x in {a..o} 
    do
        wget $REPOSITORY/riscv-gnu-toolchain/ubuntu/bionic/riscv-gnu-toolchain.tar.bz2.parta$x
    done
    cat riscv-gnu-toolchain.tar.bz2.parta* > riscv-gnu-toolchain.tar.bz2
    tar -xvf riscv-gnu-toolchain.tar.bz2
    rm -f riscv-gnu-toolchain.tar.bz2*
    sudo cp riscv-gnu-toolchain /opt/
    rm -rf riscv-gnu-toolchain
}

llvm()
{
    for x in {a..f} 
    do
        wget $REPOSITORY/llvm-riscv/ubuntu/bionic/llvm-riscv.tar.bz2.parta$x
    done
    tar -xvf llvm-riscv.tar.bz2
    rm -f llvm-riscv.tar.bz2
    sudo cp llvm-riscv /opt/
    rm -rf llvm-riscv
}

pocl()
{
    wget $REPOSITORY/pocl/ubuntu/bionic/pocl.tar.bz2
    tar -xvf pocl.bz2
    rm -f pocl.bz2
    sudo cp pocl /opt/
    rm -rf pocl
}

verilator()
{
    wget $REPOSITORY/verilator/ubuntu/bionic/verilator.tar.bz2
    tar -xvf verilator.tar.bz2
    rm -f verilator.tar.bz2
    sudo cp verilator /opt/
    rm -rf verilator
}

usage()
{
    echo "usage: toolchain_install [[-riscv] [-llvm] [-pocl] [-verilator] [-all] [-h|--help]]"
}

while [ "$1" != "" ]; do
    case $1 in
        -pocl ) pocl
                ;;
        -verilator ) verilator
                     ;;
        -riscv ) riscv
                 ;;
        -llvm ) llvm
                ;;
        -all ) riscv
               llvm
               pocl
               verilator
               ;;
        -h | --help ) usage
                      exit
                      ;;
        * )           usage
                      exit 1
    esac
    shift
done