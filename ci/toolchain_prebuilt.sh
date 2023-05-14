#!/bin/bash

# exit when any command fails
set -e

OS_DIR=${OS_DIR:-'ubuntu/bionic'}
SRCDIR=${SRCDIR:-'/opt'}
DESTDIR=${DESTDIR:-'.'}

echo "OS_DIR=${OS_DIR}"
echo "SRCDIR=${SRCDIR}"
echo "DESTDIR=${DESTDIR}"

riscv() 
{
    echo "prebuilt riscv-gnu-toolchain..."
    tar -C $SRCDIR -cvjf riscv-gnu-toolchain.tar.bz2 riscv-gnu-toolchain
    split -b 50M riscv-gnu-toolchain.tar.bz2 "riscv-gnu-toolchain.tar.bz2.part"    
    mv riscv-gnu-toolchain.tar.bz2.part* $DESTDIR/riscv-gnu-toolchain/$OS_DIR
    rm riscv-gnu-toolchain.tar.bz2
}

riscv64() 
{
    echo "prebuilt riscv64-gnu-toolchain..."
    tar -C $SRCDIR -cvjf riscv64-gnu-toolchain.tar.bz2 riscv64-gnu-toolchain
    split -b 50M riscv64-gnu-toolchain.tar.bz2 "riscv64-gnu-toolchain.tar.bz2.part"    
    mv riscv64-gnu-toolchain.tar.bz2.part* $DESTDIR/riscv64-gnu-toolchain/$OS_DIR
    rm riscv64-gnu-toolchain.tar.bz2
}

llvm() 
{
    echo "prebuilt llvm-vortex..."
    tar -C $SRCDIR -cvjf llvm-vortex.tar.bz2 llvm-vortex
    split -b 50M llvm-vortex.tar.bz2 "llvm-vortex.tar.bz2.part"    
    mv llvm-vortex.tar.bz2.part* $DESTDIR/llvm-vortex/$OS_DIR
    rm llvm-vortex.tar.bz2
}

pocl() 
{
    echo "prebuilt pocl2..."
    tar -C $SRCDIR -cvjf pocl2.tar.bz2 pocl
    mv pocl2.tar.bz2 $DESTDIR/pocl/$OS_DIR
}

verilator() 
{
    echo "prebuilt verilator2..."
    tar -C $SRCDIR -cvjf verilator2.tar.bz2 verilator
    mv verilator2.tar.bz2 $DESTDIR/verilator/$OS_DIR
}

show_usage()
{
    echo "Setup Pre-built Vortex Toolchain"
    echo "Usage: $0 [[-riscv] [-llvm] [-pocl] [-verilator] [-all] [-h|--help]]"
}

while [ "$1" != "" ]; do
    case $1 in
        -pocl ) pocl
                ;;
        -verilator ) verilator
                     ;;
        -riscv ) riscv
                 ;;
        -riscv64 ) riscv64
                 ;;
        -llvm ) llvm
                ;;
        -all ) riscv
               riscv64
               llvm
               pocl
               verilator
               ;;
        -h | --help ) show_usage
                      exit
                      ;;
        * )           show_usage
                      exit 1
    esac
    shift
done
