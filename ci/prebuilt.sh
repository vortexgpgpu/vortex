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

llvm() 
{
    echo "prebuilt llvm-riscv..."
    tar -C $SRCDIR -cvjf llvm-riscv.tar.bz2 llvm-riscv
    split -b 50M llvm-riscv.tar.bz2 "llvm-riscv.tar.bz2.part"    
    mv llvm-riscv.tar.bz2.part* $DESTDIR/llvm-riscv/$OS_DIR
    rm llvm-riscv.tar.bz2
}

pocl() 
{
    echo "prebuilt pocl..."
    tar -C $SRCDIR -cvjf pocl.tar.bz2 pocl
    mv pocl.tar.bz2 $DESTDIR/pocl/$OS_DIR
}

verilator() 
{
    echo "prebuilt verilator..."
    tar -C $SRCDIR -cvjf verilator.tar.bz2 verilator
    mv verilator.tar.bz2 $DESTDIR/verilator/$OS_DIR
}

usage()
{
    echo "usage: prebuilt [[-riscv] [-llvm] [-pocl] [-verilator] [-all] [-h|--help]]"
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
