#!/bin/bash

OS_DIR=ubuntu/bionic
SRCDIR=/opt
DESTDIR=.

riscv() 
{
    echo "prebuilt riscv-gnu-toolchain..."
    tar -cvjf -C $SRCDIR riscv-gnu-toolchain.tar.bz2 riscv-gnu-toolchain
    split -b 50M riscv-gnu-toolchain.tar.bz2 "riscv-gnu-toolchain.tar.bz2.part"    
    mv riscv-gnu-toolchain.tar.bz2.part* $DESTDIR/riscv-gnu-toolchain/$OS_DIR
    rm riscv-gnu-toolchain.tar.bz2
}

llvm() 
{
    echo "prebuilt llvm-riscv..."
    tar -cvjf -C $SRCDIR llvm-riscv.tar.bz2 llvm-riscv
    split -b 50M llvm-riscv.tar.bz2 "llvm-riscv.tar.bz2.part"    
    mv llvm-riscv.tar.bz2.part* $DESTDIR/llvm-riscv/$OS_DIR
    rm llvm-riscv.tar.bz2
}

pocl() 
{
    echo "prebuilt pocl..."
    tar -cvjf -C $SRCDIR pocl.tar.bz2 pocl
    mv pocl.tar.bz2 $DESTDIR/pocl/$OS_DIR
}

verilator() 
{
    echo "prebuilt verilator..."
    tar -cvjf -C $SRCDIR verilator.tar.bz2 verilator
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
