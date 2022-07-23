#!/bin/bash

# exit when any command fails
set -e

REPOSITORY=https://github.com/vortexgpgpu/vortex-toolchain-prebuilt/raw/master

DESTDIR="${DESTDIR:=/opt}"

OS="${OS:=ubuntu/bionic}"

riscv()
{
    case $OS in
    "centos/7") parts=$(eval echo {a..h}) ;;
    *)          parts=$(eval echo {a..o}) ;;
    esac
    rm -f riscv-gnu-toolchain.tar.bz2.parta*
    for x in $parts
    do
        wget $REPOSITORY/riscv-gnu-toolchain/$OS/riscv-gnu-toolchain.tar.bz2.parta$x
    done
    cat riscv-gnu-toolchain.tar.bz2.parta* > riscv-gnu-toolchain.tar.bz2
    tar -xvf riscv-gnu-toolchain.tar.bz2
    rm -f riscv-gnu-toolchain.tar.bz2*
    cp -r riscv-gnu-toolchain $DESTDIR
    rm -rf riscv-gnu-toolchain
}

riscv64()
{
    rm -f riscv64-gnu-toolchain.tar.bz2.parta*
    for x in {a..j} 
    do
        wget $REPOSITORY/riscv64-gnu-toolchain/$OS/riscv64-gnu-toolchain.tar.bz2.parta$x
    done
    cat riscv64-gnu-toolchain.tar.bz2.parta* > riscv64-gnu-toolchain.tar.bz2
    tar -xvf riscv64-gnu-toolchain.tar.bz2
    rm -f riscv64-gnu-toolchain.tar.bz2*
    cp -r riscv64-gnu-toolchain $DESTDIR
    rm -rf riscv64-gnu-toolchain
}

llvm()
{
    case $OS in
    "centos/7") parts=$(eval echo {a..g}) ;;
    *)          parts=$(eval echo {a..b}) ;;
    esac
    echo $parts
    rm -f llvm-riscv.tar.bz2.parta*
    for x in $parts
    do
        wget $REPOSITORY/llvm-riscv/$OS/llvm-riscv.tar.bz2.parta$x
    done
    cat llvm-riscv.tar.bz2.parta* > llvm-riscv.tar.bz2
    tar -xvf llvm-riscv.tar.bz2
    rm -f llvm-riscv.tar.bz2*
    cp -r llvm-riscv $DESTDIR
    rm -rf llvm-riscv
}

llvm2()
{
    case $OS in
    "centos/7") parts=$(eval echo {a..g}) ;;
    *)          parts=$(eval echo {a..t}) ;;
    esac
    echo $parts
    rm -f llvm-riscv2.tar.bz2.parta*
    for x in $parts
    do
        wget $REPOSITORY/llvm-riscv/$OS/llvm-riscv2.tar.bz2.parta$x
    done
    cat llvm-riscv2.tar.bz2.parta* > llvm-riscv2.tar.bz2
    tar -xvf llvm-riscv2.tar.bz2
    rm -f llvm-riscv2.tar.bz2*
    cp -r llvm-riscv $DESTDIR
    rm -rf llvm-riscv
}

pocl()
{
    wget $REPOSITORY/pocl/$OS/pocl.tar.bz2
    tar -xvf pocl.tar.bz2
    rm -f pocl.tar.bz2
    cp -r pocl $DESTDIR
    rm -rf pocl
}

pocl2()
{
    wget $REPOSITORY/pocl/$OS/pocl2.tar.bz2
    tar -xvf pocl2.tar.bz2
    rm -f pocl2.tar.bz2
    cp -r pocl $DESTDIR
    rm -rf pocl
}

verilator()
{
    wget $REPOSITORY/verilator/$OS/verilator.tar.bz2
    tar -xvf verilator.tar.bz2
    rm -f verilator.tar.bz2
    cp -r verilator $DESTDIR
    rm -rf verilator
}

usage()
{
    echo "usage: toolchain_install [[-riscv] [-riscv64] [-llvm] [-llvm2] [-pocl] [-pocl2] [-verilator] [-all] [-h|--help]]"
}

while [ "$1" != "" ]; do
    case $1 in
        -pocl ) pocl
                ;;
        -pocl2 ) pocl2
                ;;
        -verilator ) verilator
                     ;;
        -riscv ) riscv
                 ;;
        -riscv64 ) riscv64
                 ;;
        -llvm ) llvm
                ;;
        -llvm2 ) llvm2
                ;;
        -all ) riscv
               riscv64
               llvm2
               pocl2
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