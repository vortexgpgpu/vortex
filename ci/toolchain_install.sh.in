#!/bin/bash

# Copyright © 2019-2023
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# exit when any command fails
set -e

REPOSITORY=https://github.com/vortexgpgpu/vortex-toolchain-prebuilt/raw/master
TOOLDIR=${TOOLDIR:=@TOOLDIR@}
OSVERSION=${OSVERSION:=@OSVERSION@}

riscv32()
{
    case $OSVERSION in
    "centos/7") parts=$(eval echo {a..l}) ;;
    "ubuntu/bionic") parts=$(eval echo {a..j}) ;;
    *)          parts=$(eval echo {a..k}) ;;
    esac
    rm -f riscv32-gnu-toolchain.tar.bz2.parta*
    for x in $parts
    do
        wget $REPOSITORY/riscv32-gnu-toolchain/$OSVERSION/riscv32-gnu-toolchain.tar.bz2.parta$x
    done
    cat riscv32-gnu-toolchain.tar.bz2.parta* > riscv32-gnu-toolchain.tar.bz2
    tar -xvf riscv32-gnu-toolchain.tar.bz2
    mkdir -p $TOOLDIR && rm -rf $TOOLDIR/riscv32-gnu-toolchain && mv riscv32-gnu-toolchain $TOOLDIR
    rm -rf riscv32-gnu-toolchain.tar.bz2*
}

riscv64()
{
    case $OSVERSION in
    "centos/7") parts=$(eval echo {a..l}) ;;
    *)          parts=$(eval echo {a..j}) ;;
    esac
    rm -f riscv64-gnu-toolchain.tar.bz2.parta*
    for x in $parts
    do
        wget $REPOSITORY/riscv64-gnu-toolchain/$OSVERSION/riscv64-gnu-toolchain.tar.bz2.parta$x
    done
    cat riscv64-gnu-toolchain.tar.bz2.parta* > riscv64-gnu-toolchain.tar.bz2
    tar -xvf riscv64-gnu-toolchain.tar.bz2
    mkdir -p $TOOLDIR && rm -rf $TOOLDIR/riscv64-gnu-toolchain && mv riscv64-gnu-toolchain $TOOLDIR
    rm -rf riscv64-gnu-toolchain riscv64-gnu-toolchain.tar.bz2*
}

llvm()
{
    case $OSVERSION in
    "centos/7") parts=$(eval echo {a..b}) ;;
    *)          parts=$(eval echo {a..b}) ;;
    esac
    echo $parts
    rm -f llvm-vortex2.tar.bz2.parta*
    for x in $parts
    do
        wget $REPOSITORY/llvm-vortex/$OSVERSION/llvm-vortex2.tar.bz2.parta$x
    done
    cat llvm-vortex2.tar.bz2.parta* > llvm-vortex2.tar.bz2
    tar -xvf llvm-vortex2.tar.bz2
    mkdir -p $TOOLDIR && rm -rf $TOOLDIR/llvm-vortex && mv llvm-vortex $TOOLDIR
    rm -rf llvm-vortex llvm-vortex2.tar.bz2*
}

libcrt32()
{
    wget $REPOSITORY/libcrt32/libcrt32.tar.bz2
    tar -xvf libcrt32.tar.bz2
    mkdir -p $TOOLDIR && rm -rf $TOOLDIR/libcrt32 && mv libcrt32 $TOOLDIR
    rm -rf libcrt32 libcrt32.tar.bz2
}

libcrt64()
{
    wget $REPOSITORY/libcrt64/libcrt64.tar.bz2
    tar -xvf libcrt64.tar.bz2
    mkdir -p $TOOLDIR && rm -rf $TOOLDIR/libcrt64 && mv libcrt64 $TOOLDIR
    rm -rf libcrt64 libcrt64.tar.bz2
}

libc32()
{
    wget $REPOSITORY/libc32/libc32.tar.bz2
    tar -xvf libc32.tar.bz2
    mkdir -p $TOOLDIR && rm -rf $TOOLDIR/libc32 && mv libc32 $TOOLDIR
    rm -rf libc32 libc32.tar.bz2
}

libc64()
{
    wget $REPOSITORY/libc64/libc64.tar.bz2
    tar -xvf libc64.tar.bz2
    mkdir -p $TOOLDIR && rm -rf $TOOLDIR/libc64 && mv libc64 $TOOLDIR
    rm -rf libc64 libc64.tar.bz2
}

pocl()
{
    wget $REPOSITORY/pocl/$OSVERSION/pocl2.tar.bz2
    tar -xvf pocl2.tar.bz2
    mkdir -p $TOOLDIR && rm -rf $TOOLDIR/pocl && mv pocl $TOOLDIR
    rm -rf pocl2 pocl2.tar.bz2
}

verilator()
{
    wget $REPOSITORY/verilator/$OSVERSION/verilator.tar.bz2
    tar -xvf verilator.tar.bz2
    mkdir -p $TOOLDIR && rm -rf $TOOLDIR/verilator && mv verilator $TOOLDIR
    rm -rf verilator verilator.tar.bz2
}

sv2v()
{
    wget $REPOSITORY/sv2v/$OSVERSION/sv2v.tar.bz2
    tar -xvf sv2v.tar.bz2
    mkdir -p $TOOLDIR && rm -rf $TOOLDIR/sv2v && mv sv2v $TOOLDIR
    rm -rf sv2v sv2v.tar.bz2
}

yosys()
{
    case $OSVERSION in
    "centos/7") parts=$(eval echo {a..c}) ;;
    *)          parts=$(eval echo {a..c}) ;;
    esac
    echo $parts
    rm -f yosys.tar.bz2.parta*
    for x in $parts
    do
        wget $REPOSITORY/yosys/$OSVERSION/yosys.tar.bz2.parta$x
    done
    cat yosys.tar.bz2.parta* > yosys.tar.bz2
    tar -xvf yosys.tar.bz2
    mkdir -p $TOOLDIR && rm -rf $TOOLDIR/yosys && mv yosys $TOOLDIR
    rm -rf yosys yosys.tar.bz2* yosys
}

show_usage()
{
    echo "Install Pre-built Vortex Toolchain"
    echo "Usage: $0 [--pocl] [--verilator] [--riscv32] [--riscv64] [--llvm] [--libcrt32] [--libcrt64] [--libc32] [--libc64] [--sv2v] [--yosys] [--all] [-h|--help]"
}

while [ "$1" != "" ]; do
    case $1 in
        --pocl ) pocl
                ;;
        --verilator ) verilator
                ;;
        --riscv32 ) riscv32
                ;;
        --riscv64 ) riscv64
                ;;
        --llvm ) llvm
                ;;
        --libcrt32 ) libcrt32
                ;;
        --libcrt64 ) libcrt64
                ;;
        --libc32 ) libc32
                ;;
        --libc64 ) libc64
                ;;
        --sv2v ) sv2v
                ;;
        --yosys ) yosys
                ;;
        --all ) pocl
                verilator
                llvm
                libcrt32
                libcrt64
                libc32
                libc64
                riscv32
                riscv64
                sv2v
                yosys
                ;;
        -h | --help ) show_usage
                exit
                ;;
        * ) show_usage
                exit 1
    esac
    shift
done