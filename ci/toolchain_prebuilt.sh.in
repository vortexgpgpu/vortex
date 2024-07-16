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

TOOLDIR=${TOOLDIR:=@TOOLDIR@}
OSVERSION=${OSVERSION:=@OSVERSION@}

riscv32()
{
    echo "prebuilt riscv32-gnu-toolchain..."
    tar -C $TOOLDIR -cvjf riscv32-gnu-toolchain.tar.bz2 riscv32-gnu-toolchain
    split -b 50M riscv32-gnu-toolchain.tar.bz2 "riscv32-gnu-toolchain.tar.bz2.part"
    mkdir -p ./riscv32-gnu-toolchain/$OSVERSION
    mv riscv32-gnu-toolchain.tar.bz2.part* ./riscv32-gnu-toolchain/$OSVERSION
    rm riscv32-gnu-toolchain.tar.bz2
}

riscv64()
{
    echo "prebuilt riscv64-gnu-toolchain..."
    tar -C $TOOLDIR -cvjf riscv64-gnu-toolchain.tar.bz2 riscv64-gnu-toolchain
    split -b 50M riscv64-gnu-toolchain.tar.bz2 "riscv64-gnu-toolchain.tar.bz2.part"
    mkdir -p ./riscv64-gnu-toolchain/$OSVERSION
    mv riscv64-gnu-toolchain.tar.bz2.part* ./riscv64-gnu-toolchain/$OSVERSION
    rm riscv64-gnu-toolchain.tar.bz2
}

llvm()
{
    echo "prebuilt llvm-vortex2..."
    tar -C $TOOLDIR -cvjf llvm-vortex2.tar.bz2 llvm-vortex
    split -b 50M llvm-vortex2.tar.bz2 "llvm-vortex2.tar.bz2.part"
    mkdir -p ./llvm-vortex/$OSVERSION
    mv llvm-vortex2.tar.bz2.part* ./llvm-vortex/$OSVERSION
    rm llvm-vortex2.tar.bz2
}

libcrt32()
{
    echo "prebuilt libcrt32..."
    tar -C $TOOLDIR -cvjf libcrt32.tar.bz2 libcrt32
    mkdir -p ./libcrt32
    mv libcrt32.tar.bz2 ./libcrt32
}

libcrt64()
{
    echo "prebuilt libcrt64..."
    tar -C $TOOLDIR -cvjf libcrt64.tar.bz2 libcrt64
    mkdir -p ./libcrt64
    mv libcrt64.tar.bz2 ./libcrt64
}

libc32()
{
    echo "prebuilt libc32..."
    tar -C $TOOLDIR -cvjf libc32.tar.bz2 libc32
    mkdir -p ./libc32
    mv libc32.tar.bz2 ./libc32
}

libc64()
{
    echo "prebuilt libc64..."
    tar -C $TOOLDIR -cvjf libc64.tar.bz2 libc64
    mkdir -p ./libc64
    mv libc64.tar.bz2 ./libc64
}

pocl()
{
    echo "prebuilt pocl..."
    tar -C $TOOLDIR -cvjf pocl2.tar.bz2 pocl
    mkdir -p ./pocl/$OSVERSION
    mv pocl2.tar.bz2 ./pocl/$OSVERSION
}

verilator()
{
    echo "prebuilt verilator..."
    tar -C $TOOLDIR -cvjf verilator.tar.bz2 verilator
    mkdir -p ./verilator/$OSVERSION
    mv verilator.tar.bz2 ./verilator/$OSVERSION
}

sv2v()
{
    echo "prebuilt sv2v..."
    tar -C $TOOLDIR -cvjf sv2v.tar.bz2 sv2v
    mkdir -p ./sv2v/$OSVERSION
    mv sv2v.tar.bz2 ./sv2v/$OSVERSION
}

yosys()
{
    echo "prebuilt yosys..."
    tar -C $TOOLDIR -cvjf yosys.tar.bz2 yosys
    split -b 50M yosys.tar.bz2 "yosys.tar.bz2.part"
    mkdir -p ./yosys/$OSVERSION
    mv yosys.tar.bz2.part* ./yosys/$OSVERSION
    rm yosys.tar.bz2
}

show_usage()
{
    echo "Setup Pre-built Vortex Toolchain"
    echo "Usage: $0 [--pocl] [--verilator] [--riscv32] [--riscv64] [--llvm] [--libcrt32] [--libcrt64] [--libc32] [--libc64] [--sv2v] [-yosys] [--all] [-h|--help]"
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
                riscv32
                riscv64
                llvm
                libcrt32
                libcrt64
                libc32
                libc64
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
