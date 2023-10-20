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

llvm-vortex() 
{
    echo "prebuilt llvm-vortex..."
    tar -C $SRCDIR -cvjf llvm-vortex.tar.bz2 llvm-vortex
    split -b 50M llvm-vortex.tar.bz2 "llvm-vortex.tar.bz2.part"    
    mv llvm-vortex.tar.bz2.part* $DESTDIR/llvm-vortex/$OS_DIR
    rm llvm-vortex.tar.bz2
}

llvm-pocl() 
{
    echo "prebuilt llvm-pocl..."
    tar -C $SRCDIR -cvjf llvm-pocl.tar.bz2 llvm-pocl
    split -b 50M llvm-pocl.tar.bz2 "llvm-pocl.tar.bz2.part"    
    mv llvm-pocl.tar.bz2.part* $DESTDIR/llvm-pocl/$OS_DIR
    rm llvm-pocl.tar.bz2
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

sv2v() 
{
    echo "prebuilt sv2v..."
    tar -C $SRCDIR -cvjf sv2v.tar.bz2 sv2v
    mv sv2v.tar.bz2 $DESTDIR/sv2v/$OS_DIR
}

yosys()
{
    echo "prebuilt yosys..."
    tar -C $SRCDIR -cvjf yosys.tar.bz2 yosys
    split -b 50M yosys.tar.bz2 "yosys.tar.bz2.part"    
    mv yosys.tar.bz2.part* $DESTDIR/yosys/$OS_DIR
    rm yosys.tar.bz2
}

show_usage()
{
    echo "Setup Pre-built Vortex Toolchain"
    echo "Usage: $0 [[--riscv] [--llvm-vortex] [--llvm-pocl] [--pocl] [--verilator] [--sv2v] [-yosys] [--all] [-h|--help]]"
}

while [ "$1" != "" ]; do
    case $1 in
        --pocl ) pocl
                ;;
        --verilator ) verilator
                ;;
        --riscv ) riscv
                ;;
        --riscv64 ) riscv64
                ;;
        --llvm-vortex ) llvm-vortex
                ;;
        --llvm-pocl ) llvm-pocl
                ;;
        --sv2v ) sv2v
                ;;
        --yosys ) yosys
                ;;
        --all ) riscv
                riscv64
                llvm-vortex
                llvm-pocl
                pocl
                verilator
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
