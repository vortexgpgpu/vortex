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
TOOLDIR=${TOOLDIR:=/opt}
OSDIR=${OSDIR:=ubuntu/bionic}

OS="${OS:=ubuntu/bionic}"

riscv()
{
    case $OSDIR in
    "centos/7") parts=$(eval echo {a..h}) ;;
    *)          parts=$(eval echo {a..j}) ;;
    esac
    rm -f riscv-gnu-toolchain.tar.bz2.parta*
    for x in $parts
    do
        wget $REPOSITORY/riscv-gnu-toolchain/$OSDIR/riscv-gnu-toolchain.tar.bz2.parta$x
    done
    cat riscv-gnu-toolchain.tar.bz2.parta* > riscv-gnu-toolchain.tar.bz2
    tar -xvf riscv-gnu-toolchain.tar.bz2
    cp -r riscv-gnu-toolchain $TOOLDIR
    rm -f riscv-gnu-toolchain.tar.bz2*    
    rm -rf riscv-gnu-toolchain
}

riscv64()
{
    case $OSDIR in
    "centos/7") parts=$(eval echo {a..h}) ;;
    *)          parts=$(eval echo {a..j}) ;;
    esac
    rm -f riscv64-gnu-toolchain.tar.bz2.parta*
    for x in $parts
    do
        wget $REPOSITORY/riscv64-gnu-toolchain/$OSDIR/riscv64-gnu-toolchain.tar.bz2.parta$x
    done
    cat riscv64-gnu-toolchain.tar.bz2.parta* > riscv64-gnu-toolchain.tar.bz2
    tar -xvf riscv64-gnu-toolchain.tar.bz2
    cp -r riscv64-gnu-toolchain $TOOLDIR
    rm -f riscv64-gnu-toolchain.tar.bz2*    
    rm -rf riscv64-gnu-toolchain
}

llvm-vortex()
{
    case $OSDIR in
    "centos/7") parts=$(eval echo {a..b}) ;;
    *)          parts=$(eval echo {a..b}) ;;
    esac
    echo $parts
    rm -f llvm-vortex.tar.bz2.parta*
    for x in $parts
    do
        wget $REPOSITORY/llvm-vortex/$OSDIR/llvm-vortex.tar.bz2.parta$x
    done
    cat llvm-vortex.tar.bz2.parta* > llvm-vortex.tar.bz2
    tar -xvf llvm-vortex.tar.bz2
    cp -r llvm-vortex $TOOLDIR
    rm -f llvm-vortex.tar.bz2*    
    rm -rf llvm-vortex
}

llvm-pocl()
{
    case $OSDIR in
    "centos/7") parts=$(eval echo {a..b}) ;;
    *)          parts=$(eval echo {a..b}) ;;
    esac
    echo $parts
    rm -f llvm-pocl.tar.bz2.parta*
    for x in $parts
    do
        wget $REPOSITORY/llvm-pocl/$OSDIR/llvm-pocl.tar.bz2.parta$x
    done
    cat llvm-pocl.tar.bz2.parta* > llvm-pocl.tar.bz2
    tar -xvf llvm-pocl.tar.bz2
    cp -r llvm-pocl $TOOLDIR
    rm -f llvm-pocl.tar.bz2*    
    rm -rf llvm-pocl
}

pocl()
{
    wget $REPOSITORY/pocl/$OSDIR/pocl.tar.bz2
    tar -xvf pocl.tar.bz2
    rm -f pocl.tar.bz2
    cp -r pocl $TOOLDIR
    rm -rf pocl
}

verilator()
{
    wget $REPOSITORY/verilator/$OSDIR/verilator.tar.bz2
    tar -xvf verilator.tar.bz2
    cp -r verilator $TOOLDIR
    rm -f verilator.tar.bz2    
    rm -rf verilator
}

sv2v() 
{
    wget $REPOSITORY/sv2v/$OSDIR/sv2v.tar.bz2
    tar -xvf sv2v.tar.bz2
    rm -f sv2v.tar.bz2
    cp -r sv2v $TOOLDIR
    rm -rf sv2v
}

yosys()
{
    case $OSDIR in
    "centos/7") parts=$(eval echo {a..c}) ;;
    *)          parts=$(eval echo {a..c}) ;;
    esac
    echo $parts
    rm -f yosys.tar.bz2.parta*
    for x in $parts
    do
        wget $REPOSITORY/yosys/$OSDIR/yosys.tar.bz2.parta$x
    done
    cat yosys.tar.bz2.parta* > yosys.tar.bz2
    tar -xvf yosys.tar.bz2
    cp -r yosys $TOOLDIR
    rm -f yosys.tar.bz2*    
    rm -rf yosys
}

show_usage()
{
    echo "Install Pre-built Vortex Toolchain"
    echo "Usage: $0 [[--riscv] [--riscv64] [--llvm-vortex] [--llvm-pocl] [--pocl] [--verilator] [--sv2v] [--yosys] [--all] [-h|--help]]"
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
        --all ) pocl
                verilator
                sv2v
                yosys
                llvm-vortex
                riscv
                riscv64                
                ;;
        -h | --help ) show_usage
                exit
                ;;
        * ) show_usage
                exit 1
    esac
    shift
done