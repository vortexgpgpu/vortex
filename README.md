[![Build Status](https://travis-ci.com/vortexgpgpu/vortex.svg?branch=master)](https://travis-ci.com/vortexgpgpu/vortex)
[![codecov](https://codecov.io/gh/vortexgpgpu/vortex/branch/master/graph/badge.svg)](https://codecov.io/gh/vortexgpgpu/vortex)

# Vortex RISC-V GPGPU

Vortex is a full-system RISCV-based GPGPU processor.

## Specifications

- Support RISC-V RV32IMF ISA
- Performance: 
    - 1024 total threads running at 250 MHz
    - 128 Gflops of compute bandwidth
    - 16 GB/s of memory bandwidth
- Scalability: up to 64 cores with optional L2 and L3 caches
- Software: OpenCL 1.2 Support 
- Supported FPGAs: 
    - Intel Arria 10
    - Intel Stratix 10

## Directory structure

- `doc`: [Documentation](doc/Vortex.md).

- `hw`: Hardware sources.

- `driver`: Host driver software.

- `runtime`: Kernel Runtime software.

- `simX`: Cycle-approximate simulator.

- `tests`: Tests repository.

- `ci`: Continuous integration scripts.

- `miscs`: Miscellaneous resources.

## Basic Installation

### Install development tools 

    $ sudo apt-get install build-essential
    $ sudo apt-get install git

### Install gnu-riscv-tools

    $ export RISCV_TOOLCHAIN_PATH=/opt/riscv-gnu-toolchain

    $ sudo apt-get -y install \
        binutils build-essential libtool texinfo \
        gzip zip unzip patchutils curl git \
        make cmake ninja-build automake bison flex gperf \
        grep sed gawk python bc \
        zlib1g-dev libexpat1-dev libmpc-dev \
        libglib2.0-dev libfdt-dev libpixman-1-dev 
    $ git clone https://github.com/riscv/riscv-gnu-toolchain
    $ cd riscv-gnu-toolchain
    $ git submodule update --init --recursive
    $ mkdir build
    $ cd build    
    $ ../configure --prefix=$RISCV_TOOLCHAIN_PATH --with-arch=rv32im --with-abi=ilp32
    $ make -j`nproc`  
    $ make -j`nproc` build-qemu

### Install Verilator

    You need into build the latest version using the instructions on their website
    $ https://www.veripool.org/projects/verilator/wiki/Installing 

### Install Vortex 

    $ git clone --recursive https://github.com/vortexgpgpu/vortex.git
    $ cd Vortex
    $ make

### Quick Test running OpenCL vecadd sample on 2 cores

    $ ./ci/blackbox.sh --cores=2 --app=vecadd
