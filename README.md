[![Linux Build Status](https://travis-ci.org/github/vortexgpgpu/vortex.png?branch=master)](https://travis-ci.org/github/vortexgpgpu/vortex) 
[![codecov](https://codecov.io/gh/vortexgpgpu/vortex/branch/master/graph/badge.svg)](https://codecov.io/gh/vortexgpgpu/vortex)

# Vortex RISC-V GPGPU

Vortex is a full-system RISCV-based GPGPU processor.

Specifications
--------------

- Support RISC-V RV32I ISA
- Fully scalable: 1 to 16 cores with optional L2 and L3 caches
- OpenCL 1.2 Support 
- FPGA target: Intel Arria 10 @ 200 MHz peak Freq

Directory structure
-------------------

- benchmarks: OpenCL and RISC-V benchmarks
 
- docs: documentation.

- hw: hardware sources.

- driver: driver software.

- runtime: runtime software for kernels.

- simX: Vortex cycle-approximate simulator.

- evaluation: synthesis and performance data.

Basic Installation
------------------

Install development tools 

    $ sudo apt-get install build-essential
    $ sudo apt-get install git

Install gnu-riscv-tools

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

Install Verilator

    You need into build the latest version using the instructions on their website
    $ https://www.veripool.org/projects/verilator/wiki/Installing 

Install Vortex 

    $ git clone --recursive https://github.gatech.edu/casl/Vortex.git
    $ cd Vortex
    $ make

Quick Test running SGEMM kernel

    $ cd /Vortex/benchmarks/opencl/sgemm
    $ make
    $ make run
