# Vortex RISC-V GPGPU

Vortex currently supported RISC-V RV32I ISA

/benchmarks containts test benchmarks

/docs contains documentation.

/runtime contains the runtime software support for Vortex.

/emulator contains a software emulator for Vortex.

/SimX contains a cycle-approximate simulator for Vortex.

/rtl constains Vortex processor hardware description.

Basic Instructions to run OpenCL Benchmarks on Vortex
-----------------------------------------------------

Install development tools 

    $ sudo apt-get install build-essential
    $ sudo apt-get install git

Install gnu-riscv-tools

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
    $ export RISC_GNU_TOOLS_PATH=$PWD/../drops
    $ ../configure --prefix=$RISC_GNU_TOOLS_PATH --with-arch=rv32im --with-abi=ilp32
    $ make -j`nproc`  
    $ make -j`nproc` build-qemu

Install Vortex 

    $ sudo apt-get install verilator
    $ git clone https://github.gatech.edu/casl/Vortex.git

Build SimX

    $ cd Vortex/simx
    $ make

Run SGEMM OpenCL Benchmark

    $ cd Vortex/benchmarks/opencl/sgemm
    $ make
    $ make run