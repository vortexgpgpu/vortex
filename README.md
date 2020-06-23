# Vortex RISC-V GPGPU

Vortex currently supported RISC-V RV32I ISA

/benchmarks containts test benchmarks

/docs contains documentation.

/hw constains hardware sources.

/driver contains the driver software.

/runtime contains the kernel runtime software.

/SimX contains a cycle-approximate simulator for Vortex.

/evaluation contains the synthesis/runtime reports.

Basic Instructions to run OpenCL Benchmarks on Vortex
-----------------------------------------------------

Install development tools 

    $ sudo apt-get install build-essential
    $ sudo apt-get install git

Install gnu-riscv-tools

    $ export RISC_GNU_TOOLS_PATH=/opt/riscv-gnu-toolchain

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
    $ ../configure --prefix=$RISC_GNU_TOOLS_PATH --with-arch=rv32im --with-abi=ilp32
    $ make -j`nproc`  
    $ make -j`nproc` build-qemu

Install Verilator

    You need into build the latest version using the instructions on their website
    $ https://www.veripool.org/projects/verilator/wiki/Installing 

Install Vortex 

    $ git clone https://github.gatech.edu/casl/Vortex.git
    $ cd Vortex
    $ make

Run SGEMM OpenCL Benchmark

    $ cd Vortex/benchmarks/opencl/sgemm
    $ make
    $ make run
