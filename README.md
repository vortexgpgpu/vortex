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
<<<<<<< HEAD
    $ make run

Basic Instructions to build the OpenCL Compiler for Vortex
----------------------------------------------------------

Build LLVM for RiscV

    $ git clone -b release/10.x https://github.com/llvm/llvm-project.git llvm
    $ cd llvm
    $ mkdir build
    $ cd build
    $ cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DLLVM_ENABLE_PROJECTS="clang" -DBUILD_SHARED_LIBS=True -DLLVM_USE_SPLIT_DWARF=True -DCMAKE_INSTALL_PREFIX=$RISC_GNU_TOOLS_PATH -DLLVM_OPTIMIZED_TABLEGEN=True -DLLVM_BUILD_TESTS=True -DDEFAULT_SYSROOT=$RISC_GNU_TOOLS_PATH/riscv32-unknown-elf -DLLVM_DEFAULT_TARGET_TRIPLE="riscv32-unknown-elf" -DLLVM_TARGETS_TO_BUILD="RISCV" ..
    $ cmake --build . --target install

Build pocl for RISCV

    $ git clone https://github.gatech.edu/casl/pocl.git
    $ cd pocl
    $ mkdir build
    $ cd build 
    $ export POCL_CC_PATH=$PWD/../drops_riscv_cc
    $ export POCL_RT_PATH=$PWD/../drops_riscv_rt
    $ cmake -G Ninja -DCMAKE_INSTALL_PREFIX=$POCL_CC_PATH -DCMAKE_BUILD_TYPE=Debug -DWITH_LLVM_CONFIG=$RISC_GNU_TOOLS_PATH/bin/llvm-config -DNEWLIB_BSP=ON -DNEWLIB_DEVICE_ADDRESS_BIT=32 -DNEWLIB_DEVICE_MARCH=rv32im -DBUILD_TESTS=OFF -DPOCL_DEBUG_MESSAGES=ON ..
    $ cmake --build . --target install
    $ rm -rf *
    $ cmake -G Ninja -DCMAKE_INSTALL_PREFIX=$POCL_RT_PATH -DCMAKE_BUILD_TYPE=Debug -DOCS_AVAILABLE=OFF -DBUILD_SHARED_LIBS=OFF -DNEWLIB_BSP=ON -DNEWLIB_DEVICE_ADDRESS_BIT=32 -DNEWLIB_DEVICE_MARCH=rv32im -DBUILD_TESTS=OFF -DHOST_DEVICE_BUILD_HASH=basic-riscv32-unknown-elf -DCMAKE_TOOLCHAIN_FILE=../RISCV_newlib.cmake -DENABLE_TRACING=OFF -DENABLE_ICD=OFF -DPOCL_DEBUG_MESSAGES=ON ..
    $ cmake --build . --target install
=======
    $ make run
>>>>>>> fpga_synthesis
