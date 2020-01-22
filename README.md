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

Basic Instructions to build the OpenCL Compiler for Vortex
----------------------------------------------------------

Build LLVM for RiscV

    $ git clone -b release_90 https://github.com/llvm-mirror/llvm.git llvm
    $ git clone -b release_90 https://github.com/llvm-mirror/clang.git llvm/tools/clang
    $ cd llvm
    $ mkdir build
    $ cd build
    $ export LLVM_RISCV_PATH=$PWD/../drops_riscv
    $ cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=True -DLLVM_USE_SPLIT_DWARF=True -DCMAKE_INSTALL_PREFIX=$LLVM_RISCV_PATH -DLLVM_OPTIMIZED_TABLEGEN=True -DLLVM_BUILD_TESTS=True -DDEFAULT_SYSROOT=$RISC_GNU_TOOLS_PATH/riscv32-unknown-elf -DLLVM_DEFAULT_TARGET_TRIPLE="riscv32-unknown-elf" -DLLVM_TARGETS_TO_BUILD="RISCV" ..
    $ cmake --build . --target install
    $ cp -rf $LLVM_RISCV_PATH $RISC_GNU_TOOLS_PATH

Build pocl for RISCV

    $ git clone https://github.com/pocl/pocl.git
    $ cd pocl
    $ mkdir build
    $ cd build 
    $ export POCL_CC_PATH=$PWD/../drops_riscv_cc
    $ export POCL_RT_PATH=$PWD/../drops_riscv_rt
    $ cmake -G Ninja -DCMAKE_INSTALL_PREFIX=$POCL_CC_PATH -DCMAKE_BUILD_TYPE=Debug -DWITH_LLVM_CONFIG=$RISC_GNU_TOOLS_PATH/bin/llvm-config -DLLC_HOST_CPU= -DNEWLIB_BSP=ON -DNEWLIB_DEVICE_ADDRESS_BIT=32 -DBUILD_TESTS=OFF -DPOCL_DEBUG_MESSAGES=ON ..
    $ cmake -G Ninja -DCMAKE_INSTALL_PREFIX=$POCL_RT_PATH -DCMAKE_BUILD_TYPE=Debug -DOCS_AVAILABLE=OFF -DBUILD_SHARED_LIBS=OFF -DNEWLIB_BSP=ON -DNEWLIB_DEVICE_ADDRESS_BIT=32 -DBUILD_TESTS=OFF -DHOST_DEVICE_BUILD_HASH=basic-riscv32-unknown-elf -DCMAKE_TOOLCHAIN_FILE=../RISCV_newlib.cmake -DENABLE_TRACING=OFF -DENABLE_ICD=OFF -DPOCL_DEBUG_MESSAGES=ON ..
    $ cmake --build . --target install