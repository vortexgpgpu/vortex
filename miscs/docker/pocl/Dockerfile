.
#
# Dockerfile for Vortex POCL compiler
#

# set base OS
FROM ubuntu:18.04

# set build variables
ARG LOGIN=user:pass
ARG RISC_GNU_TOOLS_PATH=/opt/riscv_gnu_toolchain
ARG LLVM_HOME=/opt/llvm-10
ARG POCL_CC_PATH=/opt/pocl_cc
ARG POCL_RT_PATH=/opt/pocl_rt  
ARG VORTEX_HOME=/home/vortex 
ARG VORTEX_RUNTIME_INC=$VORTEX_HOME/runtime/sw/include
ARG VORTEX_RUNTIME_LIB=$VORTEX_HOME/runtime/sw/stub/libvortex.so
ARG VORTEX_RUNTIME_PATH=$VORTEX_HOME/runtime

# system update
RUN apt update
RUN apt upgrade -y

# install GNU RISC-V Tools dependencies
RUN apt-get -y install \
  binutils build-essential libtool texinfo \
  gzip zip unzip patchutils curl git \
  make cmake ninja-build automake bison flex gperf \
  grep sed gawk python bc \
  zlib1g-dev libexpat1-dev libmpc-dev \
  libglib2.0-dev libfdt-dev libpixman-1-dev

# install GNU RISC-V Tools dependencies
RUN git clone https://github.com/riscv/riscv-gnu-toolchain /tmp/riscv-gnu-toolchain
RUN cd /tmp/riscv-gnu-toolchain; \
    git submodule update --init --recursive
RUN cd /tmp/riscv-gnu-toolchain; \
    mkdir build
RUN cd /tmp/riscv-gnu-toolchain/build; \
    ../configure --prefix=$RISC_GNU_TOOLS_PATH --with-arch=rv32im --with-abi=ilp32;
RUN cd /tmp/riscv-gnu-toolchain/build; \
    make -j`nproc`; \
    make -j`nproc` build-qemu
RUN rm -rf /tmp/riscv-gnu-toolchain

# install LLVM 10
RUN git clone -b release/10.x https://github.com/llvm/llvm-project.git /tmp/llvm-project
RUN cd /tmp/llvm-project; \
    mkdir build
RUN cd /tmp/llvm-project/build; \
    cmake -G Ninja -DLLVM_ENABLE_PROJECTS="clang" -DBUILD_SHARED_LIBS=True -DLLVM_USE_SPLIT_DWARF=True -DCMAKE_INSTALL_PREFIX=$LLVM_HOME -DLLVM_OPTIMIZED_TABLEGEN=True -DLLVM_BUILD_TESTS=True -DDEFAULT_SYSROOT=$RISC_GNU_TOOLS_PATH/riscv32-unknown-elf -DLLVM_DEFAULT_TARGET_TRIPLE="riscv32-unknown-elf" -DLLVM_TARGETS_TO_BUILD="RISCV" ../llvm
RUN cd /tmp/llvm-project/build; \
    cmake --build . --target install
RUN rm -rf /tmp/llvm-project

# install Vortex
RUN git clone -b fpga_synthesis https://$LOGIN@github.gatech.edu/casl/Vortex.git $VORTEX_HOME
RUN cd $VORTEX_HOME; \
    make -C rtl build_config; \
    make -C runtime build_config; \
    make -C driver/sw/stub

# install POCL
RUN git clone https://$LOGIN@github.gatech.edu/casl/pocl.git /tmp/pocl
RUN cd /tmp/pocl; \
    mkdir build_cc
RUN cd /tmp/pocl/build_cc; \
    cmake -G Ninja -DCMAKE_INSTALL_PREFIX=$POCL_CC_PATH -DOCS_AVAILABLE=ON -DWITH_LLVM_CONFIG=$LLVM_HOME/bin/llvm-config -DENABLE_VORTEX=ON -DVORTEX_RUNTIME_PATH=$VORTEX_RUNTIME_PATH -DVORTEX_RUNTIME_INC=$VORTEX_RUNTIME_INC -DVORTEX_RUNTIME_LIB=$VORTEX_RUNTIME_LIB -DBUILD_TESTS=OFF -DPOCL_DEBUG_MESSAGES=ON ..
RUN cd /tmp/pocl/build_cc; \
    cmake --build . --target install
RUN cd /tmp/pocl; \
    mkdir build_rt 
RUN cd /tmp/pocl/build_rt; \
    cmake -G Ninja -DCMAKE_INSTALL_PREFIX=$POCL_RT_PATH -DOCS_AVAILABLE=OFF -DHOST_DEVICE_BUILD_HASH=riscv32-unknown-unknown-elf -DENABLE_VORTEX=ON -DVORTEX_RUNTIME_PATH=$VORTEX_RUNTIME_PATH -DVORTEX_RUNTIME_INC=$VORTEX_RUNTIME_INC -DVORTEX_RUNTIME_LIB=$VORTEX_RUNTIME_LIB -DBUILD_TESTS=OFF -DPOCL_DEBUG_MESSAGES=ON ..
RUN cd /tmp/pocl/build_rt; \
    cmake --build . --target install
RUN rm -rf /tmp/pocl

# Set the working directory to /mnt.
WORKDIR	/mnt