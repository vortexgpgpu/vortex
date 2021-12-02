# Dockerfile for setting up the development environment for vortex

# Set base OS
FROM ubuntu:18.04

# Install dependencies
RUN apt update && apt install -y \
    # verilator dependencies
    git perl python3 g++ libfl2 libfl-dev \
    zlibc zlib1g zlib1g-dev \
    ccache libgoogle-perftools-dev numactl perl-doc \
    git autoconf flex bison \
    # riscv-gnu-toolchain dependencies
    autoconf automake autotools-dev curl python3 \
    libmpc-dev libmpfr-dev libgmp-dev gawk build-essential \
    bison flex texinfo gperf libtool patchutils bc zlib1g-dev \
    libexpat-dev binutils build-essential libtool texinfo \
    # riscv-isa-sim dependencies
    device-tree-compiler 

# set environment variables
ENV RISCV32=/opt/riscv32
ENV RISCV64=/opt/riscv64
ENV VERILATOR_ROOT=/opt/verilator
ENV POCL_CC_PATH=/opt/pocl/compiler
ENV POCL_RT_PATH=/opt/pocl/runtime
ENV VORTEX_HOME=/home/vortex
ENV PATH=$PATH:${RISCV32}/bin:${RISCV64}/bin:${RISCV64}/riscv64-unknown-elf/bin:${VERILATOR_ROOT}/bin/verilator

# Install riscv-gnu-toolchain
RUN git clone https://github.com/riscv/riscv-gnu-toolchain /tmp/riscv-gnu-toolchain
RUN cd /tmp/riscv-gnu-toolchain; \
    ./configure --prefix=${RISCV64} --with-arch=rv64imfd --with-abi=lp64d; \
    make -j `nproc`
RUN cd /tmp/riscv-gnu-toolchain; \
    make clean; \
    ./configure --prefix=${RISCV32} --with-arch=rv32imf --with-abi=ilp32f; \
    make -j `nproc`
RUN rm -rf /tmp/riscv-gnu-toolchain

# Install riscv-isa-sim 
RUN git clone https://github.com/riscv-software-src/riscv-isa-sim.git /tmp/riscv-isa-sim
RUN cd /tmp/riscv-isa-sim; \
    mkdir build
RUN cd /tmp/riscv-isa-sim/build; \
    ../configure --prefix=${RISCV64}
RUN cd /tmp/riscv-isa-sim/build; \
    make -j `nproc`; \
    make install
RUN rm -rf /tmp/riscv-isa-sim

# Install riscv-pk
RUN git clone https://github.com/riscv-software-src/riscv-pk.git /tmp/riscv-pk
RUN cd /tmp/riscv-pk; \
    mkdir build 
RUN cd /tmp/riscv-pk/build; \
    ../configure --prefix=${RISCV64} --host=riscv64-unknown-elf
RUN cd /tmp/riscv-pk/build; \
    make -j `nproc`; \
    make install
RUN rm -rf /tmp/riscv-pk

# Install verilator
RUN git clone https://github.com/verilator/verilator /tmp/verilator
RUN cd /tmp/verilator; \
    git pull; \
    git checkout v4.040
RUN cd /tmp/verilator; \
    autoconf; \
    ./configure --prefix=/opt/verilator
RUN cd/tmp/verilator; \
    make -j `nproc`; \
    make install
RUN rm -rf /tmp/verilator

# set working directory
RUN mkdir -p /home/vortex
WORKDIR /home/vortex

