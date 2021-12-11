# Dockerfile for setting up the vortex development environment

FROM ubuntu:18.04

# Install dependencies
RUN apt update && apt install -y \
    git build-essential g++ libfl2 \
    libfl-dev zlibc zlib1g zlib1g-dev \
    ccache libgoogle-perftools-dev numactl perl-doc \
    python3 device-tree-compiler gdb

# Download vortex-toolchain-prebuilt
RUN git clone https://github.com/SantoshSrivatsan24/vortex-toolchain-prebuilt.git /tmp/vortex-toolchain-prebuilt

# Copy riscv-gnu-toolchain
RUN cd /tmp/vortex-toolchain-prebuilt/riscv-gnu-toolchain/ubuntu/bionic; \
    cat riscv-gnu-toolchain.tar.bz2.part* > riscv-gnu-toolchain.tar.bz2; \
    tar -xf riscv-gnu-toolchain.tar.bz2 -C /opt/; 

# Copy riscv64-gnu-toolchain
RUN cd /tmp/vortex-toolchain-prebuilt/riscv64-gnu-toolchain/ubuntu/bionic; \
    cat riscv64-gnu-toolchain.tar.bz2.part* > riscv64-gnu-toolchain.tar.bz2; \
    tar -xf riscv64-gnu-toolchain.tar.bz2 -C /opt/; 

# Copy llvm-riscv
RUN cd /tmp/vortex-toolchain-prebuilt/llvm-riscv/ubuntu/bionic; \
    cat llvm-riscv.tar.bz2.part* > llvm-riscv.tar.bz2; \
    tar -xf llvm-riscv.tar.bz2 -C /opt/; 

# Copy pocl
RUN cd /tmp/vortex-toolchain-prebuilt/pocl/ubuntu/bionic; \
    tar -xf pocl.tar.bz2 -C /opt/; 

# Copy verilator
RUN cd /tmp/vortex-toolchain-prebuilt/verilator/ubuntu/bionic; \
    tar -xf verilator.tar.bz2 -C /opt/; 

# Set environment variables
ENV RISCV_TOOLCHAIN_PATH=/opt/riscv-gnu-toolchain
ENV RISCV64_TOOLCHAIN_PATH=/opt/riscv64-gnu-toolchain
ENV VERILATOR_ROOT=/opt/verilator
ENV PATH=$PATH:/${RISCV_TOOLCHAIN_PATH}/bin:${RISCV64_TOOLCHAIN_PATH}/bin:${RISCV64_TOOLCHAIN_PATH}/riscv64-unknown-elf/bin:${VERILATOR_ROOT}/bin

# Cleanup
RUN rm -rf /tmp/vortex-toolchain-prebuilt

# Set working directory
WORKDIR /home/vortex