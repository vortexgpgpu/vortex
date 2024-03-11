# Use the official Ubuntu base image
FROM ubuntu:18.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install required packages
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update && \
    apt-get install -y build-essential zlib1g-dev libtinfo-dev libncurses5 uuid-dev libboost-serialization-dev libpng-dev libhwloc-dev gcc-11 g++-11 && \ 
    apt-get install -y wget git vim python3 python3-pip 

# Manage multiple GCC versions with update-alternatives
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11

# Clone the Vortex repository
#RUN git clone --recursive https://github.com/PSAL-POSTECH/ndp_rtl.git /Vortex
# RUN git clone --recursive https://github.com/vortexgpgpu/vortex.git /Vortex
COPY ./ /Vortex
# Set the working directory to the Vortex directory
WORKDIR /Vortex
RUN git submodule update --init --recursive

# Install Vortex's prebuilt toolchain
# Assuming `TOOLDIR` is meant to be an environment variable determining the installation directory
ENV TOOLDIR=/opt
RUN ./ci/toolchain_install.sh --all && \
    DESTDIR=$TOOLDIR ./ci/toolchain_install.sh --all

# Set up environment variables
ENV VORTEX_HOME=$TOOLDIR/vortex \
    LLVM_VORTEX=$TOOLDIR/llvm-vortex \
    LLVM_POCL=$TOOLDIR/llvm-pocl \
    POCL_CC_PATH=$TOOLDIR/pocl/compiler \
    POCL_RT_PATH=$TOOLDIR/pocl/runtime \
    RISCV_TOOLCHAIN_PATH=$TOOLDIR/riscv-gnu-toolchain \
    VERILATOR_ROOT=$TOOLDIR/verilator \
    SV2V_PATH=$TOOLDIR/sv2v \
    YOSYS_PATH=$TOOLDIR/yosys \
    PATH=$YOSYS_PATH/bin:$SV2V_PATH/bin:$VERILATOR_ROOT/bin:$PATH

# Build Vortex
#RUN make
