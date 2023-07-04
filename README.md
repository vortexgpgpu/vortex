[![Build Status](https://travis-ci.com/vortexgpgpu/vortex.svg?branch=master)](https://travis-ci.com/vortexgpgpu/vortex)
[![codecov](https://codecov.io/gh/vortexgpgpu/vortex/branch/master/graph/badge.svg)](https://codecov.io/gh/vortexgpgpu/vortex)

# Vortex RISC-V GPGPU

Vortex is a full-system RISCV-based GPGPU processor.

## Specifications

- Support RISC-V RV32IMAF & RV64IMAFD ISAs
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

- `doc`: [Documentation](docs/index.md).
- `hw`: Hardware sources.
- `driver`: Host drivers repository.
- `runtime`: Kernel Runtime software.
- `sim`: Simulators repository.
- `tests`: Tests repository.
- `ci`: Continuous integration scripts.
- `miscs`: Miscellaneous resources.

## Build Instructions
### Supported OS Platforms
- Ubuntu 18.04
- Centos 7
### Toolchain Dependencies
- [POCL](http://portablecl.org/)
- [LLVM](https://llvm.org/)
- [RISCV-GNU-TOOLCHAIN](https://github.com/riscv-collab/riscv-gnu-toolchain)
- [Verilator](https://www.veripool.org/verilator)
### Install development tools 
    $ sudo apt-get install build-essential
    $ sudo apt-get install git
### Install Vortex codebase
    $ git clone --recursive https://github.com/vortexgpgpu/vortex.git
    $ cd Vortex
### Install prebuilt toolchain
    $ ./ci/toolchain_install.sh --all

    By default, the toolchain will install to /opt folder. You can install the toolchain to a different directory by overiding DESTDIR.

    $ DESTDIR=$TOOLDIR ./ci/toolchain_install.sh --all
    $ export VORTEX_HOME=$TOOLDIR/vortex
    $ export LLVM_VORTEX=$TOOLDIR/llvm-vortex
    $ export LLVM_POCL=$TOOLDIR/llvm-pocl
    $ export RISCV_TOOLCHAIN_PATH=$TOOLDIR/riscv-gnu-toolchain
    $ export VERILATOR_ROOT=$TOOLDIR/verilator
    $ export PATH=$VERILATOR_ROOT/bin:$PATH 
### Build Vortex sources
    $ make -s
### Quick demo running vecadd OpenCL kernel on 2 cores
    $ ./ci/blackbox.sh --cores=2 --app=vecadd
