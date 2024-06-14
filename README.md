[![Build Status](https://travis-ci.com/vortexgpgpu/vortex.svg?branch=master)](https://travis-ci.com/vortexgpgpu/vortex)

# Vortex GPGPU

Vortex is a full-stack open-source RISC-V GPGPU.

## Specifications

- Support RISC-V RV32IMAF and RV64IMAFD
- Microarchitecture:
    - configurable number of cores, warps, and threads.
    - configurable number of ALU, FPU, LSU, and SFU units per core.
    - configurable pipeline issue width.
    - optional local memory, L1, L2, and L3 caches.
- Software:
    - OpenCL 1.2 Support.
- Supported FPGAs:
    - Altera Arria 10
    - Altera Stratix 10
    - Xilinx Alveo U50, U250, U280
    - Xilinx Versal VCK5000

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
More detailed build instructions can be found [here](docs/install_vortex.md).
### Supported OS Platforms
- Ubuntu 18.04, 20.04
- Centos 7
### Toolchain Dependencies
- [POCL](http://portablecl.org/)
- [LLVM](https://llvm.org/)
- [RISCV-GNU-TOOLCHAIN](https://github.com/riscv-collab/riscv-gnu-toolchain)
- [Verilator](https://www.veripool.org/verilator)
- [FpNew](https://github.com/pulp-platform/fpnew.git)
- [SoftFloat](https://github.com/ucb-bar/berkeley-softfloat-3.git)
- [Ramulator](https://github.com/CMU-SAFARI/ramulator.git)
- [Yosys](https://github.com/YosysHQ/yosys)
- [Sv2v](https://github.com/zachjs/sv2v)
### Install development tools
    $ sudo apt-get install build-essential
    $ sudo apt-get install binutils
    $ sudo apt-get install python
    $ sudo apt-get install uuid-dev
    $ sudo apt-get install git
### Install Vortex codebase
    $ git clone --depth=1 --recursive https://github.com/vortexgpgpu/vortex.git
    $ cd Vortex
### Configure your build folder
    # By default, the toolchain default install location is the /opt folder and can be overridden by setting --tooldir.
    $ mkdir build
    $ cd build
    $ ../configure --xlen=32 --tooldir=$HOME/tools
### Install prebuilt toolchain
    $ ./ci/toolchain_install.sh --all
### set environment variables
    # should always run before using the toolchain!
    $ source ./ci/toolchain_env.sh
### Building Vortex
    $ make -s
### Quick demo running vecadd OpenCL kernel on 2 cores
    $ ./ci/blackbox.sh --cores=2 --app=vecadd

### Common Developer Tips
- Installing Vortex kernel and runtime libraries to use with external tools requires passing --prefix=<install-path> to the configure script.
    ```sh
    $ ../configure --xlen=32 --tooldir=$HOME/tools --prefix=<install-path>
    $ make -s
    $ make install
    ``````
- Building Vortex 64-bit simply requires using --xlen=64 configure option.
    ```sh
    $ ../configure --xlen=32 --tooldir=$HOME/tools
    ```
- Sourcing "./ci/toolchain_env.sh" is required everytime you start a new terminal. we recommend adding "source <build-path>/ci/toolchain_env.sh" to your ~/.bashrc file to automate the process at login.
    ```sh
    $ echo "source <build-path>/ci/toolchain_env.sh" >> ~/.bashrc
    ```
- Making changes to Makefiles in your source tree or adding new folders will require executing the "configure" script again to get it propagated into your build folder.
    ```sh
    $ ../configure
    ```
- To debug the GPU, you can generate a "run.log" trace. see /docs/debugging.md for more information.
    ```sh
    $ ./ci/blackbox.sh --app=demo --debug=3
    ```
- For additional information, check out the /docs.
