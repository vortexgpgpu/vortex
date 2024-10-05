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
- Ubuntu 18.04, 20.04, 22.04, 24.04
- Centos 7
### Toolchain Dependencies
- [POCL](http://portablecl.org/)
- [LLVM](https://llvm.org/)
- [RISCV-GNU-TOOLCHAIN](https://github.com/riscv-collab/riscv-gnu-toolchain)
- [Verilator](https://www.veripool.org/verilator)
- [cvfpu](https://github.com/openhwgroup/cvfpu.git)
- [SoftFloat](https://github.com/ucb-bar/berkeley-softfloat-3.git)
- [Ramulator](https://github.com/CMU-SAFARI/ramulator.git)
- [Yosys](https://github.com/YosysHQ/yosys)
- [Sv2v](https://github.com/zachjs/sv2v)
### Install Vortex codebase
```sh
	git clone --depth=1 --recursive https://github.com/vortexgpgpu/vortex.git
	cd vortex
```
### Install system dependencies
```sh
# ensure dependent libraries are present
sudo ./ci/install_dependencies.sh
```
### Configure your build folder
```sh
    mkdir build
    cd build
    # for 32bit
    ../configure --xlen=32 --tooldir=$HOME/tools
    # for 64bit
    ../configure --xlen=64 --tooldir=$HOME/tools
```
### Install prebuilt toolchain
```sh
   ./ci/toolchain_install.sh --all
```
### set environment variables
```sh
    # should always run before using the toolchain!
    source ./ci/toolchain_env.sh
```
### Building Vortex
```sh
make -s
```
### Quick demo running vecadd OpenCL kernel on 2 cores
```sh
./ci/blackbox.sh --cores=2 --app=vecadd
```

### Common Developer Tips
- Installing Vortex kernel and runtime libraries to use with external tools requires passing --prefix=<install-path> to the configure script.
```sh
../configure --xlen=32 --tooldir=$HOME/tools --prefix=<install-path>
make -s
make install
```
- Building Vortex 64-bit requires setting --xlen=64 configure option.
```sh
../configure --xlen=64 --tooldir=$HOME/tools
```
- Sourcing "./ci/toolchain_env.sh" is required everytime you start a new terminal. we recommend adding "source <build-path>/ci/toolchain_env.sh" to your ~/.bashrc file to automate the process at login.
```sh
echo "source <build-path>/ci/toolchain_env.sh" >> ~/.bashrc
```
- Making changes to Makefiles in your source tree or adding new folders will require executing the "configure" script again without any options to get changes propagated to your build folder.
```sh
../configure
```
- To debug the GPU, the simulation can generate a runtime trace for analysis. See /docs/debugging.md for more information.
```sh
./ci/blackbox.sh --app=demo --debug=3
```
- For additional information, check out the /docs.
