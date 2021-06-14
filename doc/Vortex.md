# Vortex Documentation

### Table of Contents

- [Vortex Codebase Layout](https://github.com/vortexgpgpu/vortex-dev/blob/master/doc/Codebase.md)
- [Vortex Microarchitecture and Extended RISC-V ISA](https://github.com/vortexgpgpu/vortex-dev/blob/master/doc/Microarchitecture.md)
- [Vortex Cache Subsystem](https://github.com/vortexgpgpu/vortex-dev/blob/master/doc/Cache_Subsystem.md)
- Vortex Software
- [Vortex Simulation](https://github.com/vortexgpgpu/vortex-dev/blob/master/doc/Simulation.md)
- [FPGA Configuration, Program and Test](https://github.com/vortexgpgpu/vortex-dev/blob/master/doc/FPGA_Startup_Guide.md)
- Debugging
- Useful Links

### Quick Start

Setup Vortex environment:
```
$ export RISCV_TOOLCHAIN_PATH=/opt/riscv-gnu-toolchain
$ export PATH=:/opt/verilator/bin:$PATH
$ export VERILATOR_ROOT=/opt/verilator
```

Test Vortex with different drivers and configurations:
- Run basic driver test with rtlsim driver and Vortex config of 2 clusters, 2 cores, 2 warps, 4 threads

    $ ./ci/blackbox.sh --clusters=2 --cores=2 --warps=2 --threads=4 --driver=rtlsim --app=basic
- Run demo driver test with vlsim driver and Vortex config of 1 clusters, 4 cores, 4 warps, 2 threads

    $ ./ci/blackbox.sh --clusters=1 --cores=4 --warps=4 --threads=2 --driver=vlsim --app=demo
- Run dogfood driver test with simx driver and Vortex config of 4 cluster, 4 cores, 8 warps, 6 threads 

    $ ./ci/blackbox.sh --clusters=4 --cores=4 --warps=8 --threads=6 --driver=simx --app=dogfood