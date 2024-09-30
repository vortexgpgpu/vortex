# Vortex Documentation

## Table of Contents

- [Codebase Layout](codebase.md)
- [Microarchitecture](microarchitecture.md)
- [Cache Subsystem](cache_subsystem.md)
- [Software](software.md)
- [Simulation](simulation.md)
- [Altera FPGA Setup Guide](altera_fpga_guide.md)
- [Xilinx FPGA Setup Guide](xilinx_fpga_guide.md)
- [Debugging](debugging.md)
- [Useful Links](references.md)

## Installation

- For the different environments Vortex supports, [read this document](environment_setup.md).
- To install on your own system, [follow this document](install_vortex.md).

## Quick Start Scenarios

Running Vortex simulators with different configurations:
- Run basic driver test with rtlsim driver and Vortex config of 2 clusters, 2 cores, 2 warps, 4 threads

    $ ./ci/blackbox.sh --driver=rtlsim --clusters=2 --cores=2 --warps=2 --threads=4  --app=basic

- Run demo driver test with opae driver and Vortex config of 1 clusters, 4 cores, 4 warps, 2 threads

    $ ./ci/blackbox.sh --driver=opae --clusters=1 --cores=4 --warps=4 --threads=2 --app=demo

- Run dogfood driver test with simx driver and Vortex config of 4 cluster, 4 cores, 8 warps, 6 threads

    $ ./ci/blackbox.sh --driver=simx --clusters=4 --cores=4 --warps=8 --threads=6  --app=dogfood
