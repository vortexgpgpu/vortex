# Vortex Documentation

## Table of Contents

- [Codebase Layout](codebase.md)
- [Microarchitecture](microarchitecture.md)
- [Cache Subsystem](cache_subsystem.md)
- [Software](software.md)
- [Simulation](simulation.md)
- [FPGA Setup Guide](fpga_setup.md)
- [Debugging](debugging.md)
- [Useful Links](references.md)


## Installation

- Refer to the install instructions in [README](../README.md). 

## Quick Start Scenarios

Running Vortex simulators with different configurations:
- Run basic driver test with rtlsim driver and Vortex config of 2 clusters, 2 cores, 2 warps, 4 threads

    $ ./ci/blackbox.sh --clusters=2 --cores=2 --warps=2 --threads=4 --driver=rtlsim --app=basic
- Run demo driver test with vlsim driver and Vortex config of 1 clusters, 4 cores, 4 warps, 2 threads

    $ ./ci/blackbox.sh --clusters=1 --cores=4 --warps=4 --threads=2 --driver=vlsim --app=demo
- Run dogfood driver test with simx driver and Vortex config of 4 cluster, 4 cores, 8 warps, 6 threads 

    $ ./ci/blackbox.sh --clusters=4 --cores=4 --warps=8 --threads=6 --driver=simx --app=dogfood