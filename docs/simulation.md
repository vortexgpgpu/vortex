# Vortex Simulation Methods 

### RTL Simulation

[Verilator](https://www.veripool.org/projects/verilator/wiki) is a Verilog/SystemVerilog design simulator that converts the Verilog HDL to single- or mult-ithreaded C++/SystemC code to perform the design simulation. An installation guide for Verilator is located [here.](https://www.veripool.org/projects/verilator/wiki/Installing)

### Cycle-Approximate Simulation

SimX is a C++ cycle-level in-house simulator developed for Vortex. The relevant files are located in the `simx` folder. The [readme](README.md) has the most detailed instructions for building and running simX.

- To install on your own system, [follow this document](install_vortex.md).
- For the different Georgia Tech environments Vortex supports, [read this document](environment_setup.md).

### FGPA Simulation

The guide to build the fpga with specific configurations is located [here.](fpga_setup.md) You can find instructions for both Xilinx and Altera based FPGAs.

### How to Test (using `blackbox.sh`)

Running tests under specific drivers (rtlsim,simx,fpga) is done using the script named `blackbox.sh` located in the `ci` folder. Running command `./ci/blackbox.sh --help` from the Vortex root directory will display the following command line arguments for `blackbox.sh`:

- *Clusters* - used to specify the number of clusters (collection of processing elements) within a configuration.
- *Cores* - used to specify the number of cores (processing element containing multiple warps) within a configuration.
- *Warps* - used to specify the number of warps (collection of concurrent hardware threads) within a configuration.
- *Threads* - used to specify the number of threads (smallest unit of computation) within a configuration.
- *L2cache* - used to enable the shared l2cache among the Vortex cores.
- *L3cache* - used to enable the shared l3cache among the Vortex clusters.
- *Driver* - used to specify which driver to run the Vortex simulation (either rtlsim, opae, xrt, simx).
- *Debug* - used to enable debug mode for the Vortex simulation.
- *Perf* - used to enable the detailed performance counters within the Vortex simulation.
- *App* - used to specify which test/benchmark to run in the Vortex simulation. The main choices are vecadd, sgemm, basic, demo, and dogfood. Other tests/benchmarks are located in the `/benchmarks/opencl` folder though not all of them work wit the current version of Vortex.
- *Args* - used to pass additional arguments to the application.

Example use of command line arguments: Run the sgemm benchmark using the opae driver with a Vortex configuration of 1 cluster, 4 cores, 4 warps, and 4 threads.

    $ ./ci/blackbox.sh --clusters=1 --cores=4 --warps=4 --threads=4 --driver=opae --app=sgemm

Output from terminal:
```
Create context
Create program from kernel source
Upload source buffers
Execute the kernel
Elapsed time: 2463 ms
Download destination buffer
Verify result
PASSED!
PERF: core0: instrs=90802, cycles=52776, IPC=1.720517
PERF: core1: instrs=90693, cycles=53108, IPC=1.707709
PERF: core2: instrs=90849, cycles=53107, IPC=1.710678
PERF: core3: instrs=90836, cycles=50347, IPC=1.804199
PERF: instrs=363180, cycles=53108, IPC=6.838518
```

## Additional Quick Start Scenarios

Running Vortex simulators with different configurations and drivers is supported. For example:

- Run basic driver test with rtlsim driver and Vortex config of 2 clusters, 2 cores, 2 warps, 4 threads

    $ ./ci/blackbox.sh --driver=rtlsim --clusters=2 --cores=2 --warps=2 --threads=4  --app=basic

- Run demo driver test with opae driver and Vortex config of 1 clusters, 4 cores, 4 warps, 2 threads

    $ ./ci/blackbox.sh --driver=opae --clusters=1 --cores=4 --warps=4 --threads=2 --app=demo

- Run dogfood driver test with simx driver and Vortex config of 4 cluster, 4 cores, 8 warps, 6 threads

    $ ./ci/blackbox.sh --driver=simx --clusters=4 --cores=4 --warps=8 --threads=6  --app=dogfood