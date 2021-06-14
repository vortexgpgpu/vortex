# Vortex Simulation Methods 

### RTL Simulation

[Verilator](https://www.veripool.org/projects/verilator/wiki) is a Verilog/SystemVerilog design simulator that converts the Verilog HDL to single- or mult-ithreaded C++/SystemC code to perform the design simulation. An installation guide for Verilator is located [here.](https://www.veripool.org/projects/verilator/wiki/Installing)

### Cycle-Approximate Simulation

SimX is a C++ cycle-level in-house simulator developed for Vortex. The relevant files are located in the `simX` folder.

### FGPA Simulation

The current target FPGA for simulation is the Arria10 Intel Accelerator Card v1.0. The guide to build the fpga with specific configurations is located [here.](https://github.com/vortexgpgpu/vortex-dev/blob/master/doc/FPGA_Startup_Guide.md)

### How to Test

Running tests under specific drivers (rtlsim,simx,fpga) is done using the script named `blackbox.sh` located in the `ci` folder. Running command `./ci/blackbox.sh --help` from the Vortex root directory will display the following command line arguments for `blackbox.sh`:

- *Clusters* - used to specify the number of clusters (collection of processing elements) within a configuration.
- *Cores* - used to specify the number of cores (processing element containing multiple warps) within a configuration.
- *Warps* - used to specify the number of warps (collection of concurrent hardware threads) within a configuration.
- *Threads* - used to specify the number of threads (smallest unit of computation) within a configuration.
- *L2cache* - used to enable the shard l2cache among the Vortex cores.
- *L3cache* - used to enable the shared l3cache among the Vortex clusters.
- *Driver* - used to specify which driver to run the Vortex simulation (either rtlsim, vlsim, fpga, or simx).
- *Debug* - used to enable debug mode for the Vortex simulation.
- *Perf* - used to enable the detailed performance counters within the Vortex simulation.
- *App* - used to specify which test/benchmark to run in the Vortex simulation. The main choices are vecadd, sgemm, basic, demo, and dogfood. Other tests/benchmarks are located in the `/benchmarks/opencl` folder though not all of them work wit the current version of Vortex.
- *Args* - used to pass additional arguments to the application.

Example use of command line arguments: Run the sgemm benchmark using the vlsim driver with a Vortex configuration of 1 cluster, 4 cores, 4 warps, and 4 threads.

    $ ./ci/blackbox.sh --clusters=1 --cores=4 --warps=4 --threads=4 --driver=vlsim --app=sgemm

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