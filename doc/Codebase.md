# Vortex Codebase

The directory/file layout of the Vortex codebase is as followed:

- `hw`: 
  - `unit_tests`: contains unit test for RTL of cache and queue
  - `syn`: contains all synthesis scripts (quartus and yosys)
    - `quartus`: contains synthesis scripts for Intel Quartus toolchain
    - `opae`: contains synthesis scripts for Intel OPAE FPGA
  - `simulate`: contains RTL simulator (verilator)  
  - `rtl`: contains rtl source code
    - `cache`: contains cache subsystem code
    - `fp_cores`: contains floating point unit code
    - `interfaces`: contains code that handles communication for each of the units of the microarchitecture
    - `libs`: contains general-purpose modules (i.e., buffers, encoders, arbiters, pipe registers)
- `driver`: contains driver software implementation (software that is run on the host to communicate with the vortex processor)
  - `include`: contains vortex.h which has the vortex API that is used by the drivers
  - `opae`: contains code for driver that runs on FPGA
  - `rtlsim`: contains code for driver that runs on local machine (driver built using verilator which converts rtl to c++ binary)
  - `simx`: contains code for driver that runs on local machine (vortex)  
- `runtime`: contains software used inside kernel programs to expose GPGPU capabilities
  - `include`: contains vortex API needed for runtime
  - `linker`: contains linker file for compiling kernels
  - `src`: contains implementation of vortex API (from include folder)
- `simX`: contains simX, the cycle approximate simulator for vortex
  - `tests`: contains tests suite
    - `runtime`: contains vortex runtime tests
    - `driver`: contains vortex driver tests
    - `opencl`: contains opencl tests and benchmarks
    - `riscv`: contains official riscv tests
    - `regression`: contains regression tests
    - `vector`: tests for vector instructions (not yet implemented)
  - `ci`: contain tests to be run during continuous integration (Travis CI)
  - `miscs`: contains miscellaneous stuffs