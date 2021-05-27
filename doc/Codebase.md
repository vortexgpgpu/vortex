# Vortex Codebase

The directory/file layout of the Vortex codebase is as followed:

- `benchmark`: contains opencl, risc-v, and vector tests
  - `opencl`: contains basic kernel operation tests (i.e. vector add, transpose, dot product)
  - `riscv`: contains official riscv tests which are pre-compiled into binaries
  - `vector`: tests for vector instructions (not yet implemented)
- `ci`: contain tests to be run during continuous integration (Travis CI)
  - driver, opencl, riscv_isa, and runtime tests
- `driver`: contains driver software implementation (software that is run on the host to communicate with the vortex processor)
  - `opae`: contains code for driver that runs on FPGA
  - `rtlsim`: contains code for driver that runs on local machine (driver built using verilator which converts rtl to c++ binary)
  - `simx`: contains code for driver that runs on local machine (vortex)
  - `include`: contains vortex.h which has the vortex API that is used by the drivers
- `runtime`: contains software used inside kernel programs to expose GPGPU capabilities
  - `include`: contains vortex API needed for runtime
  - `linker`: contains linker file for compiling kernels
  - `src`: contains implementation of vortex API (from include folder)
  - `tests`: contains runtime tests
    - `simple`: contains test for GPGPU functionality allowed in vortex
- `simx`: contains simX, the cycle approximate simulator for vortex
- `miscs`: contains old code that is no longer used
- `hw`: 
  - `unit_tests`: contains unit test for RTL of cache and queue
  - `syn`: contains all synthesis scripts (quartus and yosys)
    - `quartus`: contains code to synthesis cache, core, pipeline, top, and vortex stand-alone
  - `simulate`: contains RTL simulator (verilator)
    - `testbench.cpp`: runs either the riscv, runtime, or opencl tests
  - `opae`: contains source code for the accelerator functional unit (AFU) and code which programs the fpga
  - `rtl`: contains rtl source code
    - `cache`: contains cache subsystem code
    - `fp_cores`: contains floating point unit code
    - `interfaces`: contains code that handles communication for each of the units of the microarchitecture
    - `libs`: contains general-purpose modules (i.e., buffers, encoders, arbiters, pipe registers)