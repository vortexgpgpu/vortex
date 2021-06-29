# Vortex Codebase

The directory/file layout of the Vortex codebase is as followed:

- `hw`:     
  - `rtl`: hardware rtl sources
    - `cache`: cache subsystem code
    - `fp_cores`: floating point unit code
    - `interfaces`: interfaces for inter-module communication
    - `libs`: general-purpose modules (i.e., encoder, arbiter, ...)
  - `syn`: synthesis directory
    - `opae`: OPAE synthesis scripts
    - `quartus`: Quartus synthesis scripts    
    - `synopsys`: Synopsys synthesis scripts
    - `yosys`: Yosys synthesis scripts
  - `simulate`: baseline RTL simulator (used by RTLSIM)
  - `unit_tests`: unit tests for some hardware components
- `driver`: Host driver software
  - `include`: Vortex driver public headers
  - `opae`: software driver that uses Intel OPAE
  - `vlsim`: software driver that simulates Full RTL (include AFU)
  - `rtlsim`: software driver that simulates processor RTL
  - `simx`: software driver that uses simX simulator
- `runtime`: Kernel runtime software
  - `include`: Vortex runtime public headers
  - `linker`: linker file for compiling kernels
  - `src`: runtime implementation
- `simX`: cycle approximate simulator for vortex
- `tests`: tests repository.
  - `runtime`: runtime tests
  - `regression`: regression tests
  - `riscv`: RISC-V standard tests
  - `opencl`: opencl benchmarks and tests
- `ci`: continuous integration scripts
- `miscs`: miscellaneous resources.
