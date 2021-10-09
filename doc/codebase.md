# Vortex Codebase

The directory/file layout of the Vortex codebase is as followed:

- `hw`:     
  - `rtl`: hardware rtl sources
    - `cache`: cache subsystem code
    - `fp_cores`: floating point unit code
    - `interfaces`: interfaces for inter-module communication
    - `libs`: general-purpose RTL modules
  - `syn`: synthesis directory
    - `opae`: OPAE synthesis scripts
    - `quartus`: Quartus synthesis scripts    
    - `synopsys`: Synopsys synthesis scripts
    - `modelsim`: Modelsim synthesis scripts
    - `yosys`: Yosys synthesis scripts
  - `unit_tests`: unit tests for some hardware components
- `driver`: host drivers repository
  - `include`: Vortex driver public headers
  - `stub`: Vortex stub driver library
  - `fpga`: software driver that uses Intel OPAE FPGA
  - `asesim`: software driver that uses Intel ASE simulator
  - `vlsim`: software driver that uses vlsim simulator
  - `rtlsim`: software driver that uses rtlsim simulator
  - `simx`: software driver that uses simX simulator
- `runtime`: kernel runtime software
  - `include`: Vortex runtime public headers
  - `linker`: linker file for compiling kernels
  - `src`: runtime implementation
- `sim`: 
  - `vlsim`: AFU RTL simulator
  - `rtlsim`: processor RTL simulator
  - `simX`: cycle approximate simulator for vortex
- `tests`: tests repository.
  - `runtime`: runtime tests
  - `regression`: regression tests
  - `riscv`: RISC-V standard tests
  - `opencl`: opencl benchmarks and tests
- `ci`: continuous integration scripts
- `miscs`: miscellaneous resources.
