# Vortex Documentation

## Table of Contents

- [Codebase Layout](codebase.md): Summary of repo file tree
- [Microarchitecture](microarchitecture.md): Vortex Pipeline and cache microarchitectural details and reconfigurability
- [Simulation](simulation.md): Details for building and running each simulation driver
- [Contributing](../CONTRIBUTING.md): Process for contributing your own features including repo semantics and testing
- [Debugging](debugging.md): Debugging configurations for each Vortex driver
- [Building the Toolchain from Source](building_toolchain.md): Maintainer-facing build recipes for Verilator, RISC-V GNU, LLVM (with X86 + lld + SPIR-V), compiler-rt, musl, and POCL
- [Design Documents](designs/): Detailed architectural specifications for Vortex subsystems
  - [Atomic Memory Operations](designs/atomic_memory_operations.md)
  - [Command Processor](designs/command_processor_control_plane.md)
  - [DXA (DMA)](designs/dxa_async_copy_multicast.md)
  - [Floating Point Unit](designs/floating_point_unit.md)
  - [Graphics Pipeline](designs/graphics_fixed_function_pipeline.md)
  - [HIP Support](designs/hip_on_vortex_chipstar.md)
  - [TCU (WGMMA)](designs/tensor_core_wgmma_engine.md)
  - [Virtual Memory](designs/virtual_memory_subsystem.md)
  - [Vulkan Support](designs/vortexpipe_architecture.md)
- [gem5 Integration](designs/vortex_gem5_integration.md): Running Vortex inside the gem5 simulator (x86/ARM host CPU + Vortex device over a CP regfile + BAR-mapped VRAM)
