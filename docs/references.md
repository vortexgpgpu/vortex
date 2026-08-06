# References

Foundational external resources on GPU microarchitecture and the domains Vortex
implements. Design documents for Vortex's own subsystems live in
[designs/](designs/).

## GPU architecture fundamentals

- Aamodt, Fung, Rogers — *General-Purpose Graphics Processor Architectures*,
  Morgan & Claypool Synthesis Lectures, 2018. The standard textbook on GPGPU
  microarchitecture: SIMT pipelines, divergence, memory systems.
- Lindholm, Nickolls, Oberman, Montrym — *NVIDIA Tesla: A Unified Graphics and
  Computing Architecture*, IEEE Micro, 2008. The paper that defined the modern
  unified-shader SM organization.
- NVIDIA architecture whitepapers (Fermi, Kepler, Volta, Turing, Ampere, Hopper,
  Blackwell) — [nvidia.com](https://www.nvidia.com/) — the primary public source
  for SM counts, cache geometry, tensor core throughput, and scheduling changes
  per generation.
- AMD *RDNA / CDNA Instruction Set Architecture* reference guides —
  [gpuopen.com](https://gpuopen.com/) — the most complete public GPU ISA
  documentation, including scalar/vector split, wave32/wave64, and LDS design.
- Jia, Maggioni, Staiger, Scarpazza — *Dissecting the NVIDIA Volta GPU
  Architecture via Microbenchmarking*, [arXiv:1804.06826](https://arxiv.org/abs/1804.06826).
  Measured latencies/geometry of caches, register banks, and tensor cores.
- Hennessy & Patterson — *Computer Architecture: A Quantitative Approach*,
  ch. 4 (data-level parallelism, GPUs) and the memory-hierarchy appendices.
- [GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim_distribution) and
  [Accel-Sim](https://accel-sim.github.io/) — cycle-level GPU simulators; their
  documentation and config files encode a detailed public model of NVIDIA-style
  SM microarchitecture. Vortex's Ampere baselines are produced with Accel-Sim.
- Bakhoda, Yuan, Fung, Wong, Aamodt — *Analyzing CUDA Workloads Using a Detailed
  GPU Simulator*, ISPASS 2009. The GPGPU-Sim paper; still the reference for the
  SIMT-core + interconnect + memory-partition decomposition.

## SIMT execution and control-flow divergence

- Fung, Sham, Yuan, Aamodt — *Dynamic Warp Formation and Scheduling for Efficient
  GPU Control Flow*, MICRO 2007. Baseline immediate-post-dominator reconvergence
  stack (the mechanism behind Vortex's split/join) plus warp compaction.
- ElTantawy & Aamodt — *MIMD Synchronization on SIMT Architectures*, MICRO 2016.
  Divergence deadlock and multi-path execution; motivates stackless/subgroup
  convergence schemes.
- NVIDIA Volta whitepaper §"Independent Thread Scheduling" — per-thread PC +
  convergence optimizer, the production alternative to a hardware reconvergence
  stack.
- Meng, Tarjan, Skadron — *Dynamic Warp Subdivision for Integrated Branch and
  Memory Divergence Tolerance*, ISCA 2010.

## Memory coalescing and GPU memory systems

- *CUDA C++ Best Practices Guide*, §memory optimizations —
  [docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) —
  the canonical description of coalescing rules and their performance model.
- Wong, Papadopoulou, Sadooghi-Alvandi, Moshovos — *Demystifying GPU
  Microarchitecture through Microbenchmarking*, ISPASS 2010. How to measure a
  GPU memory hierarchy from the outside.
- Nugteren, van den Braak, Corporaal, Bal — *A Detailed GPU Cache Model Based on
  Reuse Distance Theory*, HPCA 2014.
- Kroft — *Lockup-Free Instruction Fetch/Prefetch Cache Organization*, ISCA 1981.
  The original MSHR paper — the mechanism at the heart of Vortex's non-blocking
  cache banks.
- Liptay — *Structural Aspects of the System/360 Model 85, Part II: The Cache*,
  IBM Systems Journal, 1968. The original sector cache; Seznec's *Decoupled
  Sectored Caches* (ISCA 1994) is the modern treatment behind sectored L2/L3
  designs.

## Tensor cores and matrix engines

- NVIDIA *PTX ISA* reference, `wmma`/`mma`/`wgmma` sections —
  [docs.nvidia.com](https://docs.nvidia.com/cuda/parallel-thread-execution/) —
  the public contract for fragment layouts and warp-group MMA semantics.
- Markidis, Der Chien, Laure, Peng, Vetter — *NVIDIA Tensor Core Programmability,
  Performance & Precision*, IPDPSW 2018.
- Raihan, Goli, Aamodt — *Modeling Deep Learning Accelerator Enabled GPUs*,
  ISPASS 2019. Reverse-engineered tensor-core execution model used by simulators.
- Mishra et al. — *Accelerating Sparse Deep Neural Networks*,
  [arXiv:2104.08378](https://arxiv.org/abs/2104.08378). NVIDIA's 2:4 structured
  sparsity scheme.
- OCP *Microscaling (MX) Formats Specification* v1.0 — block-scaled fp8/fp6/fp4
  formats (MXFP8 etc.).

## Rasterization and the graphics pipeline

- Pineda — *A Parallel Algorithm for Polygon Rasterization*, SIGGRAPH 1988.
  Edge functions — the basis of every modern hardware rasterizer, including
  Vortex's.
- Molnar, Cox, Ellsworth, Fuchs — *A Sorting Classification of Parallel
  Rendering*, IEEE CG&A 1994. Sort-first/middle/last taxonomy for binning
  architectures.
- Fabian Giesen — *A Trip Through the Graphics Pipeline 2011* —
  [fgiesen.wordpress.com](https://fgiesen.wordpress.com/2011/07/09/a-trip-through-the-graphics-pipeline-2011-index/) —
  the best practical walkthrough of what real GPU fixed-function hardware does.
- Akenine-Möller, Haines, Hoffman — *Real-Time Rendering*, 4th ed. —
  [realtimerendering.com](https://www.realtimerendering.com/).

## Texture sampling

- Williams — *Pyramidal Parametrics*, SIGGRAPH 1983. Mipmapping.
- Heckbert — *Survey of Texture Mapping*, IEEE CG&A 1986. Filtering,
  perspective-correct interpolation, LOD selection.
- McCormack, Perry, Farkas, Jouppi — *Feline: Fast Elliptical Lines for
  Anisotropic Texture Mapping*, SIGGRAPH 1999.

## Ray tracing

- Aila & Laine — *Understanding the Efficiency of Ray Traversal on GPUs*,
  HPG 2009 (+ 2012 addendum). The canonical GPU BVH traversal kernels and the
  persistent-threads model; baseline for any hardware traversal unit.
- Meister, Ogaki, Benthin, Doyle, Guthe, Bittner — *A Survey on Bounding Volume
  Hierarchies for Ray Tracing*, Eurographics 2021 STAR. BVH construction and
  traversal design space.
- NVIDIA Turing architecture whitepaper — the public description of RT cores
  (BVH traversal + ray/triangle intersection as fixed function beside the SM).
- Shirley et al. — *Ray Tracing in One Weekend* series —
  [raytracing.github.io](https://raytracing.github.io/) — the algorithms the
  hardware accelerates, in minimal code.
- Khronos *Vulkan Ray Tracing* extension specifications
  (`VK_KHR_acceleration_structure`, `VK_KHR_ray_tracing_pipeline`,
  `VK_KHR_ray_query`) — the API contract Vortex's RTU serves.

## APIs, ISAs, and compilation stacks

- RISC-V ISA specifications (unprivileged + privileged) —
  [riscv.org/specifications](https://riscv.org/technical/specifications/) —
  the base ISA Vortex extends.
- Khronos *OpenCL* specification and registry —
  [registry.khronos.org/OpenCL](https://registry.khronos.org/OpenCL/).
- Khronos *Vulkan* specification — [registry.khronos.org/vulkan](https://registry.khronos.org/vulkan/).
- Khronos *SPIR-V* specification — [registry.khronos.org/SPIR-V](https://registry.khronos.org/SPIR-V/) —
  the IR consumed by both the OpenCL and Vulkan paths.
- AMD *HIP* programming guide —
  [rocm.docs.amd.com](https://rocm.docs.amd.com/projects/HIP/en/latest/) — the
  API surface Vortex supports via [chipStar](https://github.com/CHIP-SPV/chipStar).
- [PoCL](http://portablecl.org/) — the portable OpenCL runtime Vortex's OpenCL
  support builds on.
- [Mesa3D documentation](https://docs.mesa3d.org/) — Gallium and the software
  rasterizer stack behind the mesa-vortex Vulkan driver.
- NVIDIA *CUDA C++ Programming Guide* —
  [docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) —
  the reference programming model (grids/blocks/warps, memory spaces) that
  OpenCL/HIP concepts map onto.
