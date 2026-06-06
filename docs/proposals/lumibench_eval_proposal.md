# Evaluating LumiBench on Vortex SimX

## 1. Goal

Bring up the **LumiBench** hardware ray-tracing benchmark suite (IISWC 2023)
on the Vortex **SimX** simulator through the `mesa_vortex` (lavapipe +
vortexpipe) Vulkan stack and the **PRISM RTU**, and produce a reproducible
workload characterization: per-scene IPC plus RTU micro-architectural
counters, with the PRISM RTU measured against Vortex's own SIMT-only baseline.

The deliverable is a runnable harness + a characterization report, not a
claim of parity with discrete-GPU RT cores — the architectures differ by
design (§3).

## 2. What LumiBench is

- **LumiBench: A Benchmark for Hardware Ray Tracing**, Liu, Saed, Chou,
  Grigoryan, Nowicki, Aamodt (UBC), IISWC 2023.
- **16 scenes × 3 shaders**, chosen to span the common ray types/scenes of
  real applications while staying small enough to simulate. PCA over the
  characterization data yields **8 workload clusters** (e.g. reflection-heavy,
  shadow-heavy, primary-ray-bound), so a representative subset can stand in for
  the full suite.
- **Metric: IPC** from the simulator, geometric-mean-averaged across a subset.
- Built on **RayTracingInVulkan** (a `VK_KHR_ray_tracing_pipeline` renderer)
  and run on **Vulkan-Sim v2.0.0**, a functional-Vulkan + GPGPU-Sim-timing
  model of NVIDIA-style RT cores.
- Artifacts: Vulkan-Sim and RayTracingInVulkan repos (github.com/ubc-aamodt-group)
  + a Zenodo capsule (record 8267898) with scenes, configs, and READMEs.

The important structural facts for porting: the workloads drive Vulkan KHR ray
tracing through a **ray-tracing pipeline** (raygen → traversal → closest-hit /
miss / any-hit shaders dispatched via the **Shader Binding Table**), over
host-built **acceleration structures**, and the headline number is **IPC**.

## 3. Why Vortex, and the architectural caveat

Vulkan-Sim models a **discrete-GPU RT core** (MIMD-ish, deep BVH pipelines,
large ray queues). The **PRISM RTU** is a deliberately **mobile-class,
cluster-shared fixed-function walker** behind the SFU (CW-BVH4/6, trap-based
callback dispatch for any-hit/intersection) — the design point of PowerVR
Photon / Arm Immortalis / low-end RTX, not GA10x. So an absolute IPC
comparison against Vulkan-Sim's RT-core numbers is **not** apples-to-apples and
is not the goal.

The meaningful questions LumiBench-on-Vortex answers:

1. **RTU vs SIMT.** For each LumiBench cluster, what speedup does the PRISM RTU
   give over the same traversal run as a pure-SIMT kernel on Vortex
   (`EXT_RTU_ENABLE` on vs off)? This is the apples-to-apples comparison and
   the real value of the study.
2. **Bottleneck characterization.** Where do RT workloads stress the Vortex
   memory hierarchy (RTCache hit rate, L2 pressure), the RTU context pool /
   stack depth, and SIMT divergence on the shader side?
3. **Coverage / design feedback.** Which LumiBench clusters does the current
   RTU serve well, and which expose missing capability (any-hit, procedural,
   deep recursion) — feeding the RTU roadmap and the P1/P2/P3 capability table
   in [rtu_simx_proposal.md](rtu_simx_proposal.md).

## 4. How LumiBench maps onto the Vortex stack

```
  RayTracingInVulkan scene + RT-pipeline shaders (LumiBench)
        │  VK_KHR_ray_tracing_pipeline / VK_KHR_ray_query
        ▼
  lavapipe (Vulkan 1.3) ── GALLIUM_DRIVER=vortexpipe
        │  SPIR-V → NIR → (vp_nir_lower_ray_tracing_to_rtu) → LLVM → .vxbin
        ▼
  vortexpipe ── host AS transcode (vp_launch) → RTU scene
        │
        ▼
  SimX  ── RtuUnit (CW-BVH walker, MemoryEngine, trap dispatch) + SIMT cores
        │  metric: instrs / cycles / IPC + RTU counters
```

Reuses exactly the path validated this cycle by `tests/vulkan/rtquery` and the
RTU-accelerated `tests/vulkan/raytrace` (opaque ray query, 100% on the RTU).

Three feature gaps gate how much of LumiBench runs **on the RTU** today:

- **RT pipelines / SBT.** LumiBench's shaders are `ray_tracing_pipeline`
  (raygen + SBT), which vortexpipe currently **cap-guards to the lavapipe SW
  path** (correct, not RTU-accelerated). Two routes to RTU acceleration:
  (a) complete the **RT-pipeline → RTU dispatcher** (raygen-as-kernel + the
  SimX RTU's trap-based AHS/IS callbacks — the mechanism already exists for the
  native `tests/raytracing` smoke tests), or (b) provide **ray-query ports** of
  the LumiBench shaders (as `raytrace`/`rtquery` do) for the opaque, no-SBT
  subset. The proposal pursues (b) first (fast, already-working path) and
  scopes (a) as the stretch goal.
- **Acceleration-structure build.** lavapipe's host AS builder is
  unimplemented, so the build runs as GPU compute that falls back to CPU. As on
  every GPU, the **AS build is host/driver setup**; only **traversal** runs on
  the RTU. Measurements isolate traversal+shading IPC, excluding the one-time
  CPU build.
- **Scene format.** The Vulkan path currently transcodes the AS to a flat
  **TriList (`scene_kind=0`)**, so RTU traversal is brute-force. Wiring the
  **CW-BVH4** host transcode (the BVH walker already exists in SimX) is a
  prerequisite for representative traversal timing on the larger LumiBench
  scenes and is folded into Phase 2.

## 5. Metric alignment

LumiBench reports **IPC**; SimX reports `instrs/cycles/IPC` natively — direct
alignment. Augment with RTU-specific counters already in the SimX model:
rays launched, BVH-node tests, triangle tests, RTCache hits/misses, context-pool
occupancy, stack-overflow spills, and trap-callback counts. Report per scene and
per the 8 LumiBench clusters (geomean within cluster, per the suite's own
guidance). Always pair each RTU run with the **RTU-off SIMT baseline** so the
headline is a *speedup*, which is architecture-fair, rather than a raw IPC that
would invite an invalid cross-simulator comparison.

## 6. Scope: feasibility buckets

Classify each (scene, shader) the way the Vulkan-conformance plan buckets cases
([vulkan_conformance_proposal.md](vulkan_conformance_proposal.md)):

| Bucket | Meaning | Today |
|---|---|---|
| **A — RTU-accelerated** | opaque ray-query port, fits a Vortex CTA, transcodes to BVH | the primary measurement set |
| **B — SW-fallback** | RT-pipeline/SBT, any-hit, procedural — correct on lavapipe, not on RTU | correctness + a roadmap list; becomes A as the RT-pipeline dispatcher lands |
| **C — out of scope** | features lavapipe/vortexpipe don't support, or scenes too large for SimX wall-clock | excluded with explicit notes |

Start from the **8-cluster representative subset** rather than all 16×3 — it is
both the suite's recommended methodology and a necessary concession to SimX
speed (§8).

## 7. End-to-end pipeline

1. **Fetch LumiBench** from the Zenodo capsule + the RayTracingInVulkan repo;
   pin the commit/record in the report.
2. **Headless render path.** RayTracingInVulkan is interactive; add/confirm an
   offscreen, fixed-frame, deterministic mode (single frame, fixed camera,
   dump framebuffer) so each scene is a one-shot batch job — mirroring the
   `tests/vulkan/*` host harness.
3. **Driver selection** (identical to `tests/vulkan/common.mk`):
   `VK_ICD_FILENAMES=<lvp_icd.json>`, `GALLIUM_DRIVER=vortexpipe`,
   `MESA_VORTEX_XLEN=32`, `VORTEX_DRIVER=simx`, RTU config
   `-DVX_CFG_EXT_RTU_ENABLE`, `MESA_VORTEX_STRICT=1` for the RTU runs (proves
   traversal ran on Vortex, not the CPU fallback) and `=0` for the SW-fallback
   bucket.
4. **Run + collect.** Each scene emits IPC + RTU counters (SimX perf dump) and
   the framebuffer; a reducer aggregates per-cluster geomeans and the RTU-on /
   RTU-off speedup, and image-diffs the framebuffer against the lavapipe SW
   render as the correctness oracle.

## 8. Phased plan

- **Phase 0 — Harness.** `tests/lumibench/` (or an out-of-tree eval dir):
  headless RayTracingInVulkan runner + reducer, reusing the conformance/vulkan
  env machinery. Validate on one tiny scene end-to-end on SimX. *Exit:* one
  command → IPC + RTU counters + image-diff for a scene.
- **Phase 1 — Ray-query subset (bucket A, TriList).** Port/confirm the opaque,
  no-SBT LumiBench shaders to ray query; run the cluster-representative scenes
  small enough for flat-TriList traversal. *Exit:* RTU-on vs RTU-off speedups
  for the primary-ray / shadow clusters.
- **Phase 2 — BVH transcode.** Wire the CW-BVH4 host transcode in `vp_launch`
  so larger scenes get representative traversal timing; re-run Phase 1 scenes
  and add the mid-size clusters. *Exit:* BVH-traversal IPC + RTCache
  characterization across ≥5 clusters.
- **Phase 3 — RT-pipeline dispatcher (stretch, bucket B→A).** Stand up
  raygen-as-kernel + SBT + trap-based AHS/IS on the RTU so the native
  `ray_tracing_pipeline` LumiBench shaders run RTU-accelerated; reclassify
  reflection/any-hit clusters from B to A. *Exit:* the reflection cluster runs
  on the RTU.
- **Phase 4 — Characterization report.** Full 8-cluster results, RTU-vs-SIMT
  speedups, bottleneck analysis (RTCache, context pool, divergence), and an
  RTU design-feedback section tied to the capability table.

## 9. Risks & practical notes

- **SimX wall-clock.** RT scenes are heavy; a full 16×3 sweep on a
  cycle-approximate model is impractical. Mitigate with the 8-cluster subset,
  reduced resolution / sample counts (kept constant across RTU-on/off so the
  ratio is valid), tile/region cropping, and `.vxbin` caching keyed by SPIR-V
  hash. For scale, the same `.vxbin` runs on the **FPGA (xrt)** backend — the
  practical path for the full suite, exactly as in the conformance plan.
- **No cross-simulator IPC claims.** The report must frame results as
  RTU-vs-SIMT-on-Vortex; comparing a Vortex RTU IPC to a Vulkan-Sim RT-core IPC
  is meaningless (different ISA, issue width, memory system) and will be called
  out explicitly to avoid misreading.
- **Fallback masking.** Strict mode + the `device:.*vortex` / `MESA: error`
  gates ensure a "result" is a Vortex-RTU result, not a silent llvmpipe render.
- **Shader feature coverage.** Any-hit / intersection / deep recursion route to
  SW until Phase 3; these are bucket B, reported as coverage gaps, not failures.
- **Determinism / correctness.** Each scene is image-diffed against the
  lavapipe SW oracle; SimX is deterministic, so a mismatch is a real RTU/codegen
  bug, not noise.
- **32-bit only**, out-of-tree builds, tools under `~/tools` — same constraints
  as the rest of `vortex_ci`.

## 10. Deliverables

1. `tests/lumibench/` headless harness (RayTracingInVulkan offscreen runner +
   env wiring + reducer) reusing the existing `tests/vulkan` ICD/strict
   machinery.
2. Phased per-cluster results on SimX: IPC + RTU counters + RTU-vs-SIMT
   speedups, with framebuffer image-diffs vs the lavapipe oracle.
3. (Stretch) RT-pipeline/SBT → RTU dispatcher enabling the native LumiBench
   shaders on the RTU.
4. A versioned characterization report: per-cluster speedups, memory-hierarchy
   and RTU-occupancy bottleneck analysis, and RTU design feedback tied to
   [rtu_simx_proposal.md](rtu_simx_proposal.md).
