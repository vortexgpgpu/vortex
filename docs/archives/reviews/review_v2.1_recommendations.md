# Vortex Graphics v2 — End-to-End Review Synthesis & v2.1 Recommendations

**Date:** 2026-06-17
**Scope:** cross-cutting synthesis of the eight per-area reviews in this directory.
**Goal frame:** correctness, efficiency, performance, and alignment with mainstream
NVIDIA / AMD / Intel GPU design methodology (the "true GPU" charter,
[gfx_v2_true_gpu_charter.md](../proposals/gfx_v2_true_gpu_charter.md)).

---

## 1. Executive summary

The v2 architecture is **sound and the hard primitives are right** — the ISA-v2 RTU
ABI, the genxml-style FF register emitters, the sort-middle on-device front end, the
windowed `vx_om4`/`vx_tex4` ops — but the program is in a **half-migrated state**: the
host runtime and SimX moved to v2 while the **RTL RASTER and the Mesa driver did not**,
so the v2 path is *demonstrated in tests, not shipped in the driver*, and several
cross-layer ABIs disagree. Two findings are genuinely new: the long-open **§8 bug is
root-caused** (a SimX SFU window-handoff hazard, not scratch residual), and the
**SimX↔RTL RTU timing model is off by ~10×** (the cost model models a 4-wide PE array
the RTL doesn't have).

### Maturity grades

| # | Area | Grade | Headline |
|---|---|:---:|---|
| 1 | [mesa_vortex SW](review_mesa_vortex_sw.md) | **D** | FS still emits legacy `vx_om`/`vx_tex`; driver still orchestrates; no three-tier fallback |
| 2 | [gfx runtime](review_gfx_runtime.md) | **B−** | Emitters/builder/pool are true-GPU-correct, but driver is still a guest on llvmpipe; ABI drift |
| 3 | [gfx kernel](review_gfx_kernel.md) | **B** | Genuinely sort-middle; held back by the RTU coupling + oversized scratch |
| 4 | [RTU runtime](review_rtu_runtime.md) | **B−/C** | Single-leaf "BVH" = O(N) linear scan, no acceleration |
| 5 | [RTU kernel](review_rtu_kernel.md) | **A−** | ISA encodings agree across kernel/SimX/RTL; only doc/robustness gaps |
| 6 | [gfx SimX](review_gfx_simx.md) | **B/B+** | §8 root-caused; RASTER models zero cycles |
| 7 | [RTU SimX](review_rtu_simx.md) | **B** | Faithful oracle; PE cost model is ~10× optimistic vs RTL |
| 8 | [RTU RTL](review_rtu_rtl.md) | **B/B+** | Clean minimal RTU; multi-warp serialization + 300 MHz not yet closed |

---

## 2. Cross-cutting themes

**T1 — v2 is built and tested, but not *shipped* in the driver.** The runtime owns the
device-resident front end (`FrontEndPool` + `DrawCommands`) and it runs end-to-end as one
CP batch *in tests*, but the production UMD (`mesa-vortex/.../vp_raster.cpp`) still does
host `Binning()`, per-draw buffer create/release, and host↔device readback every draw.
The charter's residency/CP pillars are demonstrated, not delivered. (Areas 1, 2.)

**T2 — the v2 ABI migration is half-propagated, so layers disagree.** Host + SimX moved
to the 12-byte `rast_bin_header_t` and `vx_om4`/`vx_tex4`; **RTL RASTER still reads the old
8-byte `rast_tile_header_t`** (`VX_raster_mem.sv`), and **the Mesa FS wrapper still emits
legacy `vx_om`/`vx_tex`**. Plus `PIPE_PRIM_BITS=20` (producer) vs `VX_RASTER_PID_BITS=16`
(consumers). Today only the SimX backend can consume runtime binning output. (Areas 1, 2.)

**T3 — SimX↔RTL timing parity is weak — the "<5% gap" objective is far off.** The RTU
`BoxPe`/`TriPe` cost model assumes a 4-wide parallel PE array at latency 4/6; the RTL has
**one** shared box PE (depth 31) and **one** tri PE (depth 91) streaming 1 prim/cycle.
`BOX_PE`/`TRI_PE`/`CONTEXT_POOL` are **dead config in the RTL**. RASTER in SimX contributes
**zero** modeled cycles. (Areas 6, 7, 8.)

**T4 — the acceleration structures don't accelerate, and the FF datapaths aren't MIMD.**
The host BVH builder emits a single leaf (O(N)/ray, zero box tests); the RTL has 1 box +
1 tri PE, not the MIMD box/tri arrays of real RT cores. (Areas 4, 7, 8.)

**T5 — a structural gfx↔RTU coupling.** The shared graphics window lives in the RTU
register file in SimX, so `vx_om4`/`vx_tex4` *hard-depend on RTU being enabled* — the
shipped `gfx_pipeline_tex` aborts in decode without `-DVX_CFG_EXT_RTU_ENABLE`. (Areas 3, 6.)

---

## 3. §8 multi-draw determinism bug — ROOT-CAUSED (was open across sessions)

**Cause** (`sim/simx/sfu_unit.cpp:331-345`): `vx_om4` sequences 4 sub-pixels, reading the
shared per-(warp,lane) graphics window with `window_get` **once per cycle, mid-sequence**.
Both `vx_om4` and `SETW` carry **no destination register** (`decode.cpp:1003-1010,1043-1050`)
so neither is scoreboarded — the *next* fragment's `SETW` can overwrite the window while
the prior fragment's async OM sub-pixel reads are still draining. The one-batch vs
two-batch submission shapes interleave differently → a deterministic depth-write divergence
(73 scattered pixels; CBUF-diff ≡ ZBUF-diff; depths flip 0xffffffff↔0xff800000 — the far
quad's depth writes land on a different pixel set). Confirmed by experiment: forcing
`OM_MEM_QUEUE_SIZE=1` left the diff unchanged (OmCore R-M-W is innocent); the divergence
enters the OM already wrong, upstream in the SFU window handoff.

**Fix (P0):** snapshot all four sub-pixels' window payload at sub-pixel `F==0` and issue
from the snapshot (remove the per-cycle `window_get`), and/or give `vx_om4` a scoreboard
handle like `vx_tex4`'s rd-sync-handle. **Match whatever ordering the RTL uses** (parity).
This also retires T5's risk surface. → settles gfx-on-simx 6/7 → 7/7.

---

## 4. v2.1 recommendation roadmap (priority order)

### P0 — correctness / blocking (do first)

| ID | Recommendation | Area | Effort |
|---|---|---|---|
| **P0-1** | **§8 fix:** snapshot the `vx_om4` window payload at F==0 (or scoreboard `vx_om4`); match RTL ordering. Verify 146-byte diff → 0. | gfx SimX (`sfu_unit.cpp`) | S |
| **P0-2** | **Mesa FS ABI:** emit `vx_om4` (with SETW window staging) and `vx_tex4`, not legacy `vx_om`/`vx_tex`. The driver is currently broken against the v2 device. | mesa (`vp_nir_to_llvm.c`) | M |
| **P0-3** | **Decouple the gfx window from RTU** (the standalone `GfxWindow` shared by RtuUnit/TexUnit/OmUnit) **or** enable RTU in the gfx caps/Makefiles. Today `vx_om4`/`vx_tex4` abort without RTU. | gfx SimX + kernel | M |
| **P0-4** | **RTL RASTER header migration:** `VX_raster_mem.sv` must read the 12-byte `rast_bin_header_t` (dense + absolute `pids_offset`). Until then RTL can't consume runtime binning. **Release gate.** | RTL RASTER | M |
| **P0-5** | **Reconcile pid width:** `PIPE_PRIM_BITS=20` vs `VX_RASTER_PID_BITS=16` — add a bound-check + diagnostic; pick one width across producer/SimX/RTL. Silent aliasing >65 535 prims today. | gfx runtime + RTL | S |
| **P0-6** | **RTU stack truncation is a correctness bug:** `kBvhStackCap=16` silently truncates deep subtrees → can miss the closest hit. Drive cap from `VX_CFG_RTU_STACK_DEPTH` and model short-stack restart (also closes a parity gap). | RTU SimX (+ RTL restart) | M |
| **P0-7** | **Fix build-contract bug:** `vp_compile.c` reads undefined `VP_VORTEX_HOME` (meson defines `VP_VORTEX_PATH`); also honor the install-tree (pkg-config) contract, not a source tree. | mesa | S |

### P0/P1 — timing parity (the "<5% gap" objective is currently unmet)

| ID | Recommendation | Area | Effort |
|---|---|---|---|
| **P1-1** | **Re-base RTU PE cost model on the RTL:** box depth 31, tri depth 91 (symbolic from FMA=9/FDIV=17); model **one** shared box + one tri PE streaming **1 prim/cycle** with cross-context contention; **drop the ÷4**. Headline parity defect (~10× throughput overstatement). | RTU SimX (`rtu_isect.cpp`, `rtu_core.cpp`) | M |
| **P1-2** | **Charge transform latency** = `36 × instance_descents_delta`. The counter already exists (`rtu_walker.cpp:232`) — ~5 lines. | RTU SimX | S |
| **P1-3** | **Model RASTER cycles in SimX** — the walker currently runs to completion in one tick (zero modeled cycles); drive it one pipe-stage/tick for throughput parity. | gfx SimX | M |
| **P1-4** | **RTU demand-fetch** instead of whole-BVH 16 KB/lane prefetch (`rtu_memory.cpp:139-152`) — over-counts memory cycles by the tree prune ratio. | RTU SimX | M |
| **P1-5** | **Build the cycle-gap measurement harness:** a RTU-traversal-dominated workload that dumps perf and runs on **both** backends (no RTU test calls `vx_device_dump_perf`; `rt_raycast` is rtlsim-deferred). Required to *prove* <5%. | RTU SimX + tests | M |

### P1 — important (correctness-adjacent / charter delivery)

| ID | Recommendation | Area |
|---|---|---|
| **P1-6** | **RTL multi-warp serialization:** add a request queue or implement the stubbed `NUM_CTX>NUM_LANES` ray→context mapping (BRAM state already supports it) so a 2nd warp queues into idle contexts. Unblocks `rt_raycast` on rtlsim (the watchdog-timeout, which is head-of-line serialization, not a deadlock). | RTU RTL |
| **P1-7** | **Confirm 300 MHz on U55C:** land the P6 WNS measurement + production-`NUM_CTX` synth; the binding path is the `f_aligned` 1024-bit barrel-shift → node decode (`VX_rtu_scheduler.sv:250-276`) — precompute the byte-shift in SELECT. Hard constraint currently **unmet** (best WNS −0.028 ns). | RTU RTL |
| **P1-8** | **Real BVH builder** (binned-SAH or LBVH) emitting the internal-node tree the walker already consumes; validate vs the existing hand-packed fixtures + a large-scene test asserting `bvh_box_tests > 0`. | RTU runtime |
| **P1-9** | **Make the driver thin:** switch `vp_raster.cpp` to `FrontEndPool`+`DrawCommands` (on-device binning, no VS readback, no per-draw host round-trip) — the single change that makes the *driver* a true GPU. | mesa + gfx runtime |
| **P1-10** | **Residency allocator** ([gfx_v2_residency_allocator.md](../proposals/gfx_v2_residency_allocator.md)) over one pinned slab; `FrontEndPool`'s 16 separate allocations → one pooled slab; consult `VX_CAPS_VM_PINNED_*`. | gfx runtime |
| **P1-11** | **Watertight intersection:** guard `1/rd[i]` against inf (axis-aligned-ray NaN) in `ray_aabb_intersect`; the single-sided Möller-Trumbore is shared SimX↔RTL but diverges from shipping HW. | RTU SimX + RTL |
| **P1-12** | **Doc/ABI fixes:** scene-header `word2 = scene_bytes` (`raytrace.h:73`, 1 line); update proposal §5.1 to the shipped 3-lane wgather layout; size binning `thist` to stripe width not `T*B`. | RTU runtime + kernel + gfx kernel |

### P2 — alignment / roadmap (true-GPU completeness)

| ID | Recommendation | Area |
|---|---|---|
| **P2-1** | **Three-tier per-unit fallback** (native HW / HW-composed / on-device SIMT `libgfx_sw`); retire the llvmpipe runtime path; keep `MESA_VORTEX_STRICT` mandatory in CI meanwhile. | mesa + gfx kernel |
| **P2-2** | **On-device BVH build** (`vkCmdBuildAccelerationStructures`-style), reusing `rtu_cfg.h` — matches how NVIDIA/AMD/Intel build AS on the GPU. | RTU runtime + kernel |
| **P2-3** | **MIMD box/tri PE arrays + ray-coherency reordering** (NVIDIA SER / Intel TSU / AMD ray binning); wire the §8.9 octant-gather into actual memory ordering (today a no-op counter). | RTU RTL + SimX |
| **P2-4** | **Uniform-register channel** for warp-uniform trace args — lane-packing is already saturated at warp=4 (flags+cull co-pack); a real uniform-register file is the NVIDIA-since-Turing answer. | RTU kernel + core |
| **P2-5** | **Compiler stage coverage:** GS / tessellation / task+mesh in `vp_nir_to_llvm` ([gfx_v2_compiler_stage_coverage.md](../proposals/gfx_v2_compiler_stage_coverage.md)). | mesa |

---

## 5. Suggested sequencing for v2.1

1. **Unblock correctness & green the suite** — P0-1 (§8), P0-3 (RTU/gfx decouple), P0-6
   (stack-miss), P0-7 (build bug). Small, high-value; gets gfx-on-simx to 7/7 and removes
   a real ray-miss bug.
2. **Heal the ABI seams** — P0-2 (mesa `vx_om4`/`vx_tex4`), P0-4 (RTL RASTER header), P0-5
   (pid width). After this, *all four backends* consume one v2 ABI.
3. **Earn the parity claim** — P1-1/2/3/4 + P1-5 (measurement harness). Only then is the
   "<5% gap" objective testable, let alone met.
4. **Deliver the charter in the driver** — P1-9 (thin driver), P1-10 (residency), P1-8
   (real BVH), P1-6/7 (RTL multi-warp + 300 MHz).
5. **Close the true-GPU gap** — the P2 set.

> The single most leveraged item is **P1-1** (RTU PE cost model): without it, the SimX
> timing numbers that drive every RTL/architecture decision are ~10× optimistic.
