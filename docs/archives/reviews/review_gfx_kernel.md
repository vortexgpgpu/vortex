# v2 Review: prism_v3 graphics kernel stack

**Date:** 2026-06-17
**Reviewer scope:** `sw/gfx/{pipe_frontend.h, setup_math.h, pipe_abi.h, setup_types.h}`,
`sw/common/gfx_frontend_abi.h`, `sw/kernel/include/vx_graphics.h`, `sw/common/gfx_sw.h`,
`sw/runtime/graphics.cpp` (FrontEndPool/DrawCommands), and the gfx test kernels under
`tests/graphics/*` + `tests/regression/gfx_*_kernel`.
**Method:** static read of the full front-end stack against the charter / binning-redesign /
software-fallback / custom1-ISA proposals; partial sim attempt (see note in §2).

---

## 1. Overall assessment (+ maturity grade)

The graphics kernel stack is **well-structured, genuinely sort-middle, and faithful to the
gfx-v1 oracle**. The setup math (`setup_math.h`) is a clean, single-source port of host
`Binning()` (clip → HDC edge eq → Q15.16 fixed / Q7.24 attribs → screen bbox), shared
verbatim host/device so it isolates exactly the codegen/FP axis. The two-entry, nine-stage
CP-sequenced front end (`setup_k` stages 0–2, `binning_k` stages 3–8) is a legitimate
sort-middle realization: per-prim work is grid-strided and barrier-free, the three reductions
are single-CTA, and the bin-stripe histogram/scatter is the cheap multi-CTA replacement for a
global key sort (binning-redesign §6.1). The SW output-merger (`gfx_sw.h`) is a real
single-source-of-truth: the *same* code backs the host FF model and the device fallback.

Weaknesses are concentrated in (a) the still-open §8 multi-draw determinism failure, (b) a
load-bearing but undocumented coupling of the OM/TEX4 window ABI onto the **RTU** funct3=5 row
(the test cannot run without `VX_CFG_EXT_RTU_ENABLE`, and the current `build32_amo` build of
`gfx_pipeline_tex` traps in the decoder because RTU is off), and (c) efficiency debt in the
bin-stripe path (every CTA scans all keys, O(G·K)).

**Maturity grade: B.** Correct and well-architected for the validated single-draw paths;
held back from A by the unresolved §8 multi-draw diff and the implicit RTU dependency that
makes the headline test un-runnable in the default config.

---

## 2. Correctness findings

> **Sim note.** I attempted to reproduce §8 on simx. The pre-built `gfx_pipeline_tex` in
> `build32_amo` **aborts in `vortex::Decoder::decode`** immediately after `quad: tris=2 P=2
> tiles=1` — the fragment kernel issues `vx_rt_set` (RTU **SETW**, CUSTOM1 funct3=5) to stage
> the `vx_om4` window, but the build's `VX_CFG_EXT_RTU_ENABLED=0` (from `VX_config.toml`), so
> the op decodes as illegal. Rebuilding the kernel with `-DVX_CFG_EXT_RTU_ENABLE` then trips
> `"TEX/OM/RASTER not all supported!"` because the shared `libsimx.so` reports ISA caps from
> the toml (RTU=false) and a full RTU-enabled simx rebuild would mutate the shared tree. I
> restored the original kernel build and completed the §8 analysis statically.

| # | file:line | issue | severity |
|---|-----------|-------|----------|
| C1 | `tests/graphics/gfx_pipeline_tex/Makefile` (+ `kernel.cpp:2` `#include <vx_raytrace.h>`, `:45` `vx_rt_set`) | The OM-window staging path (`vx_rt_set` = RTU SETW, CUSTOM1 funct3=5) makes **OM4 hard-depend on RTU**, but the test Makefile does not add `-DVX_CFG_EXT_RTU_ENABLE` and `VX_config.toml` defaults RTU off → the shipped build **traps in the decoder**. The headline §8 test is un-runnable in the default config. | High (blocks validation) |
| C2 | §8 root cause (hypothesis below) | one-batch vs two-batch depth-multi-draw diff ~146 B, edge-localized | High |
| C3 | `vx_graphics.h:56,72` vs `gfx_v2_custom1_isa_allocation.md` (funct3 table) | `vx_tex4_single`/`vx_tex4_quad` are encoded on **funct3=5**, the row the ISA-allocation doc assigns to **RTU** (set/get/trace/wait). TEX4 and OM-window staging therefore co-occupy the RTU row; decoding correctness depends on funct7/funct2 sub-field discipline that the doc flags as "without an RTU collision" but the headers do not assert. Worth an explicit decode-table cross-check. | Medium |
| C4 | `pipe_frontend.h:212,245` vs `gfx_v2_tile_binning_redesign.md:187` | Doc specifies the bin stripe as **modulo-interleaved** (`bin_id % G == cta`); the code implements a **contiguous block** stripe (`lo = blockIdx.x*bin_stripe`). Both are valid disjoint partitions and produce identical output, but the doc/impl mismatch is a latent trap for anyone reasoning about load balance or RASTER consume-stripe matching. | Low (doc drift) |
| C5 | `setup_math.h:60` | Near-plane crossing lerp `t = fc/(fc-fn)` is unguarded; safe only because the branch is taken **only when `cur_in != nxt_in`** (so `fc` and `fn` have strictly opposite signs and `fc-fn != 0`). Correct as written, but a one-line comment asserting the invariant would harden it against future edits. | Low |
| C6 | `pipe_frontend.h:130` | `meta[0]=acc` and `offset[ntri]=acc` are written by `tid==0` **after** the in-place exclusive scan of `tsum`, then read by all threads after `__syncthreads()` (line 132). Correct. (Verified — not a bug; noted because it is the kind of single-writer/all-reader pattern that the §8 investigation must rule out, and it is sound here.) | — |

### §8 root-cause hypothesis (C2)

**Setup recap.** Path D draws a **near centered quad** (`gen_quad_centered(0.5,-0.5)`, screen-z
≈ 0.25) then a **far full-screen quad** (z=0 → screen-z 0.5) into a shared resident
color+depth attachment with `DEPTH_FUNC_LESS`, depth writemask on. `img_frame_one` (both
draws in ONE CP batch) is compared to `img_frame_seq` (the same two draws as two host-submitted
batches). Diff ≈146 bytes, edge-localized.

**What I ruled out (with confidence).** For the **128×128** render target with
`PIPE_BIN_LOG=7`, the dense bin grid is `bin_cols=bin_rows=1` → **B = 1, a single bin**. The
entire bin-stripe machinery (`thist[T*B]`, `bincount[B]`, `binbase[B]`, contiguous stripes)
degenerates: every key maps to bin 0, CTA 0 owns `[0,1)`, all other CTAs get an empty stripe.
With B=1 there is **no striping subtlety to get wrong**, and I verified every binning scratch
buffer is fully reinitialized per draw:

- `thist[tid*B+b]` is zeroed for every `tid∈[0,T)`, `b∈[lo,hi)` by the owning CTA
  (`pipe_frontend.h:216`), and every bin `b∈[0,B)` is owned by exactly one CTA, so the entire
  reachable `thist` footprint is rewritten each draw — no residual.
- `keys[0..meta[1])` are written contiguously by BEMIT (`:204-207`); `boffset` is the exact
  exclusive scan of `bcount`, so there are **no gaps** for a stale key to survive in. BHIST/
  BSCATTER read exactly `[0,meta[1])`.
- `meta[0]` (P, prim count) and `meta[1]` (K, key count) are recomputed every draw by SCAN
  and BSCAN; `bincount`/`binbase` are recomputed in BHIST/BBASE; the dense `prim`/`bbox`
  outputs are written `[0,P)` and read `[0,P)`. No accumulator survives a draw.
- The per-prim depth attribs (`attribs.z = delta(ps.z…)`) for the near quad are
  `(0,0,0.25)` and for the far quad `(0,0,0.5)`, written fresh by EMIT each draw — **no
  edge-localized residual in the front-end primbuf**.

**Therefore: for this 128×128 test the front-end binning scratch cannot be the differentiator
between one-batch and two-batch**, because the only thing that differs between those two runs
is CP batching, and the per-draw front-end output is a pure deterministic function of the
(identical) vertex input and is fully reinitialized regardless of batching.

**Where the ~146-byte edge-localized diff most plausibly originates (ranked):**

1. **(Primary) Depth read-after-write across the two in-batch draws — an FF↔FF (OM-write →
   next-draw-OM-read) ordering hazard on the resident depth buffer, not front-end scratch.**
   In the **two-batch** reference the near draw's OM depth writes drain through the OM AXI
   master and ocache *and the host event completes* before the far draw's OM reads them. In the
   **one-batch** case the two draws are back-to-back in the ring; correctness now relies on the
   inter-launch drain *also* fully ordering the OM unit's outstanding depth-buffer writes
   against the next draw's OM depth-test reads. At partially-covered **edge quads** of the
   near centered quad, the depth that survives is exactly the boundary where the far quad's
   LESS-test outcome flips — so any under-ordering shows up as a thin edge band of a few hundred
   bytes, matching "≈146 B, edge-localized." This is consistent with the §8 commit message
   (`7cb688c8`) noting the OM and its memory-write path were the sensitive surface. The task's
   prior isolation ("ocache write-through, reset invalidates every line, flush after every
   launch") rules out *cross-launch cache staleness* but does **not** rule out *intra-batch
   OM-write/OM-read completion ordering*, which is a different mechanism.

2. **(Secondary) Edge-quad depth extrapolation in the fragment shader.** `INTERPOLATE(z,
   attribs.z)` (`kernel.cpp:80`) uses RASTER bcoord gradients that, on edge quads, extrapolate
   the covered fragment's barycentric outside [0,1]; a slightly-negative interpolated z →
   `depth[i].data()` is a large negative Q7.24 int → `(uint32_t)` wrap → near-0xFFFFFF depth
   after the OM mask. This is genuinely **edge-localized**, but it is *identical* between
   one-batch and two-batch, so it cannot by itself explain `img_frame_one != img_frame_seq`
   — it can only *amplify* hazard (1) by making the boundary depth values maximally sensitive.

3. **(If the team's "front-end scratch" isolation is firm on a *larger* render target where
   B > 1)** then the one remaining front-end suspect is **`thist` row coverage tied to
   `T = blockDim.x`** in BHIST/BSCATTER (`:216,220,250`): the reduction always sums `t∈[0,T)`
   rows. This is self-consistent for B>1 too (every CTA zeros its owned columns for all T
   rows), so I could not find an actual over-read; but it is the only place where an
   implicit-zero assumption on shared device scratch exists, and it is the line to instrument
   first if §8 is re-confirmed at a resolution with B>1 (e.g. ≥256×256, which gives B≥4).

**Recommended next diagnostic step:** run §8 at **256×256** (B=4, real striping) vs **128×128**
(B=1). If the diff *vanishes* at 128×128-equivalent single-bin but *appears* only with B>1, the
bug is the bin-stripe `thist` path (hypothesis 3). If the diff persists at B=1, it is **not**
front-end scratch and the depth-RAW ordering (hypothesis 1) is confirmed — the fix is an
explicit OM-drain/barrier (or depth-buffer flush) between in-batch draws, not a kernel change.
A second cheap discriminator: temporarily memset all pool scratch between the two draws in the
one-batch path; if the diff is unchanged, scratch residual is excluded.

---

## 3. Efficiency findings

| # | file:line | issue | note |
|---|-----------|-------|------|
| E1 | `pipe_frontend.h:218,252` | **Every CTA scans all K keys** to filter its bin stripe → O(G·K) redundant reads. Acknowledged in `gfx_v2_tile_binning_redesign.md:197` as a known trade vs a global sort. Fine for small G; for many cores this is the dominant binning cost and wants the doc's "coarse pre-pass" or a per-key owning-CTA dispatch. | P2 |
| E2 | `graphics.cpp:467` `thist = T*B*4` | The histogram scratch is **T·B words** (block_dim × bins). For a large render target (B in the thousands) × a full occupancy block this is multi-MB of scratch that is mostly zero (each prim touches few bins). A CSR/compressed histogram or a per-CTA-local `stripe`-width allocation (not full B) would cut this dramatically — BHIST only ever touches `[lo,hi)` columns, so the `*B` stride wastes the rest. | P1 (memory) |
| E3 | `pipe_frontend.h:57-84` (`expand_k`) | Per-vertex varying loop re-reads `arg->varying_comps[vi]` and recomputes `16u*(1u+vi)` each iteration; the uniform `nv`/`comps` could be hoisted/unrolled. Minor vs the LSU traffic, but the inner branch on `nc` is divergent across vertices with mixed varying shapes. | P2 |
| E4 | `gfx_sw.h:258` `om_fragment` | The always-inline full depth+stencil+blend+ROP merge is large; the header itself documents (`:250-257`) that it overruns the Vortex divergence pass's 100-BB guard and needs `-vortex-divergence-max-bbs=N`. This is correct guidance but **fragile**: it is a per-kernel build flag the test Makefiles must remember to set, with a silent-miscompile failure mode (uniform reads left unselectable). Should be encapsulated in a `libgfx_sw` build fragment, not left to each call site. | P1 (build robustness) |

---

## 4. Performance findings

| # | area | finding |
|---|------|---------|
| P-1 | Occupancy / persistent threads | The fragment kernel (`kernel.cpp:74`) is a **persistent-thread `vx_rast()` pop loop** — correct and CUDARaster-aligned. The front end, by contrast, is **9 discrete CP launches** with a full drain between each (the inter-stage barrier). For tiny draws (P=2 here) the launch overhead dominates; a real GPU fuses setup→bin into far fewer dispatches. The split is justified by "no fast inter-core shared memory + coarse CP barriers" (binning-redesign §6.1) but is the main per-draw fixed cost. |
| P-2 | Atomic contention | Notably **zero atomics** in the binning path — the bin-stripe single-owner scatter (`:254`) replaces atomic bin bumps with a deterministic per-thread prefix. This is the right call for determinism and contention both; it is a genuine improvement over a naive atomic-bump binner. |
| P-3 | Single-CTA reductions | SCAN/BSCAN/BBASE run on **one CTA** (`graphics.cpp:511`). BBASE in particular is a `tid==0`-only serial loop over **all B bins** (`pipe_frontend.h:223-241`) — O(B) on a single lane. For large grids this serializes the whole device on one thread; a cooperative scan would help once B is large. |
| P-4 | Redundant memory traffic | `slot_prim`/`slot_bbox` (setup) are written then re-read+compacted into the dense `prim`/`bbox` by EMIT — a full extra round-trip of `rast_prim_t` (the largest record) through memory purely to compact away culled prims. A scan-then-scatter that writes directly to the compacted slot would halve primbuf write traffic. |

---

## 5. "True GPU" alignment vs NVIDIA / AMD / Intel

- **Sort-middle: yes, genuinely.** The front end is a real Laine & Karras 2011 / CUDARaster
  sort-middle pipeline — triangle setup distributed over SIMT lanes, a bin (coarse 128 px)
  histogram+scatter producing a per-bin sorted prim list, consumed by the FF RASTER which
  descends bin→block→quad. This is **not** a serial host-`Binning()` port shoved onto one
  core: per-prim stages are grid-strided and barrier-free, the only shared step is the tiny
  `base[B]` scan, and the output is the *identical* packed buffer the FF unit reads. That is
  architecturally the same shape as a tile-based deferred (Mali/Adreno/PowerVR) binning pass,
  with the FF unit playing the role of the per-tile rasterizer.

- **Where it diverges from real HW.** (1) **Nine CP launches** with full device drains is
  coarser than any real GPU's setup/bin fusion — real tilers bin in one pass with on-chip
  parameter buffers; here the lack of fast inter-core shared memory forces the multi-launch
  barrier model. (2) **Static contiguous bin stripe** (no dynamic work-pull) under-balances
  hot bins; the doc notes a drop-in atomic-cursor pull but it is not implemented. (3) The
  **coarse 128 px bin with no per-bin tight overlap test** over-includes prims at bin
  granularity (the redesign §8 notes a tighter count as future work) — real tilers do a finer
  bound/edge reject. (4) **B=1 for ≤128 px targets** means the test exercises essentially no
  binning; coverage of the actual stripe path needs ≥256 px.

- **OM/TEX window ABI.** The `vx_rt_set`-staged OM window (`vx_om4`) and quad-LOD `vx_tex4`
  mirror how modern GPUs batch a quad's 4 fragments through one ROP/TMU request — a sound
  "true GPU" choice. The caveat is the **funct3=5 RTU-row sharing** (§2 C1/C3): on real HW the
  ROP/TMU would not be gated on the RT unit being present. Aligning to NVIDIA, OM and TEX
  should be independently gateable; the current encoding makes OM4 require RTU.

- **SW fallback.** `gfx_sw.h` matching the FF OM bit-for-bit *because it is the same code* is
  exactly the residency-correct "no host fallback" model the charter demands, and is more
  principled than e.g. a separate reference path. The divergence-BB-guard dependency (E4) is
  the one rough edge.

---

## 6. v2.1 recommendations

### P0 (do first)
- **P0-1 (C1).** Make the OM/TEX4 window path's RTU dependency explicit and self-consistent:
  either add `-DVX_CFG_EXT_RTU_ENABLE` to the `gfx_pipeline_tex` (and any `vx_om4`) test
  Makefiles **and** ensure the simx caps reflect it, or decouple OM-window staging from the
  RTU SETW op so OM4 works with RTU off. Today the shipped test traps in the decoder.
- **P0-2 (C2/§8).** Disambiguate the §8 root cause with the **B=1 vs B>1** experiment
  (render at 256×256) and the **scratch-memset-between-draws** discriminator described in §2.
  My ranked hypothesis is that at 128×128 (B=1) the diff is **not** front-end scratch but an
  **intra-batch OM-depth write→read ordering hazard**; if confirmed, the fix is an explicit
  OM drain / depth-buffer barrier between in-batch draws, not a kernel edit. If the diff only
  appears at B>1, instrument the BHIST/BSCATTER `thist[tid*B+b]` path first.

### P1
- **P1-1 (E2).** Size the histogram scratch to the **stripe width**, not full `B`
  (`graphics.cpp:467`); BHIST/BSCATTER only touch owned columns, so `T*B` over-allocates and
  over-traffics for large render targets.
- **P1-2 (E4).** Encapsulate the `om_fragment` divergence-max-bbs requirement in a shared
  `libgfx_sw` build fragment so no fragment-kernel call site can silently miscompile.
- **P1-3 (P-3/P-4).** Replace the BBASE single-lane O(B) loop with a cooperative scan, and
  fuse the setup compaction (EMIT) so the largest record (`rast_prim_t`) is not round-tripped
  through memory twice.

### P2
- **P2-1 (E1/P-1).** Add the binning-redesign §6.1 "coarse pre-pass" (or per-key owning-CTA
  dispatch) to retire the O(G·K) all-CTAs-scan-all-keys cost as core count grows.
- **P2-2 (C4).** Reconcile the doc (modulo-interleaved stripe) with the code (contiguous
  stripe) — pick one and state it; flag the load-balance implication.
- **P2-3 (C3).** Add an explicit CUSTOM1 decode-table assertion/test that `vx_tex4`/`vx_om4`
  (funct3=5 sub-fields) and the RTU set/get/trace/wait ops never alias.
- **P2-4 (C5).** One-line invariant comments on the unguarded clip lerp (`setup_math.h:60`)
  and the single-writer meta scan (`pipe_frontend.h:127-131`).
