# v2 Review: prism_v3 graphics runtime stack

**Date:** 2026-06-17
**Reviewer scope:** `sw/runtime/graphics.cpp`, `sw/runtime/include/graphics.h`,
`sw/common/vx_gfx_abi.h`, `sw/common/gfx_frontend_abi.h`,
`sw/common/gfx_render.{h,cpp}`, `sw/common/gfx_sw.h` (as they pertain to the
runtime). `raytrace.h` and the device-side SIMT front-end kernels are out of
scope. Read/grep/build only — no code modified.

---

## 1. Overall assessment

The runtime has cleanly absorbed three of the charter's mechanisms: the genxml-style
FF emitters (`program_raster/om/tex`, `draw`) are well-factored single-source-of-truth
register packers; `DrawCommands` is a real CP-batch builder with correct
stable-storage discipline; and `FrontEndPool` + `gfx_frontend_abi.h` encode a genuine
nine-stage device-resident setup→binning pipeline that is exercised end-to-end as one
CP batch (`gfx_pipeline_tex` Path C). The host `Binning()` was correctly migrated to
the §6.3 coarse-bin schema and is **bit-validated against the migrated SimX
`raster_core`** (absolute `pids_offset`, dense-header-then-pids layout, `bin_x/bin_y`
scaled by bin size — all consistent host↔SimX). The honest caveats: (a) the production
UMD (`mesa-vortex` `vp_raster.cpp`) still runs the gfx-v1 posture — host `Binning()` +
per-draw `buffer_create`/`release` + upload/readback — so the charter's residency/CP
pillars are **demonstrated in tests, not yet shipped in the driver**; (b) RTL
`VX_raster_mem.sv` was **not** migrated to the coarse-bin header (documented-deferred,
but a live host↔RTL ABI break); and (c) there is no residency allocator yet — the
`FrontEndPool` allocates ~16 buffers per `init()` and there is no usage-routed/scoped
suballocator per `gfx_v2_residency_allocator.md`.

**Maturity grade: B−.** Strong, correct emitter/builder/pool foundation and a real
device-resident test path; held back from a B+/A by the driver still being a guest on
llvmpipe, the absent residency allocator, and stale doc/ABI drift.

---

## 2. Correctness findings (`file:line` — issue — severity)

**C1 — `graphics.h:79-84` (and `:201`) — Binning doc comment describes the *retired*
`rast_tile_header_t` layout — LOW (doc-only, but misleading).**
The header block over `Binning()` still says it emits "an array of
`rast_tile_header_t` records, each followed by the primitive-ID list for that tile" and
"number of tiles … tileLogSize typically 5 → 32×32." The implementation
(`graphics.cpp:285-309`) emits the new `rast_bin_header_t` dense block + absolute-indexed
`sorted_pids`, and the contract is now "bin at `VX_CFG_RASTER_BIN_LOGSIZE` (128 px)."
`FrontEndPool::tilebuf_addr` doc (`:201`) and `gfx_frontend_abi.h:14-28,133` likewise
still say "`rast_tile_header_t` tilebuf … RASTER's gfx-v1 buffers." The code is right; the
comments lie. Fix the comments to the §6.3 layout (the inline comment at
`graphics.cpp:285-290` is already correct — propagate it).

**C2 — host↔RTL ABI break: `VX_raster_mem.sv:127-137` still reads the 8-byte
interleaved `rast_tile_header_t` — HIGH severity, but DOCUMENTED-DEFERRED scope.**
The host now emits a 12-byte `rast_bin_header_t` (`vx_gfx_abi.h:150-154`) in a *dense
header array* followed by a *separate* absolute-indexed `sorted_pids` block. RTL still
decodes `th_tile_pos_x/y/pids_offset/pids_count` as four 16-bit fields in an 8-byte
record (`:127-130`) and computes `pids_addr = header_addr + pids_offset + 1`
(`:137`) — i.e. per-header *relative, interleaved* pids, the gfx-v1 scheme. Feeding the
new buffer to RTL RASTER reads garbage. SimX `raster_core.cpp` *is* migrated
(`:286-344,686-690`: 12-byte header, `pids_base = tbuf + num_headers*sizeof(hdr)`,
absolute offset, `bin_x*bin_size`), so the oracle is correct. This is the
`gfx_v2_tile_binning_redesign.md` §7 / charter §10-phase-3 RTL change that is explicitly
postponed until SimX-green — consistent with the "defer synth until rtlsim-green" policy.
**Not a runtime bug**, but it is the single largest correctness gap in the
host↔device↔simx↔RTL chain and must be tracked: today only the SimX backend can consume
runtime `Binning()`/`FrontEndPool` output; XRT/RTL cannot.

**C3 — `gfx_frontend_abi.h:46` `PIPE_PRIM_BITS=20` vs `VX_RASTER_PID_BITS=16`
(VX_types.toml:177) — pid truncation past 65 535 prims — MEDIUM.**
The composite-key low field and `primbuf` indices are 20-bit (1 M prims after clip), but
both the SimX consumer (`raster_core.cpp:367,696` cast pid_word → `uint16_t`) and RTL
(`RASTER_PID_BITS=16`) silently truncate to 16 bits. A draw with >65 535 *visible/clipped*
prims aliases pids and renders wrong primitives with no diagnostic. The 32-bit
`sorted_pids`/`primbuf` storage is fine; the *consumer width* is the ceiling. Either
narrow `PIPE_PRIM_BITS` to 16 to make the limit explicit at the ABI, or widen
`VX_RASTER_PID_BITS` (the proposal §3 "key width" knob anticipates this; it is currently
unreconciled between producer and consumer). At minimum, `FrontEndPool::append` /
`Binning` should bound-check and reject `num_tris*SETUP_MAX_SUB > (1<<PID_BITS)`.

**C4 — host `Binning()` and the device front-end produce *different prim sets* (cull/clip
divergence) — MEDIUM (parity hazard).**
Host `Binning()` (`graphics.cpp:204-207,231`) culls only degenerate (zero-area) and
fully-offscreen tris; it does **no** back-face cull and **no** near-plane clip. The device
front-end *does* (commit `0b130764` back-face cull §6.1; `gfx_frontend_abi.h:52-67,86`
`SETUP_CULL_*` + `SETUP_MAX_SUB=2` clip subtris). So host `Binning()` (the mesa path and
the test "gold" oracle) and `FrontEndPool` (the device path) are **not** bit-identical
whenever a draw has back-faces or near-plane crossings. `gfx_pipeline_tex` cross-checks
device==Binning() (`main.cpp:315-337`) — that check only holds for the
front-facing/unclipped inputs the tests happen to use. This is latent: the moment a
conformance scene has culling/clipping, the "oracle" disagrees with the device. Decide
which is canonical (the device, per charter §6.1) and either teach `Binning()` the same
cull/clip or stop treating it as the bit-exact oracle (demote it to a coverage-only
reference). Note `pipe_arg_t.cull_mode` exists (`gfx_frontend_abi.h:117`) but `Binning()`
has no cull parameter at all.

**C5 — `graphics.cpp:451,455-472` FrontEndPool scratch flagged `VX_MEM_WRITE`-only —
NOT a bug, but mis-intent worth a comment — INFO.**
The ~14 scratch buffers pass `W = VX_MEM_WRITE` (0x2). Kernels read several of them
(prefix-sum reads `count[]`, sort reads `keys[]`, scatter reads `binbase[]`). This is
*safe* only because `vm.cpp:28` `flags_to_pte` unconditionally sets `PTE_R` (read always
allowed; only `PTE_W` is gated). So a W-only buffer is really R+W. Correct today, but the
flag names misrepresent access and would break if the MMU ever honored a no-read bit.
Prefer `VX_MEM_READ_WRITE` for kernel-touched scratch to match actual access.

**C6 — `graphics.cpp:547-548` SCISSOR packs width/height into the high 16 bits —
silent overflow past 65 535 px — LOW.**
`emit_raster` writes `SCISSOR_X = (width<<16)|0`, decoded by SimX as
`right = value>>16` (`raster_core.cpp:682`). Fine for real targets, but `width/height`
are `uint32_t` with no range check; a >16-bit dimension corrupts the scissor. Bound-check
in `emit_raster` or document the 16-bit cap on `raster_state_t::width/height`.

**C7 — `DrawCommands::submit` pointer-lifetime — CORRECT, verified.**
`linfos.reserve(entries_.size())` before the loop (`graphics.cpp:351`) guarantees
`&linfos.back()` (`:371`) never dangles via reallocation, and `li.args_host =
e.args.data()` points into the per-`Entry` owned `std::vector` which outlives the
`vx_enqueue_commands` call (`:380`). The header contract (`graphics.h:109-113`) accurately
documents the "args valid only for the launch() call" lifetime. No issue — this is the one
place a builder usually gets UB and it's handled right.

**C8 — `emit_tex` STAGE-before-state ordering — CORRECT, but order-fragile —
INFO.** `emit_tex` writes `VX_DCR_TEX_STAGE` first (`graphics.cpp:577`), which the
`TexDCRS` mirror latches as the active stage so subsequent per-stage regs land in the
right bank (`gfx_render.h:128-137`). Correct as written; just note that any reorder of the
emit sequence (or interleaving two stages in one `DrawCommands` batch without re-emitting
STAGE) silently cross-writes banks. A brief "STAGE must lead" comment would harden it.

---

## 3. Efficiency findings

**E1 — host `Binning()` is NOT dead-weight: it is still the production path —
the charter's §6.2 goal is unmet in the driver.**
`mesa-vortex/.../vp_raster.cpp:109` calls `graphics::Binning()` on the CPU every draw,
then per-draw `vx_buffer_create` for tiles/prims/color/depth/tex (`:154-160,244`) and
`vx_enqueue_write` uploads + `vx_enqueue_read` readback (`:188-195,281`). The device
`FrontEndPool` front-end is exercised **only in tests** (`gfx_pipeline_tex` Paths B/C/D).
So the runtime has *built* the on-device binning capability but the UMD has not *adopted*
it. This is the gap between "demonstrated" and "shipped." Until `vp_raster` switches to
`FrontEndPool::append` + `DrawCommands`, host `Binning()` is load-bearing and the §6.3
migration buys nothing in production — it only keeps the CPU oracle in sync with SimX.

**E2 — `FrontEndPool::init` allocates ~16 distinct device buffers; no pooling /
sub-suballocation — MEDIUM.**
`init` does 16 `vx_buffer_create` calls (`graphics.cpp:455-472`), several sized to
worst case (`NT*MS*PRIM_SZ` twice; `T*B*4` histogram which is `block_dim*num_bins*4` — for
1024 threads × a 1080p/128px bin grid (≈128 bins) that's ~512 KB, fine, but it scales as
block_dim×bins). This is a per-pool cost, amortized across a pass (the pool is reused), so
it is not per-draw churn — good. But it is 16 separate allocator entries + 16 PTE-install
passes where the residency doc (`gfx_v2_residency_allocator.md` §5) wants **one** bump
region carved once and sub-offset. Recommend a single pinned slab + internal offsets
(also fixes C2-adjacent "all addresses static from one pool base," which the CP-frontend
doc §4.1 explicitly wants).

**E3 — emitter code duplication is well-controlled — POSITIVE.**
The `emit_*` templates with `QueueSink`/`BatchSink` (`graphics.cpp:541-621`) are exactly
the right factoring: the register layout for each unit lives once and both the immediate
(`program_*(queue)`) and batched (`program_*(DrawCommands)`) forms share it. This is the
genxml/si_emit precedent done correctly and removes the inline DCR duplication that
`graphics.h:217-231` calls out. No change needed.

**E4 — `tilebuf` over-reserved to `keys_cap` always — minor.**
`FrontEndPool::init` sizes `tilebuf = B*HDR_SZ + keys_cap*4` (`graphics.cpp:470`), and
the host `Binning()` sizes `tilebuf` to `num_bins*HDR + total_prims*4`
(`graphics.cpp:292`). The pool can't know `total_prims` pre-submit so `keys_cap` (the
high-water) is correct there. Just note that `keys_cap` doubles as the `keys` scratch size
(`:465`) and the `sorted_pids` capacity — a single knob bounding two different arrays;
fine, but make the worst-case relationship explicit (coverage entries ≤ keys_cap).

---

## 4. Performance findings

**P1 — per-draw host overhead in the production path is the gfx-v1 cost, unchanged.**
Because `vp_raster` still uploads/reads back and binds buffers per draw, per-draw latency
is dominated by the CPU `Binning()` (a `std::map<pair,vector>` scatter,
`graphics.cpp:171,298`) + 4-5 DMA round-trips. The `std::map` ordered container with a
`std::vector` per bin is allocation-heavy (one heap vector per touched bin, re-grown as
prims are appended) — acceptable for an oracle, not for a hot path. The device front-end
(when adopted) removes all of this; until then the charter's "host idle between submit and
present" is not realized in the driver.

**P2 — no residency allocator (per `gfx_v2_residency_allocator.md`) — the §6.6 pillar
is absent from the runtime.**
There is no usage-flag routing (FF-bound→pinned vs shader-only→paged), no
persistent/per-pass/per-draw scoped suballocator, and no tiling-pool bump-reset. The
`FrontEndPool` is the closest thing — it reserves once and reuses across draws
(reset-by-reuse via CP launch-drain, documented `graphics.h:160-167`) — which is a real
per-draw-transient pool for the *binning intermediates only*. Attachments/textures in the
driver are still per-draw `buffer_create`/`release` (`vp_raster.cpp:154-160,288-292`),
i.e. the opposite of "resident across the frame." The `VX_CAPS_VM_PINNED_*` query the doc
says is "already available" is not consulted anywhere in the reviewed runtime graphics
code. This is the largest *performance* gap vs the charter.

**P3 — CP-batch vs per-stage launch — the good path exists and is correct.**
`DrawCommands` + `FrontEndPool::append` produce one `vx_enqueue_commands` batch of
9 launches + DCR writes + fragment launch, polled once (`gfx_pipeline_tex/main.cpp:301-315`).
This is the genuine "ring the doorbell once" model of `gfx_v2_cp_graphics_frontend.md` §3
and is the right primitive. The overhead the CP-frontend doc §6/§7 flags (≈18 DCR writes
per launch × ~9 stages) is real and lives below this layer (KMU DCR ABI); the runtime
builder is not the bottleneck and nothing here needs changing — it's waiting on the
QMD-atomic-launch CP work.

---

## 5. "True GPU" alignment vs NVIDIA / AMD / Intel

**Where the runtime matches real UMDs:**
- **FF emitters = genxml/si_emit done right.** `raster_state_t`/`om_state_t`/`tex_state_t`
  + `program_*` is precisely the Intel-genxml `*_pack` / radeonsi `si_emit_*` /
  NVK pattern: the driver fills a plain state struct, libvortex owns the register
  encoding (`graphics.h:216-231` states this explicitly and accurately). `block_addr()`
  (64-byte block index, `graphics.cpp:537`) is the analog of GPUVA-shifted descriptor
  fields. This is the strongest "true GPU" element and is the right altitude.
- **`DrawCommands` = a command-buffer builder.** Accumulate launches+register writes,
  encode once, submit once with one completion fence — structurally a Vulkan secondary
  command buffer / PM4 IB / NV pushbuffer segment. The "build once, submit per pass,
  reusable" contract (`graphics.h:98-113`) mirrors `vkCmd*` recording into a reusable
  `VkCommandBuffer`. Device-resident counts driving static launch dims
  (`pipe_arg_t` grid-stride, `gfx_frontend_abi.h:108-135`) is the `ExecuteIndirect` /
  Vulkan-indirect idiom. This is *real*, not simulated — `gfx_pipeline_tex` Path C runs it
  through the actual CP command ring.

**Where it diverges from a true GPU (the honest gaps):**
- **The driver is still a guest on llvmpipe.** `vp_raster.cpp` host-bins, per-draw
  allocates, uploads and reads back — the exact "polite guest" posture the charter §1
  retires. The *capability* to be a true GPU exists in the runtime; the *driver* doesn't
  use it. A real radv/anv never CPU-bins or round-trips intermediates. This is the central
  alignment gap and it is a driver-adoption task, not a runtime-API defect.
- **No residency model.** A true GPU UMD (DX12 `MakeResident`, VMA suballocation,
  Vulkan memory heaps) lays the working set out once and keeps it resident. The runtime
  has the pinned-PA substrate (`device.cpp` `VX_MEM_PHYS`) and a transient binning pool,
  but no heap/scope allocator and no residency planning against `VX_CAPS_VM_PINNED_*`.
- **One queue, in-order.** Matches the baseline the CP-frontend doc §6 chooses
  (multi-queue deferred), so this is intended, not a defect — but worth noting vs the
  multi-engine reality of AMD/NV.

Net: the *encoding* layer (emitters + command builder) is at true-GPU parity; the
*memory/residency* and *driver-orchestration* layers are charter-described but not yet
shipped.

---

## 6. v2.1 recommendations (P0/P1/P2)

**P0 — must-do before this can be called "true GPU" in any backend:**
1. **Reconcile the pid-width ABI (C3).** Make `PIPE_PRIM_BITS` (`gfx_frontend_abi.h:46`)
   and `VX_RASTER_PID_BITS` (`VX_types.toml:177`) agree, and add a hard bound-check in
   `FrontEndPool::append` / `Binning` (`graphics.cpp:478`) rejecting
   `visible_prims > (1<<PID_BITS)`. Silent pid aliasing is a real-scene correctness bug.
2. **Resolve the host/device cull-clip parity hazard (C4).** Either give `Binning()`
   the `SETUP_CULL_*` + near-clip behavior of the device front-end (add a `cull_mode`
   parameter mirroring `pipe_arg_t.cull_mode`) **or** explicitly demote `Binning()` from
   "bit-exact oracle" to "coverage reference" in `graphics.h` and the
   `gfx_pipeline_*` cross-checks. As-is, the device==Binning() assert in
   `gfx_pipeline_tex/main.cpp:315` is only incidentally true.
3. **Track the RTL header migration (C2) as a release gate.** The runtime/SimX
   coarse-bin output cannot run on XRT/RTL until `VX_raster_mem.sv:127-137` adopts the
   12-byte `rast_bin_header_t` + absolute-pids layout. File it against the §7 RASTER
   front-end phase so "rtlsim-green → synth" doesn't ship a host buffer the RTL misreads.

**P1 — adopt the capabilities the runtime already built:**
4. **Switch `mesa-vortex/vp_raster.cpp` to the device front-end (E1/P1).** Replace
   host `Binning()` + per-draw tile/prim upload with `FrontEndPool::append` +
   `DrawCommands` (the `gfx_pipeline_tex` Path C recipe). This is the single change that
   moves the *driver* from "guest on llvmpipe" to "device renders." `program_*`/`draw`
   are already in place there — the binning/residency half is missing.
5. **Land the residency allocator (P2/`gfx_v2_residency_allocator.md`).** A usage-routed
   (FF→pinned, shader→paged) scoped suballocator over one pinned slab, consuming
   `VX_CAPS_VM_PINNED_*`, with persistent attachments/textures held resident across the
   pass instead of `vp_raster.cpp`'s per-draw create/release. Collapse `FrontEndPool`'s
   16 buffers (E2) into sub-offsets of one bump region while doing this.

**P2 — hygiene / hardening:**
6. **Fix the stale `rast_tile_header_t` doc comments (C1)** in `graphics.h:79-84,201`
   and `gfx_frontend_abi.h:14-28,133` to the §6.3 coarse-bin layout (the inline comment at
   `graphics.cpp:285-290` is the correct text to mirror).
7. **Range-check SCISSOR dimensions (C6)** in `emit_raster` (`graphics.cpp:547-548`) or
   document the 16-bit cap on `raster_state_t::width/height`.
8. **Use `VX_MEM_READ_WRITE` for kernel-read scratch (C5)** in `FrontEndPool::init`
   (`graphics.cpp:451`) so the flags reflect real access; add a "STAGE must lead"
   comment to `emit_tex` (C8).
