**Date:** 2026-06-06 (v2.1 callback-window addendum folded in 2026-06-07)
**Status:** implemented in SimX; supersedes the ABI surface of
[rtu_simx_proposal.md](rtu_simx_proposal.md) §4 (register-file SET/GET model).
This document consolidates the v2 ABI and the v2.1 callback-window addendum
(§12); the separate v2.1 note has been merged here.
**Branch:** `prism`
**Related:**
- [rtu_simx_proposal.md](rtu_simx_proposal.md) — the RTU microarchitecture
  and phase history. This revision changes **only** the kernel-visible ISA
  ABI; the `RtuCore` traversal pipeline, pool, PEs, and SimX TLM model are
  unchanged.
- [rtu_verilog_minimal_proposal.md](rtu_verilog_minimal_proposal.md) — the
  RTL realization; the new ABI simplifies its SFU-side decode.

# PRISM RTU — ISA ABI v2 (scope-partitioned, single-issue trace)

## 1. Motivation

The Phase-1 RTU ABI issues a single ray through a per-(warp,lane) **register
file** that the kernel populates one 32-bit slot at a time with `vx_rt_set`,
then reads back one slot at a time with `vx_rt_get`. The real op count to issue
one trace (the Phase-1 smoke kernel) is:

| Step | SFU ops |
|---|---:|
| `vx_rt_set3` origin / dir / tmin-tmax-pad + `vx_rt_set1` flags | 3+3+3+1 = **10** |
| `vx_rt_trace` + `vx_rt_wait` | 2 |
| `vx_rt_get_after` ×4 (t, u, v, primID) | 4 |
| **per trace** | **~16** |

(The "bulk" `vx_rt_set3` is *not* a bulk op — Phase-1 GAS `.insn` can't
express the R4 bulk encoding, so it expands to three single-slot stores.)

Every shipping GPU issues a trace in **one** instruction. The root cause of
PRISM's overhead is **not** that a ray has many arguments — it is that the
ABI routes **all** arguments, regardless of their SIMT scope, through the
same per-lane special-register port, one word per instruction. The fix is to
**partition arguments by scope** and give each scope its natural, cheap home —
which is exactly what NVIDIA, AMD, and Intel do, and exactly what Vortex's own
fixed-function units already do.

This revision does that. It does **not** change the RTU pipeline, the BVH
format, the async trace+wait semantics, or the opaque-only minimal-RTL scope.

## 2. Argument-scope taxonomy

A trace's arguments live at three distinct SIMT scopes. The Phase-1 ABI
ignores the distinction; v2 honors it.

| Argument | Scope | Divergence | v2 home |
|---|---|---|---|
| `origin.{x,y,z}`, `direction.{x,y,z}`, `tmin`, `tmax` | **per-thread** | divergent every lane | **FP register window** `f0–f7` (8 floats; §9.1) |
| `ray_flags`, `cull_mask` | **per-warp** | uniform per trace site (rarely divergent) | lanes 2/3 of the `rs1` config register (§5.1) |
| TLAS / AS device pointer | **per-trace** | usually warp-uniform; Vulkan allows multiple AS | lane 0 of the `rs1` config register (**per-call**, not a DCR) |
| `scene_kind`, `bvh_width`, RT enable, callback-dispatcher PC, reform threshold | **per-dispatch** | constant for the whole launch | **DCR**, host-programmed |
| `payload_ptr` | **per-warp base** | uniform base; per-thread = base + lane·stride | lane 1 of the `rs1` config register (§9.4) |

Two consequences follow immediately:

1. **The per-dispatch config belongs in DCRs**, set host-side by the runtime,
   the way Vortex's other fixed-function units program their dispatch-global
   state. The `VX_DCR_RTU_*` registers already exist but there is **no runtime
   host header** to program them. That missing host API is the real asymmetry
   vs. the other units — addressed in §5.3.

2. **The per-call TLAS pointer must NOT become a DCR.** An acceleration
   structure is a per-trace shader binding, not dispatch-global pipeline state;
   a single kernel legitimately traces multiple AS (primary vs. shadow /
   opaque-only). Every mainstream API passes it per-trace (Vulkan
   `OpTraceRayKHR`, DXR `TraceRay`, Intel `TraceRay` send, AMD
   `image_bvh_intersect_ray`). It stays in `rs1`. The Phase-1 design already
   got this right; v2 keeps it.

## 3. How mainstream GPUs handle the many arguments

The unifying principle: **partition arguments by SIMT scope; store each
partition in the cheapest medium for that scope; pass the divergent per-ray
data as one grouped object — never field-by-field.**

- **AMD RDNA2/3** — `image_bvh_intersect_ray`: the per-thread ray (origin,
  dir, inv_dir, tmax) is a **VGPR group** (~11 VGPRs) passed as one operand;
  the warp-uniform BVH descriptor (128-bit) is an **SGPR group** (read once
  per wave); results are a **4-VGPR group**. The traversal loop is shader
  SIMT. The ISA *is* "register group in, register group out."

- **NVIDIA Turing→Ada** — the per-thread ray descriptor is SASS-compiler
  register-allocated; one RT-driving instruction reads it; fixed-function
  traversal; hit attrs return in registers. Warp-uniform args (AS handle, ray
  flags, SBT offset/stride, miss index, cull mask) live in the **Uniform
  Register File** (introduced in Turing precisely so warp-uniform values are
  not replicated per lane). Pipeline-global state (SBT base) is driver-set.

- **Intel Xe-HPG** — the per-thread ray is a `MemRay` struct in the thread's
  RTStack (per-lane scratch, L1-backed); the warp-uniform BVH base + dispatch
  config ride the `send` message header; results land in a `MemHit` struct.
  Intel uses memory (not registers) because BTD reformation migrates ray state
  across EUs — a constraint Vortex does not share, which is why the AMD
  register-group model fits Vortex better than the Intel memory model.

- **Mali Immortalis / PowerVR Photon / Adreno 7xx** — fixed-function traversal
  reached through a single intrinsic over an assembled descriptor; the AS is a
  uniform descriptor binding; per-ray geometry is assembled then issued once.

PRISM Phase-1 is the only one that flattens all three scopes into per-lane
special-register writes. v2 adopts the AMD register-group shape because it (a)
keeps the hot per-ray state in registers — honoring the original §4.2 "Ray
Bank" intent — while (b) issuing in one instruction, and (c) has a direct,
shipping precedent inside Vortex itself (§4).

## 4. The two Vortex facilities v2 builds on

The v2 encoding is **not new HW**. It composes two facilities the Vortex ISA
and microarchitecture already provide and which other fixed-function units
already use; v2 is the first ray-tracing application of them.

**(1) Multi-register fixed-function ops over a register window.** Vortex
supports instructions whose source/destination is a *register group* (widths
`0-1` / `0-3` / `0-7`; the register descriptor carries a `group` field and the
LLVM backend allocates the group). The HW reads/writes the whole window by
hardware convention across the warp; the instruction's operand fields carry
only format/config, not per-element register numbers. The window is kept
resident across the op by pinning its elements to fixed registers, so the
compiler does not spill them. Because a window read exceeds the normal
operand-collector read ports, such an instruction is decoded as a **macro-op**
that the per-warp **sequencer** expands into micro-ops, each reading one
register of the group through the standard operand collector over several
cycles (§5.6).

**(2) Lane-scatter packing via `vx_wgather`.** Vortex provides
`vx_wgather(a, b, c, d)` — a single instruction that scatters four scalars
(read from lane 0) across the four lanes of one register: lane 0←a, 1←b, 2←c,
3←d. It is a pure register-domain op (no memory). This is the standard way a
small set of warp-uniform arguments is delivered to a fixed-function unit in
one register, which the unit then reads per-lane.

v2 maps the per-thread ray onto facility (1) — the `0-7` ray window (rs2) — and
the per-trace config onto facility (2) — `vx_wgather` packs `{scene, payload,
flags, cull}` into one register (rs1). Both halves are thus established Vortex
patterns, not new mechanisms.

## 5. Proposed ISA v2

Encoding space is unchanged: CUSTOM1 (opcode 0x2B), funct3=5 for RTU-prim,
funct3=6 for callback ops, 2-bit sub-op selector. v2 re-tasks the sub-ops.

### 5.1 The R2 trace encoding: lane-packed config (rs1) + ray group (rs2)

`vx_rt_trace` is a standard **R-type** instruction with two source register
operands (the "R2" form) — no R4, no per-slot ports:

- **rs1 — config register (GP), one word per lane across the warp's 4 threads.**
  At baseline `SIMD_WIDTH = 4`, one physical register spans 4 lanes (4 × 32b).
  The four warp-level config words (integer pointers/flags) ride one-per-lane;
  the `RtuUnit` reads lane *i* as config word *i*:

  | rs1 lane | config word |
  |---|---|
  | 0 | `scene_ptr` (TLAS device addr; 32b on the RV32 target) |
  | 1 | `payload_ptr` |
  | 2 | `ray_flags` |
  | 3 | `cull_mask` |

- **rs2 — ray group** (**FP registers `f0–f7`**, per-thread): the divergent ray
  geometry — eight floats kept in the FP file with no int↔float conversion
  (§9.1) — read by the register-window convention of facility (1) in §4.

  | rs2 reg | field |
  |---|---|
  | `f0..f2` | `origin.{x,y,z}` |
  | `f3..f5` | `direction.{x,y,z}` |
  | `f6` | `tmin` |
  | `f7` | `tmax` |

- **rd** ← ray handle (GP).

```c
// The kernel passes the 4 config scalars + the ray window. vx_rt_trace
// emits an implicit vx_wgather to pack {scene,payload,flags,cull} into the
// rs1 config register, then issues the trace with rs2 = the ray window.
uint32_t h = vx_rt_trace(scene_ptr, payload_ptr, ray_flags, cull_mask, ray);
```

The kernel does **not** call `vx_wgather` — it is emitted **inside the
`vx_rt_trace` intrinsic** (facility (2), §4) to scatter the four config scalars
(read from lane 0) into lanes 0–3 of the rs1 register, with **no memory
traffic**. The intrinsic marks that `vx_wgather` pure, so when the config
scalars are loop-invariant the compiler hoists it out of a bounce loop and the
steady-state per-trace config cost is ~0 ops. The ray arrives as the register
allocation the compiler already performed — the window *is* the ray. ~10 live
registers/lane (8 FP ray + 1 GP config + 1 GP handle), under AMD's
`image_bvh_intersect_ray` (~11 VGPRs).

The lane-packing exploits the narrow baseline warp directly: `SIMD_WIDTH = 4`
gives exactly 4 config slots and all 4 are used — an exact fit at the minimum
supported width, roomier at any larger warp. A 5th per-trace word (e.g. a
`scene_kind` override) would not fit at warp=4 and must instead ride a DCR
(§5.3) or an rs2 spare. Two consequences this encoding accepts (see §5.4):

- **The config is warp-uniform by construction** — one word per lane means
  `scene_ptr` / `ray_flags` / `cull_mask` are one-per-warp, not per-thread. The
  common case (all lanes trace one AS with one flag set) is exactly this;
  intra-warp multi-AS divergence is the fallback case (§5.4).
- **`payload_ptr` in lane 1 is a per-warp base**, not a per-thread pointer; a
  per-thread payload is `base + lane * stride`. If a workload needs fully
  divergent per-thread payload pointers, payload moves to an rs2 spare instead
  (open question, §9).

### 5.2 Result: register-window writeback (sub-op `WAIT`)

`vx_rt_wait` blocks on the handle and, on terminal, returns the status word in
`rd` **and** writes the hit attributes back, each to its natural register file
(HW-written through facility (1)'s window-writeback path, §4):

| Hit field | Type | File |
|---|---|---|
| `hit_t`, `bary.u`, `bary.v` | float | **FP** (reuse `f0–f2` — the ray is consumed by now) |
| `primitive_id`, `geometry_index`, `instance_id` | integer | **GP** temps |
| `status` | integer | **GP** (`rd` of `wait`) |

```c
uint32_t sts = vx_rt_wait(h, &hit);   // status in rd; HW writes t/u/v to FP, IDs to GP
```

This eliminates **all** `vx_rt_get` ops and the entire `vx_rt_get_after`
scoreboard dance (the RTU SimX proposal, §8.6) — the window write *is* the
scoreboard-ordered writeback, the same way facility (1)'s destination window
retires.

### 5.3 Per-dispatch config + host-side scene prep: the `vortex::raytrace` runtime library

Add a new runtime host header, structured as a C++ host **library** in a
`vortex::raytrace` namespace — not a thin C-style DCR-poke API. The host-side
responsibility for the RTU is two things: **transcoding the acceleration
structure into the CW-BVH byte layout the `RtuCore` walks** (so the SimX walker
and the driver share one format), and **programming the per-dispatch DCRs**.
Both live in this one library, the data-prep helper paired with the config
entry point that targets the same format.

The library uses an include guard, a `detail` sub-namespace for bit-packers,
typed structs, and free functions templated on BVH width where it earns it:

```cpp
#ifndef __VX_RAYTRACE_HOST_H__
#define __VX_RAYTRACE_HOST_H__

#include <cstdint>
#include <vector>

#include <rtu_cfg.h>     // shared host/device format constants
#include <vortex.h>      // vx_device_h, vx_dcr_write

namespace vortex {
namespace raytrace {

namespace detail {

// Pack the VX_DCR_RTU_CONFIG word (scene_kind / bvh_width / cull defaults).
inline uint32_t pack_config(uint32_t scene_kind, uint32_t bvh_width,
                            uint32_t cull_defaults) {
  return (scene_kind   << RTU_CFG_SCENE_KIND_LSB)
       | (bvh_width    << RTU_CFG_BVH_WIDTH_LSB)
       | (cull_defaults << RTU_CFG_CULL_LSB);
}

} // namespace detail

// ── Host-side scene preparation ─────────────────────────────────────
//
// Transcode a host acceleration structure into the CW-BVH<W> byte layout
// the RtuCore walks. Templated on width: W = 4 (scene_kind=2) or
// 6 (scene_kind=3). Emits the same bytes the driver's BVH builder does,
// so the SimX walker and the driver share one format.
template <uint32_t W>
bool build_bvh_scene(const host_bvh_t& src,
                     std::vector<uint8_t>& out_scene,
                     uint64_t& out_root_offset);

// ── Per-dispatch configuration (programs VX_DCR_RTU_* once per launch) ──

struct config_t {
  uint32_t scene_kind    = 2;   // CW-BVH4=2, CW-BVH6=3
  uint32_t bvh_width     = 4;
  uint32_t cull_defaults = 0;
  uint64_t callback_entry = 0;  // mtvec dispatcher PC (Phase 2)
  uint32_t reform_thresh = 0;   // Phase 3
};

// Write the config to the VX_DCR_RTU_* block. Call before vxStartKernel.
inline int program(vx_device_h dev, const config_t& cfg) {
  int ret;
#define VX_RTU__W(reg, val) do { ret = vx_dcr_write(dev, (reg), (val)); if (ret) return ret; } while (0)
  VX_RTU__W(VX_DCR_RTU_CONFIG,
            detail::pack_config(cfg.scene_kind, cfg.bvh_width, cfg.cull_defaults));
  VX_RTU__W(VX_DCR_RTU_CB_ENTRY_LO, (uint32_t)(cfg.callback_entry & 0xffffffffu));
  VX_RTU__W(VX_DCR_RTU_CB_ENTRY_HI, (uint32_t)(cfg.callback_entry >> 32));
  VX_RTU__W(VX_DCR_RTU_REFORM_THRESH, cfg.reform_thresh);
#undef VX_RTU__W
  return 0;
}

} // namespace raytrace
} // namespace vortex

#endif // __VX_RAYTRACE_HOST_H__
```

`program()` writes the existing `VX_DCR_RTU_*` block — no new DCRs; this is the
missing **host surface**, not new state. `build_bvh_scene<W>` gives the RTU a
host-side prep library, so the runtime owns both the scene format and the
per-dispatch config, and the kernel sees only the per-ray ISA (§5.1/§5.2).
Result: a clean runtime/kernel split — the runtime stages format and config
host-side, the kernel issues rays.

### 5.4 The lane-packed config register: semantics and fallback

The rs1 config register (§5.1) carries `{scene_ptr, payload_ptr, ray_flags,
cull_mask}` one word per lane across the 4-thread warp. Three properties this
mechanism rests on:

- **Construction is an implicit `vx_wgather`.** `vx_rt_trace` emits a
  `vx_wgather(scene, payload, flags, cull)` that scatters the four scalars (read
  from lane 0) into lanes 0–3 of the rs1 register — a pure register-domain op,
  no memory traffic (facility (2), §4). The kernel never writes it. It is
  loop-invariant for a fixed scene/flags and hoists out of any bounce loop, so
  the amortized per-trace config cost is zero.
- **The config lanes must be valid independent of the trace tmask.** `vx_wgather`
  reads the four scalars from lane 0, so lane 0 must be active and hold the
  config at the trace; the warp-uniform values satisfy this when the config is
  computed pre-divergence. The `RtuUnit` then reads rs1 with the full lane set,
  not the active mask.
- **Warp-uniform by construction; divergent fallback.** Because word *i* lives
  in lane *i*, `scene_ptr` / `ray_flags` / `cull_mask` are one-per-warp. This is
  the overwhelming common case (a warp of primary rays shares one AS and one
  flag set). For the rare intra-warp **multi-AS** case (different lanes tracing
  different acceleration structures), fall back to a per-thread scene operand —
  scene moves to an rs2 spare and rs1 carries only the genuinely uniform
  flags/cull/payload. This is a slow-path encoding selected by the Mesa lowering
  when it detects a divergent AS; the HW supports both rs1 interpretations under
  the TRACE sub-op.

This is the AMD/NVIDIA principle (warp-uniform args in a uniform/scalar channel,
not replicated per lane) realized through Vortex's narrow warp: the "uniform
channel" is simply the spare lanes of one register.

### 5.5 Callback payload window (retained, narrowed regfile)

The per-(warp,lane) register file is **retained only** as the in-trap callback
payload window. When the RtuCore yields a candidate hit to an AHS/IS
dispatcher (§4.6 of the SimX proposal), HW has written candidate attrs
mid-traversal; the dispatcher reads them register-fast and may write
`hit_t` / hit attrs back before `vx_rt_cb_ret`. This is the one place a
memory round-trip would hurt (it is on the traversal-yield hot path), so the
regfile earns its keep there and only there. Everything on the normal
trace→wait path goes through the §5.1/§5.2 windows.

### 5.6 How a trace executes: macro-op expansion, uop by uop

`vx_rt_trace` reads 9 registers — the 8-FP-register rs2 ray window (`f0–f7`)
plus the 1 GP rs1 config register — more than the operand collector's **3
source-read ports** deliver in one cycle. So the trace is decoded as a
**macro-op**, and the per-warp **sequencer** expands it into a short run of
ordinary uops, each reading up to 3 registers and streaming them on. Because a
uop reads from **one register file**, the GP config read and the FP ray reads
fall into separate uops: **1 GP uop (config) + ⌈8/3⌉ = 3 FP uops (ray) = 4
uops** (more RF banks → fewer). The macro-op stalls fetch until its last uop
issues. (A *register read* returns the whole `SIMD_WIDTH`-lane vector, so
reading `origin.x` delivers it for all lanes at once.)

Architecturally the trace is the R-type instruction of §5.1 (two source
operands: rs1 = GP config, rs2 = the FP ray-window base register); the
sequencer expands the `f0–f7` group hung off rs2 into the uops below. Each uop
is an ordinary micro-op using the datapath's 3 source-read ports — it is **not**
a new R4 *architectural* encoding.

The key design choice is **where the read data goes**: not into an intermediate
staging buffer, but **straight into the `RtuCore` pool slot** the ray must
occupy anyway while it traverses. The first uop allocates that slot; the run
streams the ray into it. The trace is, in effect, a *store of the ray window
into its pool slot*.

**`vx_rt_trace` uop sequence** (1 GP config uop + 3 FP ray uops = 4 uops):

| uop | file | rd | rs1 | rs2 | rs3 | description |
|---|---|---|---|---|---|---|
| 0 | GP | handle | config | — | — | unpack config lanes → `{scene, payload, flags, cull}`; **allocate a pool slot per active lane**; slot index → `rd` (the handle); **latch it as the write pointer**; write config → slot |
| 1 | FP | — | `f0` origin.x | `f1` origin.y | `f2` origin.z | write `origin` → `slot[ptr]` |
| 2 | FP | — | `f3` dir.x | `f4` dir.y | `f5` dir.z | write `dir` → `slot[ptr]` |
| 3 | FP | — | `f6` tmin | `f7` tmax | — | write `tmin`/`tmax` → `slot[ptr]`; **arm the slot** → the `RtuCore` begins traversal |

uop 0 reads the architectural rs1 (the `vx_wgather`-packed GP config); uops 1–3
read the FP `f0–f7` ray window. Only uop 0 writes a register (the handle, GP);
uops 1–3 have no register dest — their effect is the slot write.

Each uop reads up to 3 registers and writes them straight to the slot via the
latched pointer — there is **no staging buffer**. The only state held across the
4-uop expansion is the **write pointer** (the per-lane slot indices,
`SIMD_WIDTH` × ~6 bits — a latch). The handle returns early (uop 0) while the
slot keeps filling; that is safe because the kernel only consumes the handle
later at `vx_rt_wait`, by which time the slot is armed and traversing.

**`vx_rt_wait(handle, &hit)` uop sequence.** The wait issues, then **blocks on
the handle via the scoreboard** until that slot reaches terminal — no spinning.
On terminal it writes the status word and the hit fields from the slot, then
frees it. Each writeback uop drives one destination register (data sourced from
the slot), routed to its natural file — `t/u/v` → FP, IDs/status → GP:

| uop | file | rd | rs1 | description |
|---|---|---|---|---|
| 0 | GP | status | handle | block until slot terminal; status → `rd`; latch read pointer |
| 1 | FP | `f0` | — | `slot.t` → `rd` |
| 2 | FP | `f1` | — | `slot.u` → `rd` |
| 3 | FP | `f2` | — | `slot.v` → `rd` |
| 4 | GP | id0 | — | `slot.primitive_id` → `rd` |
| 5 | GP | id1 | — | `slot.geometry_index` → `rd` |
| 6 | GP | id2 | — | `slot.instance_id` → `rd`; **free the slot** |

Because the wait was already blocked until terminal, this writeback runs after
the traversal has finished — off any critical path.

**What this costs.** The ray flows FP regs `f0–f7` → (4 uop reads) → pool slot →
(traversal) → hit window, never copied into any structure other than the one
pool slot it must occupy to traverse asynchronously. Concretely:

- **No staging/marshalling SRAM** — the only transient state is the slot-index
  write pointer (a latch).
- **No register-file replication beyond the intrinsic slot** — the `f0–f7` regs
  holding the ray are caller-saved and free the moment the last uop arms the
  slot; from then on the slot is the single owner. Across the whole (long)
  traversal the only register state the kernel holds is **one GP register: the
  handle**.
- **No memory traffic for the ray** — register→slot, never through the dcache,
  so it never evicts BVH lines from L1.

The issue-side cost is just **4 uops** — config (GP) → `origin` → `dir` →
`tmin`/`tmax` (FP) — and drops further with more RF banks. Because traversal is
asynchronous, only those 4 uops sit before the ray is in flight; everything
after dispatch overlaps independent kernel work, and the wait-side writeback is
hidden under the traversal that already completed.

## 6. Before / after

```c
// ---- Phase-1 (today): ~16 SFU ops ----
vx_rt_set3(VX_RT_RAY_ORIGIN,    ox, oy, oz);     // 3
vx_rt_set3(VX_RT_RAY_DIRECTION, dx, dy, dz);     // 3
vx_rt_set3(VX_RT_T_MIN,         tmin, tmax, 0);  // 3
vx_rt_set1(VX_RT_RAY_FLAGS,     flags);          // 1
uint32_t h   = vx_rt_trace(scene);               // 1
uint32_t sts = vx_rt_wait(h);                    // 1
float t = vx_rt_get_f_imm_after(VX_RT_HIT_T, sts);          // \
uint32_t u = vx_rt_get_after(VX_RT_HIT_BARY_U, sts);        //  } 4
uint32_t v = vx_rt_get_after(VX_RT_HIT_BARY_V, sts);        // /
uint32_t pid = vx_rt_get_after(VX_RT_HIT_PRIMITIVE_ID, sts);

// ---- v2: 1 trace + 1 wait, on the compiler's existing reg allocation ----
// vx_rt_trace internally emits the config-packing vx_wgather (hoisted when
// the config is loop-invariant). rs2 ray = the 0-7 window = compiler's alloc.
vx_ray_t ray = { {ox,oy,oz}, {dx,dy,dz}, tmin, tmax };          // 0 ops: just regs
uint32_t h   = vx_rt_trace(scene, payload, flags, cull, ray);  // 1 (emits wgather+trace)
vx_hit_t hit;
uint32_t sts = vx_rt_wait(h, &hit);                            // 1; HW fills hit window
// hit.t / hit.u / hit.v / hit.primitive_id already in registers
```

~16 → **2** *architectural* SFU instructions on the hot path (`trace` + `wait`;
`vx_wgather` hoists out), NVIDIA/AMD-class issue cost. Each expands to a few
uops internally (§5.6), but only 2 instructions are fetched/decoded per trace.
No dcache traffic for the ray descriptor or hit attrs (the "Ray Bank" property
of the SimX proposal §7.3 is preserved — the data lives in registers, not a
special file *or* memory).

## 7. What stays the same (do not overcorrect)

- **Async trace+wait** (deviation D1) — kept. v2 changes how the ray is
  *passed*, not the trace lifecycle; Phase 3-B remains a strict additive
  extension.
- **Per-call TLAS pointer** — kept in `rs1` (§2, mainstream-correct).
- **RtuCore pipeline, pool, PEs, shader queues, coherency gather, RTCache** —
  untouched. The change is confined to the SFU-side decode and the
  `RtuUnit` front-end (window read/write instead of slot read/write).
- **CW-BVH4/6 format, opaque-only minimal RTL scope, SimX-as-oracle** — all
  unchanged.
- **The register-file concept** — not discarded; narrowed to the callback
  payload window (§5.5) where it is genuinely the right tool. The Phase-1
  *implementation* (field-by-field SFU marshalling) is what v2 removes.

## 8. Migration plan

| Step | Work | Validation |
|---|---|---|
| 1 | Add `vx_ray_t` / `vx_hit_t` POD types + `vx_rt_trace(scene, payload, flags, cull, ray)` / `vx_rt_wait(h,&hit)` macros to the kernel RTU header: the trace macro emits an **implicit `vx_wgather`** for the rs1 config (facility (2)) + rs2 = `0-7` ray window (facility (1), pinned-register bindings) | compiles; one smoke test ported |
| 2 | Decode `vx_rt_trace` as a macro-op (§5.6): sequencer expands the rs2 `0-7` ray-window read through the OPC; `RtuUnit` reads rs1's 4 config lanes and streams the ray into the allocated pool slot (§5.6); WAIT retires the hit window (6 words) via sequenced writeback; retire the slot-addressed SET/GET path on the normal trace flow | `rtu_smoke` passes on simx |
| 3 | Port the 23 `rtu_smoke_*` kernels from `vx_rt_set/get` to the window ABI (mechanical) | 23/23 pass |
| 4 | Add the `vortex::raytrace` runtime host library (§5.3): `config_t` + `program()` writing `VX_DCR_RTU_*`, and `build_bvh_scene<W>` host transcode | host config smoke |
| 5 | Keep the regfile SET/GET path **only** for the callback dispatcher (§5.5); AHS/IS/CHS/MISS tests unchanged | `rtu_smoke_ahs/is/chs/miss` pass |
| 6 | Driver/Mesa lowering: the RT lowering pass emits the wgather config + window-bound single trace instead of N×set; NIR→LLVM register-window codegen | Vulkan ray-query tests pass on RTU |

Acceptance: all existing RTU regression tests pass; hot-path SFU op count
drops from ~16 to 2; SimX↔(future)RTL parity unaffected (pipeline unchanged).

## 9. Open questions

1. **Register file — RESOLVED: type-split.** The ray is 8 floats
   computed/loaded in the FP file; a GPR window would force 8 `fmv.x.w`
   conversions per trace, reintroducing the per-word marshalling v2 removes.
   Since the scope-partition already separates the (integer) config from the
   (float) ray, each uses its natural file with **zero conversions**: **rs2 ray
   window + float hit outputs (`t`,`u`,`v`) → FP**; **config (rs1), handle,
   status, integer hit outputs (`primitive_id`/`geometry_index`/`instance_id`)
   → GP**. (Reverses the earlier GPR-for-ray draft; the consequence is the
   file-separated 4-uop trace of §5.6.)
2. **Register reservation — RESOLVED: ray window = `f0–f7`.** A `0-7` group
   needs an 8-aligned, all-caller-saved range, and in RISC-V that is **uniquely
   `f0–f7` (`ft0–ft7`)**: every 8-aligned GP range includes special or
   callee-saved registers (`x0–7` holds `zero/ra/sp/gp/tp`), and the other FP
   8-groups straddle callee-saved `fs` registers. So rs2 = `f0–f7`; float hits
   reuse `f0–f2` (the ray is consumed by `wait` time); config (rs1), handle, and
   integer hits sit in caller-saved GP temps (single registers, no alignment
   needed). The `vx_wgather` filling rs1 must stay hoisted/pre-divergence
   (§5.4). **Caveat:** `f0–f7` is also the tensor unit's D-fragment window, so a
   kernel cannot hold an RT ray window and a TCU accumulator live at once —
   acceptable (ray tracing and GEMM are distinct kernels). Confirm codegen on
   the heaviest kernel (`rtu_smoke_recursive`).
3. **Fused synchronous form — RESOLVED: thin wrapper, no opcode (§12.3).**
   `vx_rt_trace_sync(cfg, ray, &hit)` ships as an inline `trace`+`wait` wrapper.
   A true fused op would have to park mid-macro-op and forfeit the async overlap,
   all to save one instruction fetch (the handle never leaves a register) — not
   worth an opcode.
4. **Per-thread payload vs. lane-packed base.** rs1 lane 1 is a per-warp
   `payload_ptr` base (§5.1); per-thread payload = `base + lane*stride`. If a
   workload needs fully divergent per-thread payload pointers, payload moves to
   an rs2 spare (rs2 currently uses 8 of 8; a 9th payload word needs a `0-15`
   group or a separate operand). Resolve which model the Mesa lowering emits.
5. **Multi-AS divergence encoding — RESOLVED: dropped.** The per-lane-scene
   slow path (§5.4) was determined unnecessary. Vulkan/DXR bind one acceleration
   structure per trace, and intra-warp "different geometry" is expressed as
   instances under one TLAS — divergence rides the per-lane rays and hits, not
   the scene pointer. The `vx_rt_trace_mas` form was removed; warp-uniform
   `vx_rt_trace` covers every real case, including secondary rays under a
   narrowed callback mask (the `wgather` config materialises from the last active
   lane, so any partial-warp lane position works).
6. **Config word budget at warp=4.** rs1 holds exactly 4 words; all 4 are used.
   Any 5th per-trace word (e.g. a `scene_kind` override or `tmax`-cap) must go
   to a DCR (§5.3) or an rs2 spare — confirm none is needed on the opaque
   minimal path. (Flags fit: `ray_flags` ≤ 16b, `cull_mask` = 8b, but they now
   occupy separate lanes so width is not the constraint — lane count is.)

## 10. Risks

- **R1. Register pressure at the trace site.** The 8-FP `f0–f7` rs2 ray window
  plus the 1 GP rs1 config + 1 GP handle = ~10 live registers across the trace
  (vs. AMD's ~11 VGPRs — comparable), split across the FP and GP files so it
  does not all land in one. The lane-packing keeps the warp-level args to a
  single register, so this is *lower* pressure than per-thread config groups.
  Heavy kernels (recursion, large CHS bodies) may still spill; mitigation:
  caller-saved rs2 window so it need not survive the call; measure on
  `rtu_smoke_recursive`; fall back to a memory-descriptor variant (Intel
  `MemRay` shape) for pressure-critical sites if needed — the two can coexist
  behind one macro.
- **R2. GAS `.insn` expressiveness.** Facility (1) (§4) proves multi-register
  `.insn` with pinned-register window bindings works today; the RTU window uses
  the same construct, so no new assembler capability is required (unlike the
  Phase-1 R4 bulk-set that GAS could not express).
- **R3. Driver codegen for the window.** The NIR→LLVM lowering must emit the
  window bindings; the existing register-window codegen for facility (1) is the
  template. Scope it in step 6; the SimX path (steps 1-5) does not depend on it.
- **R-OC. Operand-collection latency & accuracy.** The trace macro-op streams
  the 9 registers (8 ray + 1 config) through the operand collector in ⌈9/3⌉ = 3
  uops at 3 read ports (fewer with more banks); the wait writeback is a handful
  of uops, hidden (§5.6). On the async trace this is bounded and hidden under
  traversal latency, but it must be **modelled**, not free: the SimX sequencer
  already does per-uop cycle accounting (existing macro-ops rely on it), so the
  trace macro-op inherits honest timing. Risk is bank-conflict / read-port contention if multiple warps
  issue traces in the same window; mitigation: the operand-collector
  partitioning (`VX_CFG_NUM_OPCS`) already arbitrates this for the existing
  macro-ops — confirm the RTU macro-op shares that arbitration rather than
  adding a private port.

## 11. Sign-off

Steps 1–5 (SimX, smoke ports, runtime header, callback retention) and the v2.1
additions (§12) are **implemented in SimX**; all hand-written
`tests/raytracing/*` kernels run on the v2 ABI. Step 6 (Mesa lowering) emits the
v2 `trace`/`wait` ops for the opaque ray-query path; the candidate/any-hit
lowering remains a follow-up. The §9 open questions are resolved (Q1/Q2 register
split, Q3 sync = inline wrapper, Q5 multi-AS dropped).

## 12. ABI v2.1 — callback-side window read + trap-safe wait (implemented)

ABI v2 collapsed the *issue* path (the ray rides `f0–f7`, the hit retires
through a register window), but the *callback read* path still read candidate
fields one slot at a time. v2.1 closes that and makes the windowed `wait`
survive a callback trap. It is additive — same encoding space, no change to the
pipeline, pool, PEs, or BVH format.

### 12.1 GETWF — FP windowed regfile read

A macro-op that reads `count` contiguous float slots starting at `start` into an
FP register group in one fetched instruction — the callback-side analog of the
`wait` writeback window. It expands (per-warp sequencer) into `count` micro-ops,
each streaming one regfile slot into one FP register, NaN-boxed, with no
int→float conversion (the slots already hold f32 bits).

Encoding (additive; CUSTOM1, funct3 = 6 — the callback-op group):

| field | meaning |
|---|---|
| funct2 = 2 | sub-op GETWF (funct2 0 = `CB_RET`, 1 = SETW, 3 = GETW) |
| funct7[6:2] | window **start slot** (0–31) |
| rs2 index | window **count** (1–8; the register field is an immediate) |
| rd | FP window **base register** (writes rd..rd+count-1) |

No source operands — the data comes from the RTU regfile (staged by the yielding
trace's `apply_callback_payload`). A dispatcher already treats its target
registers as scratch (there is no register-value save across the async trap,
only the scoreboard snapshot), so writing an FP group is no different from the
individual gets it replaces. Dispatchers that run real FP (the IS shader) carry
`__attribute__((interrupt("machine")))`, so the prologue already saves the
window registers.

Kernel API — a typed accessor for the common object-ray case:

```c
vx_objray_t objray;
vx_rt_get_objray(&objray);   // 1 macro-op -> f0..f5; was 6 get + 6 fmv
```

`GETW` is the GP twin (funct2 = 3, integer slots, no NaN-box); `vx_rt_wait` uses
both for the hit window (§12.2).

### 12.2 Trap-safe `wait`

The windowed `wait` is split so it composes with callback-yielding traces:

- **Block (single-op).** `wait` (funct3 = 7 funct2 = 1) is a SINGLE-OP block that
  reuses the park/revive path, so it survives a callback trap exactly like the
  retired register-file `WAIT` — a parked single op is revived by HW on terminal,
  whereas a parking *macro-op* could not have its writeback uops resumed after a
  trap flush.
- **Writeback.** The hit window is delivered by two separate non-blocking
  windowed reads the `vx_rt_wait` intrinsic emits next, each scoreboard-chained
  on the status word so they run post-terminal: `GETWF` (t/u/v → f0..f2) and
  `GETW` (primitive/instance/geometry IDs → t3..t5).

Two scheduler fixes are the real enablers (not the ABI shape):

1. *Resume-on-trap.* A `wstall` macro-op suspends the warp until it commits. If
   the async trap flushes it mid-flight it never commits, so its `resume_warp`
   never fires and the warp hangs. `raise_async_trap` now resumes the warp it
   takes over.
2. *mret/trap serialization.* A callback trap raised the same cycle an `mret`
   retires corrupts the warp's tmask/PC (restored vs. newly-trapped contexts
   race). The callback drain now defers a trap one cycle past an `mret`.

With these, all callback kernels (AHS/IS/CHS/MISS/SBT), reformation, and
recursion run on `wait`; the recursive dispatcher's nested ray is `trace`/`wait`
in-trap.

### 12.3 Sync trace — no new opcode

`vx_rt_trace_sync(cfg, ray, &hit)` ships as a thin inline `trace`+`wait` wrapper
(resolving §9 Q3). A true fused instruction would have to park mid-macro-op and
forfeit the async overlap that motivates the trace/wait split, to save a single
instruction fetch — not worth an opcode. When the kernel has independent work it
issues the two separately and the compiler fills the gap.

### 12.4 Status

Implemented in SimX: the trap-safe single-op `wait` block, `GETWF`/`GETW`
windowed reads, and the two scheduler fixes; `vx_rt_get_objray` /
`vx_rt_trace_sync` in the kernel header. All `tests/raytracing/*` kernels pass on
SimX on the v2 ABI. RTL decodes the v2 funct3 = 6/7 ops (`SETW` writeback
included). Trap-per-yield latency is microarchitecture, not ABI.
