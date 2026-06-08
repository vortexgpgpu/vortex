# gfx_v2 — OM v2 (mobile-class fixed-point output-merger)

**Scope:** the redesigned output-merger (ROP) — a mobile-class, **fixed-point**
(FP-free) feature set with a redesigned **ISA** (quad-rate / multi-RT `vx_om4`)
and **ABI** (per-render-target state), designed for **composition**: OM is a
*building block* whose primitives a thin SW layer wraps to reach advanced
features (MSAA, programmable blend, even FP/HDR) without dedicated datapaths,
instead of falling to pure software. Sibling of
[gfx_v2_tex_v2.md](gfx_v2_tex_v2.md); detail behind
[gfx_v2_ff_expansion_roadmap.md](gfx_v2_ff_expansion_roadmap.md) §1.1/§3.
**Reference:** mobile TBDR ROP (PowerVR/Mali/Adreno); ISA doctrine
[custom_accelerator_isa_extensions.md](../designs/custom_accelerator_isa_extensions.md).
**Tree:** `~/dev/vortex_v3/gfx_v2` (proposed branch `feature_gfx_v2`).
**Status:** Proposal — implements [gfx_v2_true_gpu_charter.md](gfx_v2_true_gpu_charter.md) §6.8 (OM).
**Date:** 2026-06-07.
**Related:** [gfx_v2_ff_expansion_roadmap.md](gfx_v2_ff_expansion_roadmap.md),
[gfx_v2_software_fallback.md](gfx_v2_software_fallback.md),
[graphics_fixed_function_pipeline.md](../designs/graphics_fixed_function_pipeline.md).

---

## 1. Current OM (gfx-v1) — already feature-rich in the ABI

From [VX_types.toml](../../VX_types.toml) + [hw/rtl/om/](../../hw/rtl/om/):

- **Depth:** 24-bit, 8 compare funcs. **Stencil:** 8-bit, full 8-op state — both
  present in HW/ABI; the gfx-v1 **driver just disables stencil**.
- **Blend:** 5 equations (ADD/SUB/REV_SUB/MIN/MAX) + a LOGICOP mode; **15 blend
  funcs** including `CONST_RGB/A` and `ALPHA_SAT`. **Logic ops:** full set.
- **ISA:** `vx_om(x, y, face, color, depth)` — CUSTOM1 `funct3=2`, R4-type.
- **Pipeline:** `VX_om_ds (depth+stencil) → VX_om_blend / VX_om_logic_op →
  VX_om_mem (RMW color+depth via ocache)`.

**Key consequence:** much of what the vortexpipe §3.7 audit flagged as "missing"
(stencil, constant-alpha, logic-op) is **already in HW/ABI — it just needs
driver wiring**, not new silicon. The only genuinely-new dedicated hardware is
**MRT**; **MSAA, programmable blend, and even FP/HDR targets are handled by
*composition*** (§3.4/§5) — a thin SW layer over OM primitives — so they need no
dedicated MSAA/float datapath. The rest is cheap additions + enablement.

## 2. v2 feature set (all fixed-point)

| Feature | In HW (v2) | Cost | Note |
|---|---|---|---|
| **Stencil** | enable | ~free | HW/ABI exist; wire the driver |
| **Logic-op** | enable | ~free | HW/ABI exist; wire |
| **Constant-alpha / full blend-func set** | enable | ~free | already in the 15-func ABI |
| **Independent RGB vs alpha** blend eq/func | yes | low | wire the alpha path (gfx-v1 dropped it) |
| **Alpha test** (func + ref, discard) | yes | low | alpha-tested foliage etc. |
| **sRGB write** (linear→sRGB encode on store) | yes | low | per-RT bit; fixed-point encode |
| **Dual-source blend** (SRC1_RGB/A) | yes | low–med | +2 funcs + 2nd FS source |
| **MRT** (N render targets) | yes | **med (new)** | per-RT state + `vx_om4` multi-RT; deferred shading |
| **MSAA 2× / 4×** | **composed** | low–med | per-sample `vx_om4` + SW resolve (§5) — no dedicated MSAA datapath |
| **Programmable blend / custom ROP / framebuffer-fetch** | **composed** | low | `vx_om_fetch` → FS math → `vx_om4(replace)` (§5) |
| **Float / HDR render targets** (RGBA16F, R11G11B10F) | **composed** | low–med | OM does raw fetch/store + addressing; SW does FP math (§5) |
| **Float depth** (D32F) | **composed** | low | same — OM memory path + SW FP compare (§5) |
| 8×+ MSAA, or no fragment interlock available | no → SW | high | pure-SW last resort (§5) |

## 3. ISA redesign — `vx_om4` is the **sole** submit op

OM is fed only by the raster/FS quad path (no compute→OM path; compute writes go
via the LSU). Fragments always arrive as 2×2 quads, and the ROP datapath is
per-fragment regardless of the op — so the gfx-v1 single-fragment `vx_om`
**is removed**: `vx_om4` with a 4-bit coverage mask subsumes full quads,
partial-coverage edges, and the degenerate single-fragment case (mask = 1 bit) at
no HW or perf cost, and reclaims the CUSTOM1 `funct3=2` encoding as R-type.

### 3.1 Encoding (R-type / R2, per doctrine §2.5)

CUSTOM1, `funct3 = 2`, **R-type** (not R4) — R-type keeps the **7-bit `funct7`**
free for sub-op fields, and the wide per-quad payload is addressed by a **base
register + group convention** (§2.3), which costs nothing on operand throughput
(§2.5). No `rd`: OM submit is a **fire-and-forget store** into the OM unit
(`rd = x0`), the analog of the custom-store pattern (§2.4), optionally async with
back-pressure (§5).

```
  vx_om4   rs1 = quad descriptor,   rs2 = payload-window base,   funct7 = sub-op
           (rd = x0)
```

### 3.2 How the arguments reach hardware

- **`rs1` — quad descriptor (one GP register).** Packs the quad origin and
  coverage: `{ qy[15:0]? , qx , cov_mask[3:0] , sample bits }` — the same
  `pos_mask` the FS already gets back from `vx_rast` (vortexpipe §3.4). One
  scalar the FS wrapper already holds; no marshalling.
- **`rs2` — base of the per-thread payload register window (§2.3).** One thread
  owns one quad; having shaded its ≤4 covered sub-pixels, their results are a
  contiguous, just-computed **register group** the op reads directly. Slot map,
  **single render target**:

  ```
    base+0 .. base+3 : color[0..3]   one packed RGBA8 word per sub-pixel (fixed-pt)
    base+4 .. base+7 : depth[0..3]   fixed-point depth per sub-pixel
  ```

  **MRT (N targets):** colors expand to `N×4` words (`base+0 .. base+4N-1`),
  depth at `base+4N .. base+4N+3`. The FS's per-RT color outputs fill the window
  in RT-major order.
- **`funct7` — sub-op:** `{ RT_count(N), sample_count (1/2/4), op/reserved }`.
  7 bits is ample (the reason for R-type).

**All GP, no FP — which reinforces the FP-free decision (§1.1):** colors are
packed fixed-point words and depth is fixed-point, so the entire window lives in
the **integer** register file. The §2.3 type-split is trivially satisfied (no
`fmv`, no FP-regfile pressure), and there is no float anywhere in the OM payload
or datapath.

### 3.3 Execution (macro-op, doctrine §3)

`vx_om4` is a **macro-op**: the per-warp sequencer expands it into a run of ROP
uops — **one per (covered sub-pixel × render target)** — each reading its
`color[r][i]` + `depth[i]` from the window via the operand collector (≤3 regs per
uop) and `cov_mask` skipping uncovered sub-pixels. Each uop runs the
depth/stencil → blend/logic-op → RMW datapath of `VX_om_*`. Uop count ≈
`⌈window / read_ports⌉` (e.g. single-RT 8-word window → ~3 uops; MRT scales by
N). The macro-op stalls fetch until the run drains; only the uops commit.

### 3.4 Composition primitives — OM as a building block

OM is **not monolithic**. Beyond the fused `vx_om4`, it exposes the *stages* of
the RMW so a thin SW layer can compose features OM doesn't natively support
**while still using OM's expensive path** — framebuffer addressing, format
encode/decode, sample/MSAA addressing, ocache locality, and per-pixel ordering.
The principle: design OM so SW wraps the fast FF block as an inner loop, rather
than bypassing it.

Primitives (modes of `vx_om4` + one new op):

- **`vx_om4` fused** — depth/stencil → blend → RMW; the common path.
- **per-sample mode** (`funct7.sample`) — RMW one MSAA sample; SW loops samples
  + resolves → **MSAA without a dedicated MSAA datapath**.
- **replace / raw mode** (`funct7`, blend = overwrite) — the **write half**:
  store a SW-computed color through OM's format/sample/ocache path.
- **`vx_om_fetch(pos)`** — the **read half** (new op; returns data, so it has an
  `rd` result window): read the current dest color/depth for the quad through
  OM's addressing/format/sample path into registers.

With the read + write halves, SW composes **programmable blend / custom ROP**:
`vx_om_fetch → FS computes any function → vx_om4(replace)` — HW does the
framebuffer memory + format + ordering, SW does **only the math**. **Even
float/HDR targets** become HW-composed: OM fetches/stores the raw pixel and owns
addressing/ocache/sample, SW does the FP decode+blend+encode — strictly better
than pure-SW LSU (which loses ocache locality, sample addressing, and per-pixel
ordering).

**Constraint — per-pixel ordering.** Fused `vx_om4` is atomic and draw-ordered
per pixel. A decomposed `fetch → compute → replace` is correct only if no other
fragment RMWs the same pixel in between — it needs a **fragment interlock**
(raster-ordered access). OM guarantees this for the fused op; the decomposed path
requires the interlock (an OM ordering primitive, or the tile-resident model).
Without it, that case falls to pure-SW with SW serialization.

## 4. ABI redesign — per-render-target state

Today OM state is a single color + single depth block
([VX_types.toml](../../VX_types.toml) `VX_DCR_OM_STATE_*`). v2 replicates the
**color** state per RT and adds the new fixed-point controls:

- **Per-RT block × N:** `CBUF_ADDR / PITCH / FORMAT / BLEND_MODE / BLEND_FUNC
  (RGB) / BLEND_FUNC_ALPHA / BLEND_MODE_ALPHA / WRITEMASK / SRGB_WRITE`.
- **Shared:** depth/stencil block (1 depth buffer); `ALPHA_TEST_FUNC + REF`;
  `SAMPLE_COUNT` (1/2/4) + sample positions; `DUAL_SRC_ENABLE`; `CONST_COLOR`.
- Independent RGB/alpha is just the separate `*_ALPHA` fields (the blend-func
  enum already exists; gfx-v1 packed one mode for both and dropped alpha).

Stays **DCR-resident** for v2 (small N, e.g. 4 RTs). MSAA sample storage lives in
the resident framebuffer (charter pillar 4); the resolve is an OM pass or the SW
path for the exotic cases.

## 5. The three-tier cut-line (composition first)

The HW/SW decision is **not binary**. OM features classify into three tiers, and
the driver picks the **highest tier a pipeline allows**. Composition (tier 2) is
preferred over pure-SW wherever OM can be a building block — so **most
"unsupported" OM features land in tier 2, not tier 3**.

| Feature | Tier | How |
|---|---|---|
| depth(8) / stencil(8) / blend(5×15) / logic-op / writemask / alpha-test / sRGB / dual-source / independent-RGBA / **MRT** | **1 — native HW** | §2 (mostly enablement + cheap additions) |
| **MSAA 2×/4×** | **2 — HW-composed** | per-sample `vx_om4` + SW resolve; no dedicated MSAA datapath |
| **Programmable blend / custom ROP** | **2 — HW-composed** | `vx_om_fetch` → FS math → `vx_om4(replace)` (needs fragment interlock) |
| **Float/HDR targets, D32F depth** | **2 — HW-composed** | OM owns raw fetch/store + addressing/ocache; SW does the FP math |
| 8×+ MSAA, or a case needing per-pixel ordering with **no** fragment interlock | **3 — pure SW** | SW-serialized ROP — last resort |

So composition cuts **both** ways: it **shrinks dedicated HW** (MSAA need not be
a datapath) *and* **shrinks pure-SW** (even FP targets reuse OM's memory path).
Pure-SW OM is the rare last resort. Conformance is preserved at every tier
([gfx_v2_software_fallback.md](gfx_v2_software_fallback.md)).

## 6. Validation & phasing

SimX-first (the SW ROP shares `gfx_render.cpp` math as the oracle), then RTL at
300 MHz on U55C:
1. **Enablement (~free):** stencil, logic-op, full blend-func set, independent
   RGB/alpha — mostly driver wiring + small HW; closes most of the §3.7 audit.
2. **Cheap additions:** alpha test, sRGB write, dual-source.
3. **MRT** (`vx_om4` multi-RT + per-RT ABI) — the first genuinely-new HW.
4. **Composition primitives** (§3.4): per-sample mode, replace mode,
   `vx_om_fetch` + the fragment-interlock guarantee — unlocks **MSAA (composed)**,
   programmable blend, and FP/HDR targets via SW orchestration, instead of
   dedicated MSAA/float datapaths.

Each diffs against the SW ROP and the lavapipe oracle.

## 7. Open items

- **Payload layout — register window (§3.2) vs lane-packed.** The doc uses a
  per-thread register window (one thread owns one quad), matching the gfx-v1 FS
  structure. An alternative maps the 2×2 quad onto 4 SIMD lanes (lane = sub-pixel,
  §2.2), shrinking the window to 1 color + 1 depth register per RT but requiring
  the FS to shade one sub-pixel per lane. Pick by FS codegen + register-pressure
  measurement.
- **Fragment interlock** — the ordering guarantee the composed `vx_om_fetch →
  compute → vx_om4(replace)` path (§3.4/§5) needs: an OM raster-ordered access
  primitive vs the tile-resident-framebuffer model. The enabler for composed
  programmable-blend and FP/HDR; without it those fall to pure-SW.
- **`vx_om_fetch` encoding** — the read-half op (returns dest into an `rd`
  window) and how it shares the CUSTOM1 budget with `vx_om4`.
- **Async submit** — `vx_om4` as fire-and-forget vs a handle+`wait` (ISA
  doctrine §5) if OM back-pressure needs to overlap; baseline is fire-and-forget
  with bounded in-flight back-pressure to the issuing warp.
- **`funct7` field budget** — `{RT_count, sample_count, op}` vs the CUSTOM1
  encoding shared with RASTER/TEX.
- **MRT count** (4 vs 8) and **DCR vs descriptor** for per-RT state.
- **MSAA storage/resolve** — tile-resident sample buffers vs ocache RMW; the
  cost driver, possibly capped at 4× in HW.
- **Stencil enablement** — confirm the existing HW path is complete end-to-end
  (it was driver-disabled, so may be under-exercised).
- **Dual-source** — the FS ABI for the second color source.
