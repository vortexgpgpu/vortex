# gfx_v2 — TEX v2 (mobile-class fixed-point texture unit)

**Scope:** the redesigned texture sampler — a mobile-class, **fixed-point**
(FP-free) feature set with a redesigned **ISA** (quad-rate `vx_tex4` for
hardware LOD) and **ABI** (expanded sampler/image state), designed for
**composition**: TEX is a *building block* whose taps a thin SW layer combines to
reach advanced filtering (anisotropic, bicubic, PCF) and formats it doesn't
natively decode — while HW always owns addressing + tcache + filtering. Because
sampling is a pure function (no RMW, no ordering), TEX composes *freely* — the
asymmetry with OM. Sibling of [gfx_v2_om_v2.md](gfx_v2_om_v2.md); detail behind
[gfx_v2_ff_expansion_roadmap.md](gfx_v2_ff_expansion_roadmap.md) §1.1/§3.
**Reference:** mobile TBDR samplers (PowerVR/Mali/Adreno); ISA doctrine
[custom_accelerator_isa_extensions.md](../designs/custom_accelerator_isa_extensions.md).
**Tree:** `~/dev/vortex_v3/gfx_v2` (proposed branch `feature_gfx_v2`).
**Status:** Proposal — implements [gfx_v2_true_gpu_charter.md](gfx_v2_true_gpu_charter.md) §6.8 (TEX).
**Date:** 2026-06-07 (rev. 2026-06-12).
**Decisions (2026-06-12, ahead of the `vx_tex4` migration):** coords are strictly
2D `(u,v)`, with cube/3D/array handled SW-composed in the FS (§3.2/§3.4); the
payload uses the **register-window** layout reusing the RTU `SET`/`GET` mechanism
(§7); v1 `vx_tex` is **removed**, not extended, and the interim mip-filter DCR bit
folds into `funct7` (§4).
**Related:** [gfx_v2_ff_expansion_roadmap.md](gfx_v2_ff_expansion_roadmap.md),
[gfx_v2_software_fallback.md](gfx_v2_software_fallback.md),
[graphics_fixed_function_pipeline.md](../designs/graphics_fixed_function_pipeline.md).

---

## 1. Current TEX (gfx-v1) — the starting point

From [VX_types.toml](../../VX_types.toml) + [hw/rtl/tex/](../../hw/rtl/tex/):

- **Stages:** `VX_TEX_STAGE_COUNT = 2` already.
- **Formats (7, fixed-point):** A8R8G8B8, R5G6B5, A1R5G5B5, A4R4G4B4, A8L8, L8, A8.
- **Filter:** POINT / BILINEAR (1 bit) — **no mip filter**.
- **Wrap:** CLAMP / REPEAT / MIRROR.
- **Mip ABI exists but unused:** `VX_DCR_TEX_MIPOFF_BASE + lod` reserves mip
  offsets, but the driver samples LOD 0 only and there is no mip-filter bit.
- **ISA:** `vx_tex(stage, u, v, lod)` — CUSTOM1 `funct3=1`, R4-type,
  `funct2 = stage`. LOD is **explicit** (no hardware derivative/LOD selection).
- **Pipeline:** `VX_tex_addr → VX_tex_mem → VX_tex_format → VX_tex_sampler/lerp
  → VX_tex_sat`, wrap in `VX_tex_wrap`.

So a real mip ABI and a 2-stage sampler already exist — v2 is mostly *enabling
mip filtering, adding hardware LOD, and growing the fixed-point format/wrap set*,
not a from-scratch unit.

## 2. v2 feature set (all fixed-point)

| Feature | In HW (v2) | Cost | Mobile rationale |
|---|---|---|---|
| **Mipmapping + trilinear** | yes | low (mip ABI exists; +mip-filter bit; one extra lerp) | essential; every textured draw |
| **Hardware LOD via `vx_tex4`** | yes | low–med (quad derivatives, §3) | enables mip/trilinear without explicit lod |
| **sRGB8 / sRGB8A8** (decode on sample) | yes | low (piecewise/LUT, fixed-point) | ubiquitous |
| **ETC2 / EAC compressed** | yes | med (block decode) | the mobile texture-compression staple; bandwidth win |
| **Wrap: CLAMP_TO_EDGE, MIRROR_CLAMP** | yes | low | common addressing |
| **Cubemap** | **composed** | low | FS does major-axis face-select + projection → 2D `(u,v)` + face; HW does per-face addressing + filtering (§3.4) |
| **Depth-compare / shadow (Dref)** | yes | low (compare after fetch) | shadow maps |
| **Stages 2 → 4** | yes | low (more state) | multitexture |
| **Anisotropic** | **composed** | low | N trilinear taps + SW weight (§3.4) — not a SW sampler |
| **Bicubic / PCF / gather** | **composed** | low | multi-tap `vx_tex4` + SW combine (§3.4) |
| **ASTC** | **composed** | med | raw-fetch the block via tcache + SW decode (§3.4) |
| **Float/integer textures** (R16F, R32F, R32I…) | **composed** | low | raw-fetch + SW FP decode/filter (§3.4) — tcache reused |
| 3D textures, large arrays | **composed** | low | HW 2D-slice/layer taps + SW interpolate (§3.4) |

## 3. ISA redesign — `vx_tex4` is the **sole** sample op

Applying the same "one op" reasoning as OM (om_v2 §3): on reflection the gfx-v1
single-sample `vx_tex` is **also removed** — `vx_tex4` with a **mode bit** in
`funct7` subsumes it. Two modes:

- **quad / derivative-LOD** (raster FS): four `(u,v)`; HW computes
  `du/dx, dv/dy` across the quad → selects LOD → trilinear → returns 4 texels.
- **single / explicit-LOD** (compute sampling): one `(u,v)` + an explicit LOD in
  `rs1`; HW skips the derivative step → returns 1 texel.

So the **compute-sampling case** (no quad, hence no derivatives — the case that
looked like it needed a separate `vx_tex`) is just the single mode, exactly as
OM's single-fragment case is just a 1-bit coverage mask. TEX therefore collapses
to one op too, symmetric with OM — no separate `vx_tex`. (`vx_tex4` differs from
`vx_om4` only in that it **returns** data, so it has an `rd` result window where
OM has none.)

### 3.1 Encoding (R-type / R2, per doctrine §2.5)

CUSTOM1, `funct3 = 1`, **R-type** — `funct7` carries the sub-op fields. Unlike
OM, there **is** an `rd` (texels are returned).

```
  vx_tex4   rd = result-window base,   rs1 = config scalar,   rs2 = (u,v)-window base
            funct7 = { stage, mode(quad|single), mip-filter, compare(Dref), format-ovr }
```

### 3.2 How the arguments reach hardware

- **`rs1` — config scalar:** explicit LOD (single mode) / LOD bias (quad mode) /
  Dref compare value. One GP register; `0` when unused.
- **`rs2` — base of the per-thread `(u,v)` input window (§2.3).** One thread owns
  the quad; the just-computed coords are a contiguous register group:

  ```
    quad mode:   base+0..3 : u[0..3]   base+4..7 : v[0..3]   (S.23 fixed-point)
    single mode: base+0    : u         base+1    : v
  ```

- **`rd` — base of the result texel window (writeback, doctrine §4):**
  `rd+0..3 : texel[0..3]` (quad) or `rd+0 : texel` (single), one packed RGBA8
  word per fragment.
- **`funct7` — sub-op:** `{ stage, mode, mip-filter, Dref, format override }`.

**All GP, no FP (reinforces §1.1):** `(u,v)` are **S.23 fixed-point** (as gfx-v1
already converts UVs) and texels are packed RGBA8 — the whole input/result lives
in the **integer** register file, no `fmv`, no FP datapath in TEX.

**Coordinates are strictly 2D `(u,v)`.** The op carries no third (`w`/`r`)
component: projective `w` is resolved upstream (RASTER `u/w,v/w,1/w` + FS divide),
and every case that needs a third coordinate is **SW-composed in the FS** over 2D
taps (§3.4) — cubemaps (FS major-axis face-select + projection → `(u,v)` + face),
3D textures, and array layers. HW always owns 2D addressing + tcache + filtering;
SW only computes the coordinate/face. This keeps the window 2 coords wide and the
unit uniform.

### 3.3 Execution (macro-op, doctrine §3)

`vx_tex4` is a **macro-op**: the sequencer reads the `(u,v)` window, computes the
cross-fragment derivatives **once per quad** (quad mode), then emits per-fragment
work — address (`VX_tex_addr`) → tcache fetch (4 texels/bilinear × 2 mips/
trilinear) → filter (`VX_tex_sampler`/`lerp`) → format decode → writeback to the
result window. Every fetch is a real tcache request (honest timing). Baseline is
**synchronous** (latency hidden by warp scheduling, as gfx-v1 `vx_tex`); an
**async handle+`wait`** variant (§5) is an option if sample latency needs deeper
overlap.

### 3.4 Composition primitives — TEX as a building block

Sampling is a **pure function** — no RMW, no per-pixel ordering — so TEX composes
**freely**, the asymmetry with OM (whose composition is bounded by the
framebuffer-ordering constraint, om_v2 §3.4). A thin SW layer orchestrates
multiple fast `vx_tex4` taps to build filters TEX doesn't do natively, while HW
always owns **addressing + tcache + format decode + filtering** (the expensive
part).

Primitives:

- **`vx_tex4` filtered tap** (single/quad, explicit LOD + coords, §3.2) — a
  bilinear/trilinear sample at a SW-chosen `(u,v,lod)`. The multi-tap building
  block.
- **raw-fetch mode** (`funct7` — `texelFetch`: integer coords, no filtering,
  optional no-decode) — the lowest-level primitive: return raw texel(s) for
  fully custom SW filtering, or for formats/decoders the HW lacks.

SW compositions (the HW-composed tier):

- **Anisotropic** = N trilinear taps along the major axis (from the quad
  derivatives) + SW weighted sum.
- **Bicubic / higher-order** = 4 bilinear taps + SW cubic weights; **PCF
  shadows** = N depth-compare taps + SW average; **`textureGather`** = taps + SW
  reorg.
- **Formats / decoders HW lacks** (ASTC, float, integer) = **raw-fetch the
  block/texel via tcache + SW decode/filter** — HW still does addressing +
  caching, SW does only the decode/math. So even **ASTC and float textures are
  HW-composed** (tcache reused), not pure-SW.
- **Cubemap** = FS computes major-axis face + face-local `(u,v)` from the
  direction vector, then a single 2D `vx_tex4` tap with the face's per-face
  addressing (`DIM`). HW does the addressing + filtering; SW does only the
  face-select math.
- **3D / large arrays** = HW 2D-slice / per-layer taps + SW interpolate.

Because there is **no ordering constraint**, every one of these keeps the
expensive memory + filter work on the HW path. Pure-SW TEX is essentially empty.

## 4. ABI redesign — expanded sampler/image state

Grow the per-stage DCR block ([VX_types.toml](../../VX_types.toml)
`VX_DCR_TEX_STATE_BEGIN`), replicated for N=4 stages:

- `FILTER` — mag/min filter (point/bilinear) + aniso level bits. **Mip-filter
  is per-op in `vx_tex4`'s `funct7`** (§3.1), not a DCR field — it varies per
  sample (e.g. explicit-LOD vs derivative-LOD). The interim
  `VX_TEX_FILTER_MIP_LINEAR` DCR bit added with the v1 trilinear work folds into
  that `funct7` field when `vx_tex4` lands, and the DCR bit is then removed with
  the rest of the v1 ISA.
- `FORMAT` — add sRGB8 / sRGB8A8, ETC2/EAC codes, and a **sample-type** field
  (color vs depth-compare).
- `WRAP` — add CLAMP_TO_EDGE, MIRROR_CLAMP_TO_EDGE; per-axis (s/t/r).
- `COMPARE` — Dref compare func (reuse the 8 depth-compare codes) + border.
- `DIM` — cube / array flags + layer count (LOGDIM already carries dims).
- Mip offsets (`MIPOFF`) — already present; now actually consumed.

The block stays **DCR-resident** for v2 (4 stages fit). A **memory-resident
sampler+image descriptor table** (bindless) is the v2.x scalability step
(roadmap Tier 3) — same datapath, descriptor fetch instead of DCRs.

## 5. The three-tier cut-line (composition first)

As for OM, the HW/SW decision is **not binary** — and because TEX composes
freely, the composed tier is large and pure-SW is nearly empty. The driver picks
the highest tier a pipeline allows:

| Feature | Tier | How |
|---|---|---|
| mip/trilinear, sRGB, ETC2, wrap modes, depth-compare, 4 stages, point/bilinear | **1 — native HW** | §2 |
| **cubemap**, anisotropic, bicubic / PCF / gather, ASTC, float/integer textures, 3D / arrays | **2 — HW-composed** | multi-tap or raw-fetch + SW combine/decode (§3.4); HW always does addressing + tcache |
| an access HW can't even raw-fetch (totally exotic addressing) | **3 — pure SW** | last resort — essentially empty |

So TEX composition keeps **everything** on the HW path for the expensive memory +
filter work; the SW layer only places taps or decodes. Conformance is preserved
at every tier ([gfx_v2_software_fallback.md](gfx_v2_software_fallback.md)).

## 6. Validation & phasing

SimX-first (the SW sampler is the oracle, since it shares
`gfx_render.cpp` math), then RTL at 300 MHz on U55C:
1. **Mip + trilinear + `vx_tex4`** (hardware LOD) — the highest-value core.
2. **sRGB + ETC2 formats**, extra **wrap** modes — format/addressing growth.
3. **Depth-compare (shadow)**, **4 stages**.
4. **Composition** (§3.4): the `vx_tex4` filtered tap + **raw-fetch mode** —
   unlocks **cubemap** (FS face-select), anisotropic, bicubic/PCF/gather, ASTC,
   float textures, and 3D/arrays as SW-orchestrated multi-tap/decode over the HW
   path, instead of pure-SW.

Each diffs against the SW sampler and the lavapipe oracle.

## 7. Open items

- **Payload layout — RESOLVED: register window** (§3.2), reusing the RTU's
  `SET`/`GET` slot-window macro-op mechanism. Chosen over the lane-packed
  alternative (mapping the quad onto 4 SIMD lanes) because it keeps the existing
  one-thread-owns-the-quad FS model and is expressible in LLVM inline asm
  (lane-packed would force the 4 quad pixels into adjacent SIMT lanes). Applies
  symmetrically to `vx_om4` (om_v2 §7).
- **`funct7` field budget** — `{stage, mode, mip-filter, Dref, format, raw-fetch}`
  vs the CUSTOM1 encoding shared with RASTER/OM.
- **Raw-fetch / `texelFetch` mode** — exact semantics (no-filter, no-decode) and
  how it shares the `vx_tex4` encoding; the lowest-level composition primitive.
- **Composed-aniso tap count / weights** vs a basic HW aniso tap — cost/quality.
- **ETC2 vs also EAC/BC** in native HW — pick by target content; ASTC is composed.
- **Aniso in HW vs SW** — basic 2-tap HW vs full SW; cost-driven.
- **DCR vs bindless descriptor** — when stage count / image count outgrows DCRs.
- **sRGB precision** — piecewise-linear vs LUT decode, fixed-point error budget.
