# gfx_v2 §5 — mesa vortexpipe per-unit HW-vs-SW selection

**Status:** design/plan for review. The device side of §5 is complete and
RTL-proven; this scopes the remaining driver-side selection. Companion to
`docs/archives/proposals/gfx_v2_software_fallback.md` (the SW-fallback charter,
§5/§7).

## 1. Goal

Make the vortexpipe Vulkan/GL driver select, **per pipeline unit (TEX / OM /
RASTER), at FS-compile time**, either the fixed-function intrinsic or the
on-device software fallback — driven by (a) device caps (`VX_CAPS_ISA_FLAGS`:
`has_tex/has_raster/has_om`) and (b) pipeline state (does the draw need a feature
the FF unit lacks). When a unit is absent or unfit, route **that unit** to its
SIMT software path — **never to llvmpipe** (full residency, charter pillar 4).

The device side is done and RTL-proven (this session): `gfx_tex4 -S`
(`gfx_sw::tex_sample_sw`), `gfx_om -S` (`gfx_sw::om_fragment`), `gfx_raster -z`
(`gfx_rast::rast_walk_primitive`) all pass on simx + rtlsim vs the FF goldens.
This note is only the **driver glue** that picks per unit and feeds the SW paths
from a real draw.

## 2. Current state (what exists today)

- `vp_screen.c` caches `has_tex / has_raster / has_om` from `VX_CAPS_ISA_FLAGS`
  (lines 108–110).
- `vp_context.c:1114` selection is **coarse / all-or-nothing**:
  `gfx_hw = has_raster && has_om`; if `gfx_hw && tex_needed && !has_tex` it sets
  `gfx_hw = false` → the whole draw **falls back to llvmpipe** (charter
  violation). `VORTEXPIPE_SW_RASTER` forces llvmpipe.
- The FS is compiled **NIR→LLVM IR** in `vp_nir_to_llvm.c`; `emit_vx_tex`
  (line 980) emits the `vx_tex4` inline-asm (`.insn r 43, 5, …`) for
  `nir_texop_tex` (line 1122). `vp_compile.c` links the IR with
  `$VORTEX_HOME/sw/kernel/libvortex2.a` + baremetal libc/compiler-rt via
  llvm-vortex clang (`-fuse-ld=lld`).
- The HW draw path is `vp_raster_draw` (VS folded into OP_DRAW; FF RASTER→OM;
  FS pulls quads via `vx_rast_fetch`).

## 3. Recommended architecture: linked-call to a C-ABI `libgfx_sw`

Emit, from the FS IR, a **call** to a precompiled device function (not inline
IR — inline duplicates the math and diverges from the §7 single-source-of-truth
headers). One SW implementation, shared by the FF model, the unit tests, and the
driver.

### 3.1 `libgfx_sw` device library (Vortex tree)

Add C-ABI entry points compiled from the existing freestanding headers for the
device target, with the libgfx_sw.mk flags (`-DGFX_SW_DIVERGENCE_OK
-mllvm -vortex-divergence-max-bbs=…`):

```c
// sw/gfx/gfx_sw_abi.h  (C ABI; thin extern "C" wrappers over the C++ headers)
extern "C" uint32_t gfx_tex_sample_sw(const gfx_sw_texstate_t* st,
                                      int32_t u, int32_t v, uint32_t lod);
extern "C" void     gfx_om_fragment_sw(const gfx_sw_omstate_t* st,
                                      uint32_t x, uint32_t y, uint32_t face,
                                      uint32_t color, uint32_t depth);
```

- `gfx_sw_texstate_t` / `gfx_sw_omstate_t` are POD mirrors of `gfx_sw::TexState`
  / `gfx_sw::om_state_t` (already POD); `gfx_sw_abi.cpp` static_asserts the
  layouts match. Implemented: `sw/gfx/gfx_sw_abi.{h,cpp}`.
- **Build realization of "fold" (decision 1), forced by divergence:**
  `libvortex2.a` is **gcc-built** and cannot carry `om_fragment`'s SIMT-divergent
  control flow (that needs the Vortex-LLVM divergence pass; a precompiled gcc
  object would be miscompiled). So instead of an archive, **co-compile
  `gfx_sw_abi.cpp` into each FS**: add it (and `-I sw/gfx -I sw/common
  -I third_party`, `-DGFX_SW_DIVERGENCE_OK -mllvm -vortex-divergence-max-bbs=512`)
  to `vp_compile.c`'s clang invocation, which already takes the FS `.ll`. clang
  compiles `.ll + .cpp` together and the divergence pass runs over the whole
  kernel → `gfx_om_fragment_sw` is transformed correctly. This still honors
  "fold": one SSOT lib from the headers, no hand-written IR, no separate archive.
  Verified: the source compiles to clang+xvortex bitcode (exports both symbols)
  and host (layout asserts pass).
- Raster SW is **not** an FS-IR call — it changes the *draw orchestration*
  (§3.3), so it stays a device kernel (the `gfx_raster -z` pattern), selected in
  `vp_raster_draw`, not in `emit_vx_tex`.

### 3.2 TEX fork in `emit_vx_tex` (the per-fragment call site)

```
if (has_tex && state_fits_FF)   emit  .insn r 43,5,…   (vx_tex4, today)
else                            emit  call gfx_tex_sample_sw(&texstate[stage], u, v, lod)
```

Plumbing the descriptor (the one real new dependency): the FS needs a **resident
`gfx_sw_texstate_t` per sampler stage**. Proposed: the driver builds a small
resident **tex-state table** (one entry per bound sampler view: base, logdim,
format, filter, wrap, mip offsets) and passes its device pointer to the FS via
the existing kernel-arg / a known resident slot; `emit_vx_tex` emits
`getelementptr` + `call`. The table is filled host-side from the bound
`pipe_sampler_view` + `pipe_sampler_state` (the same values the driver already
programs into the TEX DCRs).

### 3.3 OM + RASTER forks (draw-orchestration, in `vp_raster_draw`)

These change the **loop shape**, so they are two wrapper variants, not call-site
swaps (charter §5):

- **OM**: `has_om` → FF `vx_om4` in the FS; else the FS calls
  `gfx_om_fragment_sw(&omstate, …)` with a resident `gfx_sw_omstate_t` (built
  host-side via `resolve_om_state`, like `gfx_om -S`), writing color/depth via
  the LSU. cbuf/zbuf allocated **non-`VX_MEM_PHYS`** (LSU/dcache path).
- **RASTER**: `has_raster` → FF producer + `vx_rast_fetch` pull loop; else the
  **iterate-bin-buffer** kernel (the `gfx_raster -z` path): one thread per
  resident `rast_prim_t` walks the screen with `rast_walk_primitive`. prim_buffer
  non-`VX_MEM_PHYS`.
- **Zero-acceleration** (no FF units) = all three SW over the same bin-sort
  buffers; the natural bring-up + minimal-area config (charter §5.1).

## 4. Selection table (per draw, per unit)

| unit   | HW when… | SW path | resident state |
|--------|----------|---------|----------------|
| TEX    | `has_tex` ∧ format/filter FF-supported | `gfx_tex_sample_sw` (FS IR call) | tex-state table |
| OM     | `has_om` ∧ no MRT/exotic | `gfx_om_fragment_sw` (FS IR call) | om-state struct |
| RASTER | `has_raster` ∧ no MSAA/conservative | iterate-bin-buffer kernel | dense `rast_prim_t[]` |

No unit ever falls back to llvmpipe. The existing coarse `gfx_hw`/llvmpipe branch
in `vp_context.c` is replaced by always taking the device path with per-unit
HW/SW chosen as above.

## 5. Implementation plan (incremental, each validatable)

1. **`libgfx_sw` C-ABI + build** — add `sw/gfx/gfx_sw_abi.{h,cpp}` (extern "C"
   wrappers + POD state mirrors), compile into libvortex2.a with the
   divergence-bbs flag. Unit-test the C-ABI wrappers on host against the C++
   headers (reuse `gfx_tex_sw`/`gfx_msaa` oracles).
2. **TEX-only driver fork** — `emit_vx_tex` SW branch + resident tex-state table
   + the `vp_context.c` change so `has_raster+has_om` but `!has_tex` takes the
   device path with the SW sampler (removes the most common llvmpipe fallback).
   Validate: a textured draw on a TEX-less cap config → matches the all-HW image.
3. **OM driver fork** — FS emits `gfx_om_fragment_sw` when `!has_om`; cbuf/zbuf
   non-PHYS. Validate per depth/blend.
4. **RASTER driver fork** — `vp_raster_draw` selects the iterate-bin-buffer
   kernel when `!has_raster`. Validate a triangle draw.
5. **Zero-acceleration** end-to-end (all three SW) + the per-unit matrix.

## 6. Validation plan

- Per-unit matrix on the existing tests (already green): `gfx_tex4 -S`,
  `gfx_om -S`, `gfx_raster -z` — HW and SW each, simx + **rtlsim** (the SimX
  dcache thrash artifact makes rtlsim the oracle for LSU-heavy SW kernels; see
  `project_simx_dcache_lsu_thrash_bug`).
- Driver end-to-end: a vulkan/GL draw with caps forced per combination
  (HW-all, SW-tex, SW-om, SW-raster, zero-accel) → image matches the all-HW
  reference. Force caps via a `VORTEXPIPE_FORCE_SW=tex|om|raster|all` env knob.

## 7. Decisions (resolved)

1. **libgfx_sw location**: ✅ **fold** the C-ABI entry points into `libvortex2.a`
   (no driver link change).
2. **tex-state table delivery**: ✅ **kernel-arg pointer** (explicit, matches the
   device tests).
3. **Driver increment scope**: ✅ **all three units** (TEX + OM + RASTER) in one
   effort, not TEX-only first.

## 8. Effort / risk

- New C-ABI lib + build wiring: small.
- `emit_vx_tex` SW branch + descriptor plumbing: medium (FS IR + resident table).
- OM/RASTER wrapper variants in `vp_raster_draw`: medium-large (draw
  orchestration; reuses the proven device kernels).
- Risks: FS register/BB budget (mitigated by the divergence-bbs flag, already in
  libgfx_sw.mk); descriptor residency lifetime; cocogfx headers reaching the
  C-ABI lib (header-only, device-OK — already used by the device tests).
