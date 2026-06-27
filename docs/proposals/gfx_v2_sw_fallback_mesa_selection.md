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

- **OM** (DONE): `has_om` → FF `vx_om4`; else the FS wrapper calls
  `gfx_om_fragment_sw(&omstate, px, py, face, rgba, depth)` per covered sub-pixel
  with a resident `gfx_sw_omstate_t` (built host-side via the resolve_om_state
  mirror), writing color/depth via the LSU. Validated.

- **RASTER** (foundation DONE — `gfx_rast_walk_tile_sw` C-ABI; wrapper REMAINING).
  This is the one genuinely-distinct fork: it changes the FS kernel's whole loop
  shape and needs **per-pixel ordering** (gfx_v2_software_fallback.md §11) — so
  **one thread per screen TILE iterating all prims in draw order** (NOT one
  thread per prim, which races the OM RMW on overlapping geometry). Concrete plan:
  - C-ABI `gfx_rast_walk_tile_sw(prim, pid, tx, ty, tile_logsize, W, H, out[], max)
    → count` (DONE): walks one prim over one tile, appends covered quads
    (`gfx_rast_quad_t{pos_mask, bcoords[12]}`, FF-frag-payload layout).
  - `emit_fs_wrapper_sw_raster` (new IR variant; the HW wrapper stays untouched):
    `kernel_main(arg)` with a per-thread `gfx_rast_quad_t out[MAX]` alloca.
    `tile_idx = blockIdx*blockDim+threadIdx`; `if (tile_idx >= num_tiles) return;`
    `tx=(tile_idx%nx)*tile, ty=(tile_idx/nx)*tile`; `for pid in 0..num_prims:
    count = gfx_rast_walk_tile_sw(...); for k in count: decode pos_mask, read
    bcoords from out[k] (not the frag window), then the SAME shade+OM body`
    (dx/dy → fill_varyings → fs_main → pack → vx_om4 or gfx_om_fragment_sw).
    Factor the shade+OM body of the HW wrapper into a shared `emit_shade_quad`
    so both variants reuse it (bcoords source = frag_payload vs buffer).
  - `vp_raster_draw` (sw_raster): still runs the front end (expand/setup/bin) to
    produce the dense `rast_prim_t[]` primbuf, but skips the RASTER DCRs/producer;
    sizes the FS launch grid to cover `num_tiles` threads; passes
    `num_prims / nx / num_tiles / tile_logsize` via arg slots [3..6]; primbuf
    non-`VX_MEM_PHYS`. `num_prims` from the front-end meta (= num_tris when no
    clipping; clipping → read the meta count).
  - MAX bound: choose `tile_logsize` so a tile's quad count ≤ MAX (e.g. 16px tile
    → ≤64 quads); log if a tile overflows.
  - Open question: reusing the per-warp frag *window* (SETW-stage instead of a
    buffer) would avoid the body refactor but needs the per-warp/lane frag model
    pinned down (SETW→GETW window timing); the buffer approach above is
    deterministic and preferred.

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
