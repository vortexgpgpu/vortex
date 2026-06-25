# gfx_v2 — P2: vx_tex4 quad mode + hardware LOD

**Scope:** add `vx_tex4` quad mode — one thread owns a 2×2 fragment quad, supplies
four `(u,v)` via the shared window, and the TEX unit computes **one integer mip
LOD from the quad derivatives** and returns four texels. Builds on P1 (single
mode). **Integer-mip only** (confirmed scope): no fractional/trilinear blend.
**Status:** Proposal — implements [gfx_v2_tex_v2.md](gfx_v2_tex_v2.md) §3 (quad mode); follows [gfx_v2_tex4_p1.md](gfx_v2_tex4_p1.md).
**Tree:** `~/dev/vortex_v3/prism_v3`. **Date:** 2026-06-12.

---

## 1. Why integer-mip only

RTL has **no working trilinear**: the texture bus carries a 4-bit *integer* LOD
([VX_tex_bus_if.sv](../../hw/rtl/tex/VX_tex_bus_if.sv)); `tex_core` does mipoff-LUT
selection only; there is no two-mip frac blend. (SimX's `gfx_render` *does*
trilinear, so `mip-filter=LINEAR` already diverges SimX↔RTL — a pre-existing gap.)
So P2's "hardware LOD" computes the **integer mip** from derivatives and samples
via the existing point/bilinear path. The fractional trilinear blend — and the
new HW datapath that would close the divergence — is a separate follow-up.

## 2. Encoding & window layout

`vx_tex4` `funct7.mode = 1` (quad). One thread owns the quad; rs2 = input window
slot base, funct7[6:2] = output slot base, **rs1 = texture dims** `{logh[31:16],
logw[15:0]}` (same layout as `VX_DCR_TEX_LOGDIM`; the kernel always knows the
bound texture's size — the sampler's descriptor dims), rd = texel + sync handle
(as P1). The LOD computes from the quad derivatives + these dims **entirely in
`VX_tex_unit`** — no DCR access, no `tex_core`/bus change (the tex DCRs live on
`dcr_bus_if` at the cluster level, out of the per-core TEX unit's reach; relaying
the dims through rs1 is the clean equivalent and keeps the derivative→log2 in HW).

```
  input window  (rs2 base):  base+0..3 = u[0..3]   base+4..7 = v[0..3]   (S.23)
  output window (out_slot):  out+0..3  = texel[0..3]
  frag layout: 0=(x,y) 1=(x+1,y) 2=(x,y+1) 3=(x+1,y+1)
```

The FS `SETW`s the 8 coords, issues `vx_tex4` quad, then a handle-chained `GETW`
reads the 4 texels (`out_slot..out_slot+3`).

## 3. Bit-exact LOD formula — [sw/common/vx_tex_lod.h](../../sw/common/vx_tex_lod.h)

```
gux=|u1-u0|<<logw  guy=|u2-u0|<<logw  gvx=|v1-v0|<<logh  gvy=|v2-v0|<<logh
rho = max(gux,guy,gvx,gvy)            // texel-space gradient, 23 frac bits
LOD = clamp( floor(log2(rho)) - 23, 0, 15 )
```

`logw/logh` are the halves of `VX_DCR_TEX_LOGDIM`. The single source of truth
(`vx_tex_quad_lod`) is included by SimX and the validation kernel; RTL replicates
it with `VX_lzc` (`floor(log2(rho)) = WIDTH-1 - lzc(rho)`).

## 4. RTL plan — all in `VX_tex_unit`

`tex_core`, the tex bus, and the DCR path are **unchanged**: the TEX unit computes
the integer LOD and puts it on the existing 4-bit `lod` bus field (quad), exactly
as it forwards the explicit lod (single).

- **Window read** widens to **8** slots (`CONS_RD_PORTS` 2→8): single mode uses
  ports 0–1; quad reads `base+0..7` (`u[0..3]`, `v[0..3]`).
- **LOD datapath** (per lane): four abs-diffs (`u1-u0, u2-u0, v1-v0, v2-v0`),
  axis shifts by `logw/logh` (from rs1), 4-way max → `rho`, `VX_lzc(rho)` →
  `msb`, `clamp(msb-23, 0, 15)` → integer mip.
- **4-fragment serialized issue**: the quad holds `execute_if.ready` while a frag
  counter issues 4 `TexReq`s `(u[F], v[F], LOD)`, each acquiring a tag carrying
  the frag index + out_slot. The instruction is accepted only after all four
  responses return (single quad in-flight → one response counter, robust to
  response reordering).
- **Reassembly**: each response writes `window[out_slot+F]`; the 4th response
  retires the op and writes `rd` (the sync handle), so a handle-chained `GETW`
  sees a complete result window.

Single mode (P1) is unchanged (`mode=0`, 1 fragment, 2-slot read, explicit lod).

## 5. SimX plan

`sfu_unit` quad path: read the 8 window slots, compute `vx_tex_quad_lod`, issue 4
samples (lod = integer mip, frac forced 0 so `gfx_render` samples one mip), write
4 texels to `window[out_slot+0..3]`. Same formula header as the kernel.

## 6. Validation

`gfx_tex4` gains a quad self-check (mip-filter = NONE, so RTL==SimX): the kernel
`SETW`s a 2×2 coord quad, `vx_tex4` quad → 4 texels; for each fragment it also
computes `vx_tex_quad_lod` in SW and `vx_tex(u[i],v[i], lod)`; asserts equal.
PASS proves the HW LOD + quad path == the SW formula on both SimX and rtlsim. RTU
suite stays green.
