# WGMMA Architectural Completion — K-major bbuf + DXA Transpose

**Date:** 2026-05-29
**Status:** Draft — design only; no code changes yet.
**Owners:** RTL + tensor team
**Related:**
  - [VX_tcu_abuf.sv](../../hw/rtl/tcu/VX_tcu_abuf.sv) (row-major A fetch path, landed),
  - [VX_tcu_bbuf.sv](../../hw/rtl/tcu/VX_tcu_bbuf.sv) (B buffer, block-major-only today),
  - [VX_dxa_desc_table.sv](../../hw/rtl/dxa/VX_dxa_desc_table.sv) + [VX_dxa_gmem_req.sv](../../hw/rtl/dxa/VX_dxa_gmem_req.sv) (DXA writer + descriptor),
  - [cta_clustering_rtl_refactor_proposal.md](cta_clustering_rtl_refactor_proposal.md) (companion proposal — completed Phases 1–4),
  - NVIDIA Hopper WGMMA SS descriptor (public PTX docs, CUTLASS GMMA atoms).

---

## Summary

Complete WGMMA support on Vortex by mirroring NVIDIA Hopper's architectural choice: **K-major SMEM as the canonical layout for both A and B**, produced by the DMA engine (DXA), consumed directly by the per-operand buffers (abuf/bbuf) without re-tiling. This removes the asymmetry the current tree ships — A goes through the K-major DXA→abuf path; B is hand-staged into block-major SMEM by the kernel because bbuf cannot read row-major SMEM — and retires the kernel-side workaround as a transitional patch.

The completion has three RTL pieces and a small kernel-side migration:

1. **Add K-major fetch to VX_tcu_bbuf** — near-mirror of the abuf row-major change (~80 LOC). Honors `desc_b`'s stride field instead of `\`UNUSED_VAR`'ing it.
2. **Add K-major destination mode to DXA** — descriptor flag + axis-swap path in the writer's address generator. Replicates what NVIDIA TMA does via descriptor configuration.
3. **Migrate `sgemm_tcu_wg_dxa` + `sgemm_tcu_wg_sp_dxa` + `sgemm_tcu_wg_dxa_mcast`** off the cooperative B-store back to DXA, with B-descriptor carrying the K-major mode.

After this, the with-DXA and without-DXA WGMMA paths share one bbuf read engine, one operand SMEM layout, and one set of kernel idioms. DXA bandwidth on the B path is restored. The architectural debt the current commit ships gets paid down.

---

## Motivation

### 1. The current tree ships a workaround, not the fix

[sgemm_tcu_wg_dxa/kernel.cpp](../../tests/regression/sgemm_tcu_wg_dxa/kernel.cpp) (uncommitted at proposal time) cooperatively loops over global B element-by-element, writes each into `B_smem[ctx::b_blockmajor_idx(r, c)]`, then issues WGMMA with `desc_b` stride=0. Why: bbuf cannot fetch row-major SMEM; the kernel manually re-tiles. The matching sparse kernel does the same.

This is a kernel-side patch that *moves the layout cost off DXA and onto the LSU*. Specifically it loses:
- DXA's coalesced bulk transfer on the B path
- DXA's intra-CTA multicast on B (the most reusable operand in GEMM)
- Symmetry between the A and B kernel idioms
- The ability for `sgemm_tcu_wg_dxa_mcast` to function on rtlsim at all (mcast still uses DXA for B → fails)

### 2. Hopper does it differently

NVIDIA's WGMMA SS descriptor exposes a `leading-dim` (`ldm`) byte-stride field and a K-major / MN-major flag. Hopper's WGMMA hardware reads SMEM with K running along contiguous bytes for *both* operands:
- A `(M, K)` is laid out (M outer, K inner) → K-major
- B `(K, N)` is laid out (N outer, K inner) → K-major, which is **B transposed** relative to its mathematical orientation

The transposition for B is done by **TMA at write time** via the descriptor's box dimensions + per-axis strides — no consumer-side gather. The WGMMA matmul hardware sees the same K-major read pattern for A and B and the FEDP K-packing falls out naturally.

This is the model the abuf row-major fetch I landed already implements for A. The completion proposes the symmetric change on the B side.

### 3. Bbuf K-major is the same magnitude of work as abuf was

Bbuf row-major support is widely (mis-)characterised as expensive because of the pack-along-row mismatch with mathematically-row-major B `(K, N)`. That mismatch only exists if B is stored K-outer / N-inner. Under K-major (N outer / K inner — exactly what NVIDIA stores), each 32-bit SMEM word *already packs K-elements consecutively*, the same way abuf reads A. The bbuf change becomes a near-mirror of the abuf change:
- Per fetch: 1 LMEM bank-row read.
- Lane offset = `(N_index × ldm_words + step_k × tcK) & (BANK_ROW_WORDS - 1)`.
- Storage layout: unchanged (`storage[j × tcK + k]` indexed by the FEDP's existing scheme).
- Output mux: unchanged.

No byte-level cross-row gather. No `fmt_s` plumbing. No new pipeline stages.

### 4. DXA already has the address-generation machinery

[VX_dxa_gmem_req.sv](../../hw/rtl/dxa/VX_dxa_gmem_req.sv) walks an N-D box (host descriptor: `size0/1`, `stride0_bytes`, `tile0/1`). Adding K-major destination mode is one additional address-gen path: emit destinations in N-outer / K-inner order rather than the current row-major. The host descriptor grows one bit ("swap axes" or "dest layout = K-major"). The descriptor format already has unused bits; no host ABI break.

---

## Proposed Design

### 1. VX_tcu_bbuf K-major fetch path

Add a row-major (K-major) fetch path alongside the existing block-major one, gated on `desc_b[31:16] != 0` (= `ldm != 0`). Mirrors [VX_tcu_abuf.sv:165-205](../../hw/rtl/tcu/VX_tcu_abuf.sv#L165) structurally:

```sv
// Decode ldm (32-bit-word stride between consecutive K-rows of the
// K-major SMEM layout: rows are indexed by N, K runs inner).
wire [LDM_W-1:0] desc_b_ldm_words = LDM_W'(req_desc_b[31:16] >> 2);
wire             slot_row_major   = (desc_b_ldm_words != '0);

// Per-fetch: one bank-row per (step_n*tcN + j) N-row, tcK contiguous
// 32-bit words extracted starting at lane.
wire [BANK_ADDR_WIDTH-1:0] row_lmem_addr =
    fetch_base_r
  + BANK_ADDR_WIDTH'((req_ctr_r * slot_ldm_words_r + req_step_k_r * TCU_TC_K)
                     >> BANK_ROW_WORDS_LOG2);
wire [BANK_ROW_WORDS_LOG2-1:0] row_lane_offset =
    BANK_ROW_WORDS_LOG2'((req_ctr_r * slot_ldm_words_r + req_step_k_r * TCU_TC_K)
                         & (BANK_ROW_WORDS - 1));
```

- **State added:** `slot_row_major_r`, `slot_ldm_words_r`, row counter ~5 bits. Total: ~20 flops.
- **Storage layout:** unchanged. Same `storage[j * tcK + k]` indexed by tcu_core.
- **Output mux:** unchanged.
- **Block-major path:** retained for `sgemm_tcu_wg` (manual cooperative store style) and for backward compatibility. The two paths share storage; `slot_row_major_r` selects fetch behavior only.

**Total estimate: ~80 lines added; 0 lines removed in this phase.** Phase 5 (optional) retires the block-major path later.

### 2. VX_tcu_bbuf sparse K-major

The current bbuf has a 2-slot sparse design (slot A holds `k_blk=0` data, slot B holds `k_blk=1`) to handle the sparse 2×K-pair-per-block structure across two bank-rows. Under K-major sparse, K is contiguous within a row — so the two K-half bank-rows that legacy sparse fetches separately become *one* contiguous run of `2×tcK` words in K-major. The sparse `slot_b` becomes unnecessary; both halves come from one row's K range.

Concretely:
- Sparse-dense slot collapses to single slot (~50 LOC of sparse-pos / `slot_b_*_r` state deleted).
- Sparse permutation (`sparse_src[b]` indexing) becomes a contiguous read.

**Net for sparse: ~30 LOC added (K-major path), ~50 LOC deleted (legacy two-slot machinery) once K-major is the only sparse path.** During the transition the two-slot legacy stays.

### 3. DXA K-major destination mode

Extend the descriptor with a 1-bit `dest_kmajor` flag (or equivalently, an axis-swap bit). The DXA descriptor table ([VX_dxa_desc_table.sv](../../hw/rtl/dxa/VX_dxa_desc_table.sv)) gains one storage bit; the writer's address generator gains one path:

Current writer (row-major destination): per element at logical `(t1, t0)` within the tile, dest_offset = `(t1 × tile0 + t0) × elem_bytes`.

K-major destination: per element at logical `(t1, t0)`, dest_offset = `(t0 × tile1 + t1) × elem_bytes` — axes swapped.

This is a single index recompute; no extra LMEM transactions, no extra burst structure. The DXA bus still emits the same byte count per descriptor.

**Total estimate: ~30-40 LOC in `VX_dxa_gmem_req` (or wherever the address generator emits per-element offsets), ~5 LOC in the descriptor table for the storage bit, ~5 LOC in the host-side descriptor programmer.**

### 4. Host-side descriptor programmer

[sw/runtime/include/dxa.h](../../sw/runtime/include/dxa.h) (or the equivalent host header) gains a destination-layout parameter on `vx_dxa_program_desc_2d`. For B descriptors in WGMMA+DXA kernels, the host passes `VX_DXA_DEST_KMAJOR`. Existing callers (DXA-without-WGMMA tests, sgemm2_dxa, etc.) default to `VX_DXA_DEST_ROWMAJOR` and keep their current behavior — no ABI break.

### 5. Kernel migrations

Three kernels need updating:

**[sgemm_tcu_wg_dxa/kernel.cpp](../../tests/regression/sgemm_tcu_wg_dxa/kernel.cpp):**
- Restore the `kDescB` constant.
- Restore `bar.expect_tx(2)` (A + B both DXA-loaded).
- Delete the cooperative B-store loop.
- Issue `vx_dxa_issue_2d_wg(kDescB, ...)` again.
- Switch `desc_b` to `vt::vx_make_smem_desc(B_smem, ctx::xtileN * sizeof(input_t))` (matches the K-major SMEM stride).

**[sgemm_tcu_wg_sp_dxa/kernel.cpp](../../tests/regression/sgemm_tcu_wg_sp_dxa/kernel.cpp):**
- Same shape as the dense variant — restore DXA load for B.

**[sgemm_tcu_wg_dxa_mcast/kernel.cpp](../../tests/regression/sgemm_tcu_wg_dxa_mcast/kernel.cpp):**
- No change to kernel body (it already uses DXA for B via the mcast helper).
- Becomes correct on rtlsim for the first time — currently broken silently because bbuf can't read row-major.
- Add it to CI in `ci/regression.sh.in` as a rtlsim entry.

**[main.cpp] of each:**
- The host-side `vx_dxa_program_desc_2d` call for `kDescB` adds `VX_DXA_DEST_KMAJOR` as the destination-layout argument.

### 6. WGMMA SS descriptor — kernel-visible surface

The `vt::vx_make_smem_desc(ptr, ldm_bytes)` helper already supports a non-zero stride field. After this proposal, the field's semantics are unambiguous and documented:

| `ldm` value | Layout |
|---|---|
| 0 | Block-major (legacy) |
| > 0 | K-major; `ldm` is the byte stride between consecutive N-rows of B (or M-rows of A) in SMEM |

Document this in [vx_tensor.h](../../sw/kernel/include/vx_tensor.h) above `vx_make_smem_desc`. The MN-major variant (the third leg of NVIDIA's SS-descriptor design space) is out of scope for this proposal but the bit-encoding space leaves it as future work.

---

## Implementation Phases

| Phase | Scope | Files | Validation |
|---|---|---|---|
| **0** | This proposal | `docs/proposals/wgmma_kmajor_completion_proposal.md` | review |
| **1** | bbuf K-major dense path (additive; block-major retained) | `VX_tcu_bbuf.sv` | rtlsim `sgemm_tcu_wg` SS NRC=32 (still uses block-major); separately a small unit-test loading K-major B via cooperative store |
| **2** | bbuf K-major sparse path (additive; legacy 2-slot retained) | `VX_tcu_bbuf.sv` | rtlsim `sgemm_tcu_wg_sp` SS NRC=32 |
| **3** | DXA `dest_kmajor` descriptor flag + writer axis-swap path | `VX_dxa_desc_table.sv`, `VX_dxa_gmem_req.sv`, `sim/simx/dxa/dxa_core.cpp` (parity), `sw/runtime/.../dxa.h` (host-side) | unit test: DXA load row-major source, K-major destination; cross-check SMEM contents |
| **4** | Kernel migration: `sgemm_tcu_wg_dxa` + `sgemm_tcu_wg_sp_dxa` use DXA for B with `dest_kmajor`; delete cooperative load | kernel + main of both tests | full `--tensor_wg` rtlsim sweep |
| **5** | Enable `sgemm_tcu_wg_dxa_mcast` in CI; validate intra-CTA multicast on K-major B | `ci/regression.sh.in` + the mcast kernel/main if any change needed | rtlsim |
| **6** | (Optional) Retire legacy block-major bbuf path + migrate `sgemm_tcu_wg` / `sgemm_tcu_wg_sp` to K-major cooperative store | `VX_tcu_bbuf.sv` (~150 LOC delete) + non-DXA WGMMA kernels | rtlsim full sweep |
| **7** | PPA before/after on Yosys + OpenSTA at U55C-class config | `hw/syn/yosys/` | report |

Phases 1–4 are the must-haves to declare WGMMA "architecturally complete." Phase 5 unblocks multicast WGMMA on rtlsim (currently silently broken). Phase 6 is cleanup that retires the workaround machinery once Phases 1–5 prove the K-major path. Phase 7 quantifies the area / fmax delta.

Phases 1–4 are landable together as one commit (the kernel migration and the bbuf K-major path are tightly coupled), or split into "RTL only" + "kernel cutover" if you'd prefer two commits for bisect granularity.

---

## What This Retires

- **The kernel-side cooperative B-load workaround** in `sgemm_tcu_wg_dxa` + `sgemm_tcu_wg_sp_dxa` (deleted in Phase 4). The current uncommitted patch becomes redundant the moment Phase 4 lands.
- **The asymmetric "A through DXA, B through LSU" mental model** in WGMMA+DXA kernel authoring. Both operands flow through DXA again.
- **The latent multicast-WGMMA bug** in `sgemm_tcu_wg_dxa_mcast` (Phase 5 unblocks it).
- **The legacy block-major bbuf path** (optionally retired in Phase 6).

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| K-major bbuf storage mux misindexes the (k_pair, j) → storage slot mapping | low | wrong B operand → wrong product | abuf row-major change validated the same indexing pattern; copy the test scheme; rtlsim trace diff vs. SimX |
| DXA `dest_kmajor` axis-swap mis-strides for non-aligned tile dimensions | low | corrupted B in SMEM | unit test: DXA copy with intentional non-power-of-2 tile sizes; cross-check SMEM byte-for-byte |
| Sparse 2-slot collapse (Phase 2) breaks an edge case in the existing sparse perm | medium | sparse path regresses | keep legacy slot_b as a fallback during Phase 2; switch over only after K-major sparse passes all sparse rtlsim configs |
| Host descriptor ABI change is consumed before DXA RTL supports the flag | low | flag silently ignored → wrong data | gate the host helper on a build-time `VX_CFG_DXA_DEST_LAYOUT_ENABLE` until RTL lands; revert default if disabled |
| `sgemm_tcu_wg_dxa_mcast` doesn't actually work even with K-major bbuf (some unrelated multicast bug) | medium | Phase 5 stays broken | bisect against Phases 1–4 individually; the proposal stops short of declaring mcast a goal — call it a phase-5 outcome to verify |
| Block-major path retention causes 2× bbuf area | medium | extra silicon | Phase 6 retires it; the retention through Phases 1–5 is a transition strategy, not the steady state |

---

## What This Does NOT Change

- **abuf row-major fetch path** ([VX_tcu_abuf.sv](../../hw/rtl/tcu/VX_tcu_abuf.sv)): already K-major-aligned. Untouched.
- **CTA cluster K-span gate / dispatcher / DXA multicast addressing**: the previous proposal handled this; orthogonal to layout.
- **FEDP / matmul core**: K-packing pattern unchanged. The whole point is that the FEDP sees the same operand format regardless of how SMEM was filled.
- **Existing block-major kernels (`sgemm_tcu_wg`, `sgemm_tcu_wg_sp`)**: they keep working via the retained block-major bbuf path through at least Phase 5. Phase 6 optionally migrates them too.

---

## Out of Scope

- **MN-major SS descriptor support** for A or B (NVIDIA's third leg). The bit-encoding leaves room but no current Vortex workload needs it.
- **SMEM swizzling** (NVIDIA's bank-conflict-avoidance permutation). Worth its own proposal — orthogonal to layout direction.
- **DXA col-major source** (reading from col-major global memory). Tangential — none of our workloads stream col-major globals into SMEM today.

---

## Expected Outcome (qualitative)

| Metric | Direction | Driver |
|---|---|---|
| WGMMA+DXA B-path bandwidth | ↑↑ | DXA bulk transfer restored; per-thread LSU cooperative store deleted |
| Per-kernel LOC | ↓ | cooperative B-load loop (~15 LOC × 2 kernels) deleted |
| bbuf flop count | flat → ↓ | Phase 1–2 adds ~20 flops; Phase 6 deletes the legacy block-major machinery (~50 flops) |
| bbuf LUT / cell area | ↓ after Phase 6 | block-major's sub-block + sparse-pos selection logic retires |
| WGMMA+DXA Kernel runtime (sgemm_tcu_wg_dxa) | ↑ | B no longer bottlenecked on LSU cooperative-store latency |
| sgemm_tcu_wg_dxa_mcast on rtlsim | broken → working | Phase 5 |
| Architectural symmetry A ↔ B | asymmetric → symmetric | both paths share K-major SMEM + DXA + bbuf/abuf row-major fetch |

Phase 7 should quantify this on the Yosys + OpenSTA flow.

---

## Why Now

The current uncommitted "WGMMA bug fix" bundle is a tactical patch — it gets `sgemm_tcu_wg_dxa` and `sgemm_tcu_wg_sp_dxa` to PASS on rtlsim, but it ships the architectural debt as the answer. The owner of this codebase has explicitly stated the goal is "**area-efficient WGMMA support on Vortex, both with-DXA and without-DXA paths efficient**" — that goal is incompatible with the kernel-side workaround:

- B loses DXA bandwidth → with-DXA path is *less* efficient than the current `sgemm_tcu_wg` (cooperative load) baseline. The workaround makes DXA worse than not having it for B.
- The asymmetric kernel idiom (A via DXA, B via LSU) is a permanent pothole every WGMMA+DXA kernel author trips over.
- The multicast variant doesn't work and we'd be shipping a known-broken test in the tree.

This proposal closes those holes. The Phase 1 bbuf change is small (~80 LOC), bounded, and mirrors a fetch path that already works (abuf row-major). The DXA piece (Phase 3) is one address-gen branch. The kernel migration (Phase 4) is +15 / −20 LOC per kernel. The cleanup phases are gravy.

The right time to land this is before the current "kernel workaround" bundle gets committed — because committing the workaround locks in the kernel idiom and the test-passing illusion. The proposal is the actual fix.
