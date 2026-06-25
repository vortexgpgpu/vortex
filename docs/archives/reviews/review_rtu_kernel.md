# v2 Review: prism_v3 RTU kernel stack

**Date:** 2026-06-17
**Reviewer scope:** `sw/kernel/include/vx_raytrace.h` device intrinsics + ABI, cross-checked
against `sim/simx/decode.cpp` (EXT2/funct3=6,7), `sim/simx/rtu/rtu_unit.cpp` (RtuUopGen
macro-op expansion), `hw/rtl/core/VX_decode.sv` + `hw/rtl/gfx/VX_gfxw_uops.sv` (RTL decode +
uop expander), `docs/proposals/rtu_isa_v2_proposal.md`, and the 24 `tests/raytracing/*` kernels.
Read-only review; no code modified.

---

## 1. Overall assessment (maturity grade: **A−**)

The v2 window ABI is well-designed, internally consistent, and — critically — the three decode
surfaces (kernel `.insn` encodings, SimX decoder, RTL decoder) **agree on every field I checked**:
opcode, funct3, funct2 sub-op selectors, the `funct7[6:2]` slot, the `rs2`-as-count immediate, and
the f0–f7 ray-window / f0–f2+t3–t5 hit-window hardcoding. The macro-op expansion (4 uops for
TRACE2, `count` uops for GETWF/GETW) is mirrored bit-for-bit between `RtuUopGen` and `VX_gfxw_uops`.
This is the hard part of an ISA change and it is right.

The proposal's headline claims **verify**: the hot path collapses from ~16 architectural SFU ops to
**2** (`trace` + `wait`); register pressure is ~10 live regs (8 FP ray window + 1 GP config + 1 GP
handle), split across two register files; there is **no dcache traffic** for the ray or hit (the
ray flows f0–f7 → pool slot → hit window). The "true-GPU" alignment to AMD `image_bvh_intersect_ray`
is genuine and the strongest of the three vendor analogies.

The grade is held below A by **documentation drift** (the proposal §5.1 config-lane table and two
test-kernel comments describe an *older* encoding than the shipped code) and by one **latent
correctness hazard** (the f0–f7 ray binding is enforced only by `register __asm__("fN")` hints, not
by the architectural encoding — a fragile contract). Neither is a functional bug today (tests pass),
but both are traps for the next editor.

---

## 2. Correctness findings

| # | `file:line` | Issue | Severity |
|---|---|---|---|
| C1 | `sw/kernel/include/vx_raytrace.h:166` vs proposal §5.1 (table, lines 156–162) | **Config lane layout in code ≠ proposal table.** The proposal §5.1 table assigns `lane0=scene_ptr, lane1=payload, lane2=ray_flags, lane3=cull_mask` (4 lanes). The shipped intrinsic does `vx_wgather(0u, scene_ptr, payload_ptr, flags_cull)` → **lane0=unused(self-suppressed), lane1=scene, lane2=payload, lane3={flags[15:0]\|cull<<16}** (3 live lanes). SimX (`rtu_unit.cpp:370–381`: `cfg.at(1)`=scene, `cfg.at(2)`=payload, `cfg.at(3)`=flagscull split 16/16) and the header comment (lines 160–164) **agree with the code, not the proposal.** Code is self-consistent and arguably *better* (it resolves §9 Q6's "4-of-4 lanes used at warp=4" pressure and survives a lane-0-dead callback mask), but the proposal's normative §5.1 table is now stale and misleads anyone porting the RTL or Mesa lowering. | **Medium (doc)** |
| C2 | `sw/kernel/include/vx_raytrace.h:180`, `:204` (and all TRACE2 sites) | **f0–f7 ray binding is not in the architectural encoding — enforced only by `register float rN __asm__("fN")`.** The `.insn r ... %[cfg], x0` names only rd/rs1/rs2=x0; the FP ray regs ride the operand list purely as clobber/liveness hints. Both decoders **hardcode** f0–f7 as the ray source (`rtu_unit.cpp:326–337`, `VX_gfxw_uops.sv:73–98`). If the compiler ever fails to honor the explicit-register binding (e.g. a future codegen change, an inlining edge, or a hand-written caller that forgets the bindings), the trace **silently reads stale f0–f7** with no decode-time error. The current intrinsic does this correctly, so it is latent, not active. Mitigation today is solid (explicit register vars + `always_inline`); the risk is purely for future editors / the eventual Mesa NIR→LLVM lowering (proposal step 6), which must reproduce this exact binding. Recommend a compile-time guard or a codegen-level pin. | **Medium (latent)** |
| C3 | `tests/raytracing/rtu_smoke_is/kernel.cpp:26–28` | **Wrong comment for `vx_rt_get` funct7.** Comment claims "funct7 for vx_rt_get(slot) is `(slot << 2) \| 1`" and derives `VX_RT_CB_TYPE(29)→117`, `PAYLOAD_PTR_LO(25)→101`. `\|1` is **SETW** (write); `vx_rt_get` is **GETW = `\|3`**. The actual asm on lines 32–33 correctly uses `119` (`29<<2\|3`) and `103` (`25<<2\|3`) — i.e. the code is right and the comment is wrong (both the operator and the two derived numbers). Pure doc bug, but it is a hand-encoding cheat-sheet, so it actively misleads. | **Low (doc)** |
| C4 | `sim/simx/decode.cpp:1048,1062,1075` vs `VX_decode.sv:797` | **Slot-field width mismatch (benign).** SimX masks the slot `& 0x3F` (6 bits) for SETW/GETWF/GETW; RTL takes `funct7[6:2]` (5 bits, 0–31). `VX_RT_SLOT_COUNT=32`, so all real slots fit in 5 bits and the encoders never set bit 5. The two are functionally equivalent for the legal range, but the SimX 6-bit mask is wider than the RTL field can carry — if a future slot ≥32 were added, SimX would accept it and RTL would silently truncate. Tighten SimX to `& 0x1F` for parity. | **Low** |
| C5 | `sw/kernel/include/vx_raytrace.h:217–230` | **Hit-ID writeback order is correct but subtle — worth a guard.** GETW reads slots starting at `VX_RT_HIT_PRIMITIVE_ID(21)` count=3 → slots 21,22,23 → `t3,t4,t5` → assigned `primitive_id, instance_id, geometry_index`. This matches VX_types (`21=primitive_id, 22=instance_id, 23=geometry_index`) and the header comment (line 218). Correct. The hazard is only that the *contiguity* (21,22,23 in exactly that struct order) is an implicit ABI contract between VX_types.h and the struct field order in `vx_hit_t`; a reorder of either breaks it silently. No bug; flagging the coupling. | **Info** |
| C6 | `sw/kernel/include/vx_raytrace.h:202–207` (WAIT2) | **Trap-safety mechanism verified.** WAIT2 (funct3=7, funct2=1) is decoded as a **single op, not a macro-op** in both SimX (`decode.cpp:1106–1114`, no `set_macro_op`) and RTL (`VX_decode.sv:839` `is_wstall=(funct2!=1)` → WAIT2 not wstalled, TRACE2 is). This is exactly the proposal §12.2 requirement (a parked single op is revivable across a trap flush; a parking macro-op's writeback uops would be lost). The split into a blocking WAIT2 + two non-blocking scoreboard-chained windowed reads (GETWF on `status`, GETW on `status`) is correct and matches §12.2. No issue — this is the cleverest correct part of the stack. | **Info (correct)** |

**Opcode/encoding cross-check summary (all PASS):** `RISCV_CUSTOM1 = 0x2B` (`vx_intrinsics.h:35`) =
`Opcode::EXT2 = 0b0101011` (`instr.h:49`). funct3=6 sub-ops {0=CB_RET,1=SETW,2=GETWF,3=GETW} and
funct3=7 sub-ops {0=TRACE2,1=WAIT2} match across kernel header / SimX / RTL. `count` from `rs2[3:0]`
on both decoders; the `x3`/`x6` count registers are read as **immediates only** (not marked as source
regs), so no false scoreboard dependency — verified `decode.cpp:1058–1066` sets only `set_src_reg(0,rs1)`
and `VX_gfxw_uops.sv:113` chains the scoreboard only on uop 0's rs1=status.

---

## 3. Efficiency findings

- **E1 — Op-count claim verified.** `tests/raytracing/rtu_smoke/kernel.cpp:47–50` issues exactly one
  `vx_rt_wtrace` + one `vx_rt_wait` per ray. Architecturally 2 instructions fetched/decoded; the
  proposal's ~16→2 claim holds. The implicit `vx_wgather` (`vx_raytrace.h:166`) is one extra op but is
  pure-register and hoistable when scene/flags are loop-invariant (the common bounce-loop case), so
  steady-state config cost → ~0, as claimed.
- **E2 — No marshalling SRAM / no dcache traffic.** Confirmed in `rtu_unit.cpp:355–447`: the ray
  streams f-regs → pool slot directly; the only cross-uop state is `trace2_slot_` (the slot index
  latch, lines 366/416). The hit returns via windowed regfile reads, never memory. The "Ray Bank in
  registers, not memory" property of the proposal is real.
- **E3 — `vx_rt_get_objray` collapses 6 get+6 fmv → 1 macro-op** (`vx_raytrace.h:257–272`, GETWF
  start=`OBJECT_RAY_ORIGIN(8)` count=6 → slots 8–13 → f0–f5). This is the §12.1 win and it is correctly
  encoded. Good for IS/AHS dispatchers on the traversal-yield hot path.
- **E4 — Register-file type-split avoids fmv conversions.** Ray + float hits (t/u/v) live in FP;
  config/handle/status/int-IDs in GP (`decode.cpp:1059` GETWF→Float, `:1072` GETW→Integer). Zero
  `fmv.x.w` per trace, resolving §9 Q1 as designed.
- **E5 (minor) — `vx_rt_wtrace` always emits the wgather even at non-invariant sites.** When scene/
  flags genuinely vary per trace (rare), the wgather is 1 unavoidable op/trace; acceptable and matches
  the AMD SGPR-descriptor cost model. No action.

---

## 4. Performance findings

- **P1 — Async overlap preserved.** TRACE2 returns the handle at uop 0 (`rtu_unit.cpp:382`,
  "handle returns early") while uops 1–3 keep filling the slot; the kernel only blocks at WAIT2. The
  trace/wait split that motivates the whole design is intact (proposal §5.6, §7-D1).
- **P2 — Macro-op uop cost honest, not free.** TRACE2 = 4 uops (1 GP config + 3 FP ray;
  `uop_count` returns 4, `rtu_unit.cpp:279`), GETWF/GETW = `count` uops. The wstall on TRACE2
  (`set_wstall`, RTL `is_wstall`) correctly stalls fetch until the macro-op drains, and WAIT2 does
  **not** wstall (it blocks via scoreboard instead), so younger ops don't fetch ahead and deadlock the
  in-order warp on a callback trap — the `VX_decode.sv:831–838` comment documents this precisely. The
  uop cost is modelled in the SimX sequencer's per-uop cycle accounting (proposal R-OC), so timing is
  not optimistic.
- **P3 — Wait-side writeback hidden under traversal.** The two windowed reads in `vx_rt_wait` are
  scoreboard-chained on `status`, so they issue only post-terminal, after the (long) traversal already
  finished — off the critical path, as the proposal claims (§5.6 close).
- **P4 (caveat, not a regression) — recursive/in-trap traces hold the f0–f7 window live across the
  trap.** `rtu_smoke_recursive/kernel.cpp` issues a nested `vx_rt_wtrace`/`vx_rt_wait` inside a CHS
  dispatcher marked `__attribute__((interrupt))`, which forces the prologue to save/restore f0–f7. This
  exercises the §9 Q2 caveat (ray window vs. TCU D-fragment window aliasing). Cost is the interrupt
  save/restore of the FP window per recursion level — inherent to recursion, acceptable, and confirmed
  to run (kernel present in the passing suite).

---

## 5. "True GPU" alignment vs NVIDIA / AMD / Intel

The windowed-register approach is **sound and well-justified**; the closest precedent is real.

- **AMD `image_bvh_intersect_ray` (RDNA2/3) — strongest match, and the one the design copies.**
  AMD passes the per-thread ray as a **VGPR group** (~11 VGPRs) + the warp-uniform BVH descriptor as an
  **SGPR group**, results as a **4-VGPR group**, with SIMT traversal. PRISM v2 is a near-isomorphism:
  f0–f7 ray window = the VGPR group (~8 vs ~11), the lane-packed rs1 config = the SGPR descriptor
  (delivered through the narrow warp's spare lanes rather than a true scalar file — a fair adaptation
  given Vortex has no SGPR file), hit window = the result VGPR group. ~10 live regs vs AMD's ~11 is
  comparable (Risk R1). This is the right model for Vortex and the code realizes it faithfully.

- **NVIDIA Turing→Ada — aligned in spirit, with one honest gap.** NVIDIA register-allocates the ray
  descriptor, drives traversal with one RT instruction, returns attrs in registers, and crucially puts
  warp-uniform args (AS handle, flags, SBT offset, cull mask) in the **Uniform Register File** so they
  are not replicated per lane. PRISM has **no uniform register file**, so it emulates the uniform
  channel by stealing the spare lanes of one vector register via `vx_wgather` (§5.4). This is a
  legitimate substitute at warp=4 (exactly enough lanes), but it does *not* scale the way a true URF
  does, and the §5.1-vs-code drift (C1) shows the lane budget is already tight (flags+cull had to be
  co-packed into one lane). Honest alignment, with the lane-packing being the weakest structural point
  — correctly identified as the fallback risk in the proposal itself.

- **Intel Xe-HPG `MemRay`/`RTStack` + `send` — correctly rejected, for the right reason.** Intel uses
  *memory* (per-lane RTStack scratch) for ray state because BTD reformation migrates ray state across
  EUs. The proposal (§3) explicitly notes Vortex does **not** share that constraint, so the
  register-group model fits better — and the code bears this out: there is genuinely no memory round-trip
  on the issue path. The proposal keeps the Intel memory-descriptor variant only as a pressure-relief
  fallback (R1), which is the correct place for it. I concur with the rejection.

**Verdict:** the register-window model is the right choice for Vortex's narrow-warp, no-URF,
no-reformation-migration microarchitecture. The one place reality diverges from the cited precedent is
the *uniform channel*: NVIDIA/AMD have a dedicated scalar/uniform file; PRISM fakes it with lane-packing,
which is sound at warp=4 but is the part most likely to strain at wider warps or with a 5th uniform word
(§9 Q6). The design acknowledges this; the code's flags+cull co-packing (C1) is the first symptom.

---

## 6. v2.1 recommendations

### P0 (correctness / contract — do before Mesa lowering, proposal step 6)
- **P0-1 (C1):** Update proposal §5.1's config-lane table (and the §2 taxonomy row) to the **shipped**
  layout: lane0=unused/self-suppressed, lane1=scene, lane2=payload, lane3={flags[15:0]|cull[31:16]}.
  The §5.1 table is normative for the RTL/Mesa lowering; leaving it stale will cause the lowering pass
  to emit the wrong wgather order. Note the self-slot-suppression rationale (partial-warp/lane-0-dead
  masks) inline so it isn't "fixed" back.
- **P0-2 (C2):** Make the f0–f7 ray-window binding enforceable rather than advisory. Minimum: a
  `static_assert`/compile guard or a documented invariant that the eventual Mesa NIR→LLVM lowering must
  pin the ray to f0–f7 (the decoders hardcode it with no runtime check). This is the single largest
  silent-failure surface in the stack.

### P1 (parity / clarity)
- **P1-1 (C3):** Fix `rtu_smoke_is/kernel.cpp:26–28` comment — `vx_rt_get` is `(slot<<2)|3` (GETW),
  derived values are `119`/`103`, not `|1`/`117`/`101`. The asm is already correct; only the comment lies.
- **P1-2 (C4):** Tighten SimX slot mask from `& 0x3F` to `& 0x1F` (`decode.cpp:1048,1062,1075`) to match
  the RTL 5-bit `funct7[6:2]` field exactly, so an out-of-range slot fails identically on both paths.
- **P1-3:** Add the `vx_rt_get_objray`-style typed windowed accessor for the **hit window** too (the
  `vx_rt_wait` body open-codes two `.insn` windowed reads inline); factoring them into named `GETWF`/
  `GETW` count-typed helpers would reduce the chance of a future hand-edit desyncing slot/count.

### P2 (forward-looking)
- **P2-1:** Document the uniform-channel scaling limit explicitly as a known design boundary (the
  lane-packing only has 3 usable uniform words after self-slot suppression at warp=4; a 5th per-trace
  uniform word must go to a DCR or rs2 spare). This is the real divergence from the NVIDIA URF precedent
  and should be a first-class ABI note, not buried in §9 Q6 — especially since C1 shows the budget is
  already saturated.
- **P2-2:** Add a compile/lint check coupling `vx_hit_t` field order to the VX_types slot contiguity
  (C5) so a slot renumber or struct reorder can't silently misroute hit IDs.
- **P2-3:** Keep the Intel `MemRay` memory-descriptor fallback (R1) as a documented escape hatch behind
  the same macro for register-pressure-critical sites (recursion, large CHS bodies); the two encodings
  can coexist since the architectural encoding already names only rd/rs1.

---

### Sign-off
Encodings are consistent across kernel/SimX/RTL on every field checked; the headline efficiency and
performance claims (16→2 ops, ~10 regs, no dcache traffic, async overlap, trap-safe wait) all verify in
code. The stack is functionally sound (the full `tests/raytracing/*` suite is the cited evidence). The
remaining work is **doc-truth** (P0-1, P1-1) and **hardening the f0–f7 binding contract** (P0-2) before
the Mesa lowering reproduces these encodings by hand. Grade **A−**.
