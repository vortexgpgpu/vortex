# VX_CFG_DIVERGE_TYPE — Selectable Divergence Architecture (DEFAULT / SPLIT / NV_ITS)

**Scope:** [VX_config.toml](../../VX_config.toml), [ci/gen_config.py](../../ci/gen_config.py),
[sim/simx/{scheduler,wctl_unit,decode,types}](../../sim/simx),
[hw/rtl/core/VX_scheduler.sv](../../hw/rtl/core/VX_scheduler.sv),
[hw/rtl/core/VX_decode.sv](../../hw/rtl/core/VX_decode.sv),
[hw/rtl/core/VX_alu_int.sv](../../hw/rtl/core/VX_alu_int.sv),
[hw/rtl/core/VX_wctl_unit.sv](../../hw/rtl/core/VX_wctl_unit.sv),
[hw/rtl/VX_gpu_pkg.sv](../../hw/rtl/VX_gpu_pkg.sv), interface files, sim Makefiles,
[tests/regression/diverge](../../tests/regression/diverge), and llvm-vortex branch
`threadsplit` (`~/dev/llvm_vortex_dvg`: VortexBranchDivergence passes, RISCVInstrInfoVX.td,
RISCVTargetMachine/ISelLowering — §7)
**Reference implementation:** `~/dev/vortexz_its` (ITS on Vortex 2.0, `-DITS_ENABLE`; see the
ITS project report) and its `tests/regression/diverge/its_flowchart.md`
**Primary source:** US 10,067,768 B2, *Execution of divergent threads using a convergence
barrier*, Diamos et al. (NVIDIA), filed 2015-07-13, granted 2018-09-04 —
<https://patents.google.com/patent/US10067768B2/en> (continuation: US 11,442,795)
**Branch:** `threadsplit`
**Status:** Proposal — for review

---

## 1. Goal

Make the divergence architecture a first-class, three-valued build configuration so the
designs can be compared head-to-head on the same tree, same toolchain, same tests:

| `VX_CFG_DIVERGE_TYPE` | Design | Today |
|---|---|---|
| `DEFAULT` | Upstream baseline: IPDOM stack, strict-LIFO serialization | exists, but SCS code is fused into it |
| `SPLIT` | threadsplit / SCS: schedulable splits + `vx_yield` (deadlock-free) | current unconditional behavior of this branch |
| `NV_ITS` | NVIDIA-style ITS: per-thread PCs + convergence barriers (`bar_add`/`bar_wait`), lowest-PC group scheduling | exists only in the Vortex 2.0 snapshot at `~/dev/vortexz_its`; must be ported |

Both SimX and RTL honor the switch. `DEFAULT` must be behavior- and cycle-identical to
upstream master; `SPLIT` must be bit-identical to this branch today.

## 2. Config plumbing

The tree already has the exact pattern needed — the `TCU_TYPE`/`FPU_TYPE` string enums.

**VX_config.toml** — new section plus the enum registration:

```toml
[diverge]
# string enum: 'DEFAULT' (IPDOM baseline) | 'SPLIT' (schedulable convergence
# stack + vx_yield) | 'NV_ITS' (per-thread PCs + convergence barriers)
VX_CFG_DIVERGE_TYPE = "SPLIT"

[[enum]]
VX_CFG_DIVERGE_TYPE = ["DEFAULT", "SPLIT", "NV_ITS"]
```

Default `SPLIT` preserves this branch's current behavior (flag for review: `DEFAULT` would
instead make the tree upstream-identical unless opted in).

`gen_config.py` needs **no changes**: `[[enum]]` parsing, the companion-define synthesis
(`VX_CFG_DIVERGE_TYPE_SPLIT`), both override spellings (`-DVX_CFG_DIVERGE_TYPE=NV_ITS` and
`-DVX_CFG_DIVERGE_TYPE_NV_ITS`), and the cflags emission are all generic.

**Makefile companion re-injection** (the part that bites): RTL and SimX code select on the
*companion* define, but raw `$(CONFIGS)` may carry only the `=value` form. Mirror what
rtlsim already does for TCU/FPU:

- `sim/rtlsim/Makefile:143` (and xrtsim/opaesim/avedsim): widen the filter to include
  `-DVX_CFG_DIVERGE_TYPE_%`.
- `sim/simx/Makefile`: add `CXXFLAGS += $(filter -DVX_CFG_DIVERGE_TYPE_%,$(XCONFIGS))`
  (simx currently re-injects no enum companions at all — the `#ifdef`s would silently never
  fire on a `=value` override without this).
- Synthesis flows: the canonical place is `hw/syn/extensions.mk` (shared by all six synth
  backends), which already carries the TCU type selection and documents the two traps —
  companion flags are matched bare (no `%` needed), and `_ENABLE` guards must list all
  three gen_config spellings. Plus the two `catalog.mk` copies that deliberately duplicate
  the type selection for flatten-preserving reasons. Total: 8 edit sites (extensions.mk +
  4 sim Makefiles + simx + 2 catalog.mk).

Related docs debt picked up here: `docs/designs/build_configuration_system.md` documents
the TOML→header pipeline but not the `[[enum]]` mechanism at all (companion-define
cascade, `-DX=V` vs `-DX_V` duality). `DIVERGE_TYPE` will be the first enum consumed by
SimX, so the design doc gains a short enum section as part of Phase 1.

SimX consumes the companion via `#ifdef` exactly like the existing
`#ifdef VX_CFG_XLEN_64` precedent in `decompressor.cpp` — no new include needed;
`scheduler.cpp`/`wctl_unit.cpp` see `VX_config.h` transitively through `types.h`.

## 3. Structural approach: inline fences, not module extraction

Two options were considered for the RTL:

**(a) Sibling-dir implementations** (`hw/rtl/core/diverge/{default,split,nv_its}/` selected
by `-I`, like `hw/rtl/tcu/*`): cleanest long-term, but it requires extracting warp
PC/mask ownership plus every redirect writer (branch, trap, split, join, TMC, pred, yield,
pop-install, RVC advance) out of `VX_scheduler.sv`'s single 260-line `always @(*)` into a
new module boundary. That is major surgery on logic that just closed 300 MHz (WNS −13 ps);
any re-partitioning risks re-opening timing for zero architectural gain.

**(b) Inline `` `ifdef VX_CFG_DIVERGE_TYPE_* `` fences** inside the existing files: this is
exactly the delta structure the Vortex 2.0 ITS implementation already proved out
(clean `#ifdef/#else` at statement granularity, baseline preserved verbatim in the `#else`
arms), and it leaves the timing-closed SPLIT logic untouched down to the bit.

**Recommendation: (b)** for both RTL and SimX. The SCS additions get wrapped where they
sit; the ITS port lands as parallel guarded blocks following the old tree's structure.
Option (a) remains the documented follow-up if a fourth design ever appears.

## 4. SimX design

### 4.1 Gating the existing SCS code under `SPLIT`

The SCS delta touches five files and is mostly additive; the fences are:

- `scheduler.h`: `scs_split_t`, the eight `warp_t::scs_*` fields, the five `scs_*` method
  declarations, `yield_warp()`, and `ipdom_entry_t`'s three SCS fields
  (`passed_tmask`/`passed_pc`/`has_passed`) → `#ifdef VX_CFG_DIVERGE_TYPE_SPLIT`. Guarding
  the ipdom fields keeps DEFAULT's stack entry bit-identical to upstream.
- `scheduler.cpp`: the five `scs_*` helpers, `yield_warp`, ctor/reset/activate_warp SCS
  lines, and the three inserts inside `schedule()` (cooldown tick, pick-predicate term,
  stall watchdog) plus the SCS body of `setTmask()` (restore the upstream body in the
  `#else` arm).
- `wctl_unit.cpp`: the TMC `scs_done` block, the JOIN passed-subgroup capture, the PRED
  case (genuine behavioral fork — SCS pending-subgroup semantics vs upstream one-line
  `rs2` restore; both bodies live under `#if`/`#else`), and the YIELD case.
- `decode.cpp` / `types.h`: `WctlType::YIELD` and its funct7=5 decode stay
  **unconditional in all modes** — see §6 ISA rules.

### 4.2 Porting ITS under `NV_ITS`

The old implementation lived in `emulator.{h,cpp}`/`execute.cpp`, which no longer exist;
the port maps onto the current architecture as follows:

- **`warp_t` state** (`scheduler.h`, `#ifdef VX_CFG_DIVERGE_TYPE_NV_ITS`): `tpc[]`
  (per-thread PC), `amask` (alive), `bar_participate[]`/`bar_arrived[]`
  (`VX_CFG_ITS_NUM_BARRIERS` each — see §6; the old tree hardcoded 32). `ipdom_stack`
  remains declared (SPLIT/JOIN become no-ops that never touch it) so shared code compiles;
  `tmask`/`PC` are retained as the current-issue-group scratch, recomputed at each pick,
  exactly as the old model did.
- **Group selection** (`scheduler.cpp schedule()`): under NV_ITS the warp-pick loop gains
  the within-warp step from `emulator.cpp:199-246`: runnable = `amask & ~OR(bar_arrived)`;
  pick the lowest PC among runnable threads; issue the group of runnable threads at that
  PC into `warp.tmask`/`warp.PC`. The old model's "ibuffer non-empty → don't reselect"
  guard maps to the current per-warp fetch-stall discipline (`stalled_warps_`), which
  already prevents mid-instruction regrouping — to be confirmed during bring-up.
- **Divergent branches**: the current SimX faults on a divergent plain branch (compiler
  emits split/join). Under NV_ITS the branch execution path computes a per-thread
  taken/not-taken PC into `tpc[]` (port of `execute.cpp:399-441` + the per-thread PC
  writeback epilogue), and warp retirement moves from "tmask empty" to "amask empty".
- **`bar_add`/`bar_wait`** (`wctl_unit.cpp`, new `WctlType::BAR_ADD/BAR_WAIT`):
  participation-accumulate and arrive-until-equality-release, semantics from the old tree
  including the mask clear on release (which is what makes bid reuse across loop
  iterations work) — with one hardening found during bring-up: **arrival is masked to
  actual participants (`arrived |= tmask & participate`) and a wait on an empty barrier
  passes through**. Barrier ids are allocated per function (§7.1), so a callee's bid can
  collide with a live caller barrier — dogfood's `trigo` deadlocked because libm `sinf`'s
  internal divergent loops used bid 0 under the kernel loop's own bid 0; unmasked arrival
  poisoned the release equality forever. The patent's answer is barrier state
  save/restore across calls (BMOV); the pass-through rule is the eval-scope equivalent —
  it only weakens reconvergence packing, never blocks a thread on a barrier it never
  joined. This fix cured the bring-up hangs in trigo, bfs, and jacobi. Executed as plain SFU control ops (fetch_stall, no BarrierUnit
  involvement — these are intra-warp and thread-granular; `BarrierUnit` is warp-granular
  CTA scope and must not be conflated).
- **TMC**: sets `amask`, resets live threads' `tpc` to PC+4 — and, fixing a known latent
  bug in the old tree, **also clears the killed threads out of `bar_participate`/
  `bar_arrived`** (the old code left barriers permanently unsatisfiable if TMC killed a
  parked thread), re-evaluating each barrier's release condition after the removal.

### 4.3 Patent fidelity — documented deviations from US 10,067,768

The NV_ITS port follows the vortexz_its reference, which deviates from the patent in three
places. These are recorded here so the evaluation attributes effects correctly; the same
statements hold for the RTL port in §5.2.

1. **No YIELD / no Yielded thread state.** The patent's forward-progress guarantee is a
   per-thread *Yielded* state plus compiler-inserted yields "along any control path that
   does not terminate in a statically determined number of instructions"; yielded threads
   are excluded from the barrier-release check. The reference omits this entirely, and the
   port keeps the omission **deliberately**: adding yield to NV_ITS would reproduce the
   very mechanism SPLIT's `vx_yield` provides and blur the comparison (§10.3). The lock
   test deadlocks under NV_ITS therefore measure *barriers without yield*, not the full
   patented design — the evaluation write-up must say so.
2. **Scheduling policy.** The patent's scheduler picks the divergent path with the
   *fewest active threads* (depth-first on structured code); the reference picks the
   *lowest-PC* runnable group and the port keeps that. Both converge identically on
   structured code but interleave differently, so IPC deltas vs SPLIT partly reflect
   pick policy, not just the barrier mechanism.
3. **Thread exit.** The patent keeps exited (OPT-OUT) threads in the participation mask
   and has the release check ignore them. The reference instead leaves TMC-killed threads
   *counted*, making barriers permanently unsatisfiable — a bug. The §4.2 TMC fix (clear
   killed threads from the masks, re-evaluate release) is equivalent in effect to the
   patent's ignore-exited rule and is the one place the port corrects the reference
   toward the patent.

## 5. RTL design

### 5.1 Gating the existing SCS code under `SPLIT`

`VX_scheduler.sv` (renamed from `VX_schedule.sv` upstream): fence the identified SCS
blocks — design comment + `CS_*` localparams + state regs (57-93), `cs_pool_ram` +
`cta_warp_done` gating (120-149), comb resets + pop-install + CTA reset (244-276), the
TMC exit/rotate rewrite (295-340; upstream's two-line TMC handler returns in the `else`
arm), pred_park/pred_restore (342-375), yield (409-447), sequential resets (523-539).
`VX_wctl_unit.sv`: `is_yield`, the pred_park/restore wires, `yield_valid`, and the SCS
widening of the `wctl_reg` DATAW. `VX_warp_ctl_if.sv`: the four SCS fields. All under
`` `ifdef VX_CFG_DIVERGE_TYPE_SPLIT ``.

Verification that the fences are inert: a SPLIT build must produce bit-identical Verilator
output to today's tree (same binary hash modulo timestamps), and a DEFAULT build must diff
clean against upstream master's generated code for these files.

### 5.2 Porting ITS under `NV_ITS`

Following the old tree's additive structure, with its known defects fixed:

- **`VX_gpu_pkg.sv`**: `INST_SFU_BAR_ADD`/`INST_SFU_BAR_WAIT` (values chosen in the free
  SFU op space of the *current* package — the old 4'h9/4'hA slots must be re-checked
  against today's op map, and the `inst_sfu_is_wctl()` widening goes inside the NV_ITS
  guard, never unconditional as it was in the old tree), `its_bar_t`,
  `wctl_args_t.bid` carved from padding (guarded).
- **`VX_decode.sv`**: the new encodings (§6) decoded only under NV_ITS; under
  DEFAULT/SPLIT the slot stays reserved so the `conform` reserved-funct7 test keeps its
  meaning (the conform expectations become DIVERGE_TYPE-dependent — see §8).
- **`VX_branch_ctl_if.sv` + `VX_alu_int.sv`**: additive `taken_mask`/`tmask`/`ntaken_pc`
  fields and the per-lane branch-resolution pipe register. Two fixes over the old tree:
  the `DATAW` mixing `NUM_THREADS` with `NUM_LANES`-wide sources, and a hard elaboration
  guard (`` `STATIC_ASSERT ``) that `NUM_ALU_LANES == NUM_THREADS` under NV_ITS — per-thread
  branch masks are meaningless with lane blocking, and we make that a build error, not a
  silent wrong answer.
- **`VX_scheduler.sv`**: the per-warp `thread_pcs`/`amask`/`its_part`/`its_arr` arrays and
  the guarded forks of each writer (wspawn, TMC — including the barrier-mask cleanup fix
  from §4.2, split/join no-op'd with the unlock kept, per-thread branch write, per-thread
  PC advance, bar_add/bar_wait update, `ready_warps &= runnable_any`, lowest-PC group
  select). The min-PC comparator is written as a balanced tree, not the old serial chain —
  NT=8 is our standard config and the serial chain was written for NT=4. Under NV_ITS the
  `VX_split_join` instance is generated away entirely (the old tree left it as dead area).
- **`VX_wctl_unit.sv` / `VX_warp_ctl_if.sv`**: the `its_bar_t` payload register and
  interface fields, guarded.

### 5.3 Area/timing expectations (to be measured, not assumed)

ITS state per core: `NW×NT×PC_BITS` PC flops + `2×NW×NBAR×NT` barrier-mask flops, all
combinationally scanned. At NW=8/NT=8/NBAR=8 that is ~2.5K flops of new
always-resident state plus an 8-way OR-reduce per warp in front of the scheduler priority
encoder — precisely the cost argument the threadsplit proposal §6 makes against ITS, which
this switch finally lets us *measure*. Synthesis DUT configs get a third sandbox variant
(`its_nt8nw8`) beside the existing `scs_*` ones. Per policy: synthesis only after
rtlsim-green, never blocking.

## 6. ISA and encoding plan

The old ITS encodings (funct7=0, funct3=6/7) are **unavailable**: in the current tree
funct7=0's funct3 space is fully allocated (6 = barrier arrive/wait, 7 = WSYNC). New plan
under `EXT1` (0x0B):

| funct7 | funct3 | Op | Mode |
|---|---|---|---|
| 0x00 | 0-7 | TMC/WSPAWN/SPLIT/JOIN/BAR/PRED/BAR-arr/WSYNC | all (SPLIT/JOIN/PRED are architectural no-ops under NV_ITS, as in the old tree) |
| 0x05 | 0 | `vx_yield` | decoded in **all** modes; acts only under SPLIT, no-op (warp unlock only) under DEFAULT/NV_ITS |
| **0x06** | **0** | **`vx_bar_add`** (bid = rs1 field literal, no register read) | NV_ITS only |
| **0x06** | **1** | **`vx_bar_wait`** (bid = rs1 field literal) | NV_ITS only |

`vx_yield` decoding unconditionally is required because the threadsplit LLVM inserts it
into blocking loops; a DEFAULT/NV_ITS build must still run those binaries (and
`vx_intrinsics.h` cannot be config-conditional — `ci/check_config_boundary.sh` forbids
`VX_config.h` in `sw/`).

`VX_CFG_ITS_NUM_BARRIERS = 8` (new toml knob, NV_ITS-only): the 5-bit rs1 field allows 32
but the old tree's 32 was field-width-driven, not need-driven (the diverge test uses 8);
barrier-mask flops scale linearly with it.

Two pre-existing SPLIT-branch defects get fixed in the same series since the code is open
anyway: `INST_SFU_YIELD = 4'hE` collides with `INST_SFU_RTUW = 4'hE` under
`EXT_GFX_ANY_ENABLE` (YIELD moves to `4'hF`, the last free code with all gfx extensions
on), and `VX_trace_pkg.sv` has no YIELD trace case. The 4-bit SFU code space is otherwise
full, so `INST_SFU_BAR_ADD/BAR_WAIT` take `4'hB/4'hC` — the TEX/OM slots — under
`VX_CFG_DIVERGE_TYPE_NV_ITS` only, with a `STATIC_ASSERT` rejecting an NV_ITS+graphics
build (the NV_ITS eval configs never enable graphics).

## 7. Compiler support: `-vortex-divergence-arch` (llvm-vortex, branch `threadsplit`)

One toolchain build serves all three architectures via a **runtime backend flag** — no
per-mode LLVM rebuilds:

```
-mllvm -vortex-divergence-arch={ipdom|tsplit|its}     (default: tsplit)
```

A new `cl::opt` enum in `RISCVTargetMachine.cpp`, **orthogonal** to the existing
`-vortex-branch-divergence` (0/1/2), which stays what it is today: the
structurization-aggressiveness / master-enable knob. Overloading that int would conflate
two axes — its value is consumed positionally at `RISCVTargetMachine.cpp:539`
(`SkipRegionalBranches = (mode==1)`), and the copy stored in `VortexBranchDivergence1`
(`divergenceMode_`) is a dead field today. The default `tsplit` mirrors the proposed hw
default; reviewer call, jointly with §10.2.

### 7.1 Per-mode codegen

| | `ipdom` | `tsplit` | `its` |
|---|---|---|---|
| StructurizeCFG + `setRequiresStructuredCFG` | yes | yes | **no** (per-thread PCs make reducible-but-unstructured CFG legal) |
| `processBranches` | `vx_split`/`vx_join` | `vx_split`/`vx_join` | `vx_bar_add bid` before the divergent branch, `vx_bar_wait bid` in the join stub at the ipdom |
| `processLoops` | `vx_pred`/`vx_pred_n` + tmask snapshot | same **+ `vx_yield`** on blocking-loop back-edges | `bar_add` in preheader, `bar_wait` at loop exit; **no pred, no tmask, no yield** (§4.3.1 — deliberate) |
| ISel select lowering (`RISCVISelLowering.cpp:19711/:19914`) | split/join path | split/join path | fall through to the plain-branch path (`:19934`) |
| `VortexBranchDivergence0` | full | full | keep ret/unreachable unification (PDT must stay valid for barrier placement); **skip** the divergent-select/min-max unswitching (pure pessimization under per-thread PCs) |
| `VortexBranchDivergence2` | both modes | both modes | keep the pre-RA `VX_MOV` elimination; skip the pre-emit `BEQ/BNE` assert + split-polarity fixup (no splits exist) |

`ipdom` is `tsplit` minus exactly one thing — the yield emission block at
`VortexBranchDivergence.cpp:1439-1444` — so its output must diff-match upstream
llvm-vortex codegen; that is its Phase gate. The `its` mode reuses the existing ipdom
computation (`PDT.findNearestCommonDominator`) and join-stub/`replaceSuccessor`
machinery of `processBranches` verbatim; only the emitted ops change.

Two shared-code hazards found in the investigation must become mode-conditional:
`RISCVExpandPseudoInsts.cpp:192` hard-`abort()`s on any CC-op (currently for *all*
targets — under `its`, plain CC-ops are legal again and this guard would kill codegen
outright), and `setRequiresStructuredCFG` is consulted by generic CodeGen
(TailDuplication, MachineBlockPlacement, EarlyIfConversion), so `its` binaries will
differ beyond divergence handling — that is part of ITS's "compiler cost" story and gets
stated in the eval, not hidden.

**Barrier id allocation** (`its`): static per function by nesting depth (nested regions
get deeper ids, disjoint regions share one — safe under equality release); compile error
if a function needs more than `VX_CFG_ITS_NUM_BARRIERS` (8) live barriers — no spilling
in this iteration (the diverge tests need ≤ 3 live). Ids are reused once released (the
§4.2 mask-clear-on-release semantics make reuse across loop iterations sound).
**Known limitation**: allocation is per-function, so a non-inlined callee's bids collide
with the caller's live barriers; the hardware's masked-arrival/empty-pass rule (§4.2)
makes this safe-but-conservative rather than a deadlock. Full fidelity would need the
patent's BMOV-style barrier save/restore around calls — out of scope, documented.

### 7.2 LLVM ISA surface

New tablegen defs in `RISCVInstrInfoVX.td` (CUSTOM0 = 0x0B, matching §6 exactly):
`VX_BAR_ADD`/`VX_BAR_WAIT` at funct7=0x06, funct3=0/1, bid as a 5-bit literal in the rs1
field, with `int_riscv_vx_bar_add`/`int_riscv_vx_bar_wait` intrinsics. **Not** reusing
`int_riscv_vx_bar` (funct3=4) — that is the warp-granular workgroup barrier, semantically
unrelated. Additionally, `vx_yield` gets promoted from its current raw `.insn` inline-asm
string to a real `VX_YIELD` def (same encoding, funct7=5/funct3=0, so existing binaries
are unchanged) — giving it MC/disassembler support and removing the one asm-string
special case in the pass.

Lit tests: the tree currently has **zero** divergence-codegen lit tests
(`vortex-kernel-csr.ll` is the only vortex test). This work adds per-mode CHECK tests
for split/join, pred, yield, and bar emission — the regression net the switch needs.

### 7.3 Build-system wiring

The kernel-side flag derives from the same config the hardware reads: each
`tests/*/common.mk` (which already inject `-Xclang -target-feature -Xclang +xvortex`)
maps `$(filter -DVX_CFG_DIVERGE_TYPE_%,$(XCONFIGS))` → `-mllvm -vortex-divergence-arch=`
(`DEFAULT→ipdom`, `SPLIT→tsplit`, `NV_ITS→its`). `sw/kernel/Makefile` keeps
`-vortex-branch-divergence=0` (runtime lib builds divergence-off, unchanged). For OpenCL,
POCL's own `init_build` does not pass `+xvortex` — the mode flag rides the same channel
the feature flag already uses (the final clang invocation / `POCL_VORTEX_CFLAGS`), and
**the POCL kernel cache is keyed by source hash, so `~/.cache/pocl` must be cleared when
switching modes** — verified behavior from the vx_yield bring-up; called out as a risk in
§10 because a stale cache silently runs the *other* architecture's binary.

Toolchain discipline: these are llvm-vortex commits on branch `threadsplit`;
per standing policy, `vortex-toolchain-prebuilt` gets refreshed before any dependent
vortex push, and the manual-setup instructions in `threadsplit_proposal.md` §12 remain
valid as-is (same branch, same build recipe).

### 7.4 The binary patcher — demoted to a bring-up crosscheck

With compiler support, `its_patch.py` is no longer the NV_ITS test path. It is kept only
for one Phase-4 gate: a hand-patched diverge binary and a compiler-built one must produce
identical barrier-event traces on SimX (mutual validation of patcher mapping and compiler
emission — they were derived independently). After that gate it drops out of the flow;
it is not ported to ELF32 beyond what that single crosscheck needs.

## 8. Validation plan

| Gate | DEFAULT | SPLIT | NV_ITS |
|---|---|---|---|
| SimX regression suite | must equal upstream master results **and cycle counts** (perf_gate) | must equal this branch today (52/52) | full divergence subset (diverge/bfs/dogfood/jacobi/softmax/raycast + regressions) compiled with `-vortex-divergence-arch=its` — no longer limited to patched binaries (§7) |
| dogfood bar/gbar subtests | excluded upstream (`OPTS ?= -n64 -xbar -xgbar`); not part of any gate | same | same — CTA barriers with intra-warp divergence are out of eval scope |
| Lock tests (lockht/lclist/rangelock, EXT_A) | expected DEADLOCK (documents the baseline defect) | PASS | expected DEADLOCK — no yield in the ported design (a deliberate deviation from the patent, §4.3.1); this is a *finding*, not a bug: it quantifies that bare ITS barriers don't buy forward progress |
| rtlsim | diverge/jacobi/bfs/dogfood/raycast + legacy-spawn set | same + lock tests | diverge + bfs + dogfood (its-compiled) |
| LLVM lit tests (§7.2) | ipdom-mode CHECKs == upstream codegen | tsplit CHECKs == today's output | its CHECKs: bar placement, no split/join/pred |
| Patcher crosscheck | — | — | patched vs its-compiled diverge: identical barrier-event trace (§7.4, one-time gate) |
| conform (reserved funct7) | green as-is | green as-is | expectations adjusted for funct7=6 |
| Cycle parity | `diverge` cycle-identical upstream↔DEFAULT and (already proven) DEFAULT↔SPLIT-convergent-path | — | IPC comparison table (the §3.2 scaling study of the ITS report, re-run on this tree at NT=8/NW=8 and the report's 1c/2w/4t point) |
| Synthesis (after rtlsim-green) | existing baselines | existing `scs_*` DUTs | new `its_nt8nw8` DUT: Fmax + area vs SPLIT vs DEFAULT |

The evaluation deliverable is the three-way table the threadsplit proposal could only
estimate: area (LUT/FF/BRAM), Fmax, IPC on convergent code, IPC on divergent code, and
forward-progress capability.

## 9. Phasing

| Phase | Deliverable | Gate |
|---|---|---|
| 0 | This proposal | reviewed |
| 1 | Config plumbing (toml enum, Makefile companion filters, synth flows) | `DEFAULT/SPLIT/NV_ITS` builds all configure; SPLIT build bit-identical to today |
| 2 | SimX SPLIT fences | DEFAULT run == upstream simx; SPLIT run == today (52/52 + lock tests) |
| 3 | LLVM `-vortex-divergence-arch` switch + lit tests (§7) | ipdom output diff-matches upstream llvm-vortex; tsplit output bit-identical to today; its emits bar ops (lit-verified); prebuilt refresh staged |
| 4 | SimX NV_ITS port | its-compiled diverge PASS on simx; patcher crosscheck trace-identical (§7.4); divergence subset PASS |
| 5 | RTL SPLIT fences | DEFAULT rtlsim == upstream; SPLIT rtlsim == today |
| 6 | RTL NV_ITS port | its-compiled diverge PASS on rtlsim; SimX-as-oracle trace diff on divergence events |
| 7 | Evaluation runs + synth DUTs | the §8 three-way table |

## 10. Risks & open questions

1. **DEFAULT cycle parity**: SCS claimed cycle-identity on `diverge`; fencing must not
   perturb DEFAULT's cycle counts anywhere else — perf_gate is the arbiter.
2. **Default value of the switch** (`SPLIT` proposed) — reviewer call.
3. **ITS scheduler fairness**: lowest-PC-first starves a high-PC thread behind a
   low-PC spin loop; combined with no yield this is why lock tests deadlock. We document
   rather than fix — fixing it (adding yield to ITS) would blur the comparison the switch
   exists to make. See §4.3 for how this and the pick policy deviate from the patent.
4. **Multi-warp CTA interactions**: the old ITS was validated single-kernel on the diverge
   test only; CTA dispatch/retire under `amask`-based warp retirement needs fresh
   validation against the current KMU/CTA dispatcher (the SPLIT bring-up found two
   multi-workgroup bugs in exactly this area — expect the same class here).
5. **`NUM_LANES == NUM_THREADS`** elaboration guard means NV_ITS excludes lane-blocked
   configs; acceptable for evaluation, stated openly.
6. The stale `VX_schedule.sv` link in `threadsplit_proposal.md` §Scope gets fixed
   alongside (file was renamed upstream to `VX_scheduler.sv`).
7. **POCL kernel-cache aliasing across modes**: the cache key is the source hash, not
   the `-mllvm` flags, so switching `VX_CFG_DIVERGE_TYPE` without clearing
   `~/.cache/pocl` silently runs the previous architecture's binary. Mitigation is
   procedural (clear on switch, and the eval scripts do it unconditionally); a cache-key
   fix in POCL is out of scope but noted.
8. **`its`-mode codegen differs beyond divergence**: dropping
   `setRequiresStructuredCFG` re-enables TailDuplication/MachineBlockPlacement/
   EarlyIfConversion effects, so DEFAULT-vs-NV_ITS IPC deltas include general codegen
   differences, not just the divergence mechanism. Stated in the eval write-up; the
   SPLIT-vs-NV_ITS comparison on the same divergent kernels is the primary headline.
9. **Barrier-id exhaustion** under `its` is a compile error, not a spill (§7.1); deep
   divergence nesting in future workloads would need an allocator iteration.

---

## 11. Execution results (2026-09-08)

Phases 1–6 implemented and validated on branch `threadsplit` (vortex + llvm, both
uncommitted); Phase 7 partially complete. Status per gate:

**Config**: all three types configure; both override spellings verified; companions reach
CXXFLAGS/VL_FLAGS/synth CFLAGS (12 filter sites + simx injection).

**Fence inertness** (mechanical `unifdef`-style resolution, per macro arm):
SimX `scheduler.{h,cpp}`/`wctl_unit.cpp` and RTL `VX_scheduler.sv`/`VX_wctl_unit.sv`/
`VX_warp_ctl_if.sv` — SPLIT arm identical to pre-fence HEAD; DEFAULT arm identical to
upstream 451e04858887 except the designed always-decoded YIELD (wsync-style unlock).
tsplit toolchain output: diverge kernel **bit-identical** before/after the LLVM changes
(inline-asm→intrinsic yield promotion included).

**SimX**: SPLIT — diverge/demo/lockht(EXT_A) pass unchanged. DEFAULT — diverge/demo pass.
NV_ITS — diverge, bfs, jacobi, dogfood (22/22, upstream default opts), sgemm, vecadd,
demo, dotproduct, basic all PASS with its-compiled kernels (no binary patcher used).

**rtlsim**: SPLIT demo/diverge PASS; DEFAULT demo/diverge PASS; NV_ITS diverge/bfs/
demo/dogfood(22/22 incl. trig) PASS. NV_ITS diverge instruction count matches SimX
exactly (223063) — SimX-as-oracle.

**Two ITS defects found and fixed during bring-up** (beyond the §4.2 TMC cleanup):
1. Cross-function barrier-id collision (libm `sinf`) → masked-arrival + empty-pass rule
   (§4.2), both models.
2. Per-thread indirect-branch targets: precompiled libm jump tables diverge per lane;
   the reference RTL's warp-shared `dest` mis-steered lanes (poison-pointer load abort in
   trig). Fixed with per-thread `dest_its` through `branch_ctl_if` (SimX modeled this
   from the start).

**diverge benchmark (1 core, 4 warps × 4 threads)**:

| | instrs | SimX cycles | rtlsim cycles |
|---|---|---|---|
| DEFAULT | 248436 | 822423 | 892235 |
| SPLIT | 253191 (SimX) / 248436 (RTL) | 1056432 | **892235 — cycle-identical to DEFAULT** |
| NV_ITS | 223063 | 735480 | 828435 (−7.1% vs baseline; −10% instrs, split/join/pred gone) |

SPLIT RTL is provably inert on convergent code (no watchdog in RTL; SimX's extra instrs
come from its software watchdog safety net firing on the test's spin-wait loops —
a documented model/RTL fidelity difference, not a defect).

**Forward progress (lockht -n16, EXT_A, SimX)**: SPLIT PASS; DEFAULT and NV_ITS
deadlock (timeout), as predicted — bare ITS barriers do not buy forward progress (§4.3.1).

**Synthesis** (U55C, 300 MHz, core DUT at NT=8/NW=8): the reference's serial min-PC
comparator chain missed timing by a wide margin — WNS **−7.04 ns**, 42 logic levels from
`its_arr` through the priority-encoded warp select into the schedule buffer (the old tree
only ever ran at NT=4 in simulation and was never timing-closed). Fixed as §5.2
anticipated: the min reduction is now a balanced per-warp tournament tree computed in
parallel from registers and muxed by the selected warp; rtlsim confirms it is
cycle-identical. The pre-reset `scs_bram_nt8nw8` baseline (109K LUTs) is NOT comparable
(different upstream base); the like-for-like routed results on this tree, same flow:

| core @ NT=8/NW=8, U55C, 300 MHz | LUTs | FFs | WNS |
|---|---|---|---|
| DEFAULT (`def_nt8nw8`) | 67,946 | 65,114 | **MET (+0.060 ns)** |
| SPLIT (`split_nt8nw8`) | 69,713 (+2.6%) | 65,390 (+276) | **MET (+0.064 ns)** |
| NV_ITS (`its_tree_nt8nw8`) | 80,054 (+17.8%) | 69,041 (+3,927) | VIOLATED (−1.053 ns ≈ 228 MHz) |

SPLIT's forward-progress capability costs 2.6% core LUTs and 276 flops and closes
300 MHz with margin. NV_ITS costs 6.9× more area than SPLIT (per-thread PCs + barrier
masks ≈ +3.9K flops, matching the §5.3 estimate) and, even with the balanced tree, still
misses 300 MHz by 1.05 ns on the `its_arr`→blocked→min-PC-tree→schedule path (17 logic
levels). The remaining closure lever — registering the per-warp blocked mask at the cost
of one cycle of barrier-wakeup latency — is documented future work, deliberately not
taken: it would change NV_ITS cycle behavior mid-evaluation.

**Deferred**: full perf_gate DEFAULT-vs-upstream cycle sweep (source-level identity +
spot checks done); rtlsim SPLIT lock tests under EXT_A re-run (covered by fence
bit-identity + pre-fence validation); conform-test funct7=6 expectations only apply to
NV_ITS builds; `vortex-toolchain-prebuilt` refresh required before any dependent push.

---

## 12. Completing the ITS arm — minimal A100-equivalent (IMPLEMENTED 2026-09-09)

**Status: implemented and validated.** The ITS arm now passes all ten benchmarks on
both SimX and rtlsim under ITS+Y; the ablation (`-DVX_CFG_ITS_YIELD_DISABLE`) reproduces
the barriers-only deadlocks. Two RTL bugs surfaced during the §13 storage redesign and
were root-caused via SimX-as-oracle trace diff:

1. **bar_add stale-row regroup.** `bar_add` adds participants but changes no thread's PC;
   the running group's row-store entries are stale (their truth is the registered
   `grp_pc`). Regrouping through the min-PC tree on `bar_add` therefore mis-steered the
   warp to a stale row PC. Fix: `bar_add` keeps the current group (`srv_keep_grp`), never
   regroups. Every bar_add-bearing kernel (diverge/bfs/sgemm/jacobi/…) was corrupted
   before this fix.
2. **yield→wake row read-after-write hazard.** A full-warp `vx_yield` writes the resume PC
   into the row RAM and parks the group; the wake fired the *next* cycle and read the row
   RAM before the yield's registered write had committed, so it woke the threads at a
   stale prologue PC (observed: lockht wid=3 resumed at 0x94 instead of 0xcc → misaligned
   load). Fix: fold the wake into the yield service so the just-yielded group's fresh
   resume PC comes from `srv_wpcs` (row bypass) rather than a racing next-cycle row read.
   The separate wake event is retained as a safety net for the (hazard-free, multi-cycle)
   case of threads yielded in earlier cycles.

The subsections below are the design as implemented.

---

## 12.0 (original plan) Completing the ITS arm — minimal A100-equivalent

**Scope rule (revised per review):** implement only what (a) released Volta/Ampere hardware
demonstrably has, (b) is the *minimum* needed for the ITS arm to pass all ten evaluation
benchmarks. For calibration: A100 SASS exposes the full convergence-barrier machinery
(BSSY/BSYNC/BMOV/BREAK/YIELD over barrier registers B0–B15), while mainline
GPGPU-Sim/Accel-Sim does **not** model the ITS microarchitecture at all — PTX mode uses a
post-dominator IPDOM stack and SASS-trace mode replays recorded active masks — so simulator
precedent imposes no additional requirements. The one capability our benchmark suite
exercises and the current arm lacks is forward progress on blocking loops → **YIELD is the
single feature added.**

### 12.1 F1 — YIELD and the Yielded thread state (IN SCOPE)

Patent/§SASS: a yielded thread leaves the schedulable set and is excluded from barrier
release; the compiler inserts YIELD on control paths without statically bounded length
(exactly our blocking-loop heuristic).

- **ISA**: reuse `vx_yield` (funct7=0x05) unchanged — already decoded in every mode. Under
  NV_ITS it stops being a no-op and acts on new per-thread state. No new encodings.
- **SimX** (`scheduler.h/.cpp`, `wctl_unit.cpp`): new per-warp `ThreadMask yielded`. YIELD:
  `yielded |= group` (tpc already = PC+4). Group selection: `runnable = amask & ~blocked &
  ~yielded`; when a warp has alive threads but zero runnable ones and at least one yielded
  (not barrier-blocked) thread, clear the warp's yielded mask — wake-when-nothing-else gives
  the patent's spinner/holder alternation. Barrier release excludes yielded participants:
  release when `arrived ⊇ (participate & ~yielded)`, re-evaluated on YIELD, arrival, and TMC.
- **RTL** (`VX_scheduler.sv`): `yielded[NW][NT]` flops (+64 at NW=NT=8); the runnable term
  and the release equality change as above; the wake condition fires when a warp has alive
  threads but an empty runnable group. Lands together with the §13 storage redesign (yield
  park and wake are two more events through the shared regroup engine).
  `VX_wctl_unit`: yield under NV_ITS routes through the `its` channel (a third op kind next
  to add/wait, reusing the existing `its_pc` = PC+4 payload as the park PC) instead of the
  wsync fold.
- **Compiler**: widen the yield-emission gate in `processLoops` from `== VXDA_TSPLIT` to
  `!= VXDA_IPDOM` (heuristic and insertion point shared verbatim with tsplit), and in `its`
  mode emit in **both** exit polarities (the pred_n-arm restriction is a predication
  artifact ITS doesn't have). Lit test gains ITS-YIELD CHECK lines.
- **Ablation preserved**: toml knob `VX_CFG_ITS_YIELD_ENABLE` (default **true**; disable
  with the config system's canonical `-DVX_CFG_ITS_YIELD_DISABLE` spelling to reproduce
  the barriers-only arm already measured). The build system forwards the knob as
  `-mllvm -vortex-its-yield=0` when disabled so hardware and generated code always agree.

### 12.2 OPT-OUT — already satisfied, no work

The TMC handler clears exited threads from every barrier and re-evaluates release —
observably equivalent to the patent's ignore-exited rule (§4.3.3). F1 interaction: thread
exit also clears its `yielded` bit.

### 12.3 Explicitly OUT of scope (with rationale)

- **BMOV barrier save/restore**: real A100 has it (SASS emits BMOV around calls), but no
  benchmark in the suite requires it — the masked-arrival/empty-pass hardening already
  carries every cross-function case (trig passes) — and Accel-Sim does not model it. The
  masked-arrival rule stays the shipping mechanism. (Design sketch if ever needed:
  `vx_bar_save rd,bid` / `vx_bar_restore rs1,bid` at funct7=0x06 funct3=2/3 packing
  `{arrived,participate}` into one register, callee-saved convention in `its` mode.)
- **Fewest-active-threads pick policy**: NVIDIA has never published the shipped policy;
  GPGPU-Sim models none. Lowest-PC (the reference policy) stays; no knob.
- **New OPT-OUT/BREAK encodings**: Vortex thread exit is TMC; nothing to add.

### 12.4 Validation

| Gate | Expectation |
|---|---|
| Lock suite (lockht/lclist/rangelock), NV_ITS+YIELD, SimX + rtlsim | **PASS** → ITS arm reaches 10/10 |
| Lock suite with `VX_CFG_ITS_YIELD_ENABLE=0` | DEADLOCK (ablation reproduces the current report) |
| 10-benchmark campaign, NV_ITS+YIELD | no regression on the 7 already-passing kernels (yield fires only in blocking loops) |
| lit + conform | ITS-YIELD CHECK lines; no new encodings to cover |
| Synthesis | re-run `sched_nvits` / `its_tree` DUTs; +64 flops + release-eq masking, expected ≈neutral timing |
| Report/artifact | four-way tables: Baseline / TSPLIT / ITS-CB (ablation) / **ITS+Y**; headline becomes complete-vs-complete |

### 12.5 Phasing

| Phase | Deliverable | Gate | Status |
|---|---|---|---|
| 8a | SimX yielded state + compiler gate widening + knob | SimX lock suite PASS under ITS+Y; knob=0 reproduces deadlocks | **DONE** — SimX 10/10; ablation deadlocks |
| 8b | RTL yielded state + §13 storage redesign | rtlsim lock suite PASS; SimX-as-oracle instruction-exact on the campaign kernels | **DONE** — rtlsim 10/10; two hazards fixed (§12 top) |
| 8c | Re-synthesis + campaign re-run + report/artifact update | four-way evaluation published | **DONE** — report + design reviews updated; scheduler + core re-synthesized. §13 v2 area redesign hit target (scheduler 17.4K→7.4K LUT, core 80K→70.6K), pipelined to −1.579 scheduler / −1.894 core. Result: TSPLIT Pareto-dominates both ITS design points (smaller than v1, faster than both, matches v3 area at 60% higher Fmax) |

Review outcome (approved): knob default = 1; completed arm's report label = **ITS+Y**.

---

## 13. RTL storage efficiency — BRAM/LUTRAM banking for large divergence state

**Requirement (review revision):** both non-baseline arms must use memory primitives for
large storage, the way the baseline keeps its IPDOM stack in RAM, instead of paying LUT
fabric for flop arrays.

### 13.1 Audit of the measured scheduler DUTs (routed, U55C, 300 MHz target)

| scheduler DUT | LUT as logic | LUT as mem | FF | BRAM | WNS |
|---|---|---|---|---|---|
| DEFAULT | 3,685 | 334 | 4,009 | 0 | +0.350 |
| SPLIT | 6,006 | 334 | 4,525 | 1 | +0.323 |
| NV_ITS (v1) | 17,092 | 286 | 7,451 | 0 | −0.725 |

**SPLIT verdict — already conformant.** Its one large structure (the parked-split pool,
`NW×2NT` entries of `{tmask,pc}`) is in BRAM with a registered 1-cycle pop. The residual
+516 FF are per-warp scratch (`cs_ptmask/cs_ppc/cs_done/cs_inpool/cs_head/cs_cnt`,
~520 bits) — the same storage class as the baseline's own `warp_pcs`/`thread_masks` flops,
individually mask-scanned every cycle. No change.

**NV_ITS v1 verdict — non-conformant.** All 2,944 bits of divergence state
(`thread_pcs` 1,920 + barrier masks 1,024) are flops with full next-state mux fabric, and
the min-PC tournament tree is replicated per warp (×NW) and sits combinationally in the
schedule path — the source of both the 13.4K excess LUTs and the −0.725 ns WNS.

### 13.2 NV_ITS v2 — the group-cache microarchitecture

Key observation: per-thread PCs are only *needed* when a warp's runnable group changes,
and at most one group-changing event per warp is in flight (the warp is schedule-stalled
from issue of the causing instruction until its resolution). So:

- **Registered group cache**: per-warp `grp_pc[NW]` (PC_BITS) + `grp_mask[NW]` (NT).
  The schedule path reads these registers exactly like the baseline reads
  `warp_pcs`/`thread_masks` — the min-PC tree leaves the schedule path entirely.
  `ready_warps = active & ~stalled & (grp_mask != 0)`; issue takes `{grp_mask, grp_pc}`
  and a fired issue advances `grp_pc += 4` (register increment; RAM untouched).
- **Thread-PC store**: `NW`-deep × `NT×PC_BITS`-wide LUTRAM row array (`VX_dp_ram`,
  1W1R, async read), one row (= one warp) written per cycle with per-thread enables.
  Invariant: threads *outside* their warp's current group have their true PC in the RAM;
  group members' truth is `grp_pc` (their RAM entries are stale until the next park).
- **Barrier-mask store**: `NW`-deep × `2×NBAR×NT`-wide LUTRAM row array for
  `{participate, arrived}`. Legal because every reader/writer is a per-warp event and
  events are serialized (below); the global per-cycle "any runnable" scan v1 needed is
  gone — `grp_mask != 0` already encodes it.
- **One shared regroup engine**: a single min-PC tournament tree + release/runnable
  evaluation, fed by an event arbiter. Sources: (a) branch resolution
  (`NUM_ALU_BLOCKS = 1`), (b) wctl (`bar_add`/`bar_wait`/`yield`/TMC), (c) wake
  (a warp with alive threads, empty group, non-empty yielded set). Each source gets a
  1-deep skid; oldest-first service. Two sources with one server means every event is
  served within 2 cycles, so 1-deep skids cannot overflow. The event warp stays
  schedule-stalled until served (its unlock moves from the raw event to service
  completion), so a collision costs that warp exactly +1 cycle — cycle-identical to v1
  otherwise.
- **Service actions** (all in the service cycle: RAM row write + tree over
  {event-bypassed row} + group-register update + unlock):
  | event | row write | new group |
  |---|---|---|
  | CTA dispatch | entry PC → cta_tmask | direct: cta_tmask @ entry PC (no tree) |
  | wspawn | pc → thread 0 | direct |
  | TMC | `its_pc` → live set; masks cleaned per §4.2 | direct: live & ~blocked @ `its_pc` |
  | bar_add | none (participation only) | unchanged — **must not** regroup through the tree: the running group's row entries are stale (their truth is `grp_pc`), so a tree pass here mis-steers the warp (found in bring-up: every bar_add-bearing kernel corrupted its group) |
  | bar_wait | `its_pc` → arriving group | tree (release may free parked threads) |
  | yield | `its_pc` → yielding group | tree over runnable & ~yielded |
  | branch | per-thread dest / `ntaken_pc` → issued group | tree |
  | wake | none (clear yielded) | tree |
- **Expected result**: scheduler LUTs from ~17.1K to the SPLIT class (~6K); FFs from
  7.45K to ~4.8K (group cache + skids + `yielded`/`amask` replace 2.9K state flops);
  timing closed — the tree runs off registered event payloads into registers, and the
  schedule path is baseline-shaped.

### 13.3 Constraints and asserts

- `NV_ITS` requires `NUM_ALU_BLOCKS == 1` (single branch event source) and excludes
  `EXT_C_ENABLE` (the RVC decode-advance path bypasses the group-PC advance) — both
  become `STATIC_ASSERT`s beside the existing `NUM_ALU_LANES == NUM_THREADS` guard.
- SimX remains the oracle: instruction counts must match exactly; cycle deltas stay
  within the parity tolerance (the only new divergence is +1 cycle on event collisions).
