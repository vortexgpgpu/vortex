# Vortex Preemption — Prior Work Review

**Source reviewed:** [`/home/blaisetine/dev/vortex_preemption`](file:///home/blaisetine/dev/vortex_preemption) (single commit `ba81090 "Update"` on `main`, no PR, no test artifacts).
**Scope of this document:** what the student actually built, how it works, and what is still missing before Vortex can claim preemption support. This is a *retrospective* review of prior work, not a design proposal for `feature_preempt` — that proposal should be authored separately and informed by this review.

---

## 1. Executive summary

The directory name is misleading: the student did **not** implement preemption. They built a **minimal RISC-V S-mode trap framework inside SimX only** — the foundation a preemption mechanism would sit on top of, but not preemption itself. Concretely:

- Per-warp S-mode CSRs (`sstatus`, `stvec`, `sscratch`, `sepc`, `scause`, `stval`) added to the functional emulator.
- A synchronous `raise_trap(wid, cause, tval)` / `return_from_trap(wid)` path in `Emulator`.
- `ECALL` / `EBREAK` rewired from "kill all warps" to "trap to `stvec`"; `SRET`/`MRET`/`URET` rewired to restore PC from `sepc`.
- A reserved exception cause `GpuSyntheticFault = 0x100` (the placeholder for a future GPU-initiated preempt signal) — **never raised anywhere in the codebase**.

What is **absent**:

- No RTL changes — the hardware still has no trap CSRs, no trap vector redirect in fetch, no SRET handling.
- No asynchronous trap-injection path — nothing outside the warp's own instruction stream can cause it to trap, so there is no host-driven or timer-driven preempt.
- No GPGPU context save/restore — the register file, tmask, IPDOM stack, barrier/wspawn state, VPU/TCU state are not saved on trap entry. The IPDOM stack is *cleared* on entry, which makes resuming a divergent kernel after a trap incorrect.
- No software side — `vx_start.S` still has the trap-vector init commented out; the kernel and runtime have no handler stub, no example, no test.
- The student's `tests/` directory was deleted before the snapshot we have (`.gitignore` adds `tests/`), so there are no verification artifacts.

The work is best characterized as **"stage 1: trap plumbing in SimX"** — a useful starting point, but several stages short of preemption. The student's own filename `stage1_types.diff` corroborates this framing.

---

## 2. What was changed, by file

All meaningful changes live in [`sim/simx/`](file:///home/blaisetine/dev/vortex_preemption/sim/simx). Captured as patch text in [`sim/my_changes.patch`](file:///home/blaisetine/dev/vortex_preemption/sim/my_changes.patch) (365 lines). The unrelated `lab1.diff` (DOT8 ALU instruction) and `lab2.diff` (TF32 tensor-core format) are **separate student exercises**, not preemption work — ignore them.

### 2.1 [`sim/simx/types.h`](file:///home/blaisetine/dev/vortex_preemption/sim/simx/types.h)

Added two definitions (~60 lines):

```cpp
enum class ExcCause : uint32_t {
  None                = 0xffffffffu,
  InstrAddrMisaligned = 0,  InstrAccessFault = 1,  IllegalInstr = 2,
  Breakpoint          = 3,
  LoadAddrMisaligned  = 4,  LoadAccessFault  = 5,
  StoreAddrMisaligned = 6,  StoreAccessFault = 7,
  EcallFromUMode      = 8,  EcallFromSMode   = 9,  EcallFromMMode = 11,
  InstrPageFault      = 12, LoadPageFault    = 13, StorePageFault = 15,
  GpuSyntheticFault   = 0x100   // reserved; not used anywhere
};

namespace csr {
  static constexpr uint32_t SSTATUS  = 0x100;
  static constexpr uint32_t STVEC    = 0x105;
  static constexpr uint32_t SSCRATCH = 0x140;
  static constexpr uint32_t SEPC     = 0x141;
  static constexpr uint32_t SCAUSE   = 0x142;
  static constexpr uint32_t STVAL    = 0x143;
}
```

The numeric values 0..15 match RISC-V's standard `mcause` exception encoding. `GpuSyntheticFault = 0x100` is the bit-16 region the RISC-V spec reserves for custom causes; the intent is that some GPU-side event (host doorbell, timer, scheduler signal) would later raise this cause to *force* a kernel out to the handler. That code path was never written.

### 2.2 [`sim/simx/emulator.h`](file:///home/blaisetine/dev/vortex_preemption/sim/simx/emulator.h)

Added a per-warp CSR state block and trap-handler state to [`warp_t`](file:///home/blaisetine/dev/vortex_preemption/sim/simx/emulator.h#L54-L95):

```cpp
struct warp_csr_state_t {
  Word sstatus, stvec, sscratch, sepc, scause, stval;
};
struct warp_t {
  /* … existing fields … */
  warp_csr_state_t csr;
  bool             in_trap_handler;
};
```

And two new public methods on [`Emulator`](file:///home/blaisetine/dev/vortex_preemption/sim/simx/emulator.h#L119-L121):

```cpp
void raise_trap(uint32_t wid, ExcCause cause, Word tval);
void return_from_trap(uint32_t wid);
```

### 2.3 [`sim/simx/emulator.cpp`](file:///home/blaisetine/dev/vortex_preemption/sim/simx/emulator.cpp)

Three groups of edits.

**Trap entry / exit** ([emulator.cpp:145-194](file:///home/blaisetine/dev/vortex_preemption/sim/simx/emulator.cpp#L145-L194)) is the core of the work. Annotated:

```cpp
void Emulator::raise_trap(uint32_t wid, ExcCause cause, Word tval) {
  auto& warp = warps_.at(wid);

  // 1. Save context that hardware would save automatically.
  warp.csr.sepc   = warp.PC;                  // faulting PC
  warp.csr.scause = static_cast<Word>(cause); // why we trapped
  warp.csr.stval  = static_cast<Word>(tval);  // optional aux info
  warp.in_trap_handler = true;

  // 2. If no handler installed, kill the warp (fail-safe).
  if (warp.csr.stvec == 0) { active_warps_.reset(wid); ... return; }

  // 3. Flush any in-flight micro-ops from the old context.
  warp.ibuffer.clear();

  // 4. Throw away divergence state so the handler runs straight-line.
  while (!warp.ipdom_stack.empty()) warp.ipdom_stack.pop();

  // 5. Redirect to the handler entry point.
  warp.PC = warp.csr.stvec;
}

void Emulator::return_from_trap(uint32_t wid) {
  auto& warp = warps_.at(wid);
  warp.PC = warp.csr.sepc;        // resume at the faulting PC
  warp.in_trap_handler = false;
  warp.ibuffer.clear();
}
```

Key design decisions encoded here:

| Decision | Where | Implication |
|---|---|---|
| Trap is **per-warp**, not per-core | `wid` parameter, no fan-out to other warps | A preempt of a kernel requires every warp to trap separately — there is no built-in broadcast path. |
| `stvec` is used **literally as the new PC** | `warp.PC = warp.csr.stvec` | No support for the low-2-bit MODE field of RV `stvec` (vectored mode is impossible). Handler must be 4-byte aligned. |
| **Only PC, cause, tval are saved** | No reg file copy, no `tmask` copy, no IPDOM copy | The handler is fully responsible for saving everything else in software, and divergence state is *destroyed* on entry. |
| IPDOM stack is **cleared on entry** | step 4 above | Resuming after a trap that was taken inside a divergent region will be incorrect — there is no way to re-converge. |
| ibuffer is **cleared on entry and exit** | step 3 + return path | The micro-op currently being executed completes (the trap is taken *after* execute()), but any second/third µ-op of a multi-µop instruction is silently dropped. |
| **No `sstatus` updates** | not written by `raise_trap` | `SPP` / `SPIE` / `SIE` bits are never set or cleared — `sret` cannot meaningfully restore an interrupt-enable state. |
| **No `stvec == 0` warning to caller** | warp is silently disabled | Easy to lose a warp without noticing during bring-up. |

**CSR plumbing** ([emulator.cpp:506-511](file:///home/blaisetine/dev/vortex_preemption/sim/simx/emulator.cpp#L506-L511) for read, [emulator.cpp:635-652](file:///home/blaisetine/dev/vortex_preemption/sim/simx/emulator.cpp#L635-L652) for write) routes the six S-mode CSR addresses to/from `warp_csr_state_t`. The pre-existing M-mode CSRs (`mstatus`, `mtvec`, `mepc`, `mcause`) remain stubs that return 0 — the work added S-mode CSRs in parallel rather than backing the M-mode stubs.

**ECALL / EBREAK behavior change** ([emulator.cpp:709-716](file:///home/blaisetine/dev/vortex_preemption/sim/simx/emulator.cpp#L709-L716)):

```cpp
// before: both helpers called active_warps_.reset() (kill everything)
void Emulator::trigger_ecall (uint32_t wid) { raise_trap(wid, ExcCause::EcallFromSMode, 0); }
void Emulator::trigger_ebreak(uint32_t wid) { raise_trap(wid, ExcCause::Breakpoint,     0); }
```

Note `trigger_ecall` always reports `EcallFromSMode` regardless of which privilege the warp would actually be in — there is no privilege-tracking state in `warp_t`.

### 2.4 [`sim/simx/execute.cpp`](file:///home/blaisetine/dev/vortex_preemption/sim/simx/execute.cpp)

Three edits ([execute.cpp:171-173](file:///home/blaisetine/dev/vortex_preemption/sim/simx/execute.cpp#L171-L173), [501-524](file:///home/blaisetine/dev/vortex_preemption/sim/simx/execute.cpp#L501-L524), [1575-1584](file:///home/blaisetine/dev/vortex_preemption/sim/simx/execute.cpp#L1575-L1584)):

```cpp
bool took_trap = false;   // ECALL / EBREAK fired raise_trap()
bool did_sret  = false;   // SRET / URET / MRET fired return_from_trap()
…
case BrType::SYS: {
  trace->fetch_stall = true;     // treat like a control-flow change
  switch (brArgs.offset) {
    case 0x000: trigger_ecall (wid); took_trap = true; break; // ECALL
    case 0x001: trigger_ebreak(wid); took_trap = true; break; // EBREAK
    case 0x002:                                               // URET
    case 0x102:                                               // SRET
    case 0x302: return_from_trap(wid); did_sret = true; break;// MRET
  }
}
…
// at the bottom of execute()
if (!took_trap && !did_sret) {
  warp.PC += 4;                       // normal sequential advance
  if (warp.PC != next_pc) warp.PC = next_pc;
} // else: PC was already set by raise_trap / return_from_trap
```

The `fetch_stall = true` annotation tells the perf model the next fetch must wait — this is the right hint even though there is no actual hazard, because the PC is being rewritten from CSR state.

URET, SRET, and MRET all collapse to the same `return_from_trap()` — there is **no privilege-mode tracking and no per-mode `epc`**.

### 2.5 What did *not* change

- **No RTL changes** anywhere under [`hw/`](file:///home/blaisetine/dev/vortex_preemption/hw). Confirmed by grep for `sscratch|stvec|sepc|scause` returning nothing in `hw/rtl/`.
- **No kernel changes.** [`kernel/src/vx_start.S`](file:///home/blaisetine/dev/vortex_preemption/kernel/src/vx_start.S#L51-L53) still has the trap-vector setup commented out:
  ```asm
  # initialize trap vector
  # la t0, trap_entry
  # csrw mtvec, t0
  ```
  There is no `trap_entry` symbol defined anywhere in `kernel/` or `runtime/`.
- **No runtime API.** Nothing in `runtime/` exposes a way for the host to install a handler, signal a preempt, or read back saved state.
- **No tests.** `.gitignore` was modified to add `tests/` ([my_changes.patch:1-9](file:///home/blaisetine/dev/vortex_preemption/sim/my_changes.patch)), so any verification work the student did is not in the snapshot.
- **No other simulators.** `sim/rtlsim/`, `sim/opaesim/`, `sim/xrtsim/` are unchanged. Only SimX (the functional emulator) sees traps.

---

## 3. How it works end-to-end (today)

Given the current code, the *only* execution path that reaches the trap handler is:

```
(SimX only)
    kernel runs                        ← warp at PC = X
       │
       │ executes  ECALL  (or EBREAK)
       ▼
    BrType::SYS dispatch in execute.cpp
       │
       ▼
    Emulator::trigger_ecall(wid)
       │
       ▼
    Emulator::raise_trap(wid, EcallFromSMode, 0)
       │   sepc  ← X         (PC of the ECALL itself)
       │   scause ← 9
       │   stval ← 0
       │   ibuffer.clear()
       │   ipdom_stack.clear()    ← divergence state discarded
       │   PC    ← stvec
       ▼
    execute() returns; took_trap=true so PC is NOT post-incremented
       │
       ▼
    Next step() fetches from stvec
       │
       ▼
    [handler code]               ← must exist in the kernel image
       │   csrr  …  sscratch     ← scratch slot to spill regs
       │   save whatever it touches by hand
       │   do work
       │   csrrw  sp, sscratch, sp  (or similar to restore)
       │   sret
       ▼
    BrType::SYS dispatch (offset 0x102 = SRET)
       │
       ▼
    Emulator::return_from_trap(wid)
       │   PC ← sepc            (= X, the ECALL itself)
       │   ibuffer.clear()
       ▼
    [the kernel re-executes the ECALL!]
```

**Note the last point:** `sepc` is set to the faulting PC (the ECALL itself), and `return_from_trap` restores PC to `sepc` *unchanged*. RISC-V convention is that ECALL/EBREAK handlers must increment `sepc` by 4 themselves before `sret`. The student's code follows that convention, but there is no helper or example demonstrating it — a handler that forgets to `csrrw sepc, sepc, +4` will livelock.

Crucially, the handler is reachable **only via synchronous ECALL or EBREAK from the warp itself**. There is no way for the host, another warp, or any GPU-side event to deliver a trap to a running warp. Therefore this mechanism cannot, today, preempt anything.

---

## 4. Limitations and gaps (what would have to be added)

Roughly in order of "must fix to use" → "must fix to do real preemption":

### 4.1 Correctness gaps in the existing SimX path

1. **Divergent traps are silently broken.** `raise_trap` clears the IPDOM stack. A kernel that traps inside an `if (tid < N) { … ECALL … }` will, after `sret`, resume with all threads (because `tmask` is also not saved) at the post-ECALL instruction with no way to re-converge. *Fix:* save `tmask` + IPDOM stack into a side structure on entry, restore on exit.
2. **`sepc` PC-advance convention is not encapsulated.** The handler must remember to `csrrw sepc, sepc, sepc+4` for ECALL/EBREAK. Nothing in the codebase shows or enforces this. *Fix:* either advance `sepc` inside `raise_trap` for synchronous causes (RISC-V allows this for ECALL since the instruction *is* the trap), or provide an inline helper.
3. **`trigger_ecall` always reports S-mode** regardless of the warp's actual privilege. There is no notion of privilege in `warp_t`. *Fix:* either track privilege or downgrade to a single generic ecall cause.
4. **`stvec` low 2 bits are not masked.** RV spec uses them as MODE. A handler at an unaligned address (`stvec = addr | 1` for vectored mode) will jump into the middle of an instruction. *Fix:* mask before assigning to PC.
5. **`stvec == 0` silently kills the warp.** No diagnostic. *Fix:* log loudly; ideally fatal in debug builds.
6. **Multi-µop instructions are clipped.** `raise_trap` and `return_from_trap` both `warp.ibuffer.clear()` — if a vector or wide LSU instruction has fanned out into several µ-ops in the ibuffer, the remaining ones vanish. The trap framework was written before VPU was on this branch, so this was probably not considered.
7. **M-mode CSRs still return 0.** `mstatus`, `mtvec`, `mepc`, `mcause` are read-only-zero stubs. This is fine *iff* preemption is intended to live entirely in S-mode, but the choice should be deliberate and documented.

### 4.2 Functional gaps before preemption is possible at all

8. **No async trap injection.** `raise_trap` is only ever called from inside `execute()` (in response to ECALL/EBREAK). There is no scheduler-level path that says "next time warp W steps, force it to trap." *Fix:* make `step()` check a pending-trap bit at the top of the loop, and add a `request_preempt(wid, cause)` public API on `Emulator`/`Core`/`Processor`.
9. **No host → GPU preempt signal.** Whatever produces a host-side preempt (DCR write, doorbell MMIO, command-processor command) must propagate down to per-warp pending-trap bits. None of that wiring exists.
10. **No coordinated cross-warp trap.** Preempting a kernel means trapping *every* active warp in the device, not just one. Today the code traps one warp at a time and lets the others keep running. *Fix:* a broadcast — set pending bits across all `active_warps_`, possibly with a barrier inside the handler so all warps' saves complete before any context is migrated.
11. **No GPGPU context save.** Per-warp register file (XLEN × num_threads × 32 = up to 16 KiB for a 32-thread/8-warp/RV32 warp), `fcsr`, `tmask`, IPDOM stack, barrier membership, `wspawn_t`, and per-instance VPU/TCU/SFU/LSU state must all be saved on trap entry and restored on return. None of it is. *Fix:* either expand `raise_trap` to spill into a designated buffer (whose base is in `sscratch`), or define a software ABI that the handler obeys.
12. **No interrupts vs. exceptions distinction.** RV `scause` MSB is the interrupt bit; the student's `ExcCause` enum has no provision for it. Async preempt is most naturally modeled as an interrupt, not an exception. *Fix:* widen `scause` writes to include the interrupt bit.
13. **No delegation (`medeleg`/`mideleg`).** Fine for an S-mode-only design, but document it.
14. **No software side.** Kernel `vx_start.S` has the trap-vector init commented out; no `trap_entry` exists; no example library or test. The very first thing any user of this mechanism has to do is write a handler stub and a test, and none of that work was done.
15. **No RTL.** Everything above is SimX only. The same CSR file, the same trap-redirect logic in fetch, the same SRET handling, the same context-save discipline must be re-implemented in [`hw/rtl/core/VX_csr_data.sv`](file:///home/blaisetine/dev/vortex_preemption/hw/rtl/core/VX_csr_data.sv) + the fetch/issue pipeline. Without RTL there is no path to FPGA or to non-functional measurements.

### 4.3 Architectural questions the existing work does not answer

- **Granularity.** Is preemption per-warp, per-core, per-cluster, or device-wide? The current per-warp `raise_trap` API is the finest possible, but a usable preempt model probably wants device-wide.
- **Persistence.** Does a preempt save state such that the *exact same* warp can resume later, or is the model "kill kernel A, run kernel B, restart A from scratch"? The latter is dramatically simpler and doesn't need any of §4.2.11.
- **Handler residency.** Does the handler live in the same address space as the kernel (sharing the IPDOM stack pointer, stacks, regs) or in a separate "supervisor" image loaded by the runtime? The student's design implies same-space (sscratch is the only free register), but that gives the handler no safe stack to use until it has spilled one.
- **Interaction with the Command Processor.** `feature_preempt` will eventually have to coexist with `feature_cp`. The CP is the natural place to source host-driven preempt signals, and any persisted context needs a CP command to migrate.

---

## 5. Recommendation

Treat the prior work as **a starting reference for the SimX trap CSR layout and entry/exit micro-code**, and write a fresh `preemption_proposal.md` under [`docs/proposals/`](file:///home/blaisetine/dev/vortex_v3/feature_preempt/docs/proposals/) that:

1. Decides the architectural questions in §4.3 first.
2. Specifies the async injection API (§4.2.8–10) and the context-save discipline (§4.2.11).
3. Carries the design through both **SimX and RTL** from the start — the prior work's "SimX only, defer RTL" approach left the design un-grounded in hardware feasibility.
4. Lands a minimal end-to-end test (host triggers preempt → kernel hits handler → handler runs → kernel resumes correctly) before any incremental optimization.

The existing patch [`sim/my_changes.patch`](file:///home/blaisetine/dev/vortex_preemption/sim/my_changes.patch) is small enough (~150 lines of substantive change) that re-applying selectively after the proposal lands is cheaper than trying to forward-port it as a base.
