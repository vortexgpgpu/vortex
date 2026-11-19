# Trap / Exception Foundation (Preemption Groundwork) — Design

**Scope:** the synchronous machine-mode trap path — `ECALL` / `EBREAK` /
`MRET` with per-warp M-mode CSRs — that is the foundation for future
preemption, plus the host-side ELF/HTIF plumbing that the riscv-tests
suite rides on. Covers the RTL
([`hw/rtl/core/VX_warp_sched.sv`](../../hw/rtl/core/VX_warp_sched.sv),
[`VX_alu_int.sv`](../../hw/rtl/core/VX_alu_int.sv),
[`VX_csr_data.sv`](../../hw/rtl/core/VX_csr_data.sv)) and the SimX model
([`sim/simx/scheduler.cpp`](../../sim/simx/scheduler.cpp),
[`sim/common/elf_loader.cpp`](../../sim/common/elf_loader.cpp),
[`sim/common/host_monitor.cpp`](../../sim/common/host_monitor.cpp)).

> **Scope note:** what is implemented is the **synchronous-trap
> foundation only** (ECALL/EBREAK/MRET, single-hart `env/p/`). The named
> "preemption" capability — async preempt-from-CP, interrupts, supervisor
> mode, fault producers, register-file/IPDOM save — is future work (§4).
> The kernel-entry/dispatch mechanism is in
> [`kernel_entry_and_dispatch.md`](kernel_entry_and_dispatch.md).

---

## 1. Architecture

Each warp owns five M-mode trap CSRs held in its slice's warp scheduler
([`VX_warp_sched.sv`](../../hw/rtl/core/VX_warp_sched.sv)):
`mstatus`, `mtvec`, `mepc`, `mcause`, `mtval`. Each is stored in its legal
(WARL) form, keeping only the bits that can be nonzero (§1.1). Addresses:
`MSTATUS=0x300`, `MTVEC=0x305`, `MSCRATCH=0x340`, `MEPC=0x341`,
`MCAUSE=0x342`, `MTVAL=0x343`, with the `mstatus` field positions
(`VX_CSR_MSTATUS_{MIE,MPIE,MPP,FS}`) beside them in
[`VX_types.toml`](../../VX_types.toml).

The branch unit is the trap producer
([`VX_alu_int.sv`](../../hw/rtl/core/VX_alu_int.sv)): it
raises `is_trap_entry` / `is_mret_op` with a `br_trap_cause` (`EBREAK=3`,
`ECALL=11`). These ride the existing `VX_branch_ctl_if` (the proposed
standalone `VX_trap_if.sv` was folded in — functionally equivalent). The
scheduler redirects the warp PC: a trap goes to `mtvec`, an `mret`
returns to `mepc`. On trap entry the hardware snapshots `mepc`/`mcause`,
clears `mtval` and updates `mstatus`; `mret` updates `mstatus` too (§1.1).
A same-cycle software `csrw` and a hardware trap entry or return are ordered
hardware-wins, per field: the fields the hardware writes take the hardware
value, and the software write still lands in the others. The CSR file
([`VX_csr_data.sv`](../../hw/rtl/core/VX_csr_data.sv)) does only the read
mux and routes writes back via `sched_csr_if`; the scheduler legalises
them.

SimX mirrors this exactly:
[`scheduler.cpp`](../../sim/simx/scheduler.cpp) defines `raise_trap`,
`mret`, `trigger_ecall(wid,pc)`, `trigger_ebreak(wid,pc)`;
[`alu_unit.cpp`](../../sim/simx/alu_unit.cpp) dispatches ecall/ebreak/mret
per warp; [`csr_unit.cpp`](../../sim/simx/csr_unit.cpp) applies the same
write masks and read values as the RTL. The old "kill all warps"
(`active_warps_.reset()`) hack is gone.

### 1.1 Legal values and the `mstatus` protocol

Machine mode is the only privilege mode, there are no interrupts yet, and
`mtvec` supports Direct mode only. The trap CSRs therefore hold:

| CSR | Stored | Reads |
|---|---|---|
| `mstatus` | `MIE` (3), `MPIE` (7), `FS` (14:13) | those fields, `MPP` (12:11) hardwired `11`; every other bit 0 (`SD` is not implemented) |
| `mtvec` | `[XLEN-1:2]` | `MODE` (1:0) always 0 (Direct) |
| `mepc` | `[XLEN-1:1]` with C, `[XLEN-1:2]` without | bit 0 always 0; bit 1 also 0 without C (IALIGN=32) |
| `mcause`, `mtval` | `[XLEN-1:0]` | as written |

A write keeps only the stored bits; a CSRRS/CSRRC reads the legal value,
so it cannot set a hardwired bit. `FS` is a plain read/write field and
gates nothing yet.

| Event | `MIE` | `MPIE` | `MPP` |
|---|---|---|---|
| trap entry (ECALL/EBREAK) | 0 | old `MIE` | `11` (M) |
| `mret` | old `MPIE` | 1 | `11` (M) |

`MPP` stays `11` across `mret` because M is the only mode to return to.

### 1.2 Conformance tests

The upstream riscv-tests run unmodified through `tests/riscv/isa`
(`make run-{simx,rtlsim}-{32,64}mi` for the machine-mode suites; the
`run-*` aggregates of the matching XLEN include them). Of `rv32mi-p` and
`rv64mi-p`, `mcsr`, `scall` and `sbreak` run. The rest are skipped, with the reason beside the skip list in
[`tests/riscv/isa/Makefile`](../../tests/riscv/isa/Makefile): they need a
trap source other than ECALL/EBREAK (illegal instruction, misaligned
fetch/load/store, or `mstatus.FS`=0 trapping an FP store), or a CSR Vortex
does not model (Sdtrig triggers, the Zicntr user counters, writable
`minstret`/`mcountinhibit`, PMP).

---

## 2. Host ELF / HTIF plumbing

[`sim/common/elf_loader.{h,cpp}`](../../sim/common/elf_loader.cpp)
(`loadElfImage`, `ElfImage`, `isElfFile`) and
[`sim/common/host_monitor.{h,cpp}`](../../sim/common/host_monitor.cpp)
(an HTIF `tohost` watcher) are wired identically into both simulators'
main loops ([`sim/simx/main.cpp:119-216`](../../sim/simx/main.cpp#L119),
[`sim/rtlsim/main.cpp:92-131`](../../sim/rtlsim/main.cpp#L92)) with
HTIF-takes-precedence exit codes. This is what lets the upstream
riscv-tests (now cloned/built via `RISCV_TESTS_STAMP`, not checked in) run
on Vortex; the 616 checked-in `.bin` files and the
`riscv-tests`/`riscv-test-env` patches were removed.

---

## 3. Flow

1. A warp executes `ecall`/`ebreak` → the branch unit raises a trap with a
   cause.
2. The scheduler snapshots `mepc`/`mcause` and redirects the warp PC to
   `mtvec`.
3. The handler runs; `mret` restores the PC from `mepc`.
4. Standalone, the ELF loader places the image and the HTIF monitor
   watches `tohost` for the exit code.

---

## 4. Proposed but not yet implemented

The named **preemption** capability is future work:

1. **Async preempt-from-CP** — a `preempt_pending` port, a custom trap
   cause, and a safe-point check so the CP can preempt a running grid.
   This is the actual preemption deliverable; nothing is built.
2. **Fault producers** — illegal-instruction / misaligned-access / page-
   fault causes (2/4/6). Each one also unlocks the matching skipped
   `rv32mi-p` tests (§1.2), and the illegal-instruction trap lets `FS`
   gate the FP unit.
3. **Supervisor mode** — `stvec`/`sepc`/`scause` + `medeleg`-driven
   redirect.
4. **Interrupts** — `mie`/`mip`, async injection, `mcause` MSB.
5. **Register-file / IPDOM-stack save-restore** — required for true
   context switch (only the trap CSRs are saved today).
6. **`tests/regression/csr_smoke/`** unit test — absent.
7. **Guardrails** (open risks): same-cycle `csrw`/trap conflict assertion,
   multi-warp `tohost` race. (`mtvec` MODE≠0 can no longer happen: MODE is
   hardwired to Direct.)
8. **`misa.U`** is still set (`VX_CFG_MISA_STD` bit 20) although U-mode does
   not exist and `mstatus.MPP` is hardwired to M.

**Superseded directions** (recorded to avoid revival): the standalone
`VX_trap_if.sv` interface (folded into `VX_branch_ctl_if`); adding CSR
addresses to a hand-written `VX_csr.vh` (they live in the generated
`VX_types.vh`); and the SimX "kill all warps" trap hack (replaced by
per-warp `raise_trap`).
