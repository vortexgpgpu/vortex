# Trap / Exception Foundation (Preemption Groundwork) — Design

**Scope:** the synchronous machine-mode trap path — `ECALL` / `EBREAK` /
`MRET` with per-warp M-mode CSRs — that is the foundation for future
preemption, plus the host-side ELF/HTIF plumbing that the riscv-tests
suite rides on. Covers the RTL
([`hw/rtl/core/VX_scheduler.sv`](../../hw/rtl/core/VX_scheduler.sv),
[`VX_alu_int.sv`](../../hw/rtl/core/VX_alu_int.sv),
[`VX_csr_data.sv`](../../hw/rtl/core/VX_csr_data.sv)) and the SimX model
([`sim/simx/scheduler.cpp`](../../sim/simx/scheduler.cpp),
[`sim/common/elf_loader.cpp`](../../sim/common/elf_loader.cpp),
[`sim/common/host_monitor.cpp`](../../sim/common/host_monitor.cpp)).

> **Scope note:** what is implemented is the **synchronous-trap
> foundation only** (ECALL/EBREAK/MRET, single-hart `env/p/`). The named
> "preemption" capability — async preempt-from-CP, interrupts, supervisor
> mode, fault producers, register-file/IPDOM save — is future work (§5).

---

## 1. Architecture

Each warp owns five M-mode trap CSRs held in the scheduler
([`VX_scheduler.sv:59`](../../hw/rtl/core/VX_scheduler.sv#L59)):
`mstatus`, `mtvec`, `mepc`, `mcause`, `mtval`
(`[NUM_WARPS][XLEN]`). Addresses: `MSTATUS=0x300`, `MTVEC=0x305`,
`MSCRATCH=0x340`, `MEPC=0x341`, `MCAUSE=0x342`, `MTVAL=0x343`
([`VX_types.toml:325-334`](../../VX_types.toml#L325)).

The branch unit is the trap producer
([`VX_alu_int.sv:310-339`](../../hw/rtl/core/VX_alu_int.sv#L310)): it
raises `is_trap_entry` / `is_mret_op` with a `br_trap_cause` (`EBREAK=3`,
`ECALL=11`). These ride the existing `VX_branch_ctl_if` (the proposed
standalone `VX_trap_if.sv` was folded in — functionally equivalent). The
scheduler redirects `warp_pcs_n`: a trap goes to `mtvec & ~3`, an `mret`
returns to `mepc`
([`VX_scheduler.sv:306-311`](../../hw/rtl/core/VX_scheduler.sv#L306)).
On trap entry the hardware snapshots `mepc`/`mcause`; a same-cycle software
`csrw` and a hardware trap entry are ordered hardware-wins
([`:394-412`](../../hw/rtl/core/VX_scheduler.sv#L394)). The CSR file does
only the read mux and routes writes back via `sched_csr_if`
([`VX_csr_data.sv:253-257`](../../hw/rtl/core/VX_csr_data.sv#L253)).

SimX mirrors this exactly:
[`scheduler.cpp:271-298`](../../sim/simx/scheduler.cpp#L271) defines
`raise_trap`, `mret`, `trigger_ecall(wid,pc)`, `trigger_ebreak(wid,pc)`;
[`alu_unit.cpp:363-366`](../../sim/simx/alu_unit.cpp#L363) dispatches
ecall/ebreak/mret per warp. The old "kill all warps"
(`active_warps_.reset()`) hack is gone.

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

The named **preemption** capability is future work (the proposal scoped
these out as non-goals):

1. **Async preempt-from-CP** — a `preempt_pending` port, a custom trap
   cause, and a safe-point check so the CP can preempt a running grid.
   This is the actual preemption deliverable; nothing is built.
2. **Fault producers** — illegal-instruction / misaligned-access / page-
   fault causes (2/4/6).
3. **Supervisor mode** — `stvec`/`sepc`/`scause` + `medeleg`-driven
   redirect.
4. **Interrupts** — `mie`/`mip`, async injection, `mcause` MSB.
5. **Register-file / IPDOM-stack save-restore** — required for true
   context switch (only the trap CSRs are saved today).
6. **`tests/regression/csr_smoke/`** unit test (Milestone 1) — absent.
7. **Guardrails** (open risks): same-cycle `csrw`/trap conflict assertion,
   `mtvec` MODE≠0 assertion, multi-warp `tohost` race.

**Superseded directions** (recorded to avoid revival): the standalone
`VX_trap_if.sv` interface (folded into `VX_branch_ctl_if`); adding CSR
addresses to a hand-written `VX_csr.vh` (they live in the generated
`VX_types.vh`); and the SimX "kill all warps" trap hack (replaced by
per-warp `raise_trap`).

---

## 5. Source proposal

This design consolidates and supersedes `preemption_foundation_proposal.md`
(now removed from `docs/proposals/`). The kernel-entry/dispatch mechanism
is in [`kernel_entry_and_dispatch.md`](kernel_entry_and_dispatch.md).
