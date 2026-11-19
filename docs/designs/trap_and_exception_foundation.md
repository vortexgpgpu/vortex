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
raises `is_trap_entry` / `is_mret_op` with a `br_trap_cause` (§1.3).
These ride the existing `VX_branch_ctl_if` (the proposed
standalone `VX_trap_if.sv` was folded in — functionally equivalent), which
carries the trapping instruction's PC in `dest`. The
scheduler redirects the warp PC: a trap goes to `mtvec`, an `mret`
returns to `mepc`. On trap entry the hardware snapshots `mepc` (the
trapping instruction's PC) and `mcause`, clears `mtval` and updates `mstatus`;
`mret` updates `mstatus` too (§1.1).
A same-cycle software `csrw` and a hardware trap entry or return are ordered
hardware-wins, per field: the fields the hardware writes take the hardware
value, and the software write still lands in the others. The CSR file
([`VX_csr_data.sv`](../../hw/rtl/core/VX_csr_data.sv)) does only the read
mux and routes writes back via `sched_csr_if`; the scheduler legalises
them.

SimX mirrors this exactly:
[`scheduler.cpp`](../../sim/simx/scheduler.cpp) defines `raise_trap`,
`mret`, `trigger_ecall`, `trigger_ebreak`, `trigger_illegal` and
`trigger_misaligned_fetch`;
[`alu_unit.cpp`](../../sim/simx/alu_unit.cpp) dispatches the traps and
`mret` per warp; [`decode.cpp`](../../sim/simx/decode.cpp) produces the
same illegal-instruction uop as the RTL decoder;
[`csr_unit.cpp`](../../sim/simx/csr_unit.cpp) applies the same write masks
and read values as the RTL. The old "kill all warps"
(`active_warps_.reset()`) hack is gone.

### 1.1 Legal values and the `mstatus` protocol

Machine mode is the only privilege mode, there are no interrupts yet, and
`mtvec` supports Direct mode only. The trap CSRs therefore hold:

| CSR | Stored | Reads |
|---|---|---|
| `mstatus` | `MIE` (3), `MPIE` (7), `FS` (14:13) | those fields, `MPP` (12:11) hardwired `11`, `SD` (XLEN-1) = (`FS` == Dirty); every other bit 0 |
| `mtvec` | `[XLEN-1:2]` | `MODE` (1:0) always 0 (Direct) |
| `mepc` | `[XLEN-1:1]` with C, `[XLEN-1:2]` without | bit 0 always 0; bit 1 also 0 without C (IALIGN=32) |
| `mcause`, `mtval` | `[XLEN-1:0]` | as written |

A write keeps only the stored bits; a CSRRS/CSRRC reads the legal value,
so it cannot set a hardwired bit. `misa` is read-only: a write is ignored.

| Event | `MIE` | `MPIE` | `MPP` |
|---|---|---|---|
| trap entry | 0 | old `MIE` | `11` (M) |
| `mret` | old `MPIE` | 1 | `11` (M) |

`MPP` stays `11` across `mret` because M is the only mode to return to.

**`FS`** gates the F/D state (the FP registers and `fflags`/`frm`/`fcsr`):

- It resets to Initial (`01`) with F enabled; without F it is read-only
  `00`. Kernels never write `mstatus`, so a reset value of Off would trap
  every FP kernel; Initial keeps FP usable and still shows the first write
  as the move to Dirty, so `SD` reads 0 until the warp writes FP state.
- Like `mtvec`, it resets only with the core, not at a kernel or CTA
  launch: a kernel that turns FP off must turn it back on before it ends,
  or the next kernel on those warps traps on its first FP instruction.
- With `FS`=Off, an instruction that names an FP register or FP CSR raises
  an illegal-instruction trap (§1.3): the F/D instructions (FP
  loads/stores, `OP-FP`, the fused multiply-adds), any CSR access to
  `fflags`/`frm`/`fcsr`, and the Vortex extensions on FP registers (the
  packed, wide and matrix FP loads and stores, the TCU MMA, TFU, and the
  RTU `GETWF`/`CB_RET`). The RTU trace/wait ops, which use the FP window
  by convention rather than by encoding, are not gated.
- An instruction that writes an FP register (Vortex extensions included,
  e.g. a TCU MMA or a packed FP load), or an F/D instruction or FP CSR
  write that writes `fflags`/`frm`, sets `FS` to Dirty. It is set at
  decode, conservatively (an FP op that raises no flag still dirties).
- The check and the Dirty update are both made at decode, so they must
  see `mstatus` in program order: any CSR access to `mstatus` stalls its
  warp at decode and releases it as the CSR request fires
  (`sched_csr_if.csr_unlock` in the RTL, a warp-control release in SimX).

### 1.2 Conformance tests

The upstream riscv-tests run unmodified through `tests/riscv/isa`
(`make run-{simx,rtlsim}-{32,64}mi` for the machine-mode suites; the
`run-*` aggregates of the matching XLEN include them). Of `rv32mi-p` and
`rv64mi-p`, `mcsr`, `scall`, `sbreak`, `illegal`, `shamt` (RV32 only) and
`ma_fetch` run. The rest are skipped, with the reason beside the skip list
in [`tests/riscv/isa/Makefile`](../../tests/riscv/isa/Makefile): they need a
misaligned load/store trap, `misa.U` cleared or an unimplemented-CSR trap
(`csr`, tests 14-15), or a CSR Vortex does not model (Sdtrig triggers, the
Zicntr user counters, writable `minstret`/`mcountinhibit`, PMP).

CI runs the machine-mode suites on the default build, with C off (`isa-mi`
in `ci/testcases/isa/riscv.yaml`): only there can a branch target be
misaligned, so only there does `ma_fetch` reach the trap.
`tests/regression/trap_illegal` covers the `FS` gating and Dirty tracking
(§1.1), which no riscv-test that runs reaches, and the reserved branch and
`SYSTEM` encodings of §1.3.

### 1.3 Trap sources

| Cause | Producer | `mepc` | `mtval` |
|---|---|---|---|
| 0 instruction address misaligned | branch unit | the jump/branch | 0 |
| 2 illegal instruction | decoder (trap uop) | the instruction | 0 |
| 3 breakpoint (`EBREAK`) | branch unit | the instruction | 0 |
| 11 `ECALL` from M | branch unit | the instruction | 0 |

**Misaligned fetch.** With C disabled (IALIGN=32), a JAL, a JALR or a
*taken* conditional branch whose target has bit 1 set traps instead of
jumping; JALR clears bit 0 first, as the ISA says, so only bit 1 counts,
and a not-taken branch never traps. With C enabled nothing is misaligned.
The ALU computes the target bit before the PC is truncated to `PC_BITS`,
so the DEBUG (`PC_BITS`=XLEN) and release builds behave the same. A JAL or
JALR that traps writes no lane of `rd`. A warp takes one PC, its last
active lane's target (lane-divergent JALR targets are not otherwise
supported); the trap follows the same lane, so a warp traps exactly when
the target it would jump to is misaligned, and the trap is per-warp like
every other.

`mtval` is 0 for every cause, which the privileged spec allows; writing the
target or the instruction bits would cost a second data source into every
warp's `mtval`. Only `dest`, the trapping PC, travels with the trap.

The RTL jumps to `mtvec` as architected even when no handler is installed
(`mtvec`=0, as in every plain Vortex kernel); SimX instead aborts on an
illegal-instruction or misaligned-fetch trap with `mtvec`=0, reporting the
cause, PC and instruction bits.

**Illegal instruction.** The decoder turns the instruction into a branch
uop (`INST_BR_ILLEGAL`) that reads and writes no register and stalls its
warp like `ECALL`; the branch unit raises cause 2. It covers:

- an unknown major opcode, including the opcodes of an extension the core
  is built without (F, A, the wide/matrix loads and stores, TFU);
- a reserved F/D encoding (an unsupported format, a reserved `funct5`, a
  reserved `funct3` of `FCLASS`/`FMV`);
- a shift-immediate with `shamt[5]` set on RV32, or on an RV64 word shift;
- a conditional branch with the reserved `funct3` 2 or 3;
- a `SYSTEM` instruction other than a CSR access, `ECALL`, `EBREAK`,
  `URET`, `SRET`, `MRET` and `WFI` (`WFI` runs as a no-op: there is no
  interrupt to wait for);
- a CSR write to a read-only CSR (address `0xC00`-`0xFFF`);
- with C, a reserved compressed encoding (including `0x0000`), which the
  decompressor expands to the all-zero word; the `rd`=x0 forms the RVC
  spec calls HINTs execute as no-ops;
- an FP instruction while `FS`=Off (§1.1).

It does not cover an unknown sub-opcode inside the Vortex custom opcodes
(still decode-time assertions where checked), or an access to an
unimplemented CSR, which would need the CSR address map at decode.

With more threads than ALU lanes, an instruction runs as several lane
packets; each packet writes only its lanes, and the last one alone takes
the redirect, trap or `mret`.

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
2. **Fault producers** — misaligned load/store and page-fault causes
   (4/6, 13/15), each unlocking the matching skipped `rv32mi-p` tests
   (§1.2). The misaligned-fetch and illegal-instruction producers exist
   (§1.3).
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
   not exist and `mstatus.MPP` is hardwired to M. Clearing it also lets the
   `csr` riscv-test run.

**Superseded directions** (recorded to avoid revival): the standalone
`VX_trap_if.sv` interface (folded into `VX_branch_ctl_if`); adding CSR
addresses to a hand-written `VX_csr.vh` (they live in the generated
`VX_types.vh`); and the SimX "kill all warps" trap hack (replaced by
per-warp `raise_trap`).
