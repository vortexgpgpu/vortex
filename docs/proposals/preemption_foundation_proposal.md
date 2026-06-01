# Vortex Preemption Foundation — RISC-V Trap Path and Native riscv-tests Support

**Status:** draft proposal
**Branch:** `feature_preempt`
**Related review:** [docs/designs/preemption_review.md](../designs/preemption_review.md)

---

## 1. Summary

Today Vortex cannot run upstream [riscv-tests](https://github.com/riscv-software-src/riscv-tests) without two workarounds: a simx-side "ECALL kills all warps" hack at [`sim/simx/scheduler.cpp:247-253`](../../sim/simx/scheduler.cpp#L247-L253), and a [`miscs/patches/riscv-test-env.patch`](../../miscs/patches/riscv-test-env.patch) that rewrites the test environment's `RVTEST_PASS` / `RVTEST_FAIL` macros to bypass `ecall` and write the exit code into a Vortex-specific MMIO instead. Tests that exercise the trap mechanism itself (e.g. illegal-instruction, misaligned-access, scall) cannot be made to work this way and either silently terminate or report wrong.

The underlying gap is real: **Vortex has no synchronous exception trap path**. The RISC-V `ECALL` / `EBREAK` / `MRET` instructions are decoded but the pipeline does nothing useful with them (they currently advance PC to PC+4 — see [`hw/rtl/core/VX_decode.sv:370-394`](../../hw/rtl/core/VX_decode.sv#L370-L394) and the branch unit in [`VX_alu_int.sv`](../../hw/rtl/core/VX_alu_int.sv)). The machine-mode trap CSRs (`mtvec`, `mepc`, `mcause`, …) are placeholder-named in the CSR file ([`VX_csr_data.sv:151-160`](../../hw/rtl/core/VX_csr_data.sv#L151-L160)) but writes are silently dropped and reads return zero.

This proposal adds the smallest amount of trap machinery that makes upstream riscv-tests pass on Vortex unmodified. Specifically:

1. **RTL**: five per-warp M-mode CSRs (`mstatus`, `mtvec`, `mepc`, `mcause`, `mtval`), a trap-entry path that redirects on `ECALL` / `EBREAK`, and an `MRET` datapath. ~150 lines of Verilog, no new pipeline stage, no async/interrupt support.
2. **simx**: the same five CSRs as per-warp state, the same trap entry/return logic in the functional emulator. Replaces the `active_warps_.reset()` hack.
3. **Host-side test harness**: an ELF loader and an **HTIF-style `tohost` watcher** added to `sim/common/`, polled by both the simx and rtlsim host loops. This is the upstream protocol — the test writes `TESTNUM` to a designated memory word, the host observes the write and exits.
4. **Removals**: the two `miscs/patches/riscv-*.patch` files and the simx `trigger_ecall = kill all warps` hack go away. The Vortex-native exit-code MMIO (`IO_EXIT_CODE`) remains as a fallback for non-test workloads.

This is explicitly the **preemption foundation**, not preemption itself. There are no interrupts, no async trap injection, no host-driven preempt signal, no register-file save/restore, no privilege levels, no delegation. The trap path supports exactly the synchronous causes that riscv-tests' `env/p/` exercises. The shape, however, is forward-compatible: when the time comes to add async preempt-from-CP, illegal-instruction handling, or page faults, those are additional `trap_entry_if.cause` producers feeding the same CSR file and the same PC redirect — no surgery on what this proposal builds.

## 2. Goals and non-goals

### Goals (v1)

- **Run upstream `riscv-tests/env/p/` ISA tests on Vortex unmodified** — no patch in `miscs/patches/`, no fork of `riscv_test.h`. Pass/fail reported through the standard `tohost` protocol.
- Cover the cause codes that `env/p/` actually emits: `ECALL_M_MODE` (11), `BREAKPOINT` (3). Page-fault, misaligned-access, illegal-instr, and access-fault enums are *defined* but currently triggered only from the same ECALL/EBREAK ports — full implementations are explicit follow-ons (§13).
- One implementation, two simulators: the RTL trap path and the simx trap path are mirror images. simx is the spec; RTL conforms to it bit-for-bit on the per-warp CSR state.
- HTIF-style host monitor lives in `sim/common/` and is reused identically by simx and rtlsim. No host-side divergence.
- ELF loader added to `sim/common/`. The existing `.vxbin` / `.bin` / `.hex` loaders stay. Test binaries are ELF; Vortex apps continue to use `.vxbin`.
- Remove both `miscs/patches/riscv-tests.patch` and `miscs/patches/riscv-test-env.patch`. Remove `Scheduler::trigger_ecall` / `trigger_ebreak`'s `active_warps_.reset()` body.

### Non-goals (v1)

- **Interrupts.** No `mie`, no `mip`, no `mideleg`, no async cause delivery. `mcause` MSB (interrupt bit) is never set by hardware in v1.
- **Privilege levels.** Vortex runs in a single effective mode for the purposes of this trap path. `mret` / `sret` / `uret` all collapse to the same "restore PC from `mepc`" datapath. `mstatus.MPP` is stored but not interpreted.
- **Trap delegation.** `medeleg` / `mideleg` writes are accepted (storage) but the trap path always uses `mtvec` (no `stvec` redirect).
- **Async preempt injection.** No public API to force a warp to trap from outside its own instruction stream. No CP doorbell, no DCR-driven preempt. Adding this later is a small extension (§13.1).
- **Register-file / IPDOM / barrier-state save on trap.** Software is responsible for saving everything it touches. Same convention as standard RISC-V M-mode.
- **Page faults, illegal-instruction, misaligned-access traps.** The cause codes are defined but the producers are not wired in v1. Tests that depend on these (`env/pm/`, parts of `env/v/`) are out of scope; v1 targets `env/p/` only.
- **Multi-hart riscv-tests** (`env/v/`, `mt/`, virtual-memory tests). v1 only targets `env/p/` (physical-mode, single-hart). Multi-hart tests need `wspawn`-aware test wrappers, which is a separate problem.
- **Vector / floating-point CSR interaction with traps.** No save/restore of `fcsr`, `vcsr`, `vstart`. Tests that need those are out of scope for v1.

## 3. Terminology

| Term | Meaning in this proposal |
|---|---|
| **Trap** | A synchronous, instruction-caused control-flow redirect to `mtvec`, saving the faulting PC into `mepc` and a cause code into `mcause`. The only producers in v1 are `ECALL` and `EBREAK`. |
| **Trap return** | The `MRET` instruction's effect: PC ← `mepc`. `URET` and `SRET` collapse to `MRET` for v1. |
| **HTIF** | Host-Target Interface. spike's convention for signaling kernel termination via a memory word named `tohost`. Used here as the test pass/fail protocol. |
| **`tohost` watcher** | C++ component in `sim/common/` that polls the `tohost` memory address every host tick and reports termination + exit code to the simulator's run loop. |
| **HTIF exit code encoding** | `tohost & 1 == 1` → terminate; exit code is `tohost >> 1`. `RVTEST_PASS` writes `1` (LSB=1, code=0); `RVTEST_FAIL` writes `(testnum << 1) | 1` (LSB=1, code=testnum). |
| **The hack** | Collective name for the current state: `Scheduler::trigger_ecall = active_warps_.reset()` in simx, plus the `riscv-test-env.patch` and `riscv-tests.patch` in `miscs/patches/`. Removed by this proposal. |
| **Trap CSR file** | The five per-warp CSRs added by this proposal: `mstatus`, `mtvec`, `mepc`, `mcause`, `mtval`. (`mscratch` already exists per-warp.) |

## 4. End-to-end flow on a passing test

Running `rv32ui-p-add.elf` after this proposal lands:

```
sim/{simx,rtlsim}/main.cpp ── parses argv → "foo.elf"
                              calls ram.loadElfImage("foo.elf"), which
                                · copies PT_LOAD segments to RAM
                                · returns ehdr.e_entry  (typically 0x80000000)
                                · returns &symbol("tohost")        ──┐
                                                                     │
                              host_monitor_.init(tohost_addr, ram); ←┘

                              processor.dcr_write(VX_DCR_KMU_STARTUP_ADDR*,
                                                   ehdr.e_entry);
                              processor.run()       ──→
                                                       │ (loops below)
                                                       ▼

processor.run() loop:    while (any cluster running) {
                            tick();
                            if (host_monitor_.tick(ram)) break;
                         }
                         exitcode = host_monitor_.exit_code();

GPU side  (warp 0 thread 0):
   ┌── 0x80000000  reset_vector:  csrw mtvec, t0          ; trap_vector
   │                              la t0, 1f
   │                              csrw mepc, t0
   │                              mret                    ; → 1: in test body
   │
   │  test body                    add, sub, …            ; the actual test
   │                              li gp, 1                ; TESTNUM = 1
   │                              li a7, 93; li a0, 0
   │                              ecall                   ; ───┐
   │                                                          │ trap!
   │  trap_vector:  csrr t5, mcause                       ◄───┘
   │                li t6, CAUSE_M_ECALL                       (PC=mtvec,
   │                beq t5, t6, write_tohost                    mcause=11,
   │  write_tohost: sw gp, tohost(t5)   ── store to RAM ──→     mepc=ecall PC)
   │                j write_tohost   (spin)
   └────────────────────────────────────────────────────────
                                          ▲
                       host_monitor.tick() polls RAM[tohost_addr];
                       sees gp=1 → exit_code = 0 → terminate.
```

No simulator-side instruction interception. No test patch. The trap path is a real feature; the host monitor is the standard spike protocol.

## 5. High-level architecture

```
                              ┌────── Host side (sim/common/) ──────┐
                              │                                     │
program.elf  ───→  ElfLoader  │  · parses PT_LOAD                   │
                              │  · resolves tohost / fromhost syms  │
                              │  · returns entry_addr               │
                              │                                     │
                              │  HostMonitor (HTIF)                 │
                              │  · holds tohost_addr_               │
                              │  · tick(ram) → bool terminated      │
                              │  · exit_code() → int                │
                              └─────────────┬───────────────────────┘
                                            │ polled each host tick
                                            ▼
   ┌────── sim/rtlsim/main.cpp ──────┐   ┌────── sim/simx/main.cpp ──────┐
   │  main loop:                     │   │  main loop:                   │
   │    while (rtl_busy) {           │   │    while (proc_running) {     │
   │      processor.tick();          │   │      processor.tick();        │
   │      if (mon.tick(ram)) break;  │   │      if (mon.tick(ram)) break;│
   │    }                            │   │    }                          │
   └─────────────────────────────────┘   └───────────────────────────────┘

                              GPU side (per warp)

   ┌──────────────────────── Trap CSR file ─────────────────────────────┐
   │  Per-warp storage (NUM_WARPS × XLEN × 5):                          │
   │     mstatus   mtvec   mepc   mcause   mtval                        │
   │  Existing mscratch stays per-warp via sched_csr_if.mscratch.       │
   └────────────────────────────────────────────────────────────────────┘
                ▲                       │
   writes via   │                       │ reads via
   csrrw, etc.  │                       │ csrr / trap path
                │                       ▼
   ┌──── ALU branch unit ──┐    ┌──── Scheduler ─────┐
   │  INST_BR_ECALL ─┐     │    │  on trap_entry:    │
   │  INST_BR_EBREAK─┼─→ trap_entry_if (wid, PC,    │
   │  INST_BR_MRET   │     │    │   cause):          │
   │  INST_BR_SRET ──┴─→ mret_if (wid)              │
   │  INST_BR_URET           │  │    mepc[wid]  ← PC│
   │                          │  │    mcause[wid]← c │
   │                          │  │    warp_pc[wid] ← │
   │                          │  │     mtvec[wid]&~3 │
   │                          │  │                   │
   │                          │  │  on mret:         │
   │                          │  │    warp_pc[wid] ← │
   │                          │  │     mepc[wid]     │
   └──────────────────────────┘  └───────────────────┘
```

Three subsystems, all small:

1. **Trap CSR file** — five new CSR slots in [`hw/rtl/core/VX_csr_data.sv`](../../hw/rtl/core/VX_csr_data.sv), each `[NUM_WARPS][XLEN]`. Mirror in simx's `warp_t`.
2. **Trap entry / mret** — two new sideband interfaces (`trap_entry_if`, `mret_if`) from the ALU branch unit into the Scheduler. Wired into the existing warp-PC update path.
3. **Host runtime** — `ElfLoader` and `HostMonitor` in `sim/common/`. Both simx and rtlsim main loops call `host_monitor_.tick(ram)` once per host tick.

## 6. RTL design

### 6.1 CSR file changes

[`hw/rtl/core/VX_csr_data.sv`](../../hw/rtl/core/VX_csr_data.sv) currently has the trap CSR addresses in its write-accept case as silent no-ops (lines 151-160) and zero-read paths. The changes:

```sv
// Per-warp trap CSR storage
reg [`XLEN-1:0] mstatus  [`NUM_WARPS];
reg [`XLEN-1:0] mtvec    [`NUM_WARPS];
reg [`XLEN-1:0] mepc     [`NUM_WARPS];
reg [`XLEN-1:0] mcause   [`NUM_WARPS];
reg [`XLEN-1:0] mtval    [`NUM_WARPS];

// Write path: software csrw + hardware trap entry
always @(posedge clk) begin
    if (reset) begin
        for (int w = 0; w < `NUM_WARPS; ++w) begin
            mstatus[w] <= '0;  mtvec[w]  <= '0;
            mepc[w]    <= '0;  mcause[w] <= '0;
            mtval[w]   <= '0;
        end
    end else begin
        // Software writes (existing csr_unit plumbing)
        if (write_enable) begin
            case (write_addr)
                `VX_CSR_MSTATUS: mstatus[write_wid] <= write_data;
                `VX_CSR_MTVEC:   mtvec  [write_wid] <= write_data;
                `VX_CSR_MEPC:    mepc   [write_wid] <= write_data;
                `VX_CSR_MCAUSE:  mcause [write_wid] <= write_data;
                `VX_CSR_MTVAL:   mtval  [write_wid] <= write_data;
                default: ;
            endcase
        end
        // Hardware trap entry has priority over software write on same cycle
        if (trap_entry_if.valid) begin
            mepc  [trap_entry_if.wid] <= `XLEN'(trap_entry_if.pc);
            mcause[trap_entry_if.wid] <= `XLEN'(trap_entry_if.cause);
            mtval [trap_entry_if.wid] <= '0;  // ECALL/EBREAK report 0
        end
    end
end

// Read mux (extends existing case)
case (read_addr)
    `VX_CSR_MSTATUS: read_data_rw_w = mstatus[read_wid];
    `VX_CSR_MTVEC:   read_data_rw_w = mtvec  [read_wid];
    `VX_CSR_MEPC:    read_data_rw_w = mepc   [read_wid];
    `VX_CSR_MCAUSE:  read_data_rw_w = mcause [read_wid];
    `VX_CSR_MTVAL:   read_data_rw_w = mtval  [read_wid];
    // … existing reads
endcase

// Export to scheduler for trap-entry PC redirect and mret
assign sched_csr_if.mtvec_per_warp = mtvec;
assign sched_csr_if.mepc_per_warp  = mepc;
```

`VX_CSR_MTVAL` (0x343) is currently undefined in [`hw/rtl/VX_csr.vh`](../../hw/rtl/VX_csr.vh); add it.

`mhartid` is *already* read-mapped to the canonical per-thread ID in the existing CSR file — no change. The test's `csrr a0, mhartid` works today.

Resource cost (back-of-envelope, NUM_WARPS=8, XLEN=32): 8 × 32 × 5 = 1280 FFs. Negligible.

### 6.2 Trap-entry interface

New SV interface [`hw/rtl/interfaces/VX_trap_if.sv`](../../hw/rtl/interfaces/VX_trap_if.sv):

```sv
interface VX_trap_entry_if;
    logic                 valid;
    logic [NW_BITS-1:0]   wid;
    logic [PC_BITS-1:0]   pc;        // faulting PC
    logic [3:0]           cause;     // mcause encoding (3=EBREAK, 11=ECALL_M)
    modport master (output valid, wid, pc, cause);
    modport slave  (input  valid, wid, pc, cause);
endinterface

interface VX_mret_if;
    logic                 valid;
    logic [NW_BITS-1:0]   wid;
    modport master (output valid, wid);
    modport slave  (input  valid, wid);
endinterface
```

### 6.3 Branch-unit producer

[`hw/rtl/core/VX_alu_int.sv`](../../hw/rtl/core/VX_alu_int.sv) currently treats `INST_BR_ECALL` / `EBREAK` / `URET` / `SRET` / `MRET` as branches with `imm20 = 4` ([`VX_decode.sv:386-391`](../../hw/rtl/core/VX_decode.sv#L386-L391)), so they fall through to PC+4. We change the branch-unit output for these five opcodes:

```sv
// In the EOP cycle when is_br_op_r is asserted:
wire is_trap_entry = is_br_op_r && (br_op_r == INST_BR_ECALL || br_op_r == INST_BR_EBREAK);
wire is_mret       = is_br_op_r && (br_op_r == INST_BR_MRET  || br_op_r == INST_BR_SRET ||
                                    br_op_r == INST_BR_URET);

assign trap_entry_if.valid = br_enable && is_trap_entry;
assign trap_entry_if.wid   = br_wid;
assign trap_entry_if.pc    = result_if.data.header.PC;
assign trap_entry_if.cause = (br_op_r == INST_BR_ECALL)  ? 4'd11 :  // ECALL_M_MODE
                              (br_op_r == INST_BR_EBREAK) ? 4'd3  :  // BREAKPOINT
                              4'd0;

assign mret_if.valid       = br_enable && is_mret;
assign mret_if.wid         = br_wid;

// Suppress the existing PC+4 branch publication for these
wire branch_publishes = br_enable && !is_trap_entry && !is_mret;
assign branch_ctl_if.valid = branch_publishes;
// trap_entry_if and mret_if drive the warp PC update in the Scheduler.
```

### 6.4 Scheduler consumer

[`hw/rtl/core/VX_scheduler.sv`](../../hw/rtl/core/VX_scheduler.sv) already updates per-warp PCs from `branch_ctl_if.dest`. Add two more inputs that take precedence on the cycle they fire:

```sv
// warp_pcs_n update (combinational), in priority order:
//   1. trap_entry_if    → warp_pcs_n[wid] = mtvec[wid] & ~3
//   2. mret_if          → warp_pcs_n[wid] = mepc[wid]
//   3. branch_ctl_if    → warp_pcs_n[wid] = branch_ctl_if.dest  (existing)
//   4. fetch advance    → warp_pcs_n[wid] = warp_pcs[wid] + 4   (existing)

always_comb begin
    warp_pcs_n = warp_pcs;
    if (trap_entry_if.valid) begin
        warp_pcs_n[trap_entry_if.wid] = sched_csr_if.mtvec_per_warp[trap_entry_if.wid] & ~PC_BITS'(3);
    end else if (mret_if.valid) begin
        warp_pcs_n[mret_if.wid] = sched_csr_if.mepc_per_warp[mret_if.wid];
    end else if (branch_ctl_if.valid) begin
        warp_pcs_n[branch_ctl_if.wid] = branch_ctl_if.dest;
    end
    // … existing fetch-advance path
end
```

The `& ~3` masks `mtvec`'s low 2 bits (the spec's MODE field — vectored mode not supported in v1). The same masking is applied in simx.

### 6.5 What does *not* change in RTL

- No new pipeline stage. Trap entry happens in the same cycle as the existing branch update.
- No fetch flush logic beyond what `branch_ctl_if` already triggers — the fetch unit treats `trap_entry_if.valid` / `mret_if.valid` identically to a taken branch (re-fetch from the new PC, drop in-flight prefetch).
- No IPDOM / tmask / regfile save. The trap_vector in `env/p/riscv_test.h` is straight-line, doesn't touch divergence, and saves only the registers it uses.
- No interaction with barriers, wspawn, or KMU. Traps fire entirely within one warp; the scheduler's existing single-warp PC update is the only state mutated.

## 7. simx design

### 7.1 Per-warp CSR state

[`sim/simx/scheduler.h`](../../sim/simx/scheduler.h) — extend `warp_t`:

```cpp
struct warp_t {
    // … existing fields …
    Word                              mscratch;  // already present
    cta_csrs_t                        cta_csrs;  // already present

    // v1 trap CSRs
    Word mstatus = 0;
    Word mtvec   = 0;
    Word mepc    = 0;
    Word mcause  = 0;
    Word mtval   = 0;
};
```

### 7.2 Trap entry / mret in the emulator

[`sim/simx/scheduler.cpp`](../../sim/simx/scheduler.cpp) — replace the existing `trigger_ecall` / `trigger_ebreak` bodies:

```cpp
// Trap cause codes (subset; full table in sim/simx/types.h)
namespace cause {
    static constexpr Word BREAKPOINT      = 3;
    static constexpr Word ECALL_M_MODE    = 11;
}

void Scheduler::raise_trap(uint32_t wid, Word cause_code) {
    auto& w = warps_.at(wid);
    w.mepc   = w.PC;
    w.mcause = cause_code;
    w.mtval  = 0;
    w.PC     = w.mtvec & ~Word(3);
    // Flush any in-flight ibuffer micro-ops (decode-side)
    instr_buffers_.at(wid).clear();
}

void Scheduler::mret(uint32_t wid) {
    auto& w = warps_.at(wid);
    w.PC = w.mepc;
    instr_buffers_.at(wid).clear();
}

// Existing public API stays — body changes:
void Scheduler::trigger_ecall(uint32_t wid)  { raise_trap(wid, cause::ECALL_M_MODE); }
void Scheduler::trigger_ebreak(uint32_t wid) { raise_trap(wid, cause::BREAKPOINT); }
```

(Note: the current `trigger_ecall()` / `trigger_ebreak()` signatures don't take a `wid` — they kill all warps. We restore the per-warp `wid` argument, matching the RTL `trap_entry_if.wid` field. The caller in [`sim/simx/alu_unit.cpp:362-363`](../../sim/simx/alu_unit.cpp#L362-L363) already has the warp id in scope.)

[`sim/simx/alu_unit.cpp`](../../sim/simx/alu_unit.cpp) — wire `MRET` / `SRET` / `URET` into `Scheduler::mret(wid)` instead of falling through to PC+4. The existing `case 0x000: sched.trigger_ecall();` becomes `sched.trigger_ecall(wid)`; add `case 0x102 / 0x302 / 0x002: sched.mret(wid);`.

### 7.3 CSR plumbing

[`sim/simx/csr_unit.cpp`](../../sim/simx/csr_unit.cpp) — route the five new CSR addresses to/from `warp_t.{mstatus,mtvec,mepc,mcause,mtval}`. Today these are presumably absent or stubbed; if they exist as TODOs, this is a one-block edit.

`csrr a0, mhartid` already returns the canonical hart id and needs no change.

### 7.4 What does *not* change in simx

- The `Operands::get_exit_code()` reading `gp[0][0]` ([`sim/simx/operands.cpp:160-164`](../../sim/simx/operands.cpp#L160-L164)) is kept as a fallback for programs that terminate via `tmc 0` + `IO_EXIT_CODE` MMIO. The HTIF watcher takes precedence when a `tohost` symbol is present.
- No changes to register file, IPDOM stack, tmask, or barrier handling. Trap is straight-line and self-contained.

## 8. Host runtime — ELF loader and HTIF watcher

### 8.1 `sim/common/elf_loader.{h,cpp}`

New, ~150 lines. Standalone (no libelf dependency — small enough to hand-code with `<elf.h>` constants).

```cpp
namespace vortex {

struct ElfImage {
    uint64_t entry;                       // ehdr.e_entry
    bool     has_tohost;
    uint64_t tohost_addr;                 // st_value of `tohost` symbol, if present
    uint64_t tohost_size;                 // st_size, typically 8
};

// Reads `path`, validates ELF magic + RISC-V machine, copies each PT_LOAD
// segment to RAM at p_vaddr, fills in ElfImage. Throws on malformed ELF.
ElfImage loadElfImage(const char* path, RAM& ram);

} // namespace vortex
```

ELF24 / ELF64 both supported (riscv-tests emit ELF32 for rv32* and ELF64 for rv64*).

### 8.2 `sim/common/host_monitor.{h,cpp}`

New, ~80 lines:

```cpp
namespace vortex {

class HostMonitor {
public:
    HostMonitor() = default;

    // Empty `path` or ELF without a `tohost` symbol → monitor stays disabled
    // and tick() always returns false. This is the path for non-test apps.
    void attach(const ElfImage& img) {
        enabled_     = img.has_tohost;
        tohost_addr_ = img.tohost_addr;
    }

    // Poll. Returns true on first cycle tohost is non-zero.
    // `value` is captured at termination and exposed via exit_code().
    bool tick(RAM& ram) {
        if (!enabled_ || terminated_) return terminated_;
        uint64_t v = 0;
        ram.read(&v, tohost_addr_, sizeof(uint64_t));
        if (v != 0) {
            captured_ = v;
            terminated_ = true;
            // Conform to spike: clear tohost so guest can see the ACK.
            uint64_t zero = 0;
            ram.write(&zero, tohost_addr_, sizeof(uint64_t));
        }
        return terminated_;
    }

    // Spike encoding: LSB=1 means halt, exit code = v >> 1.
    int exit_code() const {
        if (!terminated_) return -1;
        if ((captured_ & 1) == 0) return -1; // unexpected non-halt write
        return int(captured_ >> 1);
    }

    bool terminated() const { return terminated_; }
    bool enabled() const    { return enabled_; }

private:
    bool     enabled_ = false;
    bool     terminated_ = false;
    uint64_t tohost_addr_ = 0;
    uint64_t captured_ = 0;
};

} // namespace vortex
```

### 8.3 Wiring in `sim/simx/main.cpp` and `sim/rtlsim/main.cpp`

Both files get the same three-line change in the program-load block:

```cpp
std::string program_ext(fileExtension(program));
HostMonitor monitor;
if (program_ext == "elf") {
    ElfImage img = vortex::loadElfImage(program, ram);
    startup_addr = img.entry;                       // override default STARTUP_ADDR
    monitor.attach(img);
    processor.dcr_write(VX_DCR_KMU_STARTUP_ADDR0, startup_addr & 0xffffffff);
#if (XLEN == 64)
    processor.dcr_write(VX_DCR_KMU_STARTUP_ADDR1, startup_addr >> 32);
#endif
} else if (program_ext == "vxbin") {
    ram.loadVxImage(program);
} else if (program_ext == "bin") {
    ram.loadBinImage(program, startup_addr);
} else if (program_ext == "hex") {
    ram.loadHexImage(program);
}
```

And the run loop gets the monitor poll:

```cpp
// sim/rtlsim/processor.cpp::run()
device_->start = 1;
this->tick();
device_->start = 0;
while (!device_->busy) this->tick();
while (device_->busy) {
    this->tick();
    if (monitor_.tick(*ram_)) break;  // new
}
```

(simx's [`sim/simx/processor.cpp:207-223`](../../sim/simx/processor.cpp#L207-L223) gets the same `monitor_.tick(ram_)` check inside the `do { … } while` loop.)

After the loop, the exit-code precedence is:

```cpp
if (monitor.terminated()) {
    exitcode = monitor.exit_code();
} else {
    // existing path: read IO_EXIT_CODE / get_exitcode()
}
```

## 9. Memory layout considerations

`riscv-tests/env/p/link.ld` hard-codes:

```
. = 0x80000000;
.text.init : { … }
. = ALIGN(0x1000);
.tohost : { *(.tohost) }
```

So the test entry is at `0x80000000` and `tohost` is somewhere in the 4 KiB after `.text.init`. Vortex's default `STARTUP_ADDR` is configurable via DCR ([`sim/rtlsim/main.cpp:69`](../../sim/rtlsim/main.cpp#L69)), so this *just works* once the ELF loader sets `STARTUP_ADDR = e_entry`.

The Vortex MMU configuration is `mmu_.attach(*ram, 0, 0x7FFFFFFFFF)` (XLEN=64) or `mmu_.attach(*ram, 0, 0xFFFFFFFF)` (XLEN=32) — both cover `0x80000000`. No memory map changes needed.

For RV32 tests (link.ld → `0x80000000`), this address is positive and well-defined. For RV64 tests the link script is the same; sign-extension to 64 bits is benign.

## 10. Removals

This proposal **removes**:

- [`miscs/patches/riscv-test-env.patch`](../../miscs/patches/riscv-test-env.patch) — entire file.
- [`miscs/patches/riscv-tests.patch`](../../miscs/patches/riscv-tests.patch) — entire file.
- Bodies of `Scheduler::trigger_ecall()` / `trigger_ebreak()` at [`sim/simx/scheduler.cpp:247-253`](../../sim/simx/scheduler.cpp#L247-L253) — replaced as in §7.2. Functions stay, signatures change to take `wid`.
- [`tests/riscv/isa/*.bin`](../../tests/riscv/isa/) — 616 pre-built stripped binaries, ~7.2 MB. These are obsolete: they are raw-binary images with no symbol table, which is exactly *why* the existing `riscv-test-env.patch` had to invent the `VX_IO_MPM_EXITCODE` MMIO (no `tohost` symbol → no HTIF). With the trap path landed, we use upstream ELFs (with symbols) built on demand from the [Makefile install target](#112-integration--upstream-envp-tests) below. No replacement binaries are checked in; the install target produces them in `build/tests/riscv/upstream/isa/`.

What we **keep**:

- `Operands::get_exit_code()` ([`sim/simx/operands.cpp:160`](../../sim/simx/operands.cpp#L160)) and the `IO_EXIT_CODE` MMIO path. Vortex apps that don't use HTIF (everything in `tests/regression/`, OpenCL kernels, etc.) terminate the same way they do today. HTIF is purely additive.
- `VX_IO_MPM_EXITCODE` in [`hw/VX_types.h`](../../hw/VX_types.h) — still the canonical Vortex exit-code MMIO.
- [`tests/riscv/riscv-vector-tests/`](../../tests/riscv/riscv-vector-tests/) — unchanged. That corpus has its own bringup story (Spike + Go) and uses a different exit convention; the trap-path landing does not touch it.

## 11. Test plan

### 11.1 Unit-level (per-component)

- **CSR write/read round-trip** (simx and RTL): `csrw mtvec, t0; csrr t1, mtvec; assert t0 == t1` for all five new CSRs, per warp. Smoke test runnable as a one-off `.S` under `tests/regression/csr_smoke/`.
- **ECALL → mtvec redirect** (simx and RTL): set `mtvec` to a label, fire `ecall`, verify PC reaches the label and `mcause == 11`, `mepc == ecall_pc`.
- **MRET restores PC** (simx and RTL): set `mepc` to a label, `mret`, verify PC reaches label.
- **HostMonitor unit test**: synthetic RAM, write `1` to `tohost_addr`, verify `tick()` returns true and `exit_code() == 0`. Write `7` (`(3<<1)|1`), verify `exit_code() == 3`. Write `2` (LSB=0, malformed), verify `exit_code() == -1`.
- **ElfLoader unit test**: small handcrafted ELF, verify segment placement and symbol resolution.

### 11.2 Integration — upstream `env/p/` tests

The riscv-tests corpus is installed **from the per-build Makefile**, not from `ci/toolchain_install.sh`. Rationale: LLVM/POCL/Verilator/etc. are toolchains (you build code *with* them) and rightly live in `$HOME/tools/`; riscv-tests is a test *corpus* you run *against* Vortex, and belongs alongside the other test corpora in `tests/`. Precedent: [`tests/riscv/riscv-vector-tests/`](../../tests/riscv/riscv-vector-tests/) already has its own per-directory bringup script.

[`tests/riscv/isa/Makefile`](../../tests/riscv/isa/Makefile) gains a stamp-file-driven install rule (lazy on first `run-*` invocation, also runnable explicitly via `make install`):

```make
RISCV_TESTS_REPO   := https://github.com/riscv-software-src/riscv-tests.git
RISCV_TESTS_COMMIT := <pinned hash>                                  # bumped intentionally
UPSTREAM_DIR       := $(BUILD_DIR)/tests/riscv/upstream
ISA_DIR            := $(UPSTREAM_DIR)/isa
STAMP              := $(UPSTREAM_DIR)/.installed

# Build under build/, not in-tree. Uses the riscv64-gnu-toolchain that
# toolchain_install.sh has already pulled into $HOME/tools/.
$(STAMP):
	rm -rf $(UPSTREAM_DIR)
	git clone $(RISCV_TESTS_REPO) $(UPSTREAM_DIR)
	cd $(UPSTREAM_DIR) && git checkout $(RISCV_TESTS_COMMIT) && \
	    git submodule update --init --recursive && \
	    autoconf && ./configure
	$(MAKE) -C $(UPSTREAM_DIR) XLEN=32 isa
	$(MAKE) -C $(UPSTREAM_DIR) XLEN=64 isa
	touch $@

install: $(STAMP)

# Run targets gain a dependency on the stamp so first invocation auto-installs.
TESTS_32I := $(wildcard $(ISA_DIR)/rv32ui-p-*.elf)
TESTS_32M := $(wildcard $(ISA_DIR)/rv32um-p-*.elf)
# … (analogous to existing TESTS_* but ELF, sourced from $(ISA_DIR))

run-simx-32imf: $(STAMP)
	@for t in $(TESTS_32I) $(TESTS_32M) $(TESTS_32F); do \
	    $(SIM_DIR)/simx/simx $$t || exit 1; done
# … other run-* targets unchanged in shape, just retargeted at $(ISA_DIR)

clean:
	# Note: does NOT nuke $(UPSTREAM_DIR) — preserves the install cache.
	# Use `make distclean` to force a re-clone.
distclean:
	rm -rf $(UPSTREAM_DIR)
```

Key choices:

- **Out-of-tree build** under `$(BUILD_DIR)/tests/riscv/upstream/`, per Vortex convention. Both `build32/` and `build64/` have their own copies (cheap; build is 1–2 min).
- **Pinned upstream commit.** `RISCV_TESTS_COMMIT` is an explicit SHA in the Makefile; bumping it is a deliberate one-line change with full git provenance. No floating `master`.
- **Stamp file (`.installed`)** prevents re-clone on every `make`. `make clean` preserves the install cache; `make distclean` (or `rm -rf $(BUILD_DIR)/tests/riscv/upstream`) forces a re-fetch.
- **No new env-script entry.** Unlike toolchain components, no `$(RISCV_TESTS_DIR)` export is needed — the Makefile resolves `$(ISA_DIR)` internally.
- **Lazy by default.** Developers who never run riscv-tests pay zero install cost. CI runs `make -C tests/riscv install` once after toolchain setup as a discrete step.

Smoke command from `build/`:

```bash
# One-time per build dir (or first `make run-simx` auto-installs):
make -C tests/riscv install

# Run the v1 baseline:
make -C tests/riscv run-simx-32imf
make -C tests/riscv run-rtlsim-32imf

# Or invoke a single ELF directly:
./sim/simx/simx tests/riscv/upstream/isa/rv32ui-p-add.elf
```

Per riscv-tests convention, exit code 0 = pass, exit code N = "fail at test number N" (test source has numbered subtests via the `TEST_*` macros, recovered from HTIF as `tohost >> 1`).

The v1 target subset (what we must pass):

| Suite | Count | Coverage |
|---|---|---|
| `rv32ui-p-*` | ~40 | RV32I baseline (add/sub/and/or/xor/sll/srl/sra/slt/sltu, branches, jal/jalr, lui/auipc, loads, stores, simple, fence) |
| `rv32um-p-*` | 8 | M extension (mul/mulh/mulhu/mulhsu/div/divu/rem/remu) |
| `rv32ua-p-*` | 10 | A extension (AMO ops + lr/sc); only if `EXT_A_ENABLE` |
| `rv32uc-p-*` | 1 | RVC (compressed); only if `EXT_C_ENABLE` |
| `rv64u*-p-*` | analogous | RV64 variants, gated on `XLEN=64` build |

Floating-point (`rv*uf-p-*`, `rv*ud-p-*`) is a fast follower — needs `fcsr` interaction verification but no new trap machinery.

Out of scope for v1 (tracked, not blocking):

| Suite | Why deferred |
|---|---|
| `rv*ui-pm-*` | Misaligned-access traps — need illegal-load/store trap producers (§13.2). |
| `rv*ui-v-*` | Virtual memory + supervisor — needs `stvec`, `sepc`, `scause`, page tables (§13.3). |
| `rv*mi-*`   | Machine-mode trap tests — exercises additional cause codes we don't emit. |
| `rv*si-*`   | Supervisor-mode tests — needs `sret`/`stvec` distinct from `mret`/`mtvec`. |
| `mt/`       | Multi-hart tests — needs `wspawn`-aware test wrappers, multi-warp `tohost` arbitration. |

### 11.3 Regression

- Existing `make -C tests/regression run-simx` and `run-rtlsim` continue to pass unchanged (all those apps use `tmc 0` + `IO_EXIT_CODE`, not HTIF).
- The two patches' deletion is verified by `grep -r riscv-tests miscs/patches/` returning empty.
- The simx hack removal is verified by `grep -n active_warps_.reset sim/simx/scheduler.cpp` returning empty in the trap helpers.

## 12. Milestones

The work is roughly five tightly-scoped commits (per the [no-PRs / direct commits / each commit substantial-and-testable](../../AGENTS.md) convention):

1. **CSR file + simx mirror** — add the five per-warp CSRs in RTL and simx, with csrw/csrr round-trip test. No trap path yet; ECALL/EBREAK/MRET still do what they do today.
2. **simx trap path** — `Scheduler::raise_trap` / `Scheduler::mret`, wired from `alu_unit.cpp`. Hand-written `.S` test verifies ECALL → mtvec → mret cycle in simx. Removes the simx hack.
3. **RTL trap path** — `VX_trap_entry_if` / `VX_mret_if`, branch-unit producer, scheduler consumer. Same hand-written `.S` test now passes rtlsim.
4. **ELF loader + HostMonitor** — `sim/common/elf_loader.{h,cpp}`, `sim/common/host_monitor.{h,cpp}`, unit tests. Wire into both `sim/simx/main.cpp` and `sim/rtlsim/main.cpp`. Verify with a hand-written ELF whose `tohost` symbol gets written.
5. **riscv-tests integration** — add the stamp-file-driven install rule and pinned `RISCV_TESTS_COMMIT` to [`tests/riscv/isa/Makefile`](../../tests/riscv/isa/Makefile), retarget `run-simx-*` / `run-rtlsim-*` at `$(BUILD_DIR)/tests/riscv/upstream/isa/*.elf`, verify `rv${XLEN}ui-p-*` passes on simx and rtlsim. Delete the 616 `tests/riscv/isa/*.bin` files and both `miscs/patches/riscv-*.patch` files in the same commit.

Each commit is testable end-to-end on the platforms it touches. Milestone 5 is the "feature complete" mark.

## 13. Forward-compatibility (what builds on this without redesign)

### 13.1 Async preempt-from-CP (the original `feature_preempt` story)

To deliver host-driven preemption later, add one signal:

```sv
input wire [`NUM_WARPS-1:0] preempt_pending,   // from CP via DCR
```

Then in the scheduler, between fetch and decode, check `preempt_pending[wid] && safe_point[wid]` and synthesize a `trap_entry_if.valid` with `cause = 16` (custom "GpuPreempt" cause). The CSR file and PC redirect from §6 are unchanged. The handler at `mtvec` distinguishes preempt-vs-ecall by reading `mcause`.

### 13.2 Illegal-instruction / misaligned-access

Decode-stage or LSU-stage error detection feeds `trap_entry_if.valid` with `cause = 2` (illegal) or `cause = 4/6` (misaligned load/store). Same datapath. `mtval` storage already exists (§6.1).

### 13.3 Supervisor-mode (`stvec` / `sret`)

Adds three more CSRs (`stvec`, `sepc`, `scause`) and a privilege-mode bit in `mstatus.MPP`. Trap entry inspects `medeleg[cause]` to choose between `mtvec` and `stvec`. This is the natural growth path for `env/v/` test support.

### 13.4 Interrupts

Adds `mie` / `mip` storage, an async injection port `int_pending[NUM_WARPS]`, and the interrupt bit (XLEN-1) in `mcause`. Wire `mstatus.MIE` to gate. No effect on the synchronous path built here.

None of the above require rework of the CSR file, the trap entry interface, or the host monitor. They are pure additions.

## 14. Risks and open questions

- **Verilator / RTL trap-CSR write conflict.** Spec-wise, a software `csrw mepc` on the same cycle as a hardware trap entry should write the trap's `mepc`. §6.1 priority-orders hardware over software. Verify this matches the RTL pipeline's commit ordering — if `csrw` and `ecall` cannot retire on the same cycle (they're on the same warp, sequenced by the instruction stream), the conflict is moot, but the assertion should be a `RUNTIME_ASSERT`.
- **`mtvec` low-2-bit mode.** v1 hardcodes direct mode (low bits masked). If a test sets MODE=1 (vectored), it will silently get direct mode. Document; consider an `ASSERT(mtvec[1:0] == 0)` on write for v1.
- **`tohost` 64-bit width.** spike uses a 64-bit `tohost`. On XLEN=32, the test's `sw TESTNUM, tohost` writes only the low 32 bits and the high 32 bits stay zero. `HostMonitor::tick()` reads 8 bytes; the top-half-zero case is benign because the LSB-encoded "halt" lives in the low word.
- **Test-image entry address vs Vortex reset.** `e_entry = 0x80000000` from the linker, vs Vortex's default `STARTUP_ADDR`. The ELF loader overrides `STARTUP_ADDR` via DCR before `processor.run()` — must happen *before* any `kmu_->start()`. Wiring needs care in [`sim/simx/main.cpp:100-105`](../../sim/simx/main.cpp#L100-L105).
- **Multi-warp `tohost` race.** If by accident more than one warp writes `tohost` (it should never happen in `env/p/` since only hart 0 runs), the monitor sees the first write. Acceptable for v1; revisit when adding `mt/` support.
- **Trap entry while in a divergent region.** v1 trap path does *not* save the IPDOM stack. `env/p/` tests do not trap from inside divergent regions (no SPLIT/JOIN in the trap path), so this is safe. Document as a known limitation for any future user.

## 15. Files touched (summary)

```
hw/rtl/core/VX_csr_data.sv           ~80 lines added (5 CSRs + write/read)
hw/rtl/core/VX_alu_int.sv            ~30 lines added (trap_entry_if/mret_if drivers)
hw/rtl/core/VX_scheduler.sv          ~15 lines added (warp_pcs_n priority)
hw/rtl/interfaces/VX_trap_if.sv      ~30 lines new (two interfaces)
hw/rtl/core/VX_core.sv               ~10 lines (interface instantiation + wiring)
hw/rtl/VX_csr.vh                     ~2 lines (add `VX_CSR_MTVAL` 0x343)

sim/simx/scheduler.h                 ~10 lines added (mstatus/mtvec/mepc/mcause/mtval)
sim/simx/scheduler.cpp               ~30 lines (raise_trap / mret, replace hack)
sim/simx/alu_unit.cpp                ~10 lines (mret dispatch, wid into trigger_*)
sim/simx/csr_unit.cpp                ~25 lines (CSR routing)

sim/common/elf_loader.h              ~30 lines new
sim/common/elf_loader.cpp           ~150 lines new
sim/common/host_monitor.h            ~50 lines new
sim/common/host_monitor.cpp          ~40 lines new

sim/simx/main.cpp                    ~15 lines (ELF branch + monitor attach)
sim/simx/processor.cpp               ~5 lines (monitor.tick() in run())
sim/rtlsim/main.cpp                  ~15 lines (ELF branch + monitor attach)
sim/rtlsim/processor.cpp             ~5 lines (monitor.tick() in run())

miscs/patches/riscv-tests.patch      DELETED
miscs/patches/riscv-test-env.patch   DELETED
tests/riscv/isa/*.bin                DELETED (616 files, ~7.2 MB)

tests/riscv/isa/Makefile             ~40 lines (stamp-file install + retarget at $(ISA_DIR))
tests/regression/csr_smoke/          NEW (unit test for CSRs + trap path)
```

Total: ~550 lines of new RTL/C++, ~400 lines deleted (the two patch files), 616 stripped `.bin` blobs deleted (~7.2 MB), upstream riscv-tests built on demand under `$(BUILD_DIR)/tests/riscv/upstream/`. Conservatively a 1–2 week implementation including bring-up against `rv32ui-p-*`.
