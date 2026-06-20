# VX_scheduler CTA Encapsulation — fold CTA storage into VX_cta_dispatch

Status: proposal
Scope: `hw/rtl/core/VX_scheduler.sv`, `hw/rtl/core/VX_cta_dispatch.sv`
Type: structural refactor — **zero functional / timing / IPC change**

## 1. Motivation

`VX_scheduler.sv` (812 lines) has accreted CTA (cooperative-thread-array /
workgroup) launch-and-context logic interleaved with its actual job: warp state,
barriers, split/join, warp arbitration, and machine CSRs. CTA code is spread
across the whole file with no single owner:

| Lines | CTA concern | Holds state? |
|-------|-------------|--------------|
| 54 | `cta_id_per_warp_r` table | yes (flops) |
| 67–75 | `VX_cta_dispatch` output wires | no |
| 77–123 | `cta_ctx_ram` + `cta_warp_ram` (two `VX_dp_ram`) | yes (BRAM) |
| 126 | `cta_warp_done` (TMC→retire) | no |
| 128–145 | `VX_cta_dispatch` instance | submodule |
| 154–248 | per-thread coordinate **TID ripple pipeline** + warp-table write drivers | yes (pipe regs) |
| 250–276 | ctx-table write drivers + read ports + `sched_csr_if.cta_csrs`/`cta_tid` outputs | no |
| 333–340 | `cta_fire` → activate warp / set PC+tmask (in warp-state `always@*`) | scheduler state |
| 476–479 | `cta_fire` → latch `mscratch_r` / `cta_id_per_warp_r` | mixed |
| 635 | `schedule_cta_id` lookup | no |

That is ~190 lines of CTA logic threaded through a module that should not know
how CTA context is stored. The goal is encapsulation: one named owner for CTA
launch + context, leaving the scheduler to own warp scheduling.

## 2. Alignment with SimX

The SimX model already has the structure we want to mirror:

- **One class, `CtaDispatcher`** (`sim/simx/cta_dispatcher.{h,cpp}`). There is no
  separate "CTA storage" object. `step()` returns the *full*
  `cta_warp_record_t` — both the launch fields (`PC`, `tmask`, `do_init`) and the
  context/coordinate fields (`thread_idx[3]`, `block_idx[3]`, `block_dim[3]`,
  `grid_dim[3]`, `cta_id`, `cta_rank`, `param`, `lmem_addr`, `cluster_size`).
- **The scheduler owns the dispatcher** as a child (`scheduler.cpp:68` creates
  `cta_dispatcher_`) and copies the record into per-warp state via
  `activate_warp(wid, rec)` (`scheduler.cpp:90`).
- **Read-back is per-warp** (`warp.cta_csrs`), with per-lane thread index
  derived at CSR-read time in `csr_unit.cpp:99–101`.

So the RTL should match: a **single** `VX_cta_dispatch` module that owns launch
*and* context storage, instantiated as a child of `VX_scheduler`, which keeps
warp activation. We fold, rather than add a second wrapper layer, precisely
because SimX has one class, not two.

RTL-vs-SimX difference (necessary, not a divergence): SimX keeps the per-warp
record in scheduler warp-state structs; RTL cannot afford `NUM_WARPS ×
full-record` flops, so the same data lives in BRAM (`cta_ctx_ram`,
`cta_warp_ram`). Those BRAMs fold *into* the dispatcher. This is consistent with
what `VX_cta_dispatch` already is — it already owns two `VX_dp_ram`s
(`rem_warps_ram`, `lmem_size_ram`) and a per-warp reverse-lookup flop array.

Likewise, SimX derives per-lane thread coordinates by division at read time; RTL
precomputes all lanes at launch via the TID ripple pipeline to avoid a runtime
divider. That pipeline is an RTL-only optimization and folds into the dispatcher
alongside the table it fills.

## 3. Design

Fold all CTA storage and read-back into `VX_cta_dispatch`, which becomes the
single CTA block (launch FSM + LMEM allocator + context tables + TID pipeline +
CSR read ports). `VX_scheduler` retains only the two couplings that mutate
scheduler-owned warp state.

### 3.1 What moves into `VX_cta_dispatch`

- `cta_ctx_ram` + its write drivers and `cta_ctx_raddr`/`rdata` read port.
- `cta_warp_ram` + its write drivers.
- The TID ripple pipeline (`tid_next`, `tidp_*`, `g_tid_pipe`) that fills
  `cta_warp_ram`.
- `cta_id_per_warp_r` and the `schedule_cta_id` lookup.
- The `sched_csr_if.cta_csrs` / `sched_csr_if.cta_tid` read-back composition
  (returned as plain output structs — see §3.3).

### 3.2 What stays in `VX_scheduler`

Two couplings are intrinsic to scheduler state and stay, reduced to a clean
handshake:

1. **Warp activation on launch** (lines 333–340): writing
   `active_warps_n` / `warp_pcs_n` / `thread_masks_n` is scheduler-owned. It
   already consumes only `cta_fire` / `cta_wid` / `cta_PC` / `cta_tmask` /
   `cta_init` — exactly the dispatcher's existing outputs. **No change.**
2. **`mscratch_r` latch on launch** (line 477): `mscratch` is per-warp scheduler
   state written from three sources (CTA `param`, CSR write, wspawn copy), so it
   stays in the scheduler and latches `cta_csrs.param` from the handshake.

This matches SimX exactly: `activate_warp()` copies the record into scheduler
warp-state; the dispatcher produces it.

### 3.3 CSR read interface — pass plain signals, not the interface

Only one module may be the `.master` of `VX_sched_csr_if`. The scheduler stays
sole master. The dispatcher therefore takes the read addresses as plain inputs
and returns the read data as plain output structs; the scheduler wires them into
`sched_csr_if`:

```verilog
// new dispatcher ports (added to existing port list)
    // CTA-CSR read side (combinationally wired from sched_csr_if by scheduler)
    input  wire [NW_WIDTH-1:0]   csr_rd_wid,        // = sched_csr_if.csr_rd_wid
    input  wire [NCTA_WIDTH-1:0] csr_rd_cta_id,     // = sched_csr_if.csr_rd_cta_id
    output cta_csrs_t            cta_rd_csrs,        // -> sched_csr_if.cta_csrs
    output wire [`VX_CFG_NUM_THREADS-1:0][2:0][CTA_TID_WIDTH-1:0] cta_rd_tid, // -> sched_csr_if.cta_tid

    // scheduled-warp -> its CTA id
    input  wire [NW_WIDTH-1:0]   schedule_wid,
    output wire [NCTA_WIDTH-1:0] schedule_cta_id,
```

Keeping it a plain data contract (no interface ownership transfer) is the
cleanest boundary and avoids modport/master conflicts.

### 3.4 Resulting hierarchy

```
VX_scheduler                 ← warp state, barriers, split/join, arbitration, CSRs
  └─ VX_cta_dispatch         ← KMU launch FSM + LMEM alloc + ctx/warp tables
                               + TID pipeline + cta_id table + CSR read-back
```

## 4. Migration plan

Pure lift-and-shift; each step compiles.

1. Add the new ports to `VX_cta_dispatch` (§3.3) plus `csr_wr` for `mscratch`?
   — no: `mscratch` stays in scheduler, so only the read-side and
   `schedule_wid`/`schedule_cta_id` ports are added.
2. Move `cta_ctx_ram`, `cta_warp_ram`, the TID pipeline, their write/read glue,
   `cta_id_per_warp_r`, and the read-back composition from `VX_scheduler` into
   `VX_cta_dispatch`. Inside the dispatcher these connect to the already-present
   `cta_fire` / `cta_wid` / `cta_csrs` / `cta_base_tid` signals directly (no
   longer module outputs feeding back in).
3. In `VX_scheduler`, delete the moved declarations and wire the new dispatcher
   ports: feed `sched_csr_if.csr_rd_wid` / `csr_rd_cta_id` / `schedule_wid` in;
   wire `cta_rd_csrs` / `cta_rd_tid` / `schedule_cta_id` out to `sched_csr_if`
   and `out_buf`.
4. Keep `cta_csrs` as a dispatcher output **only if** the scheduler still needs
   `cta_csrs.param` for the `mscratch_r` latch (it does) — so `cta_csrs` stays an
   output; the rest of the read-back becomes internal.
5. Lint (Verilator `UNUSEDSIGNAL` / `UNDRIVEN`), then rtlsim.

## 5. Validation

- **No functional change.** Same RAMs, same pipeline depth, same launch
  handshake. The dispatcher gains a hierarchy boundary; nothing on any timing
  path changes (the `cta_warp_ram` write-path pipeline fix stays intact, now
  inside the dispatcher).
- rtlsim trio that already exercises CTA dispatch + per-lane `CTA_THREAD_ID`:
  `vecadd`, `sgemm`, `sgemm_tcu_wg` — all must PASS.
- Verilator lint clean.
- (Optional) re-run a `dut/vortex` synth at one config to confirm WNS is
  unchanged vs. the pre-refactor `nt8nw8_vortex_ctafix` baseline — expected
  identical since it is the same logic under a new instance name.

## 6. Out of scope

- No change to the launch FSM, LMEM allocator, or KMU protocol.
- No change to the TID pipeline algorithm or its `TID_STEP` depth.
- No change to `mscratch` / trap-CSR ownership (stays in scheduler).
- No SimX change — SimX is already the reference structure.

## 7. Expected outcome

`VX_scheduler` drops to ~620 lines and reads top-to-bottom as: warp state machine
→ barriers → split/join → warp arbitration → CSRs/perf. All CTA launch and
context becomes one `VX_cta_dispatch` instance with a documented data contract,
mirroring SimX's single `CtaDispatcher`.
