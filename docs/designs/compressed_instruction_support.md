# RVC (Compressed Instructions) — Design

**Scope:** Vortex support for the RISC-V "C" (compressed) extension — the
front-end decompressor that expands 16-bit instructions to 32-bit, in both
RTL ([`hw/rtl/core/VX_decompressor.sv`](../../hw/rtl/core/VX_decompressor.sv))
and SimX ([`sim/simx/decompressor.{cpp,h}`](../../sim/simx/decompressor.cpp)).

Gated by `VX_CFG_EXT_C_ENABLE` (default off,
[`VX_config.toml:33`](../../VX_config.toml#L33)); when enabled it sets
`MISA` bit 2 (C) ([`VX_config.toml:300`](../../VX_config.toml#L300)).

---

## 1. Design choice: word-aligned fetch + a decompressor stage

The front-end stays **word-addressed** — the icache is always requested at
4-byte alignment — and RVC's 2-byte alignment is handled by a dedicated
per-warp decompressor stage between the icache response and decode, not by
the icache. PC is packed at half-word granularity when RVC is enabled
(`PC_BITS = XLEN-1`, `to_fullPC = {pc, 1'b0}`,
[`VX_gpu_pkg.sv:114-135`](../../hw/rtl/VX_gpu_pkg.sv#L114)) so 2-byte-aligned
branch targets survive; with RVC disabled the front-end keeps the direct
icache path, `PC_BITS = XLEN-2`, and unconditional `+4`.

---

## 2. RTL flow

[`VX_fetch.sv:83-153`](../../hw/rtl/core/VX_fetch.sv#L83) instantiates
`VX_decompressor` under `VX_CFG_EXT_C_ENABLE`. The scheduler issues a warp
PC; `VX_fetch` aligns it to a 4-byte icache request (drops `PC[1:0]`). The
icache response feeds [`VX_decompressor.sv`](../../hw/rtl/core/VX_decompressor.sv)
(644 LOC), which holds a per-warp halfword-buffer FSM (`BUF_EMPTY /
BUF_RVC / BUF_32HI`) with a zero-latency fast path for aligned RVC and a
2-stage slow path. It uses `PC[1]` to pick the low/high halfword, expands
16→32, and emits a full instruction on `fetch_t.instr` (existing field).

A 32-bit instruction straddling the 4-byte word triggers a `follow_req`
for `PC+4`, which `VX_fetch` prioritizes over scheduler requests; a
`sched_buffered_match` lets the scheduler advance without re-fetching when
the next halfword is already buffered. `fetch_t.is_rvc`
([`VX_gpu_pkg.sv:850`](../../hw/rtl/VX_gpu_pkg.sv#L850)) flows to decode,
which stamps `op_args.br.is_rvc` ([`:721`](../../hw/rtl/VX_gpu_pkg.sv#L721))
on branch/JAL/JALR ops. The scheduler advances `warp_pcs` by `+2`/`+4` via
the single `decode_sched_if.is_rvc` wire
([`VX_scheduler.sv:325-336`](../../hw/rtl/core/VX_scheduler.sv#L325)), and
the ALU computes JAL/JALR link addresses as
`to_fullPC(pc) + (is_rvc ? 2 : 4)`
([`VX_alu_int.sv:284-300`](../../hw/rtl/core/VX_alu_int.sv#L284)).

This `is_rvc` threading is leaner than carrying a size field through every
pipeline struct — only `fetch_t` gains a dedicated field, and the branch
path reuses `op_args.br`.

---

## 3. SimX flow

[`sim/simx/decompressor.{cpp,h}`](../../sim/simx/decompressor.cpp) provides
both a stateless `rvc_decompress(word) → {instr32, size, illegal}` and a
`Decompressor` **SimObject** (per-warp halfword buffer + refetch queue).
`Core::fetch()` always issues a 4-byte-aligned icache request; the
`Decompressor`'s `on_icache_rsp` runs the same FSM and drains its refetch
queue ahead of fresh fetches ([`core.cpp:347-414`](../../sim/simx/core.cpp#L347)).
[`decode.cpp:463-467`](../../sim/simx/decode.cpp#L463) detects RVC from
`code[1:0]` and decompresses inline, stamping `IntrBrArgs.is_rvc`. PC
advance is `scheduler_->advance_pc(wid, is_rvc ? 2 : 4)`
([`scheduler.cpp:238`](../../sim/simx/scheduler.cpp#L238)); the ALU link
address uses the same `is_rvc` selector.

---

## 4. Testing

`tests/riscv/isa/Makefile` resolves `HAS_EXT_C` from `VX_config.h`, adds
`rv32uc-p` / `rv64uc-p` targets and `run-{simx,rtlsim}-{32,64}c`
([`:16,36-142`](../../tests/riscv/isa/Makefile#L16)); the regression and
opencl `common.mk` append `c` to `-march` when `-DVX_CFG_EXT_C_ENABLE` is
set. CI has a dedicated `rvc()` job
([`ci/regression.sh.in:608-627`](../../ci/regression.sh.in#L608)) building
simx+rtlsim with RVC and running `run-simx-32c` / `run-rtlsim-32c`. ISA
test binaries are built from upstream riscv-tests, not checked in.

---

## 5. Proposed but not yet implemented

1. **`rv64uc-p-rvc` is skipped, not passing**
   ([`tests/riscv/isa/Makefile:44-52`](../../tests/riscv/isa/Makefile#L44)):
   the upstream test expects a misaligned-`C.FLD` trap that the Vortex LSU
   does not raise. RV64C coverage depends on misaligned-access trap
   support — worth tracking as a prerequisite.
2. **A-extension in the SimX LSU aborts** (orthogonal to RVC) — blocks the
   combined `imac`+A path the RVC march flags imply.

**Superseded directions** (recorded so the stale proposal narrative is not
followed): the SimX files are `decompressor.{cpp,h}` (not `decompress.*`);
the config macro is `VX_CFG_EXT_C_ENABLE` (not `EXT_C_ENABLE`); the SimX
fetch logic is a dedicated `Decompressor` SimObject (not inline
`RvcSlot`/`rvc_pending_refetch_` in `Core::fetch()`); the RTL feedback is a
single `decode_sched_if.is_rvc` wire (not a 3-port
`decompress_finished`/`pc_incr_by_2`/`decompress_wid` bundle); `is_rvc` is
carried via `op_args.br.is_rvc` + `fetch_t.is_rvc` (not threaded through
`decode_t`/`ibuffer_t`/`scoreboard_t`/etc.); and CI gating shipped (the
proposal marked it deferred). Note the proposal's own §7 timing concern
(WNS −1.765 ns ≈ 196 MHz) is attributable to pre-existing scheduler/CSR
fanout, not the 1-bit RVC additions.

---

## 6. Source proposal

This design consolidates and supersedes `rvc_migration_proposal.md` (now
removed from `docs/proposals/`).
