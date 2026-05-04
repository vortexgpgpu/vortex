# RVC (RISC-V Compressed) Migration Proposal

Migration of the compressed-instruction extension (`EXT_C_ENABLE`) from
`~/dev/vortex_rvc` (branch `project`, last commit `c5e85f6 update rtl`) into
the current `vortex_v3/feature_rvc` branch (`tinebp-patch-2`, baseline
`1f5b11f simx cache fix`).

This is not a cherry-pick: both the RTL fetch pipeline and the simx
emulator have been substantially rewritten in `vortex_v3`. The RVC logic
must be **re-implemented against the new APIs**, with `vortex_rvc` used as
a behavioral reference (especially for the half-word buffer FSM and the
16→32 bit decompression table).

## Status (as of 2026-05-02, fully resolved)

- **Phase 1 (simx): COMPLETE.** `rv32uc-p-rvc.bin` passes, full
  `run-simx-32imafd` (including the rvc test) passes with
  `EXT_C_ENABLE=1` and the default build is unchanged. All sweep
  regression apps (17 tested, demo/vecadd/vecadd_v1/dogfood/basic/
  conv3/dotproduct/dotproduct2/io_addr/printf/relu/wsync/sgemm/sgemm2/
  stencil3d/sort/softmax) pass with `EXT_C_ENABLE`.
- **Phase 2 (RTL): COMPLETE.** rtlsim builds clean for both default
  and `EXT_C_ENABLE` configs. `rv32uc-p-rvc.bin` passes on rtlsim with
  EXT_C. Full `run-rtlsim-32imf` and `run-rtlsim-32c` pass. 13 sweep
  regression apps pass on rtlsim with EXT_C (demo/vecadd/vecadd_v1/
  dogfood/basic/conv3/dotproduct/dotproduct2/io_addr/printf/relu/wsync/
  sgemm).
- **Phase 3 (tests / build glue): DONE for unit tests and
  `tests/{regression,opencl}/common.mk`.** CI gating in
  `ci/regression.sh.in` still pending (out of scope for this fix
  iteration; previously deferred for the RTL bug, now unblocked).

---

## 1. Source-side inventory (vortex_rvc)

RTL:
- `hw/rtl/core/VX_decompressor.sv` — 856 LoC, new module that **owns**
  `icache_bus_if`, holds a per-warp half-word buffer (`buffer[NUM_WARPS]`
  of `RVC_data_t`), and runs an FSM with three states (`BUF_EMPTY`,
  `BUF_RVC`, `BUF_32HI`). Issues "follow-up" I$ requests when a 32-bit
  instruction crosses a 4-byte word boundary.
- `hw/rtl/core/VX_fetch.sv` — modified to instantiate `VX_decompressor` and
  forward `icache_bus_if` / `schedule_if` through it.
- `hw/rtl/core/VX_schedule.sv` — modified to consume new schedule_if
  feedback signals: when `decompress_finished` is asserted, advance
  `warp_pcs[decompress_wid]` by `+2` (RVC) or `+4` (full).
- `hw/rtl/interfaces/VX_schedule_if.sv` — adds three feedback ports
  (`decompress_finished`, `pc_incr_by_2`, `decompress_wid`).
- `hw/rtl/VX_gpu_pkg.sv` — adds `buf_state_e` enum, `RVC_data_t` struct,
  and adds `last_in_word` flag on `fetch_t` (also renames `instr` → `word`).
- `hw/rtl/VX_config.vh` (handcoded) — `EXT_C_ENABLE` / `EXT_C_ENABLED`
  macros and bit 2 of MISA_STD.

simx:
- `sim/simx/decompress.cpp` — 314 LoC, **pure** function
  `Emulator::decompress(uint32_t)` returning
  `DecompResult{instr32, size, illegal}`. The encoder helpers
  (`ENCI/ENCR/ENCS/ENCU/ENCUJ/ENCB`) and quadrant decode are
  self-contained and directly portable.
- `sim/simx/emulator.h` — `warp_t` gains
  `{instr16, instr_size, has_valid_fetch_word}`; new `fetch_at_addr()`
  helper; `DecompResult` is in `types.h`.
- `sim/simx/emulator.cpp` (~50 LoC delta) — fetch loop with
  low/high-half pc_low logic and cross-word buffering.
- `sim/simx/execute.cpp` — replaces hard-coded `+4` with
  `warp.PC += warp.instr_size`; invalidates `has_valid_fetch_word` on PC
  redirect.
- `sim/simx/core.cpp` — gates icache request on `trace->fetch_skip`
  (avoids re-fetching when the next half-instruction came from the
  buffered halfword).
- `sim/simx/instr_trace.h` — adds `fetch_skip` field.
- `sim/simx/Makefile` — adds `decompress.cpp` to `SRCS`.

Tests:
- `tests/regression/common.mk` 32-bit branch uses `-march=rv32imafc`
  (compressed always on).
- ISA test bins `rv32uc-p-rvc.bin`, `rv32uc-v-rvc.bin`,
  `rv64uc-p-rvc.bin`, `rv64uc-v-rvc.bin` are checked in but
  `tests/riscv/isa/Makefile` does **not** add a target to run them.

---

## 2. Target-side state (vortex_v3)

What's already in place:
- **`EXT_C_ENABLE = false` exists in `VX_config.toml`** (line 38), and
  `gen_config.py` already emits `EXT_C_ENABLED` macros into both
  `hw/VX_config.vh` and `sw/VX_config.h`. The flag already feeds bit 2 of
  `MISA_STD`. The gate is wired, only the implementation is missing.
- `tests/riscv/isa/Makefile` already has the
  `EXT_A_ENABLED`-resolves-from-cpp pattern that we will reuse for
  `EXT_C_ENABLED`.
- `rv32uc-*.bin` / `rv64uc-*.bin` test binaries are already present in
  `tests/riscv/isa/` (no need to import them).

What changed and breaks a straight port:

**RTL pipeline:**
- `VX_schedule_if` only has `{valid, data, ready, ibuf_pop}` — no
  decompress feedback path.
- `VX_scheduler.sv` advances `warp_pcs[wid] = PC + 4` unconditionally
  (line 219). RVC needs a `+2/+4` decision.
- `VX_fetch.sv` directly drives `icache_bus_if` from `schedule_if` (no
  intermediate FSM). It emits a `fetch_t` carrying a single 32-bit
  `instr` field — the type has no `word`/`last_in_word`.
- `VX_gpu_pkg.sv` `fetch_t` field is `instr`, not `word`; no
  `buf_state_e`, no `RVC_data_t`.

**simx pipeline:**
- `Emulator::step()` and the imperative fetch/decode loop are **gone**.
  Replaced by a SimObject pipeline:
  `Scheduler::schedule()` → `fetch_latch_` → icache request →
  icache response → `decode_latch_` → `Decoder::decode()` → ibuffer.
  See `sim/simx/core.cpp:244-388` and `sim/simx/scheduler.cpp:123-193`.
- `Decoder::decode(uint32_t code, uuid)` is **stateless** — it takes a
  32-bit word and returns an `Instr`. There is no Emulator-owned warp
  state in the decoder anymore.
- `Scheduler::schedule()` does `warp.PC += 4` unconditionally
  (`scheduler.cpp:187`).
- `warp_t` (now in `scheduler.h:68`) holds only
  `{ipdom_stack, tmask, PC, fcsr, uuid, mscratch, cta_csrs}`. Register
  files moved to `OpcUnit`. Fields like `instr16`, `instr_size`,
  `has_valid_fetch_word` do not exist.
- `instr_trace_t` carries `code` (the 32-bit word read from icache);
  there is no `fetch_skip` field.
- Micro-op expansion lives in a dedicated `Sequencer` SimObject for
  LSU/TCU multi-uop instructions — RVC is **not** a uop expander, it is
  a fetch-stage transformation; this is a useful precedent for separating
  the FSM from the decoder.

---

## 3. Migration strategy

**Guiding principle.** Keep all RVC additions behind `EXT_C_ENABLE` so the
default build remains bit-identical. Land in three vertical slices that
each end at a green test target.

### Phase 1 — simx (start here)

simx is the cheaper target and gives us a behavioral reference for the
RTL phase. The decompression table itself ports verbatim; the rest is
glue against the new SimObject pipeline.

1. **Port the pure decompressor.** Drop `sim/simx/decompress.cpp` in
   place, but turn it into a stateless free function (or a method on
   `Decoder`) — not a method on the now-gone `Emulator`:
   ```cpp
   // sim/simx/decompress.h
   struct DecompResult { uint32_t instr32; uint8_t size; bool illegal; };
   DecompResult rvc_decompress(uint32_t word);
   ```
   Source the body verbatim from `vortex_rvc/sim/simx/decompress.cpp`
   (the `ENCI/ENCR/ENCS/ENCU/ENCUJ/ENCB` helpers + quadrant tables).
   Add to `sim/simx/Makefile` `SRCS`.

2. **Wire into the new fetch path.** The half-word FSM logic moves into
   `Core::fetch()` (in `sim/simx/core.cpp`), not `Decoder::decode()`,
   because:
   - the FSM mutates per-warp state (the buffered half-word) — adding it
     to `Decoder` would break statelessness;
   - the FSM may need to **fetch a second cache line** (the 32-bit
     instruction crosses a word), and only `Core::fetch()` has access
     to the icache request port.

   Concrete changes:
   - Add a per-warp `RvcBuf { state, hw, pc, uuid, tmask }` array to
     `Core` (or a small subobject), gated by
     `#if EXT_C_ENABLED`.
   - When an icache response arrives, run the FSM on `(low16, high16)`
     vs the buffer state. If the resulting instr32 is complete, write it
     into `trace->code` and push to `decode_latch_` exactly as today.
     If we still need the upper half, hold the trace, latch the
     half-word, and re-issue an icache request to PC+4.
   - Adjust `Scheduler::schedule()` so PC advances by `2` for an RVC and
     `4` for full — feed this back the same way `vortex_rvc` does (a
     `pc_incr_by_2` flag on the trace), or push the decision into the
     scheduler by deferring PC advance until decompression resolves
     (cleaner; mirrors the RTL design discussed below).

3. **PC advance.** Replace the unconditional `warp.PC += 4` at
   `scheduler.cpp:187` with a deferred increment driven by the fetch FSM
   when `EXT_C_ENABLED`. Default path (C disabled) keeps `+4`.

4. **Validation.**
   - `make -C tests/riscv/isa run-simx-32imac` (new target — see
     Phase 3) including `rv32uc-p-rvc.bin`.
   - Existing 32imafd / 64imafd suites must remain green with
     `EXT_C_ENABLE = false`.

### Phase 2 — RTL

Once simx is validated, port the hardware. The vortex_rvc design is a
good blueprint but **its `VX_decompressor.sv` cannot be dropped in**
because it expects different `fetch_t` / `schedule_if` shapes.

1. **Extend `fetch_t` and `schedule_if` (gated).**
   - `VX_gpu_pkg.sv`: under `` `ifdef EXT_C_ENABLE`` add
     `buf_state_e` enum and `RVC_data_t`. Either rename `fetch_t.instr`
     → `word` (touches a lot of consumers — costly) or keep `instr` and
     have the decompressor hand the decompressed 32-bit instruction
     directly into the existing field. **Recommendation:** keep `instr`
     and emit decompressed 32-bit on it, so the decode/issue path is
     untouched.
   - `VX_schedule_if.sv`: add three feedback signals
     (`decompress_finished`, `pc_incr_by_2`, `decompress_wid`) under
     `` `ifdef EXT_C_ENABLE``.

2. **Port `VX_decompressor.sv`.** The `decompress16` function and the
   FSM logic port verbatim. Adjust the I/O signal names to match the
   `fetch_t.instr` choice from step 1. Place under `hw/rtl/core/`.

3. **Modify `VX_fetch.sv` (gated).** Instantiate `VX_decompressor`
   when `` `EXT_C_ENABLED``; otherwise keep the current direct icache
   path (zero hardware cost for the disabled path).

4. **Modify `VX_scheduler.sv` (gated).** Replace the `+4` at line 219
   with a conditional `+2 / +4` driven by the new `pc_incr_by_2` /
   `decompress_finished` ports under `` `ifdef EXT_C_ENABLE``.

5. **Validation.** `make -C tests/riscv/isa run-rtlsim-32imac`
   (Phase 3) including `rv32uc-p-rvc.bin`. Verilator lint clean per
   `docs/coding_guidelines_verilog.md`.

### Phase 3 — tests / build glue

1. **`tests/regression/common.mk`.** Add a new path:
   ```make
   ifeq ($(EXT_C_ENABLE),1)
     VX_CFLAGS += -march=rv32imafc -mabi=ilp32f
   else
     VX_CFLAGS += -march=rv32imaf  -mabi=ilp32f
   endif
   ```
   Same shape for the 64-bit branch (`rv64imafdc`).

2. **`tests/riscv/isa/Makefile`.** Add `TESTS_32C` / `TESTS_64C`
   gated on `HAS_EXT_C` (resolved from `VX_config.h` exactly like
   `HAS_EXT_A` at line 18). Add `run-simx-32imac`,
   `run-rtlsim-32imac`, etc. Reuse the already-present
   `rv32uc-*.bin` / `rv64uc-*.bin`.

3. **CI.** Add one configuration to `ci/regression.sh.in` that flips
   `EXT_C_ENABLE = 1` and runs the imac / rvc tests on both simx and
   rtlsim.

---

## What was actually implemented

### simx (Phase 1)

- New `sim/simx/decompress.{h,cpp}` — pure stateless `rvc_decompress(uint32_t)`
  returning `DecompResult { instr32, size, illegal }`. Body ported verbatim
  from `vortex_rvc`.
- `sim/simx/core.cpp` — added per-warp `RvcSlot { needs_second, low_half,
  inst_pc }` (gated by `#ifdef EXT_C_ENABLE`) plus a side queue
  `rvc_pending_refetch_` for cross-word second fetches. The fetch path now
  always issues a 4-byte aligned icache request, runs the RVC FSM on
  response, decompresses if RVC, fixes up `warp.PC` (`-= 2`) for RVC, and
  re-issues fetch for cross-word 32-bit instructions.
- `sim/simx/instr_trace.h` — added `instr_size` field (default 4, set to 2
  when fetch decompresses an RVC).
- `sim/simx/alu_unit.cpp` — JAL/JALR `link_pc = trace->PC + trace->instr_size`
  (was `+4`) so c.jalr returns to `PC+2`.
- `sim/simx/Makefile` — added `decompress.cpp` to `SRCS`.
- `tests/regression/common.mk`, `tests/opencl/common.mk` — append `c` to
  `-march` when `EXT_C_ENABLE=1` (32 and 64 bit branches).
- `tests/riscv/isa/Makefile` — added `HAS_EXT_C` resolution (mirrors the
  existing `HAS_EXT_A` pattern, with `$(CONFIGS)` passed to gcc -E so
  command-line overrides take effect), `TESTS_{32,64}C`, and
  `run-{simx,rtlsim}-{32,64}c` targets. The `c` tests are also rolled
  into the `imafd` aggregate when `EXT_C_ENABLE=1`.

### RTL (Phase 2)

- `hw/rtl/VX_gpu_pkg.sv` — added `buf_state_e` enum (`BUF_EMPTY`, `BUF_RVC`,
  `BUF_32HI`) and `rvc_buf_t` struct under `` `ifdef EXT_C_ENABLE``.
- `hw/rtl/interfaces/VX_schedule_if.sv` — added `decompress_finished`,
  `pc_incr_by_2`, `decompress_wid` feedback signals under
  `` `ifdef EXT_C_ENABLE``.
- `hw/rtl/core/VX_decompressor.sv` — **new module** (≈600 LoC, ported
  from vortex_rvc with v3 adaptations). Owns `icache_bus_if`,
  per-warp halfword buffer FSM, follow-up icache request logic, and
  emits 32-bit decompressed instructions on `fetch_out_if.data.instr`.
  Adapts to v3's `fetch_t.instr` field (vortex_rvc used `.word` +
  `last_in_word`).
- `hw/rtl/core/VX_fetch.sv` — under `` `ifdef EXT_C_ENABLE`` instantiates
  `VX_decompressor`; otherwise keeps the original direct-icache path.
- `hw/rtl/core/VX_scheduler.sv` — under `` `ifdef EXT_C_ENABLE`` advances
  `warp_pcs` on `decompress_finished` (with `+2` or `+4` per
  `pc_incr_by_2`); the original unconditional `+4` is the `else` branch.
- `sim/rtlsim/verilator.vlt.in` — added `lint_off UNOPTFLAT` on
  `VX_decompressor.sv`, `VX_fetch_if.sv`, and `VX_decode_if.sv` to
  suppress a spurious combinational-loop warning across the elastic-buffer
  pass-through in decode (data is in fact a pure function of `buffer` and
  `in_data`).

## 4. Risks and open questions

- **`fetch_t.instr` vs `fetch_t.word` rename.** vortex_rvc renamed the
  field; v3 still uses `instr`. Keeping `instr` is the lower-blast-radius
  choice and is what this proposal recommends — **confirm with user**
  before phase 2.
- **Cross-word boundary fetch in simx.** The FSM has to issue a second
  icache request when a 32-bit instruction straddles a 4-byte cache word.
  In the new SimObject pipeline this means we may need to push a trace
  back into `fetch_latch_` (or stash it next to the RVC buffer). This is
  the trickiest piece of the simx port; expect to iterate on the
  back-pressure model.
- **PC redirect on branch with buffered half-word.** When a branch
  redirects PC mid-word, the buffered halfword must be invalidated. The
  vortex_rvc RTL handles this via the "Flush stale buffers when
  scheduler PC changes for a warp" block (`VX_decompressor.sv:550-560`)
  — port carefully.
- **`fetch_skip` in the new pipeline.** vortex_rvc threads `fetch_skip`
  through `instr_trace_t` to suppress redundant icache requests. In the
  new `Core::fetch()` model the natural equivalent is: don't enqueue a
  new icache request when the FSM already has a complete instr from the
  buffer. Confirm we don't need a separate `fetch_skip` flag.
- **Toolchain.** The build host's `riscv32-gnu-toolchain`,
  `llvm-vortex`, `libc32`, `libcrt32`, `verilator`, `sv2v`, and `yosys`
  live under `/opt/`, not `$HOME/tools/` where the generated
  `toolchain_env.sh` looks. Workaround: pass `TOOLDIR=/opt` (covers
  verilator/sv2v/yosys/sta) and `RISCV_TOOLCHAIN_PATH=/opt/riscv$XLEN-gnu-toolchain
  LIBC_VORTEX=/opt/libc$XLEN LIBCRT_VORTEX=/opt/libcrt$XLEN
  LLVM_VORTEX=/opt/llvm-vortex` for the regression flow.

## 6. Bugs root-caused and fixed

The RVC migration uncovered three RTL bugs and one simx bug. All four
are fixed.

### Bug A (simx) — JAL/JALR link address ignores RVC size

**Symptom.** `rv32uc-p-rvc.bin` exited with code 36 (sub-test 36 = the
`c.jalr` test) — the test set up `t0 = 0x21d2` (a 2-byte aligned target),
executed `c.jalr t0`, and verified that the link register was the PC of
the *next* instruction (PC + 2, since c.jalr is 2 bytes). The code was
reading PC + 4 for the link.

**Fix.** Added `instr_size` (uint8_t, default 4) to `instr_trace_t`. The
RVC fetch path sets it to 2 when emitting a decompressed instruction.
`VX_alu_int.sv` JAL/JALR cases compute `link_pc = trace->PC +
trace->instr_size`. (`sim/simx/instr_trace.h`, `sim/simx/core.cpp`,
`sim/simx/alu_unit.cpp`.)

### Bug B (RTL) — `PC_BITS = XLEN-2` truncates RVC branch targets

**Symptom.** `rv32uc-p-rvc.bin` aborted on rtlsim with `invalid CSR
write address: c00`. Tracing showed a JAL at PC `0x8000019c` was being
redirected to `0x80001ffc` instead of the decoded target `0x80001ffe`,
landing the warp at the *zero halfword* before the actual instruction.
The all-zero halfword decompressed to `addi x8, x2, 0`, the warp
sequenced into garbage, and many cycles later one of the spurious decoded
words happened to look like `csrrw … cycle`.

**Root cause.** In NDEBUG (release) builds, `VX_gpu_pkg.sv` packs PC at
4-byte granularity — `PC_BITS = XLEN-2`, `to_fullPC(pc) = {pc, 2'b0}`,
`from_fullPC(pc) = pc >> 2`. For a 32-bit JAL whose target is 2-byte
aligned (e.g., a target inside an RVC island), this drops bit 1 of the
byte address, rounding the destination down to the previous 4-byte word.

**Fix.** Added an `EXT_C_ENABLE` arm to the `PC_BITS` selector that
packs at half-word granularity (`PC_BITS = XLEN-1`,
`to_fullPC(pc) = {pc, 1'b0}`, `from_fullPC(pc) = pc >> 1`).
(`hw/rtl/VX_gpu_pkg.sv`.)

### Bug C (RTL) — duplicate buffered-RVC emit

**Symptom.** After fixing Bug B, the test still failed (exit 5 — the
c.addi16sp/c.nop test). RTL trace showed multiple emits at the same RVC
PC, each firing `decompress_finished` and over-advancing the warp PC.

**Root cause.** The decompressor's BUF_RVC emit fires unconditionally
whenever `buffer[wid].state == BUF_RVC` and `out_ready=1`. Meanwhile, the
scheduler — observing the warp unstall after the *previous* emit —
enqueues a new schedule for the same PC (the buffered halfword's PC).
The schedule data sits in the elastic buffer between scheduler and
decompressor for one cycle. By the time it reaches the decompressor, the
buffered RVC has already been emitted and `buffer` cleared. The
decompressor then services the schedule by issuing a redundant icache
request for the now-stale PC; the response is processed in BUF_EMPTY
case 3 as a fresh fetch and the same RVC is emitted a second time.

**Fix.** Two-part:
1. Gate the BUF_RVC emit on `have_buffered_match` (the scheduler is
   currently asking for the buffered PC for this warp). The buffered
   emit and the scheduler ack happen in the same cycle, so no stale
   schedule slips through.
2. When `have_buffered_match` is set, ack the scheduler
   (`schedule_if.ready = 1`) but suppress the icache request
   (`sched_req_valid = 0`). The data is already in `buffer[wid]`.

(`hw/rtl/core/VX_decompressor.sv`.)

### Bug D (RTL) — JAL/JALR link address ignores RVC size (same as Bug A)

**Symptom.** After Bugs B and C were fixed, exit became 36 — the same
c.jalr test that Bug A failed in simx. RTL had the same `+4` link-pc
hardcode at `VX_alu_int.sv:327`.

**Fix.** Threaded a 1-bit `is_rvc` flag through the pipeline:
- `fetch_t`, `decode_t`, `ibuffer_t`, `scoreboard_t`, `operands_t`,
  `dispatch_t`, and `alu_header_t` (via the `DECL_EXECUTE_T` macro)
  each gained an `is_rvc` field under `` `ifdef EXT_C_ENABLE``.
- The decompressor sets it (1 for RVC emits, 0 for cross-word 32-bit
  emits and full-aligned 32-bit emits).
- Decode/ibuffer/scoreboard/opc_unit/dispatcher/lane_dispatch propagate
  it through their packed concats.
- `VX_alu_int.sv` computes
  `PC_next = PC + (is_rvc ? 2 : 4)` for JAL/JALR static-branch link
  addresses.

(`hw/rtl/VX_define.vh`, `hw/rtl/VX_gpu_pkg.sv`,
`hw/rtl/core/VX_decompressor.sv`, `hw/rtl/core/VX_decode.sv`,
`hw/rtl/core/VX_ibuffer.sv`, `hw/rtl/core/VX_scoreboard.sv`,
`hw/rtl/core/VX_opc_unit.sv`, `hw/rtl/core/VX_dispatcher.sv`,
`hw/rtl/core/VX_alu_int.sv`.)

### Bug E (pre-existing v3, NOT migration-related)

`sim/simx/lsu_unit.cpp:239` calls `std::abort()` when an AMO instruction
reaches the LSU. This is a baseline limitation — the v3 simx LSU does
not implement the A extension. `EXT_A_ENABLE=1` simx builds will abort
on any `rv32ua-*` test regardless of RVC. Out of scope for this
migration; flagged for a separate change.

---

## 5. Validation matrix

| Stage      | Command                                                                                                | Result          |
|------------|--------------------------------------------------------------------------------------------------------|-----------------|
| Phase 1    | `make -s -C sim/simx` (default & `CONFIGS=-DEXT_C_ENABLE`)                                              | **PASS**        |
| Phase 1    | default-build simx ISA tests (rv32ui/m/f/d)                                                            | **PASS**        |
| Phase 1    | `CONFIGS=-DEXT_C_ENABLE make -C tests/riscv/isa run-simx-32imafd` (incl. `rv32uc-p-rvc`)               | **PASS**        |
| Phase 1    | 17-app regression sweep on simx (`EXT_C_ENABLE`)                                                       | **PASS**        |
| Phase 2    | `TOOLDIR=/opt make -s -C hw && make -s -C sim/rtlsim` (default & EXT_C)                                 | **PASS**        |
| Phase 2    | `make -C tests/riscv/isa run-rtlsim-32imf` (default rtlsim)                                            | **PASS**        |
| Phase 2    | `CONFIGS=-DEXT_C_ENABLE make -C tests/riscv/isa run-rtlsim-32imf` (EXT_C rtlsim, non-compressed tests)  | **PASS**        |
| Phase 2    | `CONFIGS=-DEXT_C_ENABLE make -C tests/riscv/isa run-rtlsim-32c` (RVC test on rtlsim)                   | **PASS**        |
| Phase 2    | 13-app regression sweep on rtlsim (`EXT_C_ENABLE`)                                                     | **PASS**        |
| Phase 2    | Vivado synthesis (`build_test32/hw/syn/xilinx/dut`, target `xcu55c-fsvh2892-2L-e`, EXT_C)               | **PASS** (functional) — see §7 |
| Phase 3    | CI gating in `ci/regression.sh.in`                                                                     | DEFERRED (out of scope) |

## 7. Vivado synthesis results (xcu55c, EXT_C build)

Run from inside the build tree:
```bash
source ~/dev/xilinx_setup.sh
cd /home/blaisetine/dev/vortex_v3/feature_rvc/build_test32/hw/syn/xilinx/dut
PREFIX=build_test1 make vortex
```
Vivado 2024.1, target part `xcu55c-fsvh2892-2L-e`, target clock 300 MHz
(3.333 ns).

**Functionality: PASS.**
- `synth_design`: 0 errors, 0 critical warnings (~2 min)
- `opt_design`: 0 errors, 0 critical warnings (~24 s)
- `place_design`: 0 errors, 0 critical warnings (~8.5 min)
- `route_design`: 0 errors, 0 critical warnings (~4.4 min)
- `phys_opt_design` (post-route): 0 errors, 0 critical warnings
- DRC (post-route): 0 errors
- Methodology check: 0 errors, 0 critical warnings
- Routing: 72,823 / 72,823 nets fully routed, 0 routing errors
- Placement: 89,300 / 89,300 cells placed, 0 errors

**Resource utilization (post-impl, % of full xcu55c):**
| Resource     | Used   | %     |
|--------------|--------|-------|
| CLB LUTs     | 37,878 | 2.91% |
| CLB Registers| 46,296 | 1.78% |
| Block RAM Tile (RAMB36) | 97 | 4.81% |
| RAMB18       | 24     | 0.60% |
| URAM         | 0      | 0.00% |
| DSP Blocks   | 24     | 0.27% |

**Timing (Slow corner, post-route + phys-opt):**
| Metric                    | Value          |
|---------------------------|----------------|
| Setup WNS                 | **−1.765 ns** (target 3.333 ns / 300 MHz) |
| Setup TNS                 | −236.160 ns    |
| Setup failing endpoints   | 645 / 104,600  |
| Hold WHS                  | +0.019 ns (PASS) |
| Hold THS                  | 0.000 ns (PASS) |
| Pulse-width WPWS          | +1.124 ns (PASS) |
| Maximum achieved frequency| ≈ 196 MHz      |

The design closes hold and pulse-width but misses setup at the aggressive
300 MHz target. The worst paths (28 logic levels, ~5 ns combinational)
run from `lane_dispatch/buf_out` through scheduler/CTA-CSR fanout into
`csr_unit/rsp_buf` — pre-existing critical paths (Vortex on this part
typically targets 150–200 MHz). The migration adds 1 bit (`is_rvc`) to
the pipeline structs and 1 bit to `PC_BITS`, neither of which appear on
the worst-path list. A baseline (no-EXT_C) synthesis run for direct
WNS-delta comparison is the next step if the absolute number matters
for production.

**Generated artifacts** (in
`build_test32/hw/syn/xilinx/dut/vortex/build_test1/`):
- `post_synth.dcp` (synthesis netlist), `post_impl.dcp` (final
  placed/routed checkpoint, written by the post-impl reporting stage)
- `post_synth_util.rpt`, `post_impl_util.rpt`
- `project_1/project_1.runs/impl_1/Vortex_timing_summary_postroute_physopted.rpt`
- `project_1/project_1.runs/impl_1/Vortex_route_status.rpt`,
  `Vortex_methodology_drc_routed.rpt`, `Vortex_power_routed.rpt`,
  `Vortex_utilization_placed.rpt`

---

## 6. Estimated diff size

| Area     | Files touched                                                                           | Net LoC added |
|----------|-----------------------------------------------------------------------------------------|---------------|
| simx     | `decompress.{h,cpp}`, `core.cpp`, `scheduler.{h,cpp}`, `Makefile`                       | ~450          |
| RTL      | `VX_decompressor.sv` (new), `VX_fetch.sv`, `VX_schedule_if.sv`, `VX_scheduler.sv`, `VX_gpu_pkg.sv` | ~900          |
| Tests/CI | `tests/regression/common.mk`, `tests/opencl/common.mk`, `tests/riscv/isa/Makefile`, `ci/regression.sh.in` | ~30           |

All gated by `EXT_C_ENABLE`; the disabled path stays bit-identical.
