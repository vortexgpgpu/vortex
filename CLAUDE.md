# CLAUDE.md — Vortex DTCU / TMA Handoff

Last updated: 2026-06-16
Owner: Chulhyung / Claude Code
Project: Vortex DTCU first, TMA next

This file is the operational handoff for Claude Code. It assumes the project source and notes are deployed under the Claude Code working directory, usually with the original source tree and notes placed under `claude_doc/`.

---

## 0. Absolute rules for this project

1. **Actual source code is the highest source of truth.** If this handoff, DTCU notes, PDFs, or prior chat disagree with the checked-out source, trust the source and report the mismatch with file path and line number.
2. **Do not implement TMA before verifying the current DTCU baseline.** First confirm that DTCU custom instructions, descriptor ABI, SimX state machine, memory request model, and regression tests are internally consistent.
3. **Do not silently paper over stale tests.** If a test uses an old descriptor or old tile shape, update the test or mark it stale. Do not interpret a stale pass/fail as architectural evidence.
4. **Every claim must cite file path + line number.** When reporting status, include exact files, line ranges, and the evidence used.
5. **Comments in code must be short English comments.** Keep implementation comments concise.
6. **Do not add a software-visible TMA ISA/API unless the user explicitly asks.** The current first target is DTCU-internal TMA/prefetch, not a general Virgo-style software-visible DMA API.

---

## 1. Expected local material

The user said the existing project material will be copied into `claude_doc/`. Expect some or all of the following:

```text
claude_doc/
  vortex-source/ or vortex-dtcu/        # current Vortex source tree
  DTCU Note/                            # unzipped DTCU Note.zip
  DTCU Test Cases.pdf
  DTCU on Vortex.png
  virgo-main/                           # Virgo reference source
  Virgo 논문.pdf
```

The uploaded `DTCU Note.zip` was rechecked. Its important files are:

```text
00. DTCU/DTCU Vortex Project.md
00. DTCU/01. Basics/DTCU Vocabularies.md
00. DTCU/01. Basics/SimX Tuturial.md
00. DTCU/01. Basics/DTCU Basic Theory.md
00. DTCU/01. Basics/DTCU Vortex Architecture.md
00. DTCU/02. Virgo/DTCU Virgo.md
00. DTCU/03. Implementation/DTCU Overview.md
00. DTCU/03. Implementation/DTCU Implementation.md
00. DTCU/03. Implementation/DTCU Increasing Tile Size.md
00. DTCU/03. Implementation/DTCU Internal Iteration.md
00. DTCU/03. Implementation/DTCU TMA.md
00. DTCU/03. Implementation/DTCU Test Cases.md
00. DTCU/04. Notes/DTCU Notes.md
```

---

## 2. Current DTCU status from source audit

Source tree audited: `vortex-source.zip` → `vortex-dtcu`.

### 2.1 DTCU is currently a SimX-level model, not a full RTL implementation

Evidence in current source:

- `sim/simx/d_tensor_core.h` defines `class DTensorCore : public SimObject` and owns `mem_req_out`, `mem_rsp_in`, descriptor state, operand buffers, accumulator buffer, and tile state.
- `sim/simx/d_tensor_core.cpp` implements the DTCU state machine and functional compute.
- `sim/simx/cluster.cpp` constructs `DTensorCore` and attaches it to an extra L2 port.
- `sim/simx/constants.h` defines `L2_NUM_REQS = NUM_SOCKETS * L1_MEM_PORTS + 1`, where the `+1` is for DTCU.
- `kernel/include/vx_tensor.h` exposes `dtensor_start()` and `dtensor_poll()` as custom instructions.

I did not find a complete RTL-side DTCU/TMA implementation in the current `vortex-dtcu` tree. Treat current work as **SimX-first** unless the user provides another RTL branch.

### 2.2 Current custom instruction path

Relevant source locations:

```text
kernel/include/vx_tensor.h
  dtensor_desc_t fields: ptrA/ptrB/ptrC/ptrD, M/N/K, ldmA/B/C/D, fmt_s/fmt_d, flags, shape_n_size, shape_policy
  dtensor_start(desc_addr): RISCV_CUSTOM0, funct7=2, funct3=1
  dtensor_poll(): RISCV_CUSTOM0, funct7=2, funct3=2

sim/simx/types.h
  FUType includes DTCU_Control and TCU
  TcuType includes WMMA, DTENSOR_START, DTENSOR_POLL

sim/simx/decode.cpp
  decodes funct7=2/funct3=1 as DTENSOR_START
  decodes funct7=2/funct3=2 as DTENSOR_POLL

sim/simx/execute.cpp
  DTENSOR_START calls cluster->dtensor()->start(desc_addr)
  DTENSOR_POLL calls cluster->dtensor()->poll() and writes the done bit back to all lanes
```

Critical thing to verify before implementation: `Core::commit()` / perf accounting appears to explicitly handle `FUType::TCU`, but `DTENSOR_START/POLL` decode as `FUType::DTCU_Control`. Confirm whether `DTCU_Control` commits cleanly or falls into a default assertion / missing perf bucket.

### 2.3 Current DTCU memory model

The DTCU notes and source agree on this two-level model:

1. **Functional data path:** descriptor, operands, and output are read/written through direct RAM access (`ram_->read`, `ram_->write`). This determines correctness.
2. **Timing/stat path:** DTCU issues representative `MemReq` lines through `mem_req_out`, receives `MemRsp` through `mem_rsp_in`, and only then performs direct RAM read/write. This approximates L2/cache timing and stats.

Source-level evidence:

```text
sim/simx/d_tensor_core.cpp
  issue_mem_req(): emits MemReq to mem_req_out
  load_desc(): direct ram_->read(desc_, desc_addr_, sizeof(desc_))
  load_operands(): direct ram_->read for C, A, B tile data
  store_output(): direct ram_->write for D
  build_req_lists_(): builds representative cache-line read/write request lists
  tick(): DESC_REQ/DESC_WAIT, OP_REQ/OP_WAIT, EXECUTE, OUT_REQ/OUT_WAIT sequencing
```

This means DTCU correctness and DTCU cache/stat modeling are intentionally not identical mechanisms. Do not compare DTCU custom `MemReq` count directly against in-core LSU instruction counters as if they were the same metric.

### 2.4 Current tile model

The old notes mention early lane-fragment tiles, but the current source has already moved to a dense-buffer DTCU style.

Current DTCU tile behavior from source:

```text
sim/simx/d_tensor_core.cpp
  DTCU_TILE_K_WORDS = 8
  fmt_d must be fp32
  shape_n_size must be nonzero
  shape_policy must currently be 0
  tile_m_ = 64
  tile_n_ = shape_n_size * 16, legal range 16..128
  tile_k_ = 8 * (4 / elem_size(fmt_s))
    fp16/bf16 => K = 16 elements
    fp32      => K = 8 elements
  M/N/K must be exact multiples of tile_m_/tile_n_/tile_k_
  a_buf_ size = tile_m_ * 8 words
  b_buf_ size = 8 * tile_n_ words
  accum_buf_ size = tile_m_ * tile_n_ floats
```

For `shape_n_size=8`, `tile_n_=128`, `accum_buf_` is `64*128*4 = 32768 B`, which matches the Virgo-like 32KB accumulator-memory framing in the notes.

### 2.5 Current compute model

Current `execute_mma()` does one logical tile accumulation by looping over:

```text
m in [0, tile_m_)
n in [0, tile_n_)
kw in [0, DTCU_TILE_K_WORDS) step cfg::tcK
```

It gathers packed A/B words from:

```text
A: a_buf_[m * DTCU_TILE_K_WORDS + kw + z]
B: b_buf_[(kw + z) * tile_n_ + n]
```

Then calls `tc_.wmma(...)` and writes back to `accum_buf_[m * tile_n_ + n]`.

Important caveat for TMA: current `execute_mma()` is effectively called as one SimX tick action. If TMA overlap is supposed to be visible in cycles, DTCU compute must either become multi-cycle or have an explicit compute-latency model. Otherwise a ping-pong TMA design may be functionally correct but its overlap benefit may be invisible or misleading.

---

## 3. Current regression-test status

### 3.1 `dtcu_basic` is stale relative to current descriptor ABI and tile rules

Evidence:

```text
tests/regression/dtcu_basic/main.cpp
  local dtensor_desc_t has reserved fields instead of shape_n_size / shape_policy
  uses M=8, N=4, K=8
```

This conflicts with the current DTCU source, which requires `M` multiple of 64, `shape_n_size != 0`, `shape_policy == 0`, and legal `tile_n_ = shape_n_size*16`.

Action: repair `dtcu_basic` before using it as baseline evidence. Recommended minimum case:

```text
fmt_s = fp16 or bf16
fmt_d = fp32
M = 64
N = 32       # shape_n_size = 2
K = 32       # two K tiles if fp16/bf16, useful for TMA prefetch testing
shape_policy = 0
flags = 0 or 1 depending C/accum behavior; document exactly
```

### 3.2 `dtcu_compare` is closer to current source, but still needs verification

Evidence:

```text
tests/regression/dtcu_compare/main.cpp
  local dtensor_desc_t includes shape_n_size and shape_policy
  DTCU case uses M=64, N=32, K=2*dtcu_tileK
  descriptor sets shape_n_size=2 and shape_policy=0

tests/regression/dtcu_compare/kernel.cpp
  mode == 0: in-core TCU path
  mode == 1: DTCU path
```

Important correction: some older notes may describe the modes in the opposite direction. The current source uses:

```text
mode = 0 -> in-core TCU
mode = 1 -> DTCU
```

Potential bug/review item: DTCU path checks `warp_id == 0 && thread_id == 0`, but unlike `dtcu_basic`, it does not obviously guard on `vx_core_id() == 0`. Confirm whether `vx_spawn_threads(2, ...)` can cause more than one core to attempt `dtensor_start()`.

---

## 4. TMA direction after rechecking `DTCU Note.zip`

The previous handoff treated a standalone `dtma_basic` / software-visible TMA path as a likely first step. After rechecking `DTCU Note.zip`, the first implementation target should be revised:

> First target: **Option C — DTCU-internal K-tile prefetch engine**.

This is an implicit/autonomous TMA-like engine inside DTCU, not a general Virgo-style software-visible DMA/TMA API.

### 4.1 Option C summary

Current DTCU does:

```text
for each output tile (M,N):
  for each K tile:
    wait for operand MemReqs
    ram_->read A/B into buffers
    execute_mma()
  wait for output MemReqs
  ram_->write D
```

Option C should become:

```text
for each output tile (M,N):
  preload K0 into operand buffer 0
  for each K tile:
    compute current buffer
    prefetch next K tile into the other buffer when possible
    if compute reaches a not-ready next buffer, count dtcu_wait_for_tma_cycles
  store D
```

The kernel API should stay simple:

```c
if (leader) {
  dtensor_start(desc_addr);
  while (!dtensor_poll()) {}
}
```

### 4.2 Difference from Virgo

Virgo-style data movement:

```text
GMEM -> DMA/TMA -> shared memory -> matrix unit
SIMT core can also access shared memory for softmax/activation/fused work
software issues virgo_dma_load / virgo_compute / virgo_fence
```

DTCU Option C data movement:

```text
GMEM/RAM -> internal TMA/prefetch -> DTCU A/B ping-pong buffers -> DTCU compute
software only issues dtensor_start/poll
```

Option C is easier to implement in the current SimX source and is the correct first milestone. Virgo-style shared memory + software-visible DMA/TMA is a later milestone if the project moves toward FlashAttention/SoftMax overlap.

---

## 5. Recommended implementation plan

### Phase 0 — Baseline repair and audit

Do this before writing TMA code.

1. Read the actual source tree and confirm all file paths below still match.
2. Verify build/config for `EXT_TCU_ENABLE`.
3. Check whether `FUType::DTCU_Control` commits cleanly in `Core::commit()`.
4. Repair `dtcu_basic` to current descriptor ABI and tile rules.
5. Run `dtcu_basic` and `dtcu_compare` on a small valid DTCU case.
6. Record baseline output, cycle counters, DTCU debug prints, and any failures.

### Phase 1 — Refactor current blocking operand load into a TMA-like abstraction

Goal: no behavior change.

Create a small internal abstraction, for example:

```text
DTensorTma / DtcuTmaEngine / internal helper methods inside DTensorCore
```

It should own or coordinate:

```text
current tile_m_idx/tile_n_idx/tile_k_idx
A/B base address generation
cache-line request coalescing
operand-buffer fill through direct ram_->read after MemRsp completion
```

Do not add overlap yet. Acceptance: existing valid DTCU tests pass exactly as before.

### Phase 2 — Add ping-pong A/B buffers

Change:

```text
a_buf_ -> a_buf_[2]
b_buf_ -> b_buf_[2]
```

Track:

```text
buf_ready[2]
buf_k_idx[2]
compute_buf
tma_buf
inflight_tma
```

Acceptance: run a two-or-more K-tile GEMM and prove both buffers are used while output remains identical.

### Phase 3 — Split FSM to model producer/consumer overlap

Needed state concepts:

```text
PREFETCH_REQ / PREFETCH_WAIT
COMPUTE
COMPUTE_WAIT if compute becomes multi-cycle
TMA_WAIT when compute needs next buffer before TMA is ready
STORE_REQ / STORE_WAIT
```

The note explicitly says `compute_done` and `tma_done` must be separate. Do not collapse them into one busy flag if you want to measure overlap.

### Phase 4 — Add counters

Add at least:

```text
tma_addr_gen_cycles
tma_mem_wait_cycles
tma_buffer_write_cycles
dtcu_compute_cycles
dtcu_wait_for_tma_cycles
tma_wait_for_buffer_cycles
```

The most important metric is:

```text
dtcu_wait_for_tma_cycles
```

If this drops after prefetching, TMA is doing useful work. If it remains high, compute is waiting on memory.

### Phase 5 — Optional output-store overlap

Only after input prefetch is stable. Output store overlap introduces arbitration because a single TMA engine may need to handle both:

```text
load A/B for next tile
store D for completed tile
```

Do not start here.

---

## 6. Open questions / blockers Claude must report before big changes

1. **Does current `DTCU_Control` commit correctly?** If not, fix this first.
2. **Do we want a real multi-cycle compute model?** Without this, TMA overlap may be functionally present but timing-invisible.
3. **Should TMA share the existing DTCU L2 port or get a logically separate port?** First implementation should probably share the existing DTCU port unless the user asks otherwise.
4. **What should happen on busy `dtensor_start()`?** Current `start()` ignores new starts while busy. Document whether this is acceptable.
5. **Should `dtcu_basic` use `flags=1` zero-accum or load C?** Choose explicitly and make host reference match.
6. **Is fp32 source supported for TMA first milestone?** Current tile rule supports it, but first tests should use fp16/bf16 source + fp32 output unless the user asks otherwise.

---

## 7. Useful source paths to inspect first

```text
kernel/include/vx_tensor.h
sim/simx/d_tensor_core.h
sim/simx/d_tensor_core.cpp
sim/simx/cluster.cpp
sim/simx/constants.h
sim/simx/types.h
sim/simx/decode.cpp
sim/simx/execute.cpp
sim/simx/func_unit.cpp
sim/simx/core.cpp
sim/simx/Makefile
tests/regression/dtcu_basic/main.cpp
tests/regression/dtcu_basic/kernel.cpp
tests/regression/dtcu_compare/main.cpp
tests/regression/dtcu_compare/kernel.cpp
```

Virgo reference paths, if `virgo-main/` is present:

```text
README.md
src/main/scala/virgo/GemminiTile.scala
src/main/scala/virgo/VirgoSharedMemComponents.scala
src/main/scala/virgo/Configs.scala
```

Use Virgo as conceptual reference for cluster-level matrix unit + DMA/shared memory design, not as code to blindly transplant into SimX DTCU.
