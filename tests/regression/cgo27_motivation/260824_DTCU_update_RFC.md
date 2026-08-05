# RFC: DTCU update — two placement levels, async completion, ragged shapes

**Status:** items 1-9 implemented and pushed (`9e2a9c198`, `e18c3bf6f`); 10-17 open
**Date:** 2026-08-24
**Supersedes parts of:** [260718_moti_RFC.md](260718_moti_RFC.md) §4 (HW modes), §8 (implementation status)

Splits the DTCU into two placement variants whose only architectural difference is
**where the GEMM output lands**, and replaces the engine-state completion bit with a
per-descriptor completion field so several cores can consume results asynchronously.

---

## 0. Definitions

Every symbol used below, defined once so the rest of the document can be read without
guessing.

| Symbol | What it is | Where |
|---|---|---|
| **DTCU** | Disaggregated Tensor Core Unit. Descriptor-driven GEMM engine that walks the whole M×N×K tile space by itself. Today: one per cluster. | `sim/simx/dtcu/` |
| **TMA** | The DTCU's memory engine — descriptor fetch, operand prefetch, D store. Owns the L2 port. Not a separate unit from software's view. | `dtcu_tma.cpp` |
| **DXA** | Data-transfer Acceleration. Warp-issued async GMEM→LMEM copy engine with OOB clamp and multicast. A *different* engine; shares no code with the DTCU. | `sim/simx/dxa/` |
| **native tile** | The M×N×K block the DTCU computes in one pass. `tile_m` build-fixed, `tile_n` descriptor-selected, `tile_k` fixed in 32-bit words. | `sw/common/dtcu_cfg.h` |
| **`shape_n_size`** | Descriptor field selecting tile-N: `tile_n = shape_n_size × DTCU_TILE_N_GRAN` (gran = 16). | `dtcu_cfg.h` |
| **dcache** | The L1 **data** cache. A `CacheCluster` owned by the **socket**, sized `VX_CFG_DCACHE_SIZE` (32 KB here). "L1" in this document always means this. | `socket.cpp:54` |
| **LMEM** | Per-**core** scratchpad (64 KB), a separate address space, *not* a cache. DXA's destination. | `core.cpp:125` |
| **`VX_CFG_MISA_EXT`** | Vortex's **custom** extension bitmask — one bit per optional HW block. Generated from an expression in `VX_config.toml:312`. See §0.1. | `VX_config.toml:312` |
| **`strsp`** | Per-request `MemFlags` bit meaning "send me a response for this store". Stores are otherwise fire-and-forget. | `types.h:415` |
| **`io`** | Per-request `MemFlags` bit meaning "uncacheable"; the cache routes it to `processBypassRequest`. | `types.h:416`, `cache.cpp:1376` |

### 0.1 `VX_CFG_MISA_EXT` in full

`vx_dev_caps(dev, VX_CAPS_ISA_FLAGS, &flags)` returns one 64-bit word assembled in
[`cmd_processor.cpp:42-46`](../../../sim/common/cmd_processor.cpp#L42):

```
bits  0..29   VX_CFG_MISA_STD   standard RISC-V `misa` extension layout (A=0, C=2, D=3, F=5, M=12, …)
bits 30..31   XLEN encoding     decoded by the VX_ISA_ARCH(flags) macro
bits 32..63   VX_CFG_MISA_EXT   Vortex custom extension field  ← this one
```

`VX_CFG_MISA_EXT` is itself a bitmask built by an expr in `VX_config.toml:312`. Current
assignment:

| bit | block | | bit | block |
|---|---|---|---|---|
| 0 | ICACHE | | 6 | TEX |
| 1 | DCACHE | | 7 | RASTER |
| 2 | L2 | | 8 | OM |
| 3 | L3 | | 9 | TCU |
| 4 | LMEM | | 10 | DXA |
| 5 | ZICOND | | **11** | **DTCU** |

Bits 12..31 are free. `sw/runtime/include/vortex2.h` mirrors each as
`VX_ISA_EXT_<block> = 1ull << (32 + bit)`; the `+32` is the shift above, nothing more.

This matters here because §1 creates a **second** DTCU variant, and software must be
able to ask which of the two a device has before choosing which start instruction to
issue. That needs a second bit (12) and a second macro.

---

## 1. What changes and why

### 1.1 The problem with one DTCU

Three limitations, all load-bearing for the paper's argument:

**It is not actually async.** `dtensor_start` is fire-and-forget, but
[`dtcu.cpp:114-118`](../../../sim/simx/dtcu/dtcu.cpp#L114) **silently drops** a second
descriptor while busy — a debug print, no error to software. In-flight is capped at 1,
so a core cannot queue work and go do something else. Worse, in a multi-core setting a
second core's submission vanishes with no signal.

**Completion cannot be observed by a consumer.** `dtensor_poll()` returns a single
cluster-wide `done_` bit ([`dtcu.cpp:168-170`](../../../sim/simx/dtcu/dtcu.cpp#L168)) that
is not per-requester and is not cleared by reading. Any warp on any core polling sees
the same bit. A ticket returned in a register does not help: the epilogue consumer is a
*different core* that never saw the ticket. Completion identity has to live somewhere
every consumer can address — i.e. in the descriptor.

**Output placement is not a variable.** The engine writes D to L2 and there is no way to
express "put it nearer the cores that will consume it". That is exactly the axis the
paper wants to measure.

### 1.2 The two variants

Both engines are otherwise identical — same descriptor ABI, same FSM, same MMA
datapath. **Only the output target differs**, and the tile geometry follows from the
capacity of that target.

| | DTCU_socket | DTCU_cluster |
|---|---|---|
| Placement | one per socket (N instances) | one per cluster |
| Scratchpad | separate SRAM (`-D` selectable) | separate SRAM (`-D` selectable) |
| Operand read | L2, bypassing L1 — **N engines share one port** | L2, own port |
| **D output target** | **the socket's dcache** | **L2** |
| `DTCU_TILE_M` | 32 | 64 |
| `DTCU_TILE_N_MAX` | 16 | 128 |
| D tile footprint | 2 KB | up to 32 KB |
| Scratchpad size | ~7 KB × N | ~76 KB |

**Why 32×16 for socket.** The in-core TCU tile at `NUM_THREADS=32` is 16×16
(`wmma_config_t<32>`), and DTCU_cluster is 64×(16..128). The socket tile must sit
between the two to be a meaningful middle point: 32×16 is 2× the TCU tile and 1/16 of
the cluster tile.

The binding constraint is **output residency, not SRAM budget**. A D tile is
`tile_m × tile_n × 4 B`; at 32×16 that is 2 KB, so a 32 KB dcache holds 16 of them.
At `SOCKET_SIZE=4` — where one dcache serves 4 cores
(`VX_CFG_NUM_DCACHES = up(SOCKET_SIZE/4)`) — that is 4 tiles per consumer, enough for
double-buffering with slack. At 64×16 (4 KB/tile) it would be 2 per consumer, and at
64×32 exactly 1, i.e. no double-buffering at all.

**Why the tile-N choice disappears for socket.** With `DTCU_TILE_N_GRAN = 16` and
`TILE_N_MAX = 16`, `shape_n_size` has exactly one legal value. That is a *consequence*
of the output target's capacity, not a defect — and it is itself a finding: the socket
variant trades away tile-shape freedom for locality.

### 1.3 Why the socket variant needs a dcache port

To land a line in a cache you must send it a request. The socket's dcache is created
with exactly as many request inputs as it has cores
([`socket.cpp:54`](../../../sim/simx/socket.cpp#L54)):

```cpp
dcaches_ = CacheCluster::Create(sname, cores_per_socket, VX_CFG_NUM_DCACHES, …);
//                                     ^^^^^^^^^^^^^^^^ num_inputs
```

and every slot is bound to a core ([`socket.cpp:110-118`](../../../sim/simx/socket.cpp#L110)).
There is no free input. `num_inputs` becomes `cores_per_socket + 1` and the extra slot
goes to the DTCU — the same arrangement the DTCU already has on the L2 arbiter, where
it owns `kDtcuRow` ([`cluster.cpp:183`](../../../sim/simx/cluster.cpp#L183)).

Without this port the socket variant writes to L2 like the cluster variant, the two
become indistinguishable, and §1.2's comparison measures nothing.

### 1.4 Why a dummy read is needed, and why it is nearly free

Both caches are **write-through, no-write-allocate** — the Vortex default
(`VX_CFG_{DCACHE,L2,L3}_WRITEBACK = 0`, `VX_config.toml:188/207/221`). On a write miss
the line is **not installed**; the store is forwarded and nothing stays behind
([`cache.cpp:1093-1095`](../../../sim/simx/mem/cache.cpp#L1093)). D's first write always
misses, so "output target: L1" alone would leave D nowhere.

A **read** miss *does* allocate ([`cache.cpp:1143`](../../../sim/simx/mem/cache.cpp#L1143)),
and a write **hit** updates the cached line even under write-through
([`cache.cpp:1055`](../../../sim/simx/mem/cache.cpp#L1055) — `line_merge` runs before the
policy branch). So reading a D tile before writing it installs the line, and the
subsequent stores hit and update it.

Cost: one extra read per D line, which the TMA overlaps with compute. Total DRAM
traffic is unchanged versus today (2 transfers per line either way); what changes is
that the epilogue's read becomes a cache hit instead of a DRAM fetch.

**It can be free.** The engine already reads C for the accumulator preload when
`FLAG_ZERO_ACC` is clear, at the same addresses and tile shape as the D store. If
software sets `ptrC == ptrD` (in-place accumulate), the C preload *is* the dummy read.
An explicit dummy read is only needed under `FLAG_ZERO_ACC`.

Alternative considered and rejected: switching L2 to write-back. DRAM traffic per D line
would be identical (write-back fetches on write-miss, then evicts later — 2 transfers),
so it buys nothing without a full-line-write-allocate path, and that is a change to the
shared cache model affecting every benchmark.

### 1.5 Why completion must be a memory field, checked with an AMO

**Where it lives.** The consumer is a different core than the submitter, so a register
return value cannot reach it. The descriptor is the one object both sides address.
A `uint32_t` completion field goes in the existing `reserved2` slot — no ABI break.

**Why AMO.** A plain load of that field installs the line in the consumer's dcache
(read miss allocates), and the consumer then re-reads its own stale copy forever. An
atomic access takes the `AmoProbe` path
([`cache.cpp:813-826`](../../../sim/simx/mem/cache.cpp#L813)), which **invalidates the
local line, forwards to the LLC, and installs no fill** — a coherent read every time.
This requires `VX_CFG_EXT_A_ENABLE`, off by default.

**It must be a read-modify-write, not an atomic load** — learned the hard way. RISC-V
lowers `__atomic_load_n(p, ACQUIRE)` to `lw` plus a fence: an ordinary load that hits
the core's own cached copy, with the fence ordering memory operations and doing nothing
about staleness. The check spun forever. `__atomic_fetch_or(p, 0, ACQUIRE)` is the
read-only RMW — it returns the value, writes it back unchanged, and emits a real
`amoor.w`, which is what reaches AmoProbe.

**The check is software, not an instruction.** `dtensor_check()` is an atomic load on
the descriptor address; it never talks to the engine. No opcode is spent on it.

**Cost is lower than it looks.** The engine fetches the descriptor at `DESC_REQ`, and a
read miss allocates — so the descriptor line is resident in L2. The engine's completion
store then *hits* in L2, and the consumer's AMO resolves there. No DRAM round-trip
unless the line is evicted, which one 64 B line in a 1 MB L2 rarely is.

### 1.6 Why store completion must become response-based

This is the correctness prerequisite for §1.5 and it is easy to miss, because the FSM
*already* waits for `store_active()` at `TILE_STORE` and `FINAL_TILE_STORE`. The problem
is what that flag means:

```cpp
// dtcu_tma.cpp:567 — "done" == all lines pushed into the channel
if (out_req_idx_ >= out_req_lines_.size() && tma_store_accread_left_ == 0)
    tma_store_active_ = false;
```

[`issue_store_`](../../../sim/simx/dtcu/dtcu_tma.cpp#L140) does not set `strsp`, so the
cache sends no response and the engine never learns when a store landed. Its own comment
says so: *"we track store completion by 'all lines issued', not by responses"*.

So a completion flag written at that point can become visible **before** the last D
lines — different addresses go to different L2 banks with independent pipelines, and the
consumer's AMO would faithfully report "done" over a half-written tile. AMO fixes
visibility; it does nothing for ordering.

The fix is one concept: make `store_active()` mean *acked*, not *issued*. The mechanism
exists — `need_core_rsp` already honours a per-request `strsp`
([`cache.cpp:803-805`](../../../sim/simx/mem/cache.cpp#L803)). Because the FSM
serialises stores one output tile at a time, fixing this one flag makes every
downstream wait correct, including the final one.

Side effect, wanted: `store_drain` starts measuring the real drain. Today it stops at
issue, so the reported value is an undercount.

### 1.6b L2 is mandatory for the DTCU (found during implementation)

The completion flag only works if the engine writes it to the **consumer's point of
coherence**. The consumer's AMO resolves at its last-level cache, so the engine has to
reach that same cache.

With L2 enabled that holds: L2 is the LLC and the DTCU's port lands there. Without L2
it does not. `socket.cpp:69` makes the dcache the LLC when L2 and L3 are both off, and
the DTCU bypasses the dcache entirely — so the engine writes to memory, the consumer's
AMO resolves against a dcache line the engine never touched, and the check spins
forever. `dtcu_basic` did exactly that for ten minutes at 199% CPU.

`dtcu_basic` and `dtcu_compare` therefore enable L2. This is not a tuning choice: the
engine has always been described as "own TMA → L2", and this makes the dependency real
instead of implied. Their cycle counts shift accordingly; `cgo27_motivation` already had
L2, so its in-core modes are untouched.

The general rule, worth keeping in mind for the socket variant: **whatever cache the
engine writes D into must also be where the consuming core's atomics resolve.** For
DTCU_socket that is the socket's dcache, which is why item 11 gives it a port there
rather than letting it write past.

### 1.7 Why `-s` goes away

`-s N` expands to `M = N × dtcu_tileM`, `N = N × dtcu_tileN`, `K = N × dtcu_tileK`
([`main.cpp:515-517`](main.cpp#L515)) — "N times the DTCU's native tile". With two
engines whose tiles differ (32×16 vs 64×128) there is no longer a single native tile,
so the flag has nothing to multiply.

It cannot expand per-engine either. The harness's premise is that **every mode runs the
same GEMM**; that is the only reason the cycle comparison means anything. A flag that
produced a different shape per mode would quietly destroy that.

Pinning `-s` to one engine's tile would work, but it encodes a policy ("sizes are
multiples of the cluster tile") in the harness that only the sweep scripts care about.
Since ragged shapes now run correctly on both engines, that policy no longer buys
anything at the harness level.

So `-M/-N/-K` become the only shape input, and the size ladder moves into the sweep
scripts, which expand each rung to explicit dimensions. Consequences, all in item 15/16:

- `SIZE_MULT`, `g_size_mult`, and `case 's'` come out of `main.cpp`.
- The `[MOTI]` line loses `size=`. Both scripts parse it
  (`sweep_exp1.py:25`, `sweep_exp2.py:36`) and both write it to CSV, so their regex and
  headers change together.
- `sweep_exp1.py` is *itself* a size sweep (`--sizes 1,2,4,8,16`): it keeps the ladder
  as a script-side constant and emits `-M/-N/-K` per rung. `sweep_exp2.py` does the same
  for its single `--size`.
- The README's `-s` documentation goes with it.

---

## 2. Work items

Ordered by dependency, not importance. DONE items are implemented, verified
against all three tests, and pushed.

| # | Item | Touches |
|---|---|---|
| 1 ✅ | **Store completion by response.** Set `strsp` in `issue_store_`; count store responses in `drain_responses()`; clear `tma_store_active_` on response count. Then emit the completion-flag store. | `dtcu_tma.{h,cpp}` |
| 2 ✅ | **Counter reset split.** `start()` currently re-zeros every perf counter, so a second submission erases the first's numbers. Move counters to `on_reset()` only; factor per-GEMM state into one `begin_descriptor_()` the queue can also call. | `dtcu.{h,cpp}` |
| 3 ✅ | **`start()` reports.** Return accept/reject instead of dropping silently. `rd` is currently `x0` in the intrinsic; the SFU already writes `rd` for the old poll, so no new encoding. | `dtcu.{h,cpp}`, `sfu_unit.cpp`, `vx_dtensor.h` |
| 4 ✅ | **Descriptor queue.** Depth: socket `NUM_CORES × 2`, cluster `NUM_SOCKETS × 2`. An entry is `{desc_addr, requester}` ≈ 12 B, so depth is effectively free; the constraint is only that no sharer can be starved. | `dtcu.{h,cpp}`, `dtcu_params.h` |
| 5 ✅ | **Enable atomics.** `VX_CFG_EXT_A_ENABLE`. | build CONFIGS |
| 6 ✅ | **Completion field + `dtensor_check()`.** `reserved2` becomes the completion word. Add the SW helper; **remove `dtensor_poll`** and its decode/SFU handling. | `dtcu_cfg.h`, `vx_dtensor.h`, `decode.cpp`, `sfu_unit.cpp` |
| 7 ✅ | **Descriptor buffer writable.** `VX_MEM_READ` → `VX_MEM_READ_WRITE` in all three tests. | `*/main.cpp` |
| 8 ✅ | **D dummy read.** Socket reads through L1, cluster through L2, so the line is allocated where the output is meant to live. Skipped when `ptrC == ptrD` (the C preload already did it). | `dtcu_tma.cpp` |
| 9 ✅ | **Per-engine geometry.** Split `DTCU_TILE_M` / `DTCU_TILE_N_MAX` (and the `dtcu_config_t` traits) per engine. Each engine validates its own limits — a `shape_n_size` beyond its `TILE_N_MAX` must be rejected. | `dtcu_cfg.h`, `tensor_cfg.h` |
| 10 | **Two start instructions.** `RISCV_CUSTOM2` funct3=1 `START_SOCKET`, funct3=2 `START_CLUSTER` (funct3=2 is freed by removing poll). Two config enables, two MISA_EXT bits (11 keeps DTCU_cluster, 12 adds DTCU_socket), two `VX_ISA_EXT_*` macros. | `decode.cpp`, `sfu_unit.cpp`, `VX_config.toml`, `vortex2.h`, `vx_dtensor.h` |
| 11 | **dcache input port.** `num_inputs` → `cores_per_socket + 1`; bind the spare slot to DTCU_socket. | `socket.cpp` |
| 12 | **Socket engine placement + shared L2 read port.** N engines at socket scope; their operand reads funnel through one arbiter into a single `l2arb` row so `kL2Rows` stays independent of socket count. Response routing: engine id in the **high** tag bits (arbiters add bits at the LSB — `cache.cpp:1243`). | `socket.cpp`, `cluster.cpp`, `dtcu_tma.*` |
| 13 | **Perf aggregation.** Socket and cluster engines report separately. | `csr_unit.cpp`, `cluster.cpp`, `socket.cpp` |
| 14 | **cgo27 modes.** Drop the DTCU_TMA mode; make modes for DTCU_cluster and DTCU_socket. Delete the NO_TMA tripwire ([`main.cpp:644-650`](main.cpp#L644)). **Keep `DTENSOR_FLAG_NO_TMA` in the ISA** — the engine paths stay, only the harness mode goes. | `main.cpp`, `k_dtcu.h` |
| 15 | **Remove `-s`.** With two engines there is no single native tile to multiply, so the flag has no definition left (§3.1). Drop `-s`, `SIZE_MULT`, `g_size_mult`, and the `size=` field from the `[MOTI]` line; `-M/-N/-K` become the only way to set a shape, defaulting to one cluster tile. | `main.cpp`, `Makefile` |
| 16 | **Sweep scripts.** Two changes. (i) `MODES` is hardcoded in both (`sweep_exp1.py:21`, `sweep_exp2.py:20`) and feeds CSV headers, so a stale table silently mislabels results. (ii) Both drive `-s` and parse `size=`; the size ladder moves into the scripts, which expand each rung to `-M/-N/-K`. | `sweep_exp*.py` |
| 17 | **Test coverage.** All three tests use `while (0 == dtensor_poll())` and must move to `dtensor_check()`. `dtcu_basic` and `dtcu_compare` should exercise **both** variants. | `dtcu_basic/`, `dtcu_compare/`, `k_dtcu.h` |

---

## 3. Open decisions

### 3.1 `L2_NUM_REQS` (deferred, not dropped)

`VX_CFG_L2_NUM_REQS` is declared `"int"` in `VX_config.toml:335` — a placeholder with no
value. The generated C++ header references it anyway
(`VX_config.h:876`), so the preprocessor folds the undefined identifier to 0 and
`L2_NUM_BANKS` collapses to 1. RTL computes it properly from a localparam
(`VX_gpu_pkg.sv:1409`: `NUM_SOCKETS × L1_MEM_PORTS = 4`), so **SimX models a 4× narrower
L2 than the design**.

Deferred because it buys no performance here — measured L2 utilisation on one bank peaks
at 29% (mode 1) and is 5-8% in the DTCU modes — while changing it invalidates every
baseline. Revisit when core count grows or when N socket engines start contending.

| mode | L2 requests | per cycle | 1-bank utilisation |
|---|---|---|---|
| 0 SIMT | 10242 | 0.054 | 5% |
| 1 TCU | 4204 | 0.289 | **29%** |
| 2 TCU+DXA | 4660 | 0.266 | 27% |
| 3 DTCU | 1797 | 0.054 | 5% |
| 4 DTCU+TMA | 1797 | 0.081 | 8% |

---

## 4. Verification

Everything before item 9 keeps a single engine and must be **numerically inert**: all
three tests report identical cycles and MPM counters to the current baseline. Any change
there is a bug, not an improvement.

- After each of items 1-8: full run of `dtcu_basic`, `dtcu_compare`, `cgo27_motivation`;
  diff cycles and every MPM counter against baseline (ignore `host_ms`).
- Item 1 additionally: `store_drain` **will** increase — it starts counting the real
  drain. That is the one expected delta, and it should be explainable as the
  issue-to-ack latency of the final store.
- Item 6: a consumer core on a *different* core than the submitter must observe
  completion. A same-core test does not exercise the staleness path.
- Items 9-14: socket and cluster variants must produce bit-identical **results** (only
  cycles differ), verified against the CPU reference at ragged shapes as well as
  aligned — the OOB path is already covered by 100×48×20, 1×1×1, 65×33×17, 63×31×16,
  64×32×15, 100×48×21, 129×17×3.

---

## 5. Not doing

- **Hardware barrier for completion.** `vx_barrier`'s `wait()` blocks, which turns an
  async engine into a cluster-wide rendezvous. The descriptor field lets a consumer
  check and keep working.
- **Per-tile completion.** Completion is per **descriptor**. Finer overlap is expressed
  by submitting more, smaller descriptors — a software knob, no extra hardware state.
- **L2 write-back.** Same DRAM traffic as write-through here (§1.4); the real fix would
  be full-line write-allocate, which is a shared-cache change.
- **Scratchpad in L1/L2.** No way-reservation or address-range carve-out exists in the
  cache model, so this is a new cache mode, not a flag. Both variants use separate SRAM.
- **Backpressure from consumer to engine.** If the epilogue lags, D evicts and the
  locality benefit silently disappears. Measure it (dcache miss rate on the epilogue
  read) rather than build flow control; the crossover point is itself a result.
