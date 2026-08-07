# RFC: DTCU update — two placement levels, async completion, ragged shapes

**Status:** CLOSED. All items 1-17 implemented, verified and pushed
(`9e2a9c198`, `e18c3bf6f`, `69fb84d4f`, `e827b57b9`, `fc39497b3`, `1c5c7c61e`,
`6b9514922`), then audited and corrected (`d666f4159`, `9d1e968fd`, `29d8efa45`,
`fa8588e4c`).
Four findings changed the plan and are recorded below: item 12's tag-bit scheme turned
out to be unnecessary (§2.1), the L2 arbiter's row indexing was already wrong before this
work (§2.2), a post-implementation audit found one item specified backwards and three
latent configuration bugs (§2.3), and measuring at three GEMM shapes rather than one
changed what the numbers support (§4.2).
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
[`cmd_processor.cpp:42-46`](../../../../sim/common/cmd_processor.cpp#L42):

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
| 4 | LMEM | | 10 ✅ | DXA |
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
[`dtcu.cpp:114-118`](../../../../sim/simx/dtcu/dtcu.cpp#L114) **silently drops** a second
descriptor while busy — a debug print, no error to software. In-flight is capped at 1,
so a core cannot queue work and go do something else. Worse, in a multi-core setting a
second core's submission vanishes with no signal.

**Completion cannot be observed by a consumer.** `dtensor_poll()` returns a single
cluster-wide `done_` bit ([`dtcu.cpp:168-170`](../../../../sim/simx/dtcu/dtcu.cpp#L168)) that
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
| `DTCU_TILE_N_MAX` | 16 ✅ | 128 |
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
([`socket.cpp:54`](../../../../sim/simx/socket.cpp#L54)):

```cpp
dcaches_ = CacheCluster::Create(sname, cores_per_socket, VX_CFG_NUM_DCACHES, …);
//                                     ^^^^^^^^^^^^^^^^ num_inputs
```

and every slot is bound to a core ([`socket.cpp:110-118`](../../../../sim/simx/socket.cpp#L110)).
There is no free input. `num_inputs` becomes `cores_per_socket + 1` and the extra slot
goes to the DTCU — the same arrangement the DTCU already has on the L2 arbiter, where
it owns `kDtcuRow` ([`cluster.cpp:183`](../../../../sim/simx/cluster.cpp#L183)).

Without this port the socket variant writes to L2 like the cluster variant, the two
become indistinguishable, and §1.2's comparison measures nothing.

### 1.4 Why a dummy read is needed, and why it is nearly free

Both caches are **write-through, no-write-allocate** — the Vortex default
(`VX_CFG_{DCACHE,L2,L3}_WRITEBACK = 0`, `VX_config.toml:188/207/221`). On a write miss
the line is **not installed**; the store is forwarded and nothing stays behind
([`cache.cpp:1093-1095`](../../../../sim/simx/mem/cache.cpp#L1093)). D's first write always
misses, so "output target: L1" alone would leave D nowhere.

A **read** miss *does* allocate ([`cache.cpp:1143`](../../../../sim/simx/mem/cache.cpp#L1143)),
and a write **hit** updates the cached line even under write-through
([`cache.cpp:1055`](../../../../sim/simx/mem/cache.cpp#L1055) — `line_merge` runs before the
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
([`cache.cpp:813-826`](../../../../sim/simx/mem/cache.cpp#L813)), which **invalidates the
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

[`issue_store_`](../../../../sim/simx/dtcu/dtcu_tma.cpp#L140) does not set `strsp`, so the
cache sends no response and the engine never learns when a store landed. Its own comment
says so: *"we track store completion by 'all lines issued', not by responses"*.

So a completion flag written at that point can become visible **before** the last D
lines — different addresses go to different L2 banks with independent pipelines, and the
consumer's AMO would faithfully report "done" over a half-written tile. AMO fixes
visibility; it does nothing for ordering.

The fix is one concept: make `store_active()` mean *acked*, not *issued*. The mechanism
exists — `need_core_rsp` already honours a per-request `strsp`
([`cache.cpp:803-805`](../../../../sim/simx/mem/cache.cpp#L803)). Because the FSM
serialises stores one output tile at a time, fixing this one flag makes every
downstream wait correct, including the final one.

Side effect, wanted: `store_drain` starts measuring the real drain. Today it stops at
issue, so the reported value is an undercount.

### 1.6c What "acked" means differs between the two engines (audit finding)

`strsp` buys an acknowledgement from **the first cache the store reaches**, not from the
point of coherence, and the two engines reach different first caches:

| engine | D goes to | so the ack proves |
|---|---|---|
| cluster | L2 directly | the L2 line is merged — the same place a consumer's AMO resolves |
| socket | that socket's dcache | only the **dcache** line is merged; the write-through to L2 is still in flight |

`cache.cpp` emits the ack in the *same bank cycle* it pushes the write-through
downstream, and the write-through `MemReq` does not copy `flags`, so `strsp` never
propagates. For the socket engine the completion flag and the tail of its own D data are
therefore racing to L2 along **two independent paths** with no fence: the data rides the
socket egress (l2arb **row 0**), the flag rides the DTCU socket row.

**For the consumer the socket engine exists to serve, none of that matters.** A core in
the *same socket* reads D out of the very cache the engine merged it into, so the
timeline is:

| | |
|---|---|
| t1 | dcache write-hit: `line_merge` — D is now visible to every core in this socket |
| t1 | the ack is emitted in the same cycle |
| t3 | the flag is issued only after **all** acks are in, so t3 > t1 |
| t4 | the consumer's AMO observes the flag at L2 |
| t5 | the consumer loads D and hits that same dcache line |

t1 < t3 < t4, so D precedes the flag **by construction**. The property that looked like a
weakness — `strsp` acking at the first cache rather than at the point of coherence — is
exactly right here, because for this engine the first cache *is* the consumer's cache.

The priority argument is therefore only load-bearing on the two fallback paths: a reader
in a **different socket**, and a same-socket reader whose D line was **evicted** before
it got there. Both resolve at L2, and there the flag and the tail of the write-through
race. What keeps them ordered is that `PriorityArbiter::grant()` returns the lowest
requesting index, so socket egress (row 0) always beats the DTCU row, plus the AMO round
trip the consumer needs before it can see the flag at all. Two independent verifiers
failed to turn it into a live bug. `cluster.cpp` carries a `static_assert` pinning
`kDtcuSocketRow > 0` so the row layout cannot be reordered silently — see also §B6,
whose suggestion to promote that row for latency is precisely what would break it.

Worth separating clearly, because it is easy to conflate:

* **socket scope = the performance claim.** D lands where that socket's cores read it
  cheaply. That is the entire reason to pick this variant over the cluster one.
* **cluster scope = the correctness range.** The dcache is write-through, so D reaches L2
  regardless — it has to, or the host could not read D back at all — and the completion
  flag is deliberately routed out the read port to L2 so that *any* core can poll it.

Not fixed, deliberately. The clean fix is to make `strsp` mean "acked at the point of
coherence" — propagate it into the write-through and withhold the core response until
the downstream one returns, using a pending table shaped like the existing
`amo_passthru_`. The DTCU would need no change at all, since its FSM already waits for
all acks before writing the flag. It is not worth it now: `need_core_rsp` is upstream
Vortex code shared by every cache level, the DTCU is its only `strsp` client, the
ordering currently holds, and the change would lengthen the socket engine's store drain
and move mode 4's numbers.

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
([`main.cpp:515-517`](../main.cpp#L515)) — "N times the DTCU's native tile". With two
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
| 4 ✅ | **Descriptor queue.** Depth keyed to each engine's **sharer set**: cluster `NUM_CORES × 2`, socket `SOCKET_SIZE × 2`. ⚠️ This row originally said the opposite (socket `NUM_CORES × 2`, cluster `NUM_SOCKETS × 2`) and was implemented that way; the audit in §2.3 caught it. An entry is just a descriptor address, so depth is effectively free — the only constraint is that no sharer can be starved. | `dtcu.{h,cpp}`, `dtcu_params.h` |
| 5 ✅ | **Enable atomics.** `VX_CFG_EXT_A_ENABLE`. | build CONFIGS |
| 6 ✅ | **Completion field + `dtensor_check()`.** `reserved2` becomes the completion word. Add the SW helper; **remove `dtensor_poll`** and its decode/SFU handling. | `dtcu_cfg.h`, `vx_dtensor.h`, `decode.cpp`, `sfu_unit.cpp` |
| 7 ✅ | **Descriptor buffer writable.** `VX_MEM_READ` → `VX_MEM_READ_WRITE` in all three tests. | `*/main.cpp` |
| 8 ✅ | **D dummy read.** Socket reads through L1, cluster through L2, so the line is allocated where the output is meant to live. Skipped when `ptrC == ptrD` (the C preload already did it). | `dtcu_tma.cpp` |
| 9 ✅ | **Per-engine geometry.** Split `DTCU_TILE_M` / `DTCU_TILE_N_MAX` (and the `dtcu_config_t` traits) per engine. Each engine validates its own limits — a `shape_n_size` beyond its `TILE_N_MAX` must be rejected. | `dtcu_cfg.h`, `tensor_cfg.h` |
| 10 ✅ | **Two start instructions.** `RISCV_CUSTOM2` funct3=1 `START_SOCKET`, funct3=2 `START_CLUSTER` (funct3=2 is freed by removing poll). Two config enables, two MISA_EXT bits (11 keeps DTCU_cluster, 12 adds DTCU_socket), two `VX_ISA_EXT_*` macros. | `decode.cpp`, `sfu_unit.cpp`, `VX_config.toml`, `vortex2.h`, `vx_dtensor.h` |
| 11 ✅ | **dcache input port.** `num_inputs` → `cores_per_socket + 1`; bind the spare slot to DTCU_socket. | `socket.cpp` |
| 12 ✅ | **Socket engine placement + shared L2 read port.** N engines at socket scope; their operand reads funnel through one arbiter into a single `l2arb` row so `kL2Rows` stays independent of socket count. Response routing: engine id in the **high** tag bits (arbiters add bits at the LSB — `cache.cpp:1243`). | `socket.cpp`, `cluster.cpp`, `dtcu_tma.*` |
| 13 ✅ | **Perf aggregation.** Socket and cluster engines report separately. | `csr_unit.cpp`, `cluster.cpp`, `socket.cpp` |
| 14 ✅ | **cgo27 modes.** Drop the DTCU_TMA mode; make modes for DTCU_cluster and DTCU_socket. Delete the NO_TMA tripwire ([`main.cpp:644-650`](../main.cpp#L644)). **Keep `DTENSOR_FLAG_NO_TMA` in the ISA** — the engine paths stay, only the harness mode goes. | `main.cpp`, `k_dtcu.h` |
| 15 ✅ | **Remove `-s`.** With two engines there is no single native tile to multiply, so the flag has no definition left (§3.1). Drop `-s`, `SIZE_MULT`, `g_size_mult`, and the `size=` field from the `[MOTI]` line; `-M/-N/-K` become the only way to set a shape, defaulting to one cluster tile. | `main.cpp`, `Makefile` |
| 16 ✅ | **Sweep scripts.** Two changes. (i) `MODES` is hardcoded in both (`sweep_exp1.py:21`, `sweep_exp2.py:20`) and feeds CSV headers, so a stale table silently mislabels results. (ii) Both drive `-s` and parse `size=`; the size ladder moves into the scripts, which expand each rung to `-M/-N/-K`. | `sweep_exp*.py` |
| 17 ✅ | **Test coverage.** All three tests moved to `dtensor_check()`. `dtcu_basic` and `dtcu_compare` now run **both** variants, and `dtcu_compare` additionally asserts the two engines produce **byte-identical** D from the same descriptor. New `dtcu_xcore` covers the cross-core completion path (§4.1). | `dtcu_basic/`, `dtcu_compare/`, `dtcu_xcore/`, `k_dtcu.h` |

---

## 2.1 Correction to item 12: no engine id in the tag

Item 12 above specifies "engine id in the **high** tag bits". That turned out to be
unnecessary, and the reasoning behind it was half right in a way worth recording.

The half that was right: arbiters add their bits at the **LSB**, so you cannot put an id
there. The half that was wrong: they do not merely add bits, they *shift the tag left*
and OR the input index in on the way down (`TxRxArbiter`, `types.h`), then shift right by
the same amount on the way back. The round trip is therefore lossless and
**self-routing** — a response returns to the exact input that issued it, carrying the tag
the requester originally wrote. Since the N socket engines fan in through a real
`MemArbiter`, each engine's `mem_rsp_in` only ever receives its own responses and no id
is needed at all.

What the shifting DOES require is *headroom*: a tag whose high bits are already occupied
loses them off the top of the `uint32_t` going down and can never get them back. An
engine id in the high bits would have been destroyed by the first arbiter — the exact
opposite of what item 12 assumed. So instead of adding an id, the implementation MASKS
the tag allocation to 16 bits, well below what the deepest plausible fan-in consumes
(socket arbiter + l2arb + the L2's bank crossbar ≈ 7 bits here). Tag 0 is skipped,
because `main_done()` is `pending_tag_ == 0` and a masked counter can now wrap onto it
where a free-running one could not.

## 2.2 The cgo27 baseline moved, and mostly not because of the DTCU

Item 12 exposed a pre-existing bug in `Cluster::Impl`'s L2 wiring. `TxArbiter` serves
input `i` only from output `i / R`, where `R = 1 << log2ceil(num_inputs / num_outputs)`.
The row indexing used throughout that constructor is `kL2Rows * port + row`, which agrees
with that grouping **only when `kL2Rows` is a power of two**. In the cgo27 config it was
3: sockets 0 and 1 shared one arbiter output together with DXA and the DTCU, and L2
request lane 3 was never driven at all. Rounding the row count up to a power of two fixes
it, and each socket now gets its own lane.

Two independent effects therefore move the cgo27 numbers, and neither is the socket
engine doing work (nothing submits to it in that harness's cluster modes):

1. **The dcache requester slot (item 11).** At `SOCKET_SIZE=1` this takes `CacheCluster`
   from 1 input to 2, replacing a bypass link with a real arbiter and adding a tick to
   every dcache request. This is a config-specific cliff — larger sockets already had an
   arbiter — and it slows the memory-bound modes.
2. **The row-count padding (item 12).** Isolated by building with
   `-DVX_CFG_EXT_DTCU_SOCKET_DISABLE`, which applies the padding without the dcache slot:
   the DTCU modes then come out **bit-identical to the old baseline**, confirming the
   cluster engine itself is untouched by either change.

The direction is mixed per mode rather than uniformly better, because
`VX_CFG_L2_NUM_BANKS` is 1 here: freeing a request lane redistributes arbitration into
the same single-bank bottleneck instead of adding bandwidth. That is the same L2 width
issue recorded as deferred in §3.1, now with a concrete cost attached.

| mode | before | after |
|---|---|---|
| 0 SIMT | 188712 | 190995 |
| 1 TCU | 14533 | 14626 |
| 2 TCU+DXA | 17492 | 15468 |
| 3 DTCU_cluster | 25057 (was mode 4) | 25061 |
| 4 DTCU_socket | — | 25553 |
| 5 TCU-pipe | 18142 | 17351 |
| 6 TCU+DXA-pipe | 19920 | 23170 |

**Any measurement taken before `fc39497b3` must be re-run before it is compared with one
taken after.**

---

## 2.3 Post-implementation audit (`d666f4159`, `9d1e968fd`, `29d8efa45`)

With everything passing, the submit and completion paths were audited adversarially —
48 agents attacking the claims "any core can submit to either engine" and "any core can
observe completion at any time". Nothing the tests exercise was wrong. Everything below
is a configuration the suite does not build, which is exactly why it survived.

**One item was specified backwards in this document.** Item 4 gave the cluster engine
`NUM_SOCKETS × 2` queue entries and the socket engine `NUM_CORES × 2`. The sharer sets are
the other way round: the cluster engine is shared by every core in the cluster, the socket
engine only by its own socket's cores. At `SOCKET_SIZE > 2` the cluster queue was therefore
*smaller* than its own sharer set — the starvation the depth exists to prevent. It matters
more than it looks, because acceptance is not fair once the queue overflows: SimObjects
tick in creation order, so the lowest-index core wins the last slot every time. (Once
queued, service is strict FIFO.) Now cluster `NUM_CORES × 2`, socket `SOCKET_SIZE × 2`.

**Three latent bugs, all now compile-time failures rather than silent ones.**

| trigger | old behaviour | now |
|---|---|---|
| `SIMD_WIDTH < NUM_THREADS` | The dispatcher splits one instruction into a trace copy per SIMD group, and the DTCU submit branch — the only side-effecting SFU op with no sop/eop guard — re-entered on each: one start instruction, N submissions, N runs of the same GEMM. CI builds `SIMD_WIDTH=1` but never with the DTCU. | Submit on the `sop` packet only, ticket latched per warp. Not an `eop` guard: the op is warp-scalar, so every packet's lanes must write back the *same* ticket or a warp-wide retry loop diverges. |
| cluster-only build, L2 off | `socket.cpp` asserted L2 was present; the cluster engine had no counterpart, so the build compiled clean (L2 defaults off) and then spun forever on a flag that could never arrive. | Compile error. The real requirement is a shared cache *below* the per-socket dcache, so the assert is `L2 ‖ L3` — with L2 off and L3 on, `l2cache_` degrades to a pass-through and both the flag and the consumer's AMO resolve at L3. |
| `SOCKET_SIZE ≥ 8` | `CacheCluster` gives each lane a `MemArbiter(inputs → NUM_DCACHES)`, and `TxArbiter` serves input *i* only from output *i/R* with *R* a power of two, so all inputs are served only when `I ≤ O·R`. The core-only shape satisfies that exactly; the DTCU's extra requester slot is the one that falls off the end. At `SOCKET_SIZE=8` input 8 is never arbitrated — the engine hangs in its first D store. | Compile error naming the escape hatch. There is no free fix inside `CacheCluster`: raising *R* to cover the 9th input collapses all 8 cores onto one cache unit. `-DVX_CFG_NUM_DCACHES=1` restores a served mapping at any socket size. |

**One thing deliberately left alone.** `strsp` buys an acknowledgement from the *first*
cache a store reaches, not from the point of coherence — see §1.6c. For the socket engine
that is correct by construction for its own socket's consumers and upheld only by arbiter
priority for the cross-socket fallback. Fixing it properly means changing `need_core_rsp`,
upstream Vortex code shared by every cache level, to buy an ordering that currently holds.
Pinned with a `static_assert` instead.

## 2.4 Mode renumbering

Mode ids were assigned as paths were added, so the DTCU modes had landed at 3/4 —
*before* the two in-core pipeline modes they are meant to be read against. The list now
runs in-core → in-core+DXA → engine, with a hole where the old DTCU ids were:

| id | name | unit |
| --: | --- | --- |
| 0 | SIMT | cores, scalar MAC loop |
| 1 | TCU | cores, WMMA |
| 2 | TCU + DXA | cores, WMMA on DXA-staged smem |
| 3, 4 | *(reserved hole)* | — |
| 5 | TCU + DXA, 2-stage | cores, smem pipeline |
| 6 | TCU + DXA, 3-stage | cores, smem pipeline |
| 7 | DTCU_socket | 4 socket engines, D → the submitting core's L1 |
| 8 | DTCU_cluster | 1 cluster engine, D → L2 |
| 9, 10, 11 | *(planned)* | in-core TCU + engines on one GEMM |

3 and 4 are left empty rather than reused: they appear in every result table recorded
before this change, and silently rebinding them would make old and new logs collide
without any diff to show for it. `mode_state()` reports them `Reserved` and the runner
skips them, so a stale `-m 3` fails loudly instead of measuring mode 3's replacement.

9-11 are **numbered but not built**. A first attempt is described in §3.2.

## 2.5 The socket engines now run concurrently

§4.2 previously closed by admitting that the socket mode "leaves three of its four socket
engines idle — the multi-engine question is untested, not answered." It is answered now.

The kernel submitted one descriptor for the whole GEMM, and a socket engine is only
reachable from a core inside that socket, so exactly one of the four ever ran. The GEMM is
now split by rows and each socket's core builds and submits its own band:

```c
const uint32_t core = (uint32_t)vx_core_id();
if ((core % VX_CFG_SOCKET_SIZE) != 0) return;     // one submitter per socket
const uint32_t sock = core / VX_CFG_SOCKET_SIZE;
const uint64_t d = arg->desc_addr + (uint64_t)sock * sizeof(dtensor_desc_t);
moti_fill_desc(...);
moti_publish_desc(d);                             // fence + AMO -- see below
while (0 == dtensor_socket_start(d)) ;
```

Only the **row** origin moves per slice: A and C/D are row-major so a slice is a
contiguous band, and B is shared untouched. Each slice's D lands in the L1 of the socket
that computed it, which is the placement the variant exists to model — with
`SOCKET_SIZE=1` (the measured config, §1.2) that is one engine per core writing into
that core's own dcache.

**The KERNEL builds the descriptor, not the host,** and needs nothing added to
`kernel_arg_t` to do it: A/B/C/D and M/N/K are already there, the element format ids are
`constexpr`, and the engine's tile-N is a build constant. An earlier version of this
change added `socket_size`, `num_slices`, `m_tcu` and `ctl_addr` to carry the same
information; all four were removable, and §2.6 is why that mattered.

It is also the honest accounting: a host-staged descriptor is written before the launch
and costs **zero measured cycles**, hiding exactly the per-GEMM control cost §1.1 claims
the DTCU reduces. Building it in the kernel costs mode 7 a fixed ~500 cycles — +3.6 % at
128×64×32, +0.5 % at 512×256×128.

**A fence does not publish it.** Core stores are write-through and fire-and-forget —
nothing acknowledges them, the same property §1.6 needed `strsp` for — so `fence` has no
completion to wait on and the engine's descriptor read can pass the fill.
`moti_publish_desc` follows the fence with `dtensor_check()`'s AMO, which takes the
AmoProbe path and resolves at the LLC, forcing the fill out ahead of the start. Without
it, mode 8's four slices produced 6,144 errors — because **a zeroed descriptor is a valid
descriptor**: `fmt_d = 0` is fp32 so `init_tile_state_` accepts it, `M = N = K = 0`
retires instantly, and the engine sets `done`. Each of the three the engine read as
still-zero therefore satisfied its submitter's poll while computing nothing.

**A launch that misses a core is silent wrong output, not a hang.** Mode 8 kept its 1×1×1
launch after being switched to a per-core split, so three of four slices were never
submitted — again 6,144 of 8,192 elements wrong, with no hang, no timeout, and a plausible
cycle count. Both engine modes now launch `grid_dim = NUM_CORES`.

### The same split applied to the cluster engine costs, it does not pay

Mode 8 now also splits four ways, one descriptor per core into the single cluster engine's
queue. That isolates tiling from engine count, and the answer is unambiguous:

| mode 8 | 128×64×32 | 256×128×64 | 512×256×128 |
| --- | --: | --: | --: |
| 1 descriptor | 25,061 | 149,305 | 1,097,497 |
| 4 descriptors | 51,461 | 168,613 | 1,140,573 |
| cost | **2.05×** | 1.13× | 1.04× |

One engine gains no parallelism from four descriptors — only four `DESC_REQ`/`DESC_WAIT`
round trips and four pipeline fills, plus half-empty tiles when a quarter of M is under
the cluster's 64-row tile. The penalty is nearly all fixed, so it amortises from 105 % to
3.9 %. **Mode 7's advantage is the engine count, not the tiling.**

## 2.6 `kernel_arg_t`'s **size** perturbs every mode's cycle count

`common.h` carries a warning that inserting a field mid-struct shifts later offsets and
moves the numbers. Appending four fields at the end — the documented-safe position —
moved them anyway, because the struct grew 64 → 80 B and every kernel reads it:

| mode | 64 B struct | 80 B struct | Δ |
| --- | --: | --: | --: |
| 1 TCU | 14,626 | 14,487 | −1.0 % |
| 2 TCU + DXA | 15,468 | 17,912 | **+15.8 %** |
| 5 2-stage | 23,170 | 15,548 | **−32.9 %** |
| 6 3-stage | 17,351 | 17,526 | +1.0 % |
| 8 DTCU_cluster | 25,061 | 25,217 | +0.6 % |

Reverting the struct restored all five to the digit, so the struct is the cause and not
a coincident change. **The rule is stronger than the comment says: `kernel_arg_t`'s size
is part of the experiment's configuration.** Anything a kernel can derive from a build
constant or from `desc_addr` must not go in it.

**Mode 5 is bimodal, and that is a result about mode 5.** A 16-byte struct growth is not
supposed to be worth 33 %. Its stall profile against the modes either side of it:

| mode | cycles | instrs | `stall_lsu` | `stall_sfu` | unattributed |
| --- | --: | --: | --: | --: | --: |
| 2 single-buffer | 15,468 | 1,368 | 8,427 | 3,389 | 3,045 |
| **5 2-stage** | **23,170** | 1,632 | 8,485 | 3,041 | **11,062** |
| 6 3-stage | 17,351 | 1,608 | 7,308 | 1,022 | 8,307 |

Mode 5's per-unit stalls are within a few percent of mode 2's, yet it spends 7,700 more
cycles; the difference is in the 11,062 cycles attributed to no functional unit, i.e.
warps idle at a barrier. With two buffers the DXA transfer and the stage's compute are
close enough in length that which finishes first is decided by the instruction schedule,
and the struct's 16 bytes were enough to flip it. Three buffers give enough slack that it
stops mattering — mode 6's `stall_sfu` is a third of mode 5's.

So a 2-stage smem pipeline that is *slower than no pipeline at all* (23,170 vs mode 2's
15,468) is not a stable measurement of anything. **Mode 5 should not be quoted as a
single number.** It is not clear it should be a reported mode at all, as opposed to a
data point about how much slack the DXA path needs; that is §3.3.

## 2.7 One device program per mode

A mode's cycle count was depending on which OTHER modes existed. Adding modes 3 and 4 to
the shared `kernel.vxbin` moved mode 2 from 15,468 to 24,106 cycles with a **byte-identical**
`moti_tcu_dxa`: same 423 instructions, same `0x698` size, same 1,368 executed instructions,
same 1,120 instruction fetches. The only difference was the start address — icache set 41
became set 62 — and the average instruction-fetch latency, 54.0 → 101.8 cycles. Data-side
counters were unchanged.

Note the mechanism is **not** that unused kernels occupy cache: an icache only holds what
is fetched, and the other modes' code never is. What a bigger binary changes is the
ADDRESS of the code that *does* run, and therefore which set it maps to
(`set = (addr >> 6) & 63`) relative to the per-block spawn/dispatch runtime it shares the
cache with. That is ordinary set-associative conflict, not a Vortex defect — and it is why
the fix is layout control rather than a cache change. (Which of conflict or L2 queueing
dominates is not established: SimX exposes `ifetches`/`ifetch_lt` but no icache
hit/miss split.)

Each mode now builds `kernel_modes/kernel_m<N>.cpp` into its own `kernel_m<N>.vxbin`,
holding that kernel and nothing else. Every mode's code starts at `0x180000034` whatever
else exists, and each program is 536–2,940 B against the old combined 14,700 B.

**Verified, not assumed:** growing mode 3 with a dummy kernel and rebuilding moves no
other mode's entry address by a byte.

## 2.7b The merge with upstream moved the floor, and it moved it the engine's way

Every number in this RFC was re-measured after merging 445 upstream commits
(`00ea949a1`). The ordering changed, and the reason is worth recording because it is the
sharpest evidence §1.1 has produced.

Upstream rebuilt the memory path: L2 went from a 64 B line to a **sectored 128 B** one,
`LSUQ_IN_SIZE`/`LSUQ_OUT_SIZE` were replaced by a single `LSU_PENDING_SIZE` queue, and
roughly a thousand lines landed in `sim/simx/mem/`. At 128×64×32, mode 1:

| | pre-merge | post-merge |
| --- | --: | --: |
| loads | 7,936 | 7,936 |
| **average load latency** | 64.5 cyc | **351.4 cyc** |
| `stall_lsu` | 8,427 | 12,920 |
| cycles | 14,584 | 23,513 |

**Not one extra load — each costs 5.4x more.** Mode 1 spends 55 % of its cycles in
`stall_lsu`, so that is the entire 61 % slowdown.

**Mode 7 did not feel it, and got faster**: 14,389 -> 11,912. Its core-side counters are
`loads=116` with every stall category at zero. The GEMM traffic leaves through the
engine's own TMA port; the core submits a descriptor and polls. So a coarser memory
hierarchy taxes a core that issues every load and costs an engine nothing, while the
engine turns the wider line into bandwidth. Mode 7's margin over in-core WMMA at the
largest shape went 1.16x -> 1.27x.

**This is 1.1's argument arriving from the outside.** The case for the DTCU was control
cost per GEMM and completion a non-submitting core can observe. Add: the engine is
insulated from the memory hierarchy getting coarser, because it is not the core that
waits. Nothing in this branch was changed to produce that -- upstream changed the machine
and the engine came out ahead.

One ordering flipped. Mode 2 (TCU+DXA) used to beat mode 1 at 512x256x128 and now loses,
446,129 against 386,994: DXA stages through Local Memory but the fragments still reach the
TCU as LSU loads, so it pays the new latency on the fill and again on the smem read. That
is the same property 2.8 identifies as the reason the single-warp DXA modes never showed a
gain.

**A measurement limit, recorded rather than solved.** Modes 3/4/5/6 and 12/13 do not
complete at 512x256x128. Mode 4 scales linearly to 256x128x32 (231,610 -> 345,311 ->
581,372 cycles, 7/12/21 s of simulation) and then 384x192x32 does not finish in an hour;
K depth is not the cause (231,610 / 225,279 / 302,906 at K = 32/64/128). Their 512 column
is empty for that reason, not because it was skipped.

---

## 2.8 What a copy engine is worth, isolated (modes 12/13)

§4.2 reported the DXA modes landing within 7 % of mode 1, which stages nothing at all,
and that reading was real but it was a statement about the KERNELS, not about DXA. Modes
2/5/6 launch one warp per block: a warp stages a tile, issues one `mma_sync` against it
and discards it, so there is nothing to amortise the copy over, and the sixteen warps
resident on a core are sixteen unrelated CTAs each copying its own private tile. Three
things have to hold before a copy engine can pay, and those modes have none of them:

1. **Reuse** — the staged tile feeds more than one MMA.
2. **Warp specialisation** — a producer warp separate from the consumers, so the async
   copy overlaps compute. Modes 2/5/6 already contain `is_dxa = get_sub_group_id() == 0`,
   but with one warp per block producer and consumer are the same warp.
3. **The consumer reads shared memory directly.** `load_matrix_sync` moves the fragment
   into registers, so the LSU load COUNT does not drop — measured 49,632 → 47,520, 4 %.
   DXA only makes each load cheaper (95.5 → 65.8 cycles) while paying issue and barrier
   traffic on the SFU: `stall_sfu` 13,360 → 27,741.

Modes 12 and 13 have all three. A CTA of `ISSUE_WIDTH` warps stages one A tile spanning
all of them plus one B tile they all read, warp 0 issues the copy, and `wgmma_sync` takes
B as a shared-memory descriptor. 12 and 13 differ **only** in whether that copy is a DXA
descriptor or the CTA's own loads.

| | 128×64×32 | 256×128×64 | 512×256×128 |
| --- | --: | --: | --: |
| 12 DXA, C pass removed | 14,093 | 71,583 | 335,171 |
| 13 SW copy, C pass removed | 16,634 | 91,418 | 494,464 |
| **what DXA is worth** | **1.18×** | **1.28×** | **1.48×** |

**The engine does pay, and by more as the shape grows** — against 0.98–1.0× for the
single-warp modes. The variable is the kernel's shape, not the engine.

**The C pass, and why these two numbers are quoted with it removed.** A wgmma context
refuses to load an accumulator from memory
([`vx_tensor.h:789`](../../../../sw/kernel/include/vx_tensor.h#L789)), and the refusal is
correct: the warpgroup accumulator is distributed differently from a per-warp WMMA
fragment even at the same tile shape, so seeding it through the WMMA layout puts C in the
wrong lanes — 24,173 of 32,768 elements wrong, exactly one warp in four correct,
identically for both modes. `D = C + A*B` therefore splits into: accumulate A*B from
zero, store it, then read D, read C, write D. Four M*N accesses where the in-core modes
fuse C into the accumulator and make one. That is **58-79 % of these two modes**, measured
by compiling the pass out (`-DMOTI_WG_NO_C`, a build whose D is wrong on purpose).

**Worked around, not solved.** The fix is to combine C while the accumulator is still in
registers and store once — two M*N accesses instead of four, which is what CUTLASS does
in its Hopper epilogue: drain the accumulator through shared memory and apply
`alpha*AB + beta*C` on the way out. The Local Memory for it is there (a CTA uses 2.5 KB of
64 KB). Not done yet.

**Deeper staging makes it worse, and that is a result about the machine.** Holding two
K-steps per staged tile instead of one costs 1.40×/1.82× at 128×64×32 and 1.39×/1.83× at
256×128×64 — every shape, both modes. Local Memory is a **per-CTA** resource, so doubling
the stage halves the CTAs resident on a core, and halving the number of copies does not
pay for halving the latency hiding. Reuse has to grow along **N** — one staged tile
feeding several output tiles, leaving the stage size alone — not along K.

**The epilogue is free here and expensive for the engines.** Modes 12/13 at app 2 and app
6 land within 0.3 % of app 1 (257,844 and 258,601 against 258,543), because the C pass
they are already forced to make absorbs it. Modes 7/8 pay a second kernel launch for the
same thing: mode 7 goes 14,389 → 73,973 at app 2, 5.1×.

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

(Mode ids in this table predate §2.4; 3/4 are today's 8/7.)

### 3.2 Hetero modes 9-11: numbered, attempted, not working

Modes 9-11 split one GEMM's rows between the in-core TCU and the engines — the
configuration the design is actually for, since the point of an engine is that the cores
keep working while it runs. A first implementation is **not in the tree**: the host built
per-slice descriptors and a claim counter, the kernel ran WMMA over the leading rows and
submitted the rest, and it did not work. The claim flags came back set and the
descriptors were correct in memory, but the engine reported `active=0, done=0` — it never
started. Ruled out: warp divergence around the submit (it fails at `block_dim=1` too), a
stale device binary, and the host-side row arithmetic.

It was backed out rather than left disabled because the version that existed depended on
the four `kernel_arg_t` fields §2.6 shows must not exist. Reimplementing it means finding
a way to reach the engine from inside a divergent WMMA kernel that needs no new argument
fields — which is a real design question, not a port of the working code.

### 3.3 Is mode 5 a mode?

§2.6 shows the 2-stage smem pipeline is bimodal: same stall profile as unpipelined, 50 %
more cycles, and a 16-byte struct change flips it. Either it gets reported as a range
with the mechanism stated, or it gets dropped and the DXA pipeline story is told by mode
6 alone with mode 5 as the "two buffers is not enough slack" data point. Not decided here
because it is a presentation call, not an implementation one.

---

## 4. Verification

Everything before item 9 keeps a single engine and had to be **numerically inert**: all
three tests report identical cycles and MPM counters to the baseline. That held.

- Items 1-8: verified inert against `dtcu_basic`, `dtcu_compare`, `cgo27_motivation`.
- Item 1: `store_drain` increased as predicted, once it began counting the real
  issue-to-ack latency of the final store.
- Items 9b/12a (`e827b57b9`): inert to the cycle on all three tests, even though they
  rewrote the operand-SRAM indexing — `dtcu_basic` 8237, `dtcu_compare` 158578/27021,
  cgo27 188712/14533/17492/36465/25057/18142/19920.
- Items 11-13 (`fc39497b3`): NOT inert, deliberately. See §2.2.

### 4.1 What the tests now actually prove

- **Both engines exist and are distinct.** `dtcu_basic` runs one native tile on each:
  cluster 64×32 in 8293 cycles, socket 32×16 in 4191. Built one variant at a time
  (`-DVX_CFG_EXT_DTCU_{SOCKET,CLUSTER}_DISABLE`) each build runs its own engine and
  reports the other as skipped, so the ISA gate is real rather than decorative.
- **They compute the same thing.** `dtcu_compare` submits the same descriptor to both
  and asserts D matches **byte for byte**, not to within a ULP. Identical arithmetic on
  identical inputs means any difference at all is a geometry bug, and this is the
  assertion that would catch a wrong per-engine `shm_b_` stride.
- **The counters are separated.** The two MPM classes report different values from the
  same run: cluster `op_reqs=1792`, socket `op_reqs=2560`. Same GEMM, same 32768 FEDPs;
  the socket engine's smaller tile simply gets less operand reuse. Had item 13 been
  skipped, both would have read the cluster engine's numbers.
- **Ragged shapes work on both.** `-m 4 -M 100 -N 96 -K 20` passes with zero errors, so
  the hardware OOB clamp holds at the socket engine's 32×16 tile too.
- **Cross-core completion works** — the property nothing in the tree tested before.
  `dtcu_xcore` (4 cores, `SOCKET_SIZE=2`) has one core submit and the others observe,
  and asserts that a non-submitting core actually did observe. It reported
  `submitter=core1 observers=3 cross_core=3 cross_socket=2` for both engines: a
  different-socket consumer saw completion and read correct D.

  Roles are assigned by atomic ticket rather than by core id, and that mattered — the
  submitter landed on core **1**, so a test keyed on `vx_core_id() == 0` would have
  deadlocked.

  **Negative control.** Replacing `dtensor_check()`'s AMO with a plain volatile load
  makes the test fail, which is what proves it is not vacuous. Two of the three
  consumers hit the spin limit and never saw the flag; the third got lucky on timing and
  did. That the failure is a *race* rather than deterministic is the strongest argument
  for the AMO: a plain load is not merely slower, it is intermittently wrong. The same
  experiment showed the original 20,000,000-iteration spin bound never terminates inside
  a simulator, so the bound is now 200,000 — still ~40× the whole run, but it reports a
  hang in seconds instead of being indistinguishable from one.

### 4.2 Measuring at three shapes changed what the numbers support

The full result table lives in [README.md](../README.md#measured-results--2026-08-05) —
seven modes × three shapes, with cycles, aggregate `MAC/cyc`, and `MAC/cyc` per **unit**
(a core for the in-core modes, all 4 active; an engine for the DTCU modes, of which
exactly 1 is). Recorded here is only what it changes about this RFC's claims.

**A single shape would have misled us, in the engine's favour.** The RFC was developed
against 128×64×32, which fills only 32 of the cluster's 64 warp slots. That handicaps the
in-core path, and the aggregate ratio against mode 1 is not monotonic in size —
1.71× → 1.54× → 2.85×. Any figure quoted from one shape is quoting an artifact of that
shape's occupancy.

**The two paths are bound by different walls, and growing the GEMM only lowers one.**
In-core is memory-latency bound: `stall_lsu` is 93-94 % of cycles, so cycles track total
load latency. From 256×128×64 to 512×256×128 its throughput doubles, and neither half of
that is occupancy — both shapes are already full. It is (a) ×1.22 fewer loads per MAC,
because doubling K amortizes the per-block C load and D store over twice the MACs, and
(b) ×1.74 cheaper loads, because the grid grows 8×16 → 16×32 blocks so each A/B panel is
reused by twice as many blocks and DRAM reads per L2 read fall 58 % → 26 %.

The engine sees none of it. At the same shape it is **compute bound** — `compute` is
95.9 % of its cycles, `tma_mem_wait` 6.9 %, and its loader sits idle 62 % of the time
waiting for a free operand buffer. Better L2 reuse buys it ×1.09 where it buys the cores
×2.02.

**What this does and does not say about the design.** Per unit the cluster engine is ahead
of a core at every shape — 2.3× at the smallest, 2.6× in the middle, 1.4× once the cluster
is full. Against the *whole* machine that one engine is not competitive and the gap grows
with size (1.71× → 1.54× → 2.85× slower than mode 1), which is arithmetic, not a defect:
one MAC array against four cores' worth.

**With the socket engines tiled (§2.5), that conclusion no longer generalises to the
design — only to the cluster variant.** Four socket engines are the fastest mode at all
three shapes:

| | 128×64×32 | 256×128×64 | 512×256×128 |
| --- | --: | --: | --: |
| 0 SIMT · 4 cores | 190,995 | 1,145,460 | 9,581,708 |
| 1 TCU · 4 cores | 14,584 | 97,240 | 377,131 |
| 2 TCU+DXA · 4 cores | 15,647 | 100,281 | 354,814 |
| **7 DTCU_socket · 4 engines** | **14,389** | **56,449** | **325,477** |
| 8 DTCU_cluster · 1 engine, 4 descriptors | 51,305 | 168,725 | 1,140,949 |

Measured with one device program per mode (§2.7), so a mode's number no longer depends on
which other modes are in the tree. 27 runs, 27 passes, zero mismatches.

51.55 MAC/cyc against mode 2's 47.28 at the largest shape. Per unit the ordering is the
reverse — the cluster engine's 64×32 tile gets 4× the operand reuse of the socket
engine's 32×16, so it is the most efficient single unit in the table (14.70 against a
socket engine's 12.89 and a core's 11.82) and still loses aggregate by 3.5×. **Tile
efficiency and throughput point in opposite directions, and the placement decision is
which of the two is being bought.**

The third row isolates why. Applying the *same* four-way row split to the single cluster
engine makes it slower at every shape — 2.05× at the smallest, 1.04× at the largest —
because four descriptors into one engine add no parallelism, only four descriptor fetches
and four pipeline fills. **Mode 7's win is the engine count, not the tiling.**

This also reframes the widening result, and the two placements turn out to respond to
width in opposite ways. Scaling all three terms of the compute model together at
512×256×128:

| width | 7 socket ×4 | speedup | 8 cluster | speedup |
| --- | --: | --: | --: | --: |
| 1× (16/2/2) | 324,469 | — | 1,140,573 | — |
| 2× (32/4/4) | 243,869 | 1.33× | 663,865 | 1.72× |
| 4× (64/8/8) | 223,977 | 1.45× | 411,997 | **2.77×** |

**The cluster engine is compute-bound; the socket engines are not.** A 4× wider datapath
pays the cluster engine 2.77× and the socket engines 1.45×, and the socket variant has
visibly saturated by then (2× → 4× buys 1.09×). Its 32×16 tile is a quarter the area, so
per tile a far larger share of its time is descriptor fetch, operand fill and store
drain — none of which a wider MAC array touches. **Widen the cluster engine; replicate the
socket engine.**

At equal silicon replication still wins: four unmodified socket engines reach 51.71
MAC/cyc against a 4×-widened single cluster engine's 40.72, and those are the same budget
either way (4 MAC arrays and 4 accumulators). The default engine is not undersized, it is
**under-replicated**. At 4× the socket variant reaches 74.91 MAC/cyc — 1.60× the whole
four-core cluster.

None of it displaces what §1.1 claimed — control cost per GEMM, freeing the cores for
other work, and completion a non-submitting core can observe. Those remain the case for
the DTCU; the throughput result is now simply not an argument against it.

**Superseded: the socket engines no longer sit idle.** This section originally closed by
noting that the harness submitted one descriptor from one thread, so three of the four
socket engines never ran, and that "the multi-engine question is untested, not answered."
§2.5 answers it: the GEMM is split into one row-band per socket and each socket submits
its own, so all four run concurrently. Every mode-7 number in the result table is from
the tiled version; the single-descriptor numbers this section was written against are
gone. What remains untested is the *hetero* question — cores and engines working on one
GEMM at the same time — which is §3.2.

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
