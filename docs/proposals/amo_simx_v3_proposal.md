# AMO (RVA) SimX v3 — Proposal

**Date:** 2026-05-01
**Status:** v1.0 — Ready for implementation
**Owners:** SimX team
**Related:**
[simx_v3_proposal.md](simx_v3_proposal.md),
[dxa_simx_v3_proposal.md](dxa_simx_v3_proposal.md),
`feedback_simx_tlm_design`, `feedback_simx_perf_goal`,
`project_simx_tlm_refactor`.

---

## Summary

Add support for the RISC-V `A` extension (LR, SC, AMO\*) to SimX v3
under the existing `EXT_A_ENABLE` build flag. Atomicity for global
addresses is provided by a per-bank `AmoUnit` instantiated **only at
the last-level cache (LLC)** of whatever hierarchy SimX is configured
with (L1-only, L1+L2, or L1+L2+L3). Caches above the LLC operate in
*AMO-passthrough* mode: they probe-and-invalidate the affected line
and forward the request downstream. All atomic state — RMW commit,
reservation table — lives at the LLC. AMOs targeting Shared (LMEM)
or IO regions are out of scope (§6).

---

## 1. Constraints (load-bearing)

The same three rules that govern the rest of SimX v3 govern AMO
support. Any design that breaks one is wrong.

1. **NoC-only memory access.** Every AMO byte read or written flows
   through `LsuReq` / `LsuRsp` (LSU↔switch) and `MemReq` / `MemRsp`
   (cache↔memory). No `core->mem_read`, no `core->mem_write`, no
   shadow `MemoryUnit::amo_reserve` table mutated from `Emulator`.
2. **Functional and timing coupled.** The value `rd` receives is the
   value the cache line carried at the cycle the bank's RMW
   committed. The byte that lands in DRAM is the byte the bank wrote
   (write-back: defer; write-through: send the new value). Nothing
   is precomputed at execute time.
3. **Mirror RTL where it exists; for AMO, RTL has nothing to copy.**
   This is a clean-sheet TLM design. The constraint becomes: build
   the AMO engine at the LLC — the single point of serialization
   across all cores in a non-coherent hierarchy — so it composes with
   the existing MSHR / fill / replay machinery instead of side-
   stepping it.

The legacy SimX (in `vortex_main`) violates Rule 1 wholesale: AMO
bodies execute via `mem_read` / `mem_write` against `MemoryUnit`,
with a single-entry per-MMU reservation
([emulator.cpp:534-547](../../../vortex_main/sim/simx/emulator.cpp#L534),
[execute.cpp:872-1048](../../../vortex_main/sim/simx/execute.cpp#L872),
[common/mem.cpp:281-304](../../../vortex_main/common/mem.cpp#L281)).
**Do not port it.** The compute kernels (read-modify-write logic per
op-type) are correct and reusable; the placement and reservation
storage are not.

---

## 2. Design decisions

The starting sketch was: one `AmoUnit` per cache bank, with a
reservation table of size `Q = VX_AMO_RS_SIZE` (default 2), and
"extend cache requests with additional info." Eight decisions
refined that sketch into the design in §3.

| # | Question | Decision |
|---|---|---|
| D1 | Reservation entry shape and size? | Entry is `{hart_id, line_addr, valid, lru}`. `Q` per LLC bank, replacement = LRU. RVA permits *spurious* SC failure but not spurious success — small `Q` is conformant. Floor `Q ≥ 2`; default `4`. |
| D2 | Where does the AMO engine live in a multi-level hierarchy? | At the **LLC only**. Vortex caches are non-coherent; the LLC is the single point of serialization. Intermediate caches operate in passthrough mode (§3.1). |
| D3 | What additional fields does `MemReq` need? | `op` already encodes AMO\_\* ([types.h:960](../../sim/simx/types.h#L960)). Add `amo_width`, `amo_rhs`, `hart_id`. `MemReq::write` is unconditionally `false` for AMOs (a missing line under SC must miss-and-return-failure, not write-and-succeed). |
| D4 | When does the RMW happen relative to fill / replay? | One bank cycle on hit (matches a write-hit). On miss: enqueue MSHR (matches a read-miss); fill arrives; Replay runs the commit. No phantom RMW cycle, no separate channel. |
| D5 | What address types are in scope? | Global only. Shared (LMEM) atomics are a separate proposal — different hardware site, different reservation domain. The local-mem switch hard-asserts on a Shared-AMO so divergence is loud. IO atomics: not architecturally specified; bypass path asserts. |
| D6 | Coalescing same-line AMO lanes? | No. RVA gives no commutativity guarantee across operands. Each AMO lane is a distinct bank request; the bank xbar serializes same-line lanes at `pipe_req_`. |
| D7 | How are reservations broken by ordinary stores from other harts? | LLC bank invalidates every reservation entry whose `hart_id` differs on every committed write reaching the LLC tag array. Correctness depends on **every store from a hart reaching the LLC** — i.e. write-through above the LLC. Build assert in `processor.cpp` enforces this (§3.1). |
| D8 | How is SC's success/failure signal carried back? | In the lane payload. Bank writes 0 (success) or 1 (failure) at the AMO's byte offset of an otherwise-zero `mem_block_t`. The LSU's existing load formatter sign/zero-extends correctly because the response goes through the same path as a load-word. |

---

## 3. Target architecture

### 3.1 Cache-hierarchy placement (load-bearing)

Vortex's cache hierarchy is configurable: L1 (per socket) → optional
L2 (per cluster) → optional L3 (system) → memory. The "LLC" is
whichever level is the highest enabled — `L3_ENABLE` ? L3 : `L2_ENABLE`
? L2 : L1.

**The AMO engine and the reservation table live at the LLC, and only
at the LLC. Caches above the LLC operate in AMO-passthrough mode.**

#### 3.1.1 Why the LLC

Vortex caches are **non-coherent** — no MESI, no directory, no snoop.
Two cores doing `AMOADD` on the same address from their private L1s
would each operate on a private stale copy of the line: silent loss
of atomicity. The LLC is the single level at which "the line at this
address" is unambiguous, so the RMW must commit there.

#### 3.1.2 Why intermediate caches need pre-flush, not just bypass

After the LLC commits the AMO, any clean copy still cached at L1/L2
serves stale data on the next normal load. Any dirty copy at L1/L2
is fresher than the LLC's bytes; bypassing without writeback would
let the AMO operate on stale LLC bytes and lose the dirty update. So
non-LLC caches must, on the way down: writeback-if-dirty, invalidate
the line, then forward the AMO via the bypass path. On the way back
up, response is forwarded without installing — same shape as the
existing IO bypass at [cache.cpp:1073](../../sim/simx/mem/cache.cpp#L1073)
plus the new probe-and-invalidate step.

```
L1 (non-LLC):
   AMO_in  ──► tag-probe ──► writeback if dirty
                         ──► invalidate line
                         ──► forward AMO (bypass)
   AMO_rsp ──► forward up (no install)

L2 (non-LLC, only when L3_ENABLE):
   same as L1.

L3 (LLC, when L3_ENABLE) — or L2 (LLC if L3 off) —
or L1 (LLC if both off):
   AMO_in  ──► full bank pipeline (tag, MSHR, fill,
               RMW commit via AmoUnit, reservation
               table update, response).
```

#### 3.1.3 The same `Cache` class is used at every level — the role is a flag

[mem/cache.cpp](../../sim/simx/mem/cache.cpp) defines a single
generic cache. The L1 dcache, L2, and L3 are all instances of this
class, differing only by their `Cache::Config`
([socket.cpp:52](../../sim/simx/socket.cpp#L52),
[cluster.cpp:53](../../sim/simx/cluster.cpp#L53),
[processor.cpp:63](../../sim/simx/processor.cpp#L63)). The cache
class itself has no idea where it sits in the hierarchy — the
instantiator does, and must tell it.

```cpp
struct Cache::Config {
  // ... existing fields ...
  bool is_llc = false;   // when true: AMO commits here; reservation
                         // table active. when false: AMO is probe-
                         // invalidate-and-bypass.
};
```

Set true at exactly one site:

| `EXT_A_ENABLE` | `L3_ENABLE` | `L2_ENABLE` | Site that sets `is_llc=true` |
|---|---|---|---|
| true  | true  | *    | `processor.cpp` (L3 ctor)  |
| true  | false | true | `cluster.cpp` (L2 ctor)    |
| true  | false | false| `socket.cpp` (dcache ctor) |
| false | *     | *    | nowhere (flag stays false) |

A cache configured with `bypass = true` (transparent arbiter, no
banks) has no AMO behavior of its own — it forwards `core_req_in`
straight to the next level, so the AMO walks down to the first
non-bypass cache that has `is_llc = true`. The build wiring must
guarantee that exactly one cache on the AMO path is so flagged.

#### 3.1.4 LLC writeback policy is independent

The LLC may be configured **write-back or write-through** — both are
correct for AMO. The bank's RMW path uses the existing store-commit
branch: `mem_req_out.send(line)` for write-through, `line.dirty =
true` for write-back. AMO ordering and reservation tracking key on
the LLC tag array, not on DRAM contents, so when bytes hit DRAM is
irrelevant.

#### 3.1.5 Write-through-strictly-above-LLC invariant

Putting the reservation table at the LLC works *if and only if* the
LLC sees every store that could break a reservation. A write-through
path from each core down to the LLC guarantees this. Current Vortex
defaults satisfy it: `DCACHE_WRITEBACK = 0`
([VX_config.toml:183](../../VX_config.toml#L183)),
`L2_WRITEBACK = 0`.

**AMO support requires every cache *strictly above* the LLC to be
write-through.** A write-back intermediate cache could absorb a
normal store from hart B to line X without the LLC learning about
it; a subsequent `SC` from hart A on the same line would spuriously
succeed against a stale view — RVA spec violation (RVA permits
spurious failure, not spurious success).

The LLC itself is exempt. Build-time assert in `processor.cpp` ctor:

```cpp
if (EXT_A_ENABLED) {
  if (L3_ENABLED) {
    // L3 is LLC. L1, L2 must be WT.
    static_assert(!DCACHE_WRITEBACK, "AMO requires write-through L1");
    static_assert(!L2_WRITEBACK,     "AMO requires write-through L2");
  } else if (L2_ENABLED) {
    // L2 is LLC. L1 must be WT.
    static_assert(!DCACHE_WRITEBACK, "AMO requires write-through L1");
  }
  // L1-only: L1 is the LLC; its writeback policy is unconstrained.
}
```

Reservation invalidation at the LLC bank is then driven by every
committed bank write that reaches the LLC's tag array — exactly the
set of writes visible to other harts:

- ordinary writethroughs from L1/L2 above,
- AMO RMW commits (which break other harts' reservations on the same
  line),
- LLC's own evictions (write-back) and writethroughs (write-through)
  go to DRAM, *not* to the LLC tag array — they do not trigger
  reservation invalidation. (Consistent with RVA: eviction is not a
  write.)

### 3.2 Module sketch

```
┌──────────────────────────────────────────────────────────────────┐
│ Core                                                             │
│                                                                  │
│  Execute (execute.cpp)                                           │
│    AMO branch: NO mem_read/mem_write. Set trace->op_type to      │
│    AmoType::*; rs2_data flows as the RMW operand. The AMO        │
│    return value is filled by the LSU response path, like a load. │
│                                                                  │
│  LsuUnit                                                         │
│    Recognises AmoType in addition to LsuType.                    │
│    compute_addrs(): per-lane (addr, size, rs2_data, tid).        │
│    process_request_step(): AMO miss-or-hit allocates MSHR like   │
│    a load (we always need a return). Packs LsuReq.is_amo,        │
│    .amo (op, width), .amo_rhs[lane].                             │
│                                                                  │
│  LocalMemSwitch (mem/local_mem_switch.cpp)                       │
│    Routes by addr-type. Shared+AMO is hard-asserted (§3.13).     │
│                                                                  │
│  LsuMemAdapter (mem/lsu_mem_adapter.cpp)                         │
│    Lane → MemReq fan-out. Sets MemReq.op = AMO_*,                │
│    .amo_width, .amo_rhs, .hart_id (derived from cid+wid+lane).   │
│    MemReq.write = false even for SC.                             │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ Cache (mem/cache.cpp)                                            │
│                                                                  │
│  Cache::Config gains is_llc : bool. Sites in §3.1.3.             │
│                                                                  │
│  CacheBank: behavior on AMO depends on is_llc.                   │
│                                                                  │
│  ── is_llc = false (passthrough) ──────────────────────────      │
│   AMO core_req_in → schedule AmoProbe pipe entry.                │
│   processRequests AmoProbe:                                      │
│     hit & dirty → emit writeback, invalidate                     │
│     hit & clean → invalidate                                     │
│     miss        → no-op                                          │
│     forward original AMO MemReq via nc_mem_arbs (bypass).        │
│   AMO MemRsp from below → forward to core_rsp_out, no install.   │
│                                                                  │
│  ── is_llc = true (commit) ────────────────────────────────      │
│   Bank owns AmoUnit (sim/simx/amo/).                             │
│   bank_req_t gains is_amo, amo_op, amo_width, amo_rhs, hart_id.  │
│                                                                  │
│   Core+is_amo:                                                   │
│     hit  → AmoUnit::compute(line, rhs) → {new_word, ret_word}    │
│            store path (LR or SC-fail: skip; else line_merge +    │
│            writethrough/dirty).                                  │
│            core_rsp_out.send({ret_word at byte_off}).            │
│            On store: amo_unit.invalidate(line_addr, hart_id).    │
│     miss → MSHR enqueue (read-miss path). Fill triggers Replay;  │
│            Replay+is_amo runs the commit.                        │
│                                                                  │
│  AmoUnit (sim/simx/amo/amo_unit.{h,cpp})                         │
│    compute(op, width, old, rhs) → {new, ret}     // pure         │
│    reserve(hart_id, line_addr)                                   │
│    check(hart_id, line_addr)        → bool       // for SC       │
│    invalidate(line_addr, except_hart_id)                         │
│    reservations_: vector<Reservation>, size = AMO_RS_SIZE        │
└──────────────────────────────────────────────────────────────────┘
```

### 3.3 New files

```
sim/simx/amo/
  amo_unit.h        // class AmoUnit — compute kernel + reservation table
  amo_unit.cpp
  amo_ops.h         // pure helpers: amo_compute(op, width, lhs, rhs),
                    //               sext/zext, AmoType ↔ MemOp mapping
```

`AmoUnit` is **not** a `SimObject` — it has no channels and no ticking
state of its own. It is a per-bank helper held by `CacheBank` and
exercised synchronously from `processRequests()`. AMO commit happens
in the same bank cycle as a regular store commit; a separate ticked
object would add a phantom cycle that has no RTL counterpart.

### 3.4 Channel-type extensions

```cpp
// sim/simx/types.h

struct AmoInfo {
  AmoType  op;      // existing enum (LR, SC, AMOADD, AMOSWAP, …)
  uint8_t  width;   // 2 (.W) or 3 (.D); matches IntrAmoArgs.width
};

struct LsuReq {
  // ... existing fields ...
  bool                  is_amo = false;
  AmoInfo               amo;        // valid iff is_amo
  uint32_t              wid    = 0; // needed by LsuMemAdapter for hart_id
  std::vector<uint64_t> amo_rhs;    // per-lane rs2 (size = num_lanes)
};

struct MemReq {
  // ... existing fields ...
  // op already encodes AMO_LR / AMO_SC / AMO_ADD …
  uint8_t  amo_width = 0;   // 2|3 when op is AMO_*
  uint64_t amo_rhs   = 0;
  uint32_t hart_id   = 0;
};
```

`MemReq::write` is unconditionally `false` for AMO requests, including
SC. The cache must treat every AMO as a "read with side-effects":
allocate a fill on miss, respond with data, optionally schedule a
writeback after the RMW. Treating SC as `write = true` would route it
into the write-through write-miss fast path (no fill, no MSHR), where
a missing line would write-through-and-succeed instead of
miss-and-return-failure.

### 3.5 hart_id encoding

```cpp
// sim/simx/types.h
static inline uint32_t make_hart_id(uint32_t cid, uint32_t wid, uint32_t tid) {
  // Globally unique per simulated hart. Bit layout: thread in low
  // bits, then warp, then core. Lower bits carry more entropy → fewer
  // hash collisions on the small per-bank reservation table.
  return (cid << (LOG_NUM_WARPS + LOG_NUM_THREADS))
       | (wid << LOG_NUM_THREADS)
       | tid;
}
```

Derived per-lane in `LsuMemAdapter` from `(LsuReq.cid, LsuReq.wid,
lane_index)` — no per-lane field on `LsuReq`. `hart_id` rides on
`MemReq` from the adapter onward.

### 3.6 Cache `bank_req_t` extensions (LLC bank only)

```cpp
// inside cache.cpp
struct bank_req_t {
  // ... existing fields ...
  bool      is_amo    = false;
  AmoType   amo_op    = AmoType::LR;
  uint8_t   amo_width = 0;
  uint64_t  amo_rhs   = 0;
  uint32_t  hart_id   = 0;
};
```

Captured from incoming `MemReq` in `processInputs()` (the existing
`bank_req.write = core_req.write` line gets a sibling block guarded
on `core_req.op` ∈ AMO_\*).

### 3.7 LLC bank request flow

MSHR reservation policy at processInputs() time (matches the existing
conservative pre-reservation for predicted-miss-path requests):

```
use_mshr = !core_req.write              // any read
        || config_.write_back           // any write-back write
        || (core_req.op is AMO_*);      // any AMO (may miss → fill)
```

Pipeline body:

```
processInputs():
  ... existing fill-bypass and Replay/Core scheduling unchanged ...
  if (core_req.op is AMO_*):
    capture amo_* fields into bank_req_t.

processRequests():
  Replay or Core, when bank_req.is_amo:
    tag-match line.
    if (Replay) assert(hit);   // fill installed it
    if (Core && miss):
      MSHR enqueue (read-miss path); Fill→Replay path runs commit.
      break;

    // — commit —
    // Decide do_store first so we can gather all stall conditions
    // before any mutation (mirrors the existing write-hit pattern).
    sc_fail = (op == SC) && !amo_unit.check(hart_id, line_addr);
    do_store = (op != LR) && !sc_fail;

    // Stall checks (collect all before mutating).
    if (core_rsp_out.full())                                 stall;
    if (do_store && !config_.write_back && mem_req_out.full()) stall;

    // Compute (pure; no side effects on the cache or reservations).
    old = bytes at offset of hit_line.data, width-sized;
    {new, ret} = amo_unit.compute(op, width, old, rhs);
    if (op == LR) amo_unit.reserve(hart_id, line_addr);
    if (op == SC) ret = sc_fail ? 1 : 0;

    // Commit.
    if (do_store) {
      line_merge(hit_line, new_block, byteen);
      if (config_.write_back) {
        hit_line.dirty = true;
      } else {
        // Write-through: emit the updated line bytes downstream.
        mem_req_out.send({addr, write=true, data=new_block, byteen});
      }
      amo_unit.invalidate(line_addr, /*except_hart=*/hart_id);
    }
    core_rsp_out.send({tag, cid, uuid, data = ret_block});
```

Crucial property: the AMO commit is **one bank cycle**, exactly
matching a write-hit. No phantom RMW cycle, no second channel.
Latency to `rd` = `tag_lookup_latency + (miss ? miss_penalty : 0)` —
identical to a load.

### 3.8 Non-LLC bank request flow (AMO passthrough)

Non-LLC banks (`config.is_llc == false`) own no `AmoUnit` and no
reservation table. AMO requests arriving at `core_req_in` are
detected by `core_req.op` ∉ {`READ`, `WRITE`}, scheduled onto a new
pipe-entry type:

```
processInputs():
  ... existing scheduling ...
  if (core_req.op is AMO_*):
    schedule AmoProbe on pipe_req_ (no MSHR alloc).
    capture in bank_req_t: addr, hart_id, the full original MemReq
    for downstream forwarding.

processRequests():
  case AmoProbe:
    if (mem_req_out.full()) stall;   // worst case = writeback emit
    tag-match line:
      hit & dirty → emit writeback MemReq, invalidate (valid=false,dirty=false)
      hit & clean → invalidate
      miss        → no-op
    forward the original AMO MemReq through the bank's nc_mem_arb
      egress (same path as IO bypass).
    pop pipe entry. No tag install on the eventual response.
```

The corresponding AMO response from below is handled by extending
[cache.cpp's processBypassResponse](../../sim/simx/mem/cache.cpp#L1060)
to accept AMO returns (today only IO returns flow this way). Encoding:
pack a 1-bit "is_bypass" tag prefix into the bypass-mux tag, mirroring
the existing `tag << log2_num_inputs` scheme.

The AMO line is never installed in this cache — the next normal load
to that address takes a fresh miss and refetches from below.

**Hit-dirty stall.** The "writeback before forward" sequence is at
least one cycle: writeback occupies `mem_req_out` this cycle; the
AMO forward needs the bypass arb on the next. SimX accounts for this
as a real cycle.

### 3.9 Reservation invalidation (LLC bank only)

Trigger: every committed write that reaches the LLC bank's tag array
calls `amo_unit.invalidate(line_addr, /*except_hart=*/cur_hart)`.
That set is, under §3.1.5, exactly:

- writethroughs from above (L1 always; L2 when L3 is the LLC),
- AMO RMW commits at the LLC,
- write-back write-hit commits if the LLC itself is configured
  write-back.

Line evictions do **not** invalidate. The reservation key is
`line_addr`, not "physical memory contents" — a SC after the
reservation set was evicted but never written by another hart still
succeeds (and is required to per RVA).

Cross-bank invalidation is unnecessary by construction: address
striping ties each line to exactly one LLC bank, so all writes to
that line route there.

### 3.10 The `AmoUnit` class

```cpp
// sim/simx/amo/amo_unit.h
namespace vortex {

class AmoUnit {
public:
  struct Reservation {
    uint32_t hart_id;
    uint64_t line_addr;
    bool     valid;
    uint32_t lru;
  };

  explicit AmoUnit(uint32_t reservation_size);

  // Pure compute — no mutation, no reservation touch.
  // old_word / rhs are 64-bit holders; width selects meaningful bits.
  // LR returns {old, old} (no store will happen).
  // SC's caller decides ret based on check() before calling compute().
  struct Result { uint64_t new_word; uint64_t ret_word; };
  Result compute(AmoType op, uint8_t width,
                 uint64_t old_word, uint64_t rhs) const;

  // Reservation table.
  void reserve  (uint32_t hart_id, uint64_t line_addr);
  bool check    (uint32_t hart_id, uint64_t line_addr) const;
  void invalidate(uint64_t line_addr, uint32_t except_hart_id);

  void reset();

private:
  std::vector<Reservation> reservations_;   // size = AMO_RS_SIZE
  uint32_t lru_clock_ = 0;
};

} // namespace vortex
```

`compute()` covers `AMO_ADD/SWAP/AND/OR/XOR/MIN/MAX/MINU/MAXU` —
ports the legacy `execute.cpp` per-op kernels
([reference](../../../vortex_main/sim/simx/execute.cpp#L905)) but
operating on extracted/packed words rather than going through
`mem_read` / `mem_write`.

### 3.11 LSU integration

`lsu_unit.cpp:236-240` currently aborts on AMO. Replace with:

```cpp
auto* lsu_tag = std::get_if<LsuType>(&trace->op_type);
auto* amo_tag = std::get_if<AmoType>(&trace->op_type);
if (!lsu_tag && !amo_tag) std::abort();

const bool is_amo   = (amo_tag != nullptr);
const bool is_fence = lsu_tag && (*lsu_tag == LsuType::FENCE);
const bool is_write = lsu_tag && (*lsu_tag == LsuType::STORE);
// ... existing fence path unchanged ...

// MSHR alloc: AMOs always need a return → reserve like a load.
if ((is_amo || !is_write) && state.mshr.full()) { stall; }
```

`compute_addrs()` already loops per-thread. Extend to stash
`rs2_data[t].u64` into `addr_list[t].rhs` when `is_amo`.

`process_request_step()` packs the per-lane batch:

```cpp
LsuReq lsu_req(NUM_LSU_LANES);
lsu_req.is_amo = is_amo;
lsu_req.wid    = trace->wid;
if (is_amo) {
  auto amoArgs = std::get<IntrAmoArgs>(trace->instr_ptr->get_args());
  lsu_req.amo.op    = *amo_tag;
  lsu_req.amo.width = amoArgs.width & 0x3;
  lsu_req.amo_rhs.assign(NUM_LSU_LANES, 0);
}
// per enabled lane i:
//   ...
//   if (is_amo) lsu_req.amo_rhs[i] = state.addr_list[t0+i].rhs;
```

`process_response_step()` already does load-formatting. The AMO
return path is functionally a load — read the response payload, sext
to width, place into `rd_data`. No new code.

### 3.12 LsuMemAdapter

The lane fan-out in `lsu_mem_adapter.cpp:103-136` widens the per-lane
`MemReq` build with:

```cpp
if (in_req.is_amo) {
  out_req.op        = amo_to_memop(in_req.amo.op);   // AmoType→MemOp
  out_req.amo_width = in_req.amo.width;
  out_req.amo_rhs   = in_req.amo_rhs.at(i);
  out_req.hart_id   = make_hart_id(in_req.cid, in_req.wid, /*tid=*/i);
  out_req.write     = false;          // see §3.4
}
```

`amo_to_memop` is a tiny mapping in `amo/amo_ops.h`. The bypass-mode
path (`num_inputs == 1`) needs the same translation.

### 3.13 LocalMemSwitch — out-of-scope guard

LMEM atomics are explicitly not supported (§6). The switch
([mem/local_mem_switch.cpp:61-77](../../sim/simx/mem/local_mem_switch.cpp#L61))
must hard-assert when a Shared-typed AMO lane is observed:

```cpp
assert(!(in_req.is_amo && type == AddrType::Shared)
       && "AMO on Shared (LMEM) is unsupported in this build");
```

This keeps a future LMEM-AMO mistake from silently routing through
the LMEM path (which has no AMO machinery).

---

## 4. Why this is correct (against the three rules)

- **Rule 1.** AMO functional bytes flow exclusively through TLM
  channels: `LsuReq.amo_rhs` → `MemReq.amo_rhs` → LLC bank line bytes
  → `MemRsp.data` → `LsuRsp.data` → `rd_data`. No `mem_read`, no
  `mem_write`, no `mmu_.amo_reserve`/`amo_check`. The legacy
  `MemoryUnit::amo_reservation_` is left dead — it is not referenced
  from v3.
- **Rule 2.** AMO commit is one bank cycle gated by the same
  `pipe_req_` scheduler as every other cache request. The byte that
  lands in DRAM (write-through) or is marked dirty (write-back) is
  the byte computed from the line present at that cycle. SC's
  success/failure outcome is decided in the same cycle, against the
  reservation table state at that cycle. Nothing is precomputed.
- **Rule 3.** RTL has no AMO unit yet, so the design has no RTL
  counterpart to mirror. The TLM design picks the LLC bank because
  it is the single point of serialization across cores in a
  non-coherent hierarchy, and because every reservation-breaking
  write is visible there (build assert enforces write-through above
  the LLC; §3.1.5). Intermediate caches stay simple — probe-
  invalidate-and-forward, no AMO state. When RTL grows AMO support,
  the recommended mirror is one `VX_amo_unit` per LLC bank with the
  same `Q`-entry reservation CAM and the same passthrough behavior
  at non-LLC levels — the SimX module name and config knob are
  chosen with that future symmetry in mind.

---

## 5. Phased implementation

Each phase compiles, runs, and is independently reviewable. Validate
via `build_test32/` (per `feedback_build_dir`) and the SimX-vs-RTLsim
CSV trace diff (per `reference_csv_trace_debugging`). Until RTL grows
AMO, only the SimX leg of the diff is meaningful — but the SimX leg
must remain green against the RVA test suite (`riscv-tests/rv32ua/*`,
`rv64ua/*`).

### Phase 0 — Config + plumbing (no behavior change)

- Enable `EXT_A_ENABLE` gating in `VX_config.toml` (default stays
  `false` until Phase 4 lands).
- Add `AMO_RS_SIZE` (default `4`, floor `2`) to `VX_config.toml` →
  `VX_config.h`.
- Add `Cache::Config::is_llc` (default `false`). Wire `is_llc = true`
  per the §3.1.3 table at `processor.cpp` (L3), `cluster.cpp` (L2),
  `socket.cpp` (dcache).
- Add the build-time write-through-strictly-above-LLC assert in
  `processor.cpp` ctor (§3.1.5). The assert does **not** constrain
  the LLC's own writeback policy.
- Extend `MemReq` / `LsuReq` per §3.4 — defaulted-zero fields, no
  consumer yet. Build remains green with `EXT_A_ENABLE = false`.

### Phase 1 — `AmoUnit` standalone

- Create `sim/simx/amo/{amo_unit.h, amo_unit.cpp, amo_ops.h}`.
- Hook into `sim/simx/Makefile`.
- Unit-test under `tests/unittests/amo/`: every AmoType across W/D
  widths, reservation alloc/check semantics under capacity pressure,
  invalidation paths. **No cache or LSU touched yet.** The compute
  kernel is provably correct in isolation before plumbing.

### Phase 2 — Cache bank wiring (LLC and non-LLC paths)

- LLC path: extend `bank_req_t` and `processInputs` to capture AMO
  fields. Extend `processRequests` Core+is_amo and Replay+is_amo
  branches per §3.6 / §3.7. Each LLC bank instantiates
  `AmoUnit(AMO_RS_SIZE)`. Wire reservation invalidation (§3.9).
- Non-LLC path: add the `AmoProbe` pipe entry and the writeback +
  invalidate + bypass-forward sequence per §3.8.
- Extend `processBypassResponse` to route AMO returns back to the
  core (today only IO returns flow this way). Encode the
  is-bypass bit in the bypass-mux tag.
- LSU still aborts on AMO — Phase 2 is testable via a synthetic
  `MemReq` injection harness in the cache unit-tests, run against
  L1-only, L1+L2, and L1+L2+L3 hierarchies.

### Phase 3 — LSU + Adapter + Execute

- LSU recognises `AmoType` ops (§3.11). MSHR alloc on AMOs.
- `LsuMemAdapter` maps `LsuReq.amo`/`amo_rhs`/`wid` → `MemReq.amo_*`/
  `hart_id` (§3.12).
- `execute.cpp` AMO branch: replace the `mem_read`/`mem_write` body
  with a builder that fills `rd_data` from the LSU's response —
  treat AMO like a load with side-effects, using the same
  `LsuTraceData` path. No `dcache_amo_reserve` / `dcache_amo_check`
  calls remain.
- `LocalMemSwitch` guard against Shared-AMOs (§3.13).
- (Verify) the legacy `Emulator::dcache_amo_reserve` /
  `dcache_amo_check` shells were not ported into v3; if any did
  slip in, delete them.

### Phase 4 — Conformance & multi-hart contention

- Run `riscv-tests/rv32ua/*` and `rv64ua/*` under SimX against three
  hierarchy configs: L1-only, L1+L2, L1+L2+L3. Same binary, same
  result — only timing differs.
- Author `tests/regression/atomics/` covering:
  1. **`AMOADD` hammer.** All harts increment a shared counter; final
     value equals `NUM_HARTS × iters × increment`.
  2. **LR/SC mutex.** Spin lock under contention; eventual progress
     required. Tune default `Q` if a typical N-warp pattern thrashes.
  3. **Cross-cache reservation invalidation.** Hart A on socket 0
     does `LR`; hart B on socket 1 does an ordinary store to the
     same line; A's `SC` must fail. Proves the §3.1.5 invariant is
     wired correctly.
- Diff against legacy `vortex_main` SimX results on the same
  binaries — values must match (timing won't).
- Flip `EXT_A_ENABLE` default to `true` once green.

### Phase 5 — Perf counters + docs

- Update `docs/cache_subsystem.md` with the AMO commit path and the
  reservation-table sizing.
- Add SimX perf counters: `amo_total`, `amo_sc_fail`,
  `amo_reservation_evictions`, rolled up at cache and core scope.

---

## 6. Out of scope

- **LMEM (Shared) atomics.** Distinct hardware site (LMEM bank),
  distinct reservation domain. Switch hard-asserts on Shared-AMO so
  divergence is loud, not silent (§3.13).
- **MMIO atomics.** Architecturally unspecified for Vortex's IO
  region. Cache bypass path asserts on AMO if it ever sees one.
- **Write-back intermediate caches.** Build asserts every cache
  *strictly above* the LLC is write-through (§3.1.5). The LLC itself
  is unconstrained — write-back or write-through, both correct.
  Lifting the strictly-above-LLC restriction would require a
  coherence protocol Vortex doesn't implement.
- **Forward-progress guarantees under arbitrary contention.** RVA
  permits livelock; we commit only to *eventual* progress when
  `AMO_RS_SIZE ≥ 2`. Default `Q = 4` is chosen so a typical 4-warp
  lock-free pattern doesn't thrash.
- **Acquire/Release ordering bits (`aq` / `rl`).** Decoded into
  `IntrAmoArgs` already; SimX's single-cluster cache hierarchy is
  trivially sequentially consistent so they have no effect today.
  Decoder retains them; bank ignores them.
- **RTL implementation.** Future work. The proposal is structured so
  that an RTL `VX_amo_unit` with the same per-LLC-bank shape and the
  same `AMO_RS_SIZE` knob can be added without touching the SimX
  side.
