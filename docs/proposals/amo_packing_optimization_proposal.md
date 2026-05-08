# Vortex Memory Interface — Refactor Proposal (RTL + SimX)

# Part A — RTL refactor

**Date:** 2026-05-07
**Status:** v1.2 — Part B (SimX) **implemented and validated**;
Part A (RTL) still proposed. v1.2 promotes `hart_id` from inside
`tag_t` to a **top-level `req_data_t` field** (matches Part B's
shape; tag stays opaque — see §4.3).
**Validation:** SimX passes RISC-V AMO ISA tests (10/10), regression
`amo` (12/12), `dxa_copy`, and `draw3d`.
**Scope:** Two parts. **Part A** refactors the RTL memory interface
(`hw/rtl/`). **Part B** mirrors the same refactor in SimX
(`sim/simx/`) so the two stay bit-compatible at trace level.
**Goal:** Replace the current `VX_mem_bus_if` / `VX_lsu_mem_if` shape
(scalar `rw` + per-lane `amo_req_t` sideband + ad-hoc `flags`) with
an **opcode-based** interface that is:
  1. **Smaller** — eliminates the AMO sideband as a first-class type
     and recovers per-lane `amo_req_t` bits across all carriers.
  2. **Extensible** — gives prefetch, eviction hints, write-no-rsp,
     cache-scope, fences, and future research features a clean place
     to land **without re-plumbing every interface**.
  3. **Layering-clean** — identity in `tag`, operation in `op`,
     orthogonal attributes in `flags`, payload in `data`/`byteen`.
     One concept per field.
  4. **Minimal-disruption** — touches only producer (LSU), consumer
     (cache), and the two interface declarations.
     **No edits under `hw/rtl/libs/`.**

**Related:**
[amo_rtl_v3_proposal.md](amo_rtl_v3_proposal.md) (the original AMO
RTL plan that introduced `amo_req_t`, which this proposal subsumes
and replaces).

---

## 1. Motivation

The AMO (RVA) implementation in commit `0487e1e8` added a per-lane
`amo_req_t` sideband to `VX_mem_bus_if` / `VX_lsu_mem_if`:

```sv
typedef struct packed {
    logic                       valid;     // 1 bit
    logic [INST_AMO_BITS-1:0]   op;        // 4 bits
    logic [1:0]                 width;     // 2 bits
    logic [`XLEN-1:0]           rhs;       // XLEN bits
    logic [HART_ID_WIDTH-1:0]   hart_id;   // HART_ID_WIDTH bits
} amo_req_t;
```

For a 4-lane RV32 LSU (HART_ID_WIDTH=4): `4 × (1+4+2+32+4) = 172 bits`
of per-lane AMO sideband on **every** interface that propagates
`req_data` — `lsu_mem_if`, `mem_bus_if`, the queues inside
`VX_mem_scheduler`, the seed/output flops in `VX_mem_coalescer`, the
pipe registers in `VX_cache_bank` (`amo_sel/st0/st1/replay_amo`),
and so on.

A field-by-field audit (§3) shows that **every bit of `amo_req_t` is
either redundant with an existing field, derivable from one, or
better placed elsewhere in the request shape**. The right fix isn't
to shrink the struct — it's to **dissolve it entirely** by adopting
the opcode-based request shape that essentially every modern cache
fabric uses.

### Why the current shape will keep biting

Two design pressures argue for a refactor now, not later:

1. **Future cache features keep asking for new sideband fields.**
   Prefetch hints, eviction-priority hints (NVIDIA `.EL/.EF`),
   write-no-response, cache-scope (warp/CTA/global), streaming
   non-allocate, compression hints, memory-ordering hints — every
   one of these would, today, mean another `\`ifdef`-guarded struct
   bolted onto `req_data`, with the same per-lane multiplication, the
   same updates to every consumer, and the same lib-tree parameter
   widening. We need a shape that absorbs new attributes without
   structural churn.
2. **`VX_mem_scheduler`, `VX_mem_coalescer`, `VX_mem_arb`,
   `VX_lsu_adapter`, `VX_stream_*`** are reusable IP under
   `hw/rtl/libs/` and **must not be modified for a feature change**.
   The current shape forces them to know about `amo_req_t` (its
   ports `core_req_amo` / `mem_req_amo` are explicit on
   `VX_mem_scheduler.sv:63,90` and similar in the coalescer). The
   right shape lets them stay opaque-by-`$bits()` forever.

---

## 2. What the world does — a survey

I sampled ~30 cache interface designs across industry, academia,
research codebases on GitHub, and reference texts. The patterns that
recur:

### 2.1 Opcode-based request

Almost every modern fabric uses a **single typed opcode field** that
enumerates the request kind:

| Fabric / source                         | Opcode field                                   | Example values                                                           |
|-----------------------------------------|------------------------------------------------|--------------------------------------------------------------------------|
| **TileLink** (SiFive, Rocket, BOOM)     | A-channel `opcode` + `param`                   | `PutFullData`, `PutPartialData`, `Get`, `ArithmeticData`, `LogicalData`, `AcquireBlock`, `Intent` (prefetch) |
| **ARM AMBA AXI4**                       | `AxLOCK` + `AxCACHE` + `AxPROT` (no opcode per se; direction by channel) | normal/exclusive, cacheable/bufferable, secure/priv                       |
| **ARM AMBA CHI** (CMN, neoverse)        | REQ `Opcode`                                    | `ReadOnce`, `ReadShared`, `WriteUnique`, `WriteBack`, `Atomic`, `PrefetchTgt` |
| **OpenPiton**                           | `MSG_TYPE` enum                                 | `LOAD`, `STORE`, `PREFETCH`, `EVICT`, `INV`                              |
| **gem5 Packet (`MemCmd`)**              | enum                                            | `ReadReq`, `WriteReq`, `SoftPFReq`, `HardPFReq`, `WritebackDirty`, `SwapReq` (atomic) |
| **ChampSim** (microarch research)       | `type` enum                                     | `LOAD`, `STORE`, `PREFETCH`, `RFO`, `WRITEBACK`, `TRANSLATION`           |
| **NVIDIA SASS / PTX memory**            | instruction-encoded                             | `LD`, `ST`, `LDG.E.CG`, `ATOM.ADD`, `RED`                                |
| **AMD GCN/CDNA**                        | dedicated `BUFFER_*` instr family               | `BUFFER_LOAD`, `BUFFER_STORE`, `BUFFER_ATOMIC_ADD`                       |
| **IBM POWER L2 (per US Patent 9,229,866)** | `req_type` field                              | `LD`, `ST`, `LARX`/`STCX` (RVA-equivalent), `PREFETCH`, `SYNC`           |
| **MIT BlueDB / Bluespec L1**            | tagged-union `MemReq`                           | `Ld`, `St`, `Atom`                                                       |

Notable point: **none of these carry a separate "valid" bit per op**;
the channel `valid` plus the opcode tell the consumer everything.
The "is this an AMO" question is the value of `opcode`, not a
separate boolean.

### 2.2 Flags / attributes

Orthogonal to the opcode is a small set of **attribute bits**:

| Fabric | Attribute bits |
|--------|----------------|
| AXI    | `AxCACHE` (4 bits), `AxPROT` (3 bits), `AxLOCK` (1 bit) |
| TileLink | `corrupt`, `denied`, `param` (op-specific sub-encoding) |
| CHI    | `Allocate`, `Cacheable`, `EWA`, `Order`, `MemAttr`, `SnpAttr` |
| GPU (NVIDIA) | `.CG`, `.CA`, `.CV`, `.WT`, `.EF`, `.EL` |
| gem5   | `Request::Flag` (~30 flags incl. `LOCKED_RMW`, `PREFETCH`, `STRICT_ORDER`, `UNCACHEABLE`, `CLEAN`, `INVALIDATE`) |

The shared idea: **op = "what to do", flags = "how to do it"**. AMO
*signedness* is a flag, not part of the op. Cacheability is a flag.
Eviction priority is a flag. The op enum stays small; the flag set
grows over time.

### 2.3 Identity / tag carrying source info

Every coherent fabric carries a **source identifier** in the tag:

| Fabric  | Source field            |
|---------|-------------------------|
| TileLink | A.`source` (per-master ID) — used to route response back |
| AXI/CHI | `AxID` / `SrcID`         |
| OpenPiton | `mshrid` + `cid`        |
| Gem5 Request | `_contextId` (hart ID) |
| gem5 Packet | `senderState` chain     |

The Vortex `tag_t` already separates `uuid` from `value` for exactly
this reason. Hart identity for AMO reservation tracking belongs in
the same identity prefix — alongside `uuid`, not as a separate
sideband.

### 2.4 User / extension sideband

The "we don't know what we'll need next" problem is universally
solved with a **user sideband**:

- AXI4: `AWUSER`, `WUSER`, `BUSER`, `ARUSER`, `RUSER` — variable-width,
  IP-specific.
- CHI: `RSVDC` reserved fields per channel.
- TileLink: `param` (opcode-dependent) plus `user` (TL-UH).
- gem5: `Request::extraData` + `Packet::senderState`.

The pattern: a **compile-time-parameterized opaque vector** that the
fabric propagates without inspecting. New attributes get added there
first (proven in research), promoted to first-class flags or opcodes
once the design crystallizes.

### 2.5 What this survey says for Vortex

A best-practice request shape for a Vortex-class cache is:

```
{ op_e op, flags_t flags, tag_t tag,
  addr_t addr, data_t data, byteen_t byteen, [user_t user] }
```

- **`op`** — typed enum (LD / ST / LR / SC / AMO_* / PREFETCH /
  FLUSH / FENCE / …). Direction (load vs store) is implied by `op`,
  not a separate `rw`. Atomic operation is *which* AMO, not a separate
  `valid + op` pair.
- **`flags`** — orthogonal Boolean attributes (cacheability,
  signedness, no-rsp, eviction hint, scope, …). Growable.
- **`tag`** — identity. `{uuid, hart_id, value}`. The tag is the
  routing/coherence cookie; it should carry source identity natively.
- **`addr` / `data` / `byteen`** — payload, unchanged. `byteen`
  already encodes access width, so a separate `width` is redundant.
- **`user`** *(optional, parameterized width)* — a "research lane"
  for sideband fields that aren't worth promoting to first-class
  flags yet. Defaults to width 0.

The current shape has scattered the `op` concept across `rw + flags +
amo_req_t.valid + amo_req_t.op`, with `width` duplicated in `byteen`,
`rhs` duplicated in `data`, and `hart_id` in a separate per-lane
struct rather than the tag. It's structurally what the survey
recommends *not* to do.

---

## 3. Field-by-field audit of today's `amo_req_t`

| Field      | Bits | Status                                                                 | Disposition                            |
|------------|-----:|------------------------------------------------------------------------|----------------------------------------|
| `valid`    | 1    | Set 1:1 with `lsu.amo_valid`; per-lane today, but warp-uniform.        | Replace with `op != LD/ST` membership. |
| `op`       | 4    | 11 RVA ops; warp-uniform.                                               | Fold into `MEM_OP_e`.                  |
| `width`    | 2    | `.W=2`, `.D=3`. RVA mandates natural alignment, so `byteen` already encodes width: `&byteen ⇒ .D`, else `.W`. ([cache_bank](../../hw/rtl/cache/VX_cache_bank.sv#L788), [amo_alu](../../hw/rtl/cache/VX_amo_alu.sv#L35)) | Drop. Derive at the AMO unit.          |
| `rhs`      | XLEN | LSU drives it from the same `rs2_data[i]` as `data`. Bit-identical for AMO traffic. ([VX_lsu_slice.sv:198-214,327](../../hw/rtl/core/VX_lsu_slice.sv#L198-L214)) | Drop. Reuse `data` (`write_word_st1` at the bank). |
| `hart_id`  | HART_ID_WIDTH | Reservation-table key in `VX_amo_unit`'s CAM ([amo_unit:88](../../hw/rtl/cache/VX_amo_unit.sv#L88)). Per-lane today; only `tid` varies within a warp. | Promote to top-level `req_data_t.hart_id` (per-lane on LSU side, scalar on bus side); see §4.3. |

**Every field is redundant, derivable, or homed in the wrong place.**
The struct as a whole has no reason to exist as a separate sideband.

---

## 4. Proposed interface

### 4.1 `mem_op_e` — the operation enum

```sv
// VX_gpu_pkg.sv
typedef enum logic [3:0] {
    // Always-present (independent of EXT_A_ENABLE)
    MEM_OP_LD       = 4'h0,   // ordinary load
    MEM_OP_ST       = 4'h1,   // ordinary store
    MEM_OP_FLUSH    = 4'h2,   // cache flush (was MEM_REQ_FLAG_FLUSH; also
                              //   covers `fence`, which today aliases to
                              //   FLAG_FLUSH at VX_lsu_slice.sv:69)

    // Atomic family (only meaningful when EXT_A_ENABLE; contiguous range
    //   keeps `mem_op_is_atomic` a 2-comparator bound check)
    MEM_OP_AMO_LR   = 4'h3,   // RVA load-reserved
    MEM_OP_AMO_SC   = 4'h4,   // RVA store-conditional
    MEM_OP_AMO_SWAP = 4'h5,
    MEM_OP_AMO_ADD  = 4'h6,
    MEM_OP_AMO_AND  = 4'h7,
    MEM_OP_AMO_OR   = 4'h8,
    MEM_OP_AMO_XOR  = 4'h9,
    MEM_OP_AMO_MIN  = 4'hA,   // signed/unsigned via MEM_FLAG_AMO_UNSIGNED
    MEM_OP_AMO_MAX  = 4'hB
    // 4'hC..4'hF reserved for future extensions (see §6).
} mem_op_e;

// Helper functions (compile-time evaluated, free):
function automatic logic mem_op_is_write(mem_op_e op);
    return (op == MEM_OP_ST) || (op == MEM_OP_AMO_SC) ||
           ((op >= MEM_OP_AMO_SWAP) && (op <= MEM_OP_AMO_MAX));
endfunction
function automatic logic mem_op_is_atomic(mem_op_e op);
    return (op >= MEM_OP_AMO_LR) && (op <= MEM_OP_AMO_MAX);
endfunction
function automatic logic mem_op_is_amo_rmw(mem_op_e op);
    return (op >= MEM_OP_AMO_SWAP) && (op <= MEM_OP_AMO_MAX);
endfunction
```

`rw` is **derived** from `op` via `mem_op_is_write(op)`. We keep
the function as a compile-time helper (no flop saving lost), and old
code that still wants `rw` can call it. Phase-2 work can delete the
`rw` field outright.

The ordering puts always-present ops (LD/ST/FLUSH) at slots 0–2 and
the `EXT_A_ENABLE`-gated atomic family contiguously at 4'h3–4'hB.
That alignment matches the source-level ifdef gating: when AMO is
disabled, only slots 0–2 are populated, and the atomic-range check
in helpers stays a clean 2-comparator bound.

12 named ops in 4 bits, **4 reserved slots** for future extensions
(prefetch, fence variants, reductions, invalidate, texture — see §6).

### 4.2 `flags` — orthogonal attributes

```sv
// VX_gpu_pkg.sv
// Bit 0 is the most-checked flag on the request critical path:
// the cache uses it to gate response-queue allocation for stores.
localparam MEM_FLAG_STRSP         = 0;  // store-response enabled (opt-in;
                                        //   plain stores default to no rsp,
                                        //   matching VX_lsu_slice.sv:126)
localparam MEM_FLAG_IO            = 1;  // existing — uncacheable IO
localparam MEM_FLAG_LOCAL         = 2;  // existing — LMEM port
localparam MEM_FLAG_AMO_UNSIGNED  = 3;  // MIN/MAX signedness (1=unsigned)
localparam MEM_FLAGS_WIDTH        = 4;  // grow as new flags land (see §6)
```

**On the polarity of `STRSP`:** GPU stores are overwhelmingly
fire-and-forget for throughput; only ordering-sensitive or
`wb`-needing writes want an ack. Defaulting to **no response** with
an opt-in flag (`STRSP=1` ⇒ enabled) matches:
- Vortex's existing behavior at
  [VX_lsu_slice.sv:126](../../hw/rtl/core/VX_lsu_slice.sv#L126) —
  plain stores already skip the response unless `wb=1`.
- NVIDIA / AMD GPU convention — store ack is opt-in for specific ops
  (.WT, .CV, atomics).
- AXI is the exception that proves the rule: it defaults to mandatory
  B-channel writes for CPU ordering, which costs throughput Vortex
  doesn't want.

Putting `STRSP` at bit 0 also matches the convention that the
most-frequently-checked flag sits at the LSB — the cache's response
queue allocation reads it on every store request fire.

The other ops have implicit response semantics:
- `LD`/`LR`/`AMO_*` always respond (they return loaded data).
- `SC` always responds (returns success/fail).
- `PREFETCH` never responds.
- `FLUSH`/`FENCE` respond with a 1-bit "done" pulse.

So `STRSP` is meaningful only when `op == ST`. Other op responses
are governed by `op` itself, not by this flag.

**Note:** `MEM_REQ_FLAG_FLUSH` becomes `MEM_OP_FLUSH` (it's not really
an attribute of a load/store — it *is* the operation). Existing
flush call sites change `flags[FLUSH]=1` ⇒ `op=MEM_OP_FLUSH`.

`MEM_FLAG_IO` and `MEM_FLAG_LOCAL` stay as flags (they're
attributes of any op — a load can be IO, a store can be LMEM).

`MEM_FLAG_AMO_UNSIGNED` is the *only* flag tied to a specific op
(MIN/MAX). That's an acceptable price for collapsing 4 ops
(MIN/MAX × signed/unsigned) into 2 + 1 flag. Alternative: keep 4
separate `MEM_OP_AMO_MIN/MAX/MINU/MAXU` ops (still fits in 4 bits,
14 → 16 used). Both work; the flag form leaves 2 reserved op slots
free, the explicit form leaves 0.

### 4.3 `hart_id` — top-level identity field

```sv
// VX_mem_bus_if.sv and VX_lsu_mem_if.sv
// `hart_id` is a *top-level* req_data_t field (see §4.4),
// not a member of `tag_t`.
logic [HART_ID_WIDTH-1:0]  hart_id;   // always present
```

`hart_id` rides as a **first-class request attribute**, alongside
`op` / `addr` / `data` / `byteen` / `flags` — not nested inside
`tag_t`. Two roles, two structs:

- **`tag_t`** = the *callback identifier*. Producer hands it to the
  cache; cache returns it on the response so the producer can match
  request↔response. The cache **does not read tag contents** — it
  round-trips them opaquely. Routers/arbiters append sel bits on the
  request and strip them on the response.
- **`hart_id`** = the *who-asked* attribute. Read by the *consumer*
  (cache reservation table for AMO LR/SC, future per-hart prefetch
  tracking, perf counters, QoS arbitration, observability). The
  producer does not need it back on the response — it knows its own
  identity.

Keeping these separate has three concrete benefits:
1. **Cache code is a hot-path read shorter** —
   `req_data_st1.hart_id` instead of `req_data_st1.tag.hart_id`.
   The cache no longer needs to know `tag_t`'s internal layout.
2. **`tag_t` stays evolvable** — each producer/cache pair can pick
   its own `{uuid, value}` shape; arbiters that compute tag widths
   don't need to thread `HART_ID_WIDTH` through.
3. **SimX↔RTL parity is structural** — SimX's `MemReq::hart_id`
   is already a top-level field (Part B, §13.2). Mirroring at the
   field level keeps the trace-compare contract simple.

**Always-present (not `EXT_A_ENABLE`-gated).** A general-purpose
hart-identity field has uses well beyond AMO reservation. Gating it
on `EXT_A_ENABLE` would force every future feature that wants hart
identity to either (a) re-introduce its own conditional sideband
(the same anti-pattern this proposal is fixing) or (b) flip an
unrelated `EXT_A_ENABLE` switch as a side-effect. The cost —
`HART_ID_WIDTH` bits per request flop — is small (typ. 4–6 bits,
vs. the 30+ bits of `addr` and 32+ of `data` that ride alongside)
and uniform across configs.

`tag_t` itself stays untouched by this proposal — same shape as
today, just no longer required to carry hart identity.

### 4.4 `req_data_t` — the assembled request

**`VX_mem_bus_if`** (per-bus):

```sv
typedef struct packed {
    mem_op_e                  op;       // 4 bits — replaces rw + amo.valid + amo.op
    logic [ADDR_WIDTH-1:0]    addr;
    logic [DATA_SIZE*8-1:0]   data;     // for ST/SC/AMO: store data / rhs
    logic [DATA_SIZE-1:0]     byteen;   // also encodes AMO access width
    logic [FLAGS_WIDTH-1:0]   flags;    // orthogonal attributes
    logic [HART_ID_WIDTH-1:0] hart_id;  // who-asked (always present, §4.3)
    tag_t                     tag;      // opaque round-trip token
} req_data_t;
```

- `rw` field deleted (use `mem_op_is_write(op)`).
- `amo` field deleted entirely.
- 2 new fields added (`op`, 4 bits; `hart_id`, `HART_ID_WIDTH` bits);
  `rw` removed (1 bit); `amo` removed (`AMO_REQ_BITS`).
- Net change vs today (RV32, `HART_ID_WIDTH=4`): `op (4) +
  hart_id (4) − rw (1) − amo_req_t (43) = −36 bits` per `mem_bus_if`.

**`VX_lsu_mem_if`** (per-lane):

```sv
typedef struct packed {
    logic [NUM_LANES-1:0]                  mask;
    mem_op_e                               op;       // SCALAR — warp-uniform
    logic [NUM_LANES-1:0][ADDR_WIDTH-1:0]  addr;
    logic [NUM_LANES-1:0][DATA_SIZE*8-1:0] data;
    logic [NUM_LANES-1:0][DATA_SIZE-1:0]   byteen;
    logic [NUM_LANES-1:0][FLAGS_WIDTH-1:0] flags;
    logic [NUM_LANES-1:0][HART_ID_WIDTH-1:0] hart_id; // PER-LANE (see note)
    tag_t                                  tag;
} req_data_t;
```

- `op` is **scalar** (warp-uniform — same as `rw` is today). All
  lanes within a SIMD issue execute the same instruction, so per-lane
  storage of `op` is wasteful.
- `hart_id` is **per-lane**: `make_hart_id(cid, wid, tid)` differs
  by `tid` across SIMD lanes. The LLC reservation table keys on the
  per-lane hart_id (Part B Stage 2 implementation discovered this —
  collapsing to scalar broke AMO test 7 lrsc_counter across
  divergent lanes).
- `rw` field deleted.
- Per-lane `amo[NUM_LANES]` field deleted entirely.
- Net change vs today (4-lane RV32): `op (4) + 4 × hart_id (16) −
  rw (1) − 4 × amo_req_t (172) = −153 bits` per `lsu_mem_if`.

Note: `flags` stays per-lane because some flags (`MEM_FLAG_IO`,
`MEM_FLAG_LOCAL`) can legitimately differ per lane in a divergent
scenario. AMO-related flags (`MEM_FLAG_AMO_UNSIGNED`) are
warp-uniform but ride along in flags for shape consistency.

### 4.5 Optional `user` sideband (deferred)

A parameterized opaque vector for research extensions:

```sv
parameter USER_WIDTH = 0;
...
logic [USER_WIDTH-1:0] user;   // 0-width by default → zero cost
```

When a research feature wants to carry, say, a 16-bit prefetch PC
hint or a 4-bit confidence value, set `USER_WIDTH=20` at one cache
boundary; the bus, mem_scheduler, and arbs propagate it opaquely via
`$bits()`. Promote to a first-class `flag` / `op` only after the
feature crystallizes. **Default 0-width means zero silicon cost
unless explicitly enabled.**

---

## 5. Silicon savings

For RV32, `NUM_LANES=4`, `HART_ID_WIDTH=4`, `LMEM_ENABLED=1`:

### 5.1 Per `lsu_mem_if`

| Component                       | Today                  | Proposed              | Δ        |
|---------------------------------|-----------------------:|----------------------:|---------:|
| `rw`                            | 1                      | 0 (derived)           | −1       |
| `op` (scalar)                   | —                      | 4                     | +4       |
| `flags` (per-lane × 4)          | 4 × 3 = 12             | 4 × 4 = 16            | +4       |
| `amo[NUM_LANES]`                | 4 × 43 = 172           | 0                     | −172     |
| `hart_id` (per-lane × 4)        | 0 (in tag, ad-hoc)     | 4 × 4 = 16            | +16      |
| **Total**                       |                        |                       | **−149** |

### 5.2 Per `mem_bus_if`

| Component                       | Today | Proposed | Δ      |
|---------------------------------|------:|---------:|-------:|
| `rw`                            | 1     | 0        | −1     |
| `op`                            | —     | 4        | +4     |
| `flags`                         | 3     | 4        | +1     |
| `amo`                           | 43    | 0        | −43    |
| `hart_id`                       | 0     | 4        | +4     |
| **Total**                       |       |          | **−35** |

### 5.3 RV64 / 8-lane / `HART_ID_WIDTH=6`

| Carrier | Δ |
|---------|--:|
| `lsu_mem_if`: `−1 + 4 + (8 × (4−3)) − (8 × 77) + (8 × 6) = −553` |
| `mem_bus_if`: `−1 + 4 + 1 − 77 + 6 = −67` |

These are **per-instance** savings. With `mem_bus_if` instances at
every cache port, every adapter, every queue stage in
`VX_mem_scheduler` and `VX_mem_coalescer`, and the `amo_st0/st1` /
`replay_amo` pipe stages in `VX_cache_bank`, the cumulative flop
count saved per LSU+LLC chain is in the **low thousands** for the
RV64 8-lane case. A representative-config Yosys/Vivado spot-check
would tighten this estimate.

### 5.4 Note: response path is unaffected

Because `hart_id` rides as a top-level **request** field — not in
`tag_t` — the response path does **not** carry it. The producer
already knows its own hart_id and matches the response by `tag`
alone. Compared to the earlier (v1.0-rev1) "hart_id-in-tag" plan,
this saves `HART_ID_WIDTH` bits per response flop on every carrier
in the response chain (cache→adapter→arb→producer), at no
correctness cost.

---

## 6. Future-extension catalog (the real point)

The v1.0 shape (12 ops in 4 bits with 4 reserved, 4 flag bits with
plenty of room to grow, plus an opt-in `user` sideband) lets future
features land in one of three places **without re-plumbing any IP**.

The features below are **out of scope for v1.0**; they are listed
here to demonstrate that the new shape has a place for each of them.
Each row identifies *where* the feature would land if/when it's
proposed — the bit/op slot is **not** allocated until that work is
in flight.

| Future feature                                    | Where it would land                                  | Notes                                                                  |
|---------------------------------------------------|------------------------------------------------------|------------------------------------------------------------------------|
| **Prefetch hint** (compiler-emitted)              | new `MEM_OP_PREFETCH` (reserved op slot)             | One reserved slot consumed.                                            |
| **Prefetch w/ stride hint**                       | `MEM_OP_PREFETCH` + stride in `user`                 | Promote stride to a flag once it earns its keep.                       |
| **Eviction hint .EF / .EL / NEU**                 | new flag bits `MEM_FLAG_EVICT_HI`/`_LO` (or 2-bit field) | Add flag bits; no struct change.                                   |
| **Streaming non-allocate**                        | new flag bit `MEM_FLAG_NOALLOC`                      | Single flag bit.                                                       |
| **Pin in cache (research)**                       | new flag bit `MEM_FLAG_PRESERVE`                     | Single flag bit.                                                       |
| **Scope: warp / CTA / cluster / global**          | 2-bit `MEM_FLAG_SCOPE_*` field in flags              | Add 2 flag bits, no struct change.                                     |
| **Memory ordering: acquire / release / SC**      | new `MEM_OP_FENCE` op + 2 ordering bits in flags     | Today's `fence` aliases to `MEM_OP_FLUSH`; if richer ordering is needed, split. |
| **Reductions** (NVIDIA `RED.ADD` etc.)            | new `MEM_OP_RED_*` ops in reserved slots             | Same dispatch path as AMO RMW.                                         |
| **Cache invalidate without writeback**            | new `MEM_OP_INV` op                                  | One reserved slot.                                                     |
| **Compression hint** (data is pre-compressed)     | flag bit + `user` payload size                       | Flag bit + sideband for size.                                          |
| **Speculative load** (squashable)                 | new flag bit `MEM_FLAG_SPEC`                         | Single flag bit.                                                       |
| **Atomic broadcast / multicast**                  | `MEM_OP_AMO_*` + `user` mask                         | User sideband for the mask.                                            |
| **Texture fetch path**                            | new `MEM_OP_TEX_*` op(s)                             | Reserved op slot or upgrade `op` to 5 bits later.                      |
| **Tagged loads** (research color-tagging)         | `user` sideband                                      | Research-grade until proven.                                           |

The contract: **research and one-off features start in `user`;
proven features promote to `flags`; mutually-exclusive request
classes promote to `op`.** The lib-tree IPs never need to know.

v1.0 keeps the flag set lean (`STRSP`, `IO`, `LOCAL`, `AMO_UNSIGNED`)
and the op set focused on what's already needed (basic LD/ST + RVA +
FLUSH). Everything in this catalog is a separate proposal.

---

## 7. AMO mapping (sanity check)

To prove the design preserves AMO functionality without a sideband:

| RVA op | New encoding                                                              |
|--------|---------------------------------------------------------------------------|
| `LR.W`/`LR.D`  | `op=MEM_OP_AMO_LR`, `byteen` selects `.W`/`.D`, `hart_id` set      |
| `SC.W`/`SC.D`  | `op=MEM_OP_AMO_SC`, `data=rs2`, `hart_id` set                     |
| `AMOSWAP`      | `op=MEM_OP_AMO_SWAP`, `data=rs2`, `hart_id` set                   |
| `AMOADD`       | `op=MEM_OP_AMO_ADD`, `data=rs2`                                   |
| `AMOMIN`       | `op=MEM_OP_AMO_MIN`, `flags[AMO_UNSIGNED]=0`                      |
| `AMOMINU`      | `op=MEM_OP_AMO_MIN`, `flags[AMO_UNSIGNED]=1`                      |
| `AMOMAX/MAXU`  | `op=MEM_OP_AMO_MAX`, `flags[AMO_UNSIGNED]=0/1`                    |
| `AMOAND/OR/XOR`| `op=MEM_OP_AMO_AND/OR/XOR`, `data=rs2`                            |

`VX_amo_unit`'s ALU
([VX_amo_alu.sv:42-58](../../hw/rtl/cache/VX_amo_alu.sv#L42-L58))
becomes a case on `mem_op_e` directly:

```sv
case (op)
    MEM_OP_AMO_LR:    new_word = a_u;
    MEM_OP_AMO_SC,
    MEM_OP_AMO_SWAP:  new_word = b_u;
    MEM_OP_AMO_ADD:   new_word = a_u + b_u;
    MEM_OP_AMO_AND:   new_word = a_u & b_u;
    MEM_OP_AMO_OR:    new_word = a_u | b_u;
    MEM_OP_AMO_XOR:   new_word = a_u ^ b_u;
    MEM_OP_AMO_MIN:   new_word = unsigned_flag ? min_u(a,b) : min_s(a,b);
    MEM_OP_AMO_MAX:   new_word = unsigned_flag ? max_u(a,b) : max_s(a,b);
    default:          new_word = a_u;
endcase
```

`compute_width` is `&byteen ? 2'd3 : 2'd2`. `compute_rhs` is the
existing `write_word_st1` flop. `compute_hart_id` is `tag_st1.hart_id`.
Reservation table behaviour is identical to today.

---

## 8. Migration plan

A 4-stage landing keeps each step under 10 files:

### Stage 1 — Introduce `mem_op_e` alongside the existing fields

- Add the `mem_op_e` typedef and helper functions to `VX_gpu_pkg.sv`.
- Add a new `op` field to `VX_mem_bus_if.req_data` and
  `VX_lsu_mem_if.req_data` (do **not** remove `rw` or `amo` yet).
- LSU populates the new `op` field from `lsu.amo_op` /`rw`/load-or-store
  in addition to the old fields.
- Cache reads `op` in parallel with old fields, asserting they match.

This is purely additive. Run full regression. Ship.

### Stage 2 — Add top-level `hart_id` field to `req_data_t`

- Add `hart_id` to `req_data_t` in both `VX_mem_bus_if.sv` and
  `VX_lsu_mem_if.sv` (per-lane on the LSU side, scalar on the bus
  side). `tag_t` is **untouched**.
- Producers populate it: LSU drives
  `req.hart_id[i] = make_hart_id(cid, wid, tid_i)`; DXA/TEX/RASTER/OM/TCU
  engines drive their owning warp/CTA's hart_id.
- Cache reads `req.hart_id` in parallel with `amo[*].hart_id`,
  asserting equality on every AMO request.
- Pre-flight grep: `grep -rn "amo\[.*\]\.hart_id\|amo\..*\.hart_id" hw/rtl/`
  to enumerate the call sites the cache cross-checks against.

Regress. Ship.

### Stage 3 — Delete the redundant fields

- Delete `rw` (replace with `mem_op_is_write(op)`).
- Delete `amo[NUM_LANES]` from both interfaces.
- Delete `amo_req_t`, `AMO_REQ_BITS`, `INST_AMO_*` from `VX_gpu_pkg.sv`.
- Delete `core_req_amo` / `mem_req_amo` lines from `VX_lsu_slice.sv`'s
  `VX_mem_scheduler` instantiation. The `\`ifdef EXT_A_ENABLE` blocks
  in `VX_mem_scheduler.sv` and `VX_mem_coalescer.sv` become
  no-ops; no edit needed there (they auto-disappear when the field
  goes away from the interface).
- Update `VX_amo_unit` / `VX_amo_alu` to take `mem_op_e` and the
  derived `width`/`rhs`.

Regress. Ship.

### Stage 4 — Reorganize `flags`

- Move `MEM_REQ_FLAG_FLUSH` ⇒ `MEM_OP_FLUSH` (was a flag
  masquerading as an op).
- Add the new flag bits (NORSP, EVICT_HI/LO, NOALLOC, PRESERVE,
  AMO_UNSIGNED) per §4.2.
- Update `VX_dcr_flush.sv` to set `op=MEM_OP_FLUSH` instead of the
  flag.
- Update `VX_cache_init.sv` to dispatch on `op == MEM_OP_FLUSH`.

Regress. Ship.

### 8.5 Affected modules

Confirmed by grep over `hw/rtl/`. Modules fall into four categories:

#### A. Interface declarations — **change** (the heart of the refactor)

| File | Stage | What changes |
|------|------|--------------|
| `hw/rtl/mem/VX_mem_bus_if.sv` | 1, 2, 3 | `req_data_t` adds `op` and `hart_id` (top-level), drops `rw` and `amo`. `tag_t` untouched. |
| `hw/rtl/mem/VX_lsu_mem_if.sv` | 1, 2, 3 | `req_data_t` adds scalar `op`, per-lane `hart_id[NUM_LANES]`, drops `rw` and per-lane `amo[NUM_LANES]`. `tag_t` untouched. |

#### B. Producers — **change** (set the new fields)

| File | Stage | What changes |
|------|------|--------------|
| `hw/rtl/VX_gpu_pkg.sv` | 1, 3, 4 | new `mem_op_e`, helper functions, flags enum; **delete** `amo_req_t`, `INST_AMO_*`, `AMO_REQ_BITS` |
| `hw/rtl/core/VX_lsu_slice.sv` | 1, 2, 3 | populate `op`/per-lane `hart_id`; drop `core_req_amo` block; drop the `core_req_amo` connection in the `VX_mem_scheduler` instantiation |
| `hw/rtl/core/VX_dcr_flush.sv` | 4 | drive `op = MEM_OP_FLUSH` instead of `flags[FLUSH] = 1` |
| `hw/rtl/dxa/VX_dxa_*.sv`, `hw/rtl/tex/VX_tex_*.sv`, `hw/rtl/raster/VX_raster_*.sv`, `hw/rtl/om/VX_om_*.sv`, `hw/rtl/tcu/VX_tcu_*.sv` | 2 | DXA/TEX/RASTER/OM/TCU mem clients also produce `req_data`; set `hart_id` from the engine's owning warp/CTA id (or 0 for unowned). No tag changes (tag stays opaque). |

#### C. Consumers — **change** (read the new fields)

| File | Stage | What changes |
|------|------|--------------|
| `hw/rtl/cache/VX_cache.sv` | 1, 3 | dispatch on `op`; drop per-bank `amo` plumbing |
| `hw/rtl/cache/VX_cache_bank.sv` | 1, 3 | read AMO state from `op_st1`/`hart_id_st1` (top-level field, not nested in tag); derive `width` from `byteen_st1`, `rhs` from `write_word_st1`; **delete** `amo_st0/1`, `replay_amo` |
| `hw/rtl/cache/VX_cache_init.sv` | 4 | dispatch on `op == MEM_OP_FLUSH` |
| `hw/rtl/cache/VX_cache_bypass.sv` | 1 | reads `op` instead of `rw`+`amo.valid` |
| `hw/rtl/cache/VX_cache_wrap.sv`, `hw/rtl/cache/VX_cache_cluster.sv` | (none) | parameter-thread only; `$bits()`-opaque |
| `hw/rtl/cache/VX_amo_alu.sv` | 3 | input port type changes from `[INST_AMO_BITS-1:0]` to `mem_op_e` |
| `hw/rtl/cache/VX_amo_unit.sv` | 3 | same; takes `byteen` instead of `width` (or leaves derivation to caller) |

#### D. Opaque pass-through — **NO change** (propagate `req_data`/`tag` via `$bits()`)

These modules carry the bus generically and shrink automatically
when the AMO sideband is removed. **No edits needed.**

| File | Why no change |
|------|--------------|
| `hw/rtl/mem/VX_mem_arb.sv` | unpacks `req_data` field-by-field; the field list shrinks but the per-field assigns are identical (no `amo` reference today) |
| `hw/rtl/mem/VX_mem_xbar.sv` | wholly opaque-by-`$bits()` |
| `hw/rtl/mem/VX_mem_switch.sv` | wholly opaque |
| `hw/rtl/mem/VX_lmem_switch.sv` | wholly opaque |
| `hw/rtl/mem/VX_lsu_adapter.sv` | unpacks/repacks but the new field set just takes its place |
| `hw/rtl/mem/VX_lsu_mem_arb.sv` | wholly opaque |
| `hw/rtl/mem/VX_local_mem.sv` | wholly opaque |
| `hw/rtl/core/VX_core.sv`, `hw/rtl/core/VX_execute.sv`, `hw/rtl/core/VX_lsu_unit.sv`, `hw/rtl/core/VX_mem_unit.sv` | parameter-thread only |
| `hw/rtl/core/VX_fetch.sv` | uses `mem_bus_if` for icache; no AMO fields touched |

#### E. `hw/rtl/libs/` — **revert AMO-commit additions** (return to pre-AMO state)

The AMO commit `0487e1e8` modified two libs files to add explicit
`core_req_amo` / `mem_req_amo` ports. With `amo_req_t` deleted from
`VX_gpu_pkg.sv`, those port declarations no longer compile.

The fix is to **revert** the AMO-commit additions in these files
(returning them to their pre-`0487e1e8` state). This is *aligned*
with the "no edits under `hw/rtl/libs/`" rule, not a violation of
it — the AMO commit itself broke that rule, and v1.0 restores it.

| File | Change |
|------|--------|
| `hw/rtl/libs/VX_mem_scheduler.sv` | revert the `core_req_amo`/`mem_req_amo` port additions and the internal `reqq_amo*`/`mem_req_amo_*` flop arrays from `0487e1e8` |
| `hw/rtl/libs/VX_mem_coalescer.sv` | revert the `in_req_amo`/`out_req_amo` port additions and the `seed_amo_r`/`out_req_amo_r` flop arrays from `0487e1e8` |

After this revert, `hw/rtl/libs/` is back to pre-AMO and stays
that way for v1.0 and every future feature in the §6 catalog.

#### Summary

| Category | Files | Edits per file |
|----------|------:|----------------|
| A — Interface declarations | 2 | structural (field-list edit; `tag_t` untouched) |
| B — Producers | 5–8 | small; populate `op` and `hart_id` |
| C — Consumers | 7 | read new field names; ALU op-set update |
| D — Opaque pass-through | 11 | **none** |
| E — Libs revert | 2 | revert AMO-commit additions to pre-`0487e1e8` state |
| **Total touched** | **~15–19** | of which **none stay-modified under `hw/rtl/libs/`** |

---

## 9. Open questions

**Q1.** Is it safe to derive AMO `width` from `byteen` everywhere?
The bank pipes `byteen_st1` alongside `amo_st1` already, but the
LSU's byteen-shaper
([VX_lsu_slice.sv:162-187](../../hw/rtl/core/VX_lsu_slice.sv#L162-L187))
operates on `req_align` which is 0 for AMOs (naturally aligned). Worth
walking the byteen subtree exhaustively before stage 3.

**Q2.** When `write_word_st1` is reused as `compute_rhs`, is the
bank's writeback FSM
([VX_cache_bank.sv:317-326](../../hw/rtl/cache/VX_cache_bank.sv#L317-L326))
guaranteed to *not* overwrite `write_word_st1` with the post-RMW value
before the AMO unit reads it within S1? Looks safe per the existing
`amo_wb_data_r` separation, but a co-sim assertion would harden it.

**Q3.** Per-lane `hart_id` correctness across producers. Stage 2
needs a full grep to ensure every producer drives `req.hart_id`
correctly (per-lane on the LSU side, scalar on the bus side). A
small hooked-PR test that asserts `req.hart_id == expected_hart_id`
at each producer (matching what the cache cross-checks against
`amo[*].hart_id` during the parallel-assert window) would catch
regressions. Discovered during Part B: per-lane is required because
the LSU compaction step renumbers lanes, so the lane index ≠
original tid for divergent warps.

**Q4.** `flags` per-lane vs. scalar in `lsu_mem_if`. Most flags are
per-lane today (the field is `[NUM_LANES-1:0][FLAGS_WIDTH-1:0]`).
For warp-uniform flags (`MEM_FLAG_AMO_UNSIGNED`), per-lane storage
is redundant. Possible future optimization: split `flags` into
`scalar_flags` + `per_lane_flags`. Out of scope for this proposal —
the savings are second-order vs. the AMO sideband elimination.

**Q5.** Should we widen `op` to 5 bits now (32 slots) to leave more
headroom? 4 bits leaves 2 reserved; 5 bits leaves 18. The cost is
one extra bit per request flop everywhere. Recommend 4 for now;
reserve the upgrade for when reserved slots fill.

**Q6.** Default `USER_WIDTH=0` (parameterized opaque sideband):
should we plumb the parameter even at width 0, so a future
research branch can flip it on without a structural change?
Recommended: yes — it's cheap (no flops at width 0) and removes the
"oh, we need to add a sideband everywhere" pain when a future
feature wants it.

---

## 10. Part A summary

The current `amo_req_t` is a microcosm of a deeper design pressure:
the Vortex memory interface needs an **opcode-based** shape (with
operation in `op`, who-asked in `hart_id`, attributes in `flags`,
payload in `data`/`byteen`, opaque round-trip in `tag`, and a
research sideband in `user`) to absorb future cache features
without per-feature interface churn.

Adopting that shape:

- **Eliminates `amo_req_t` entirely** (~149 bits saved per
  `lsu_mem_if` at RV32-4-lane; ~553 at RV64-8-lane).
- **Touches no files under `hw/rtl/libs/`** after v1.0 lands; the
  AMO commit's libs additions revert to their pre-`0487e1e8` state,
  and the lib IPs propagate `req_data` opaquely going forward.
- **Aligns the Vortex bus with industry/academic best practice**
  (TileLink, AXI/CHI, gem5, ChampSim — every modern fabric uses
  opcode + flags + identity + user).
- **Future-proofs** prefetch, eviction hints, no-rsp writes, scope,
  fences, reductions, compression, speculation — each lands in one
  reserved op slot, one new flag bit, or the `user` sideband. Zero
  re-plumbing.

The 4-stage migration (~17–21 files total, no `libs/` edits) is the
roadmap; the new shape is the contract.

---

# Part B — SimX refactor

SimX is the C++ functional/cycle-approx model that runs alongside
the RTL. It must stay **bit-compatible at trace level** with the
RTL, so the Part A interface change implies a parallel SimX change.

## 11. Current SimX state — half-refactored

A grep of `sim/simx/types.h` reveals SimX is *already* moving in the
direction Part A advocates, but the migration was never completed:

```cpp
// sim/simx/types.h:1057
enum class MemOp : uint8_t {
    READ      = 0,  WRITE     = 1,
    AMO_LR    = 2,  AMO_SC    = 3,
    AMO_ADD   = 4,  AMO_SWAP  = 5,  AMO_XOR  = 6,  AMO_OR  = 7,
    AMO_AND   = 8,  AMO_MIN   = 9,  AMO_MAX  = 10,
    AMO_MINU  = 11, AMO_MAXU  = 12,
};

inline bool memop_is_amo(MemOp op);   // already exists
inline MemOp amo_to_memop(AmoType t); // already exists
inline AmoType memop_to_amo(MemOp op);// already exists
```

`MemOp` is a typed `enum class` and already enumerates exactly the
ops Part A's `mem_op_e` enumerates. **The hard work is done.** What's
missing:

1. **`MemReq.write` is still the dispatch** ([cache.cpp](../../sim/simx/mem/cache.cpp)
   reads `req.write` on the critical path; `op` is set in the
   constructor but rarely read).
2. **`amo_req_t` carries duplicate state**:
   - `amo_req_t::valid` duplicates `memop_is_amo(req.op)`.
   - `amo_req_t::op` (an `AmoType`) duplicates `req.op` (a `MemOp`).
   - `amo_req_t::width` is derivable from `byteen` popcount (same
     as RTL §3).
   - `amo_req_t::rhs` is the rs2 value at the AMO byte offset within
     `req.data` — extractable by the AmoUnit.
   - `amo_req_t::hart_id` duplicates the conventional use of
     `MemReq::cid` documented at types.h:1144 ("the LLC bank's
     reservation table rides on `cid` … not on a separate field").
3. **Two parallel enums** (`AmoType` and `MemOp` AMO subset) require
   constant `amo_to_memop`/`memop_to_amo` conversions on every code
   path.

So SimX has the *destination shape* in code but never deleted the
*origin shape*. Part B is largely a cleanup.

---

## 12. Field-by-field audit (SimX)

| Field                  | Status                                                           | Disposition                                                              |
|------------------------|------------------------------------------------------------------|--------------------------------------------------------------------------|
| `MemReq::write` (bool) | Dispatched everywhere; redundant with `op`.                      | **Keep as derived** via `memop_is_write(op)` helper; phase-out reads.    |
| `MemReq::op` (`MemOp`) | Already exists; partially populated.                             | **Promote to primary dispatch**; cache reads `op` on critical path.      |
| `MemReq::cid` (uint32) | Documented as carrying `make_hart_id(cid,wid,tid)` per-lane.     | **Rename** to `hart_id` (or add explicit `hart_id` field); deprecate `cid` semantic overload. |
| `MemReq::amo` (`amo_req_t`) | All 5 fields redundant or derivable (per RTL §3).            | **Delete the struct**.                                                   |
| `AmoType` enum         | Exists in parallel with `MemOp` AMO subset; conversion helpers everywhere. | **Delete**; use `MemOp` directly.                                |
| `amo_to_memop` / `memop_to_amo` | Conversion helpers needed only because of dual enums.   | **Delete** with `AmoType`.                                               |

Net effect: SimX's `MemReq` shrinks (struct + nested struct
collapse) and the cache's hot path drops a level of indirection
(no more `req.amo.op` — just `req.op`).

---

## 13. Proposed SimX shape

### 13.1 `MemOp` — implemented

Re-namespaced existing constants to match RTL ordering exactly so
the trace-comparison harness can `static_assert` parity:

```cpp
// sim/simx/types.h (as implemented)
enum class MemOp : uint8_t {
    // Always-present
    LD       = 0,   // was READ
    ST       = 1,   // was WRITE
    FLUSH    = 2,   // matches RTL MEM_OP_FLUSH

    // Atomic family (EXT_A_ENABLE; contiguous, mirrors RTL ordering)
    AMO_LR   = 3,
    AMO_SC   = 4,
    AMO_SWAP = 5,
    AMO_ADD  = 6,
    AMO_AND  = 7,
    AMO_OR   = 8,
    AMO_XOR  = 9,
    AMO_MIN  = 10,  // signed/unsigned via flags.amo_unsigned
    AMO_MAX  = 11,
    // 12..15 reserved for future extensions (Part A §6).
};

constexpr bool memop_is_write(MemOp op);   // ST / AMO_SC / AMO_SWAP..AMO_MAX
constexpr bool memop_is_atomic(MemOp op);  // AMO_LR..AMO_MAX
constexpr bool memop_is_amo_rmw(MemOp op); // AMO_SWAP..AMO_MAX
```

AMO renumbering: previous SimX had `AMO_LR=2..AMO_MAXU=12`; the
new ordering shifts `AMO_LR` to 3 and collapses `AMO_MINU`/
`AMO_MAXU` into `AMO_MIN`/`AMO_MAX` + `flags.amo_unsigned`.
The LSU sets `flags.amo_unsigned = 1` for the unsigned MIN/MAX
variants and the AmoUnit branches on the flag.

### 13.2 `MemReq` and `MemFlags` — implemented

```cpp
// sim/simx/types.h (as implemented)
struct MemFlags {
  union {
    uint32_t raw;
    struct {
      uint32_t strsp        : 1;  // bit 0: opt-in store response
      uint32_t io           : 1;  // bit 1: uncacheable I/O
      uint32_t local        : 1;  // bit 2: LMEM port
      uint32_t amo_unsigned : 1;  // bit 3: MIN/MAX signedness
    #ifdef EXT_DXA_ENABLE
      uint32_t dxa_notify_done   : 1;  // bit 4
      uint32_t dxa_notify_bar_id : 8;  // bits 5..12
    #endif
    };
  };
  MemFlags() : raw(0) {}
  MemFlags(uint32_t r) : raw(r) {}
  bool any()  const { return raw != 0; }
  bool none() const { return raw == 0; }
  // operator==/!=/<<
};
static_assert(sizeof(MemFlags) == sizeof(uint32_t));

struct MemReq {
  uint64_t addr;
  uint32_t tag;
  uint32_t hart_id;                   // was `cid` (semantic clarity)
  uint64_t uuid;
  MemOp    op = MemOp::LD;            // primary dispatch
  MemFlags flags;                     // bitfield struct
  uint64_t byteen = 0;
  std::shared_ptr<mem_block_t> data;

  MemReq(MemOp _op = MemOp::LD,
         uint64_t _addr = 0,
         std::shared_ptr<mem_block_t> _data = nullptr,
         uint64_t _byteen = 0,
         uint32_t _tag = 0,
         uint32_t _hart_id = 0,
         uint64_t _uuid = 0);

  bool      is_write()  const { return memop_is_write(op); }    // semantic
  AddrType  addr_type() const { return get_addr_type(addr); }   // derived
};
```

Notes vs. original proposal:
- `flags` shipped as a **`MemFlags` bitfield struct** (typed
  accessors `flags.io`, `flags.amo_unsigned`, …), not a raw
  `uint32_t`. `sizeof == 4` is locked by `static_assert`.
- The DXA sideband (`notify_done`, `notify_bar_id`) was promoted
  **into `MemFlags`** as `dxa_notify_done` / `dxa_notify_bar_id`,
  gated on `EXT_DXA_ENABLE`. The bare fields on `MemReq` are gone.
- `AddrType type` field was **dropped** — `addr_type()` derives it
  from `addr` via `get_addr_type()` (cheap, single source of truth
  with `get_addr_type` already used in adapters).
- `bool write` field is gone; `is_write()` derives from `op`.
- `is_write()` is **semantic** (returns `true` for ST and the AMO
  RMW family). The cache uses a **separate path-oriented**
  `bank_req.write = (core_req.op == MemOp::ST)` to route through
  the store pipeline (AMO uses a dedicated path). This distinction
  was learned during implementation — see comment at the
  `bank_req.write =` assignment site in `cache.cpp`.
- `MemRsp` mirrors the request: `{tag, hart_id, uuid, [data]}`.

### 13.3 AmoUnit — implemented

```cpp
// sim/simx/amo/amo_unit.h (as implemented)
class AmoUnit {
public:
    uint64_t compute(MemOp    op,
                     uint8_t  width,
                     uint64_t old_word,
                     uint64_t rhs,
                     bool     unsigned_minmax = false);

    void reserve   (uint32_t hart_id, uint64_t line_addr);
    void clear     (uint32_t hart_id, uint64_t line_addr);
    bool check     (uint32_t hart_id, uint64_t line_addr) const;
    void invalidate(uint64_t line_addr, uint32_t except);
};
```

`width` is computed by the caller (cache.cpp) from
`__builtin_popcountll(req.byteen)`; `rhs` is extracted from
`req.data` at the AMO byte offset via `amo_load_word()`. This
mirrors the RTL bank's pattern of slicing `byteen_st1` /
`write_word_st1` when feeding the AMO unit. The
`unsigned_minmax` flag comes from `req.flags.amo_unsigned`.

**Kept (not deleted):** `AmoType` is retained because it is part of
**decode** (`IntrAmoArgs::type` carries the decoded AMO sub-op
through the pipeline before LSU translates it into `MemOp`). The
proposal's "delete `AmoType`" line was scoped down during
implementation: the *memory-interface* uses `MemOp` exclusively,
but the *decode→LSU* path keeps `AmoType` as a domain-specific
type. Removed: `amo_to_memop` / `memop_to_amo` conversion helpers
(unused after the LSU switches `AmoType → MemOp` in one place).

### 13.4 Cache dispatch — `op` instead of `write` (implemented)

`mem/cache.cpp` now reads `req.op` (and a small number of typed
`req.flags.*` accessors) on the bank-request critical path:

```cpp
// after (implemented)
bank_req.op      = core_req.op;
bank_req.flags   = core_req.flags;
bank_req.hart_id = core_req.hart_id;
// path-oriented routing — distinct from is_write() (semantic).
// AMO has its own dedicated path; only ST goes through the store path.
bank_req.write   = (core_req.op == MemOp::ST);
…
// store-response gate now opt-in per request:
bool need_core_rsp = !bank_req.write
                   || config_.write_reponse
                   || bank_req.flags.strsp;
…
if (bank_req.flags.io)  { /* uncacheable path */ }
…
if (memop_is_atomic(bank_req.op)) {
    // AMO commit derives width from byteen popcount, rhs from data,
    // and unsigned_minmax from bank_req.flags.amo_unsigned.
    auto rhs = amo_load_word(bank_req.data, bank_req.byteen);
    auto res = amo_unit_->compute(bank_req.op, width, old, rhs,
                                  bank_req.flags.amo_unsigned);
    …
}
```

Notes from implementation:
- `local_mem.cpp` mirrors the same gating: bank rsp uses
  `bank_req.flags.strsp || …` for stores.
- `mem_coalescer.cpp` propagates `flags`, and forwards per-lane
  `tids` for the AMO seed lane. It now merges `data`/`byteen` for
  **stores AND AMOs** (the original `if (in_req.write)` guard was
  too narrow and was the cause of an early AMO RMW data mismatch).
- `local_mem_switch.cpp` propagates `op`, `flags`, `tids` on the
  dc_req path.
- The `#if EXT_A_ENABLED` blocks around the `AmoUnit` member
  declaration and reset stay (the class itself is gated via the
  simx Makefile).

---

## 14. Affected SimX modules — implemented

#### A. Type declarations — **changed**

| File | What changed |
|------|--------------|
| `sim/simx/types.h` | `MemOp` re-numbered to match RTL ordering. `MemFlags` bitfield struct introduced (`strsp`/`io`/`local`/`amo_unsigned`, plus DXA-gated `dxa_notify_done`/`dxa_notify_bar_id`). `MemReq` now `{addr, tag, hart_id, uuid, op, flags, byteen, data}` with derived `is_write()`/`addr_type()`. `MemRsp` is `{tag, hart_id, uuid, [data]}`. `LsuReq` adds per-lane `tids[]` plus warp-uniform `op`/`flags`. **Deleted** `amo_req_t`, `MemReq::amo`, `MemReq::write`, `MemReq::type`, `MemReq::thread_id`, `MemRsp::thread_id`, `LsuReq::write`, `amo_to_memop`, `memop_to_amo`. **Kept** `AmoType` (decode-side type, not memory-interface). |

#### B. Producers — **changed** (set the new fields)

| File | What changed |
|------|--------------|
| `sim/simx/lsu_unit.cpp` | Sets `lsu_req.op` from instruction decode, sets `lsu_req.flags.amo_unsigned` for unsigned MIN/MAX, sets per-lane `lsu_req.tids[i] = entry.tid`, packs rs2 into `data` block + sets `byteen` for AMO via the same path as stores. |
| `sim/simx/lsu_unit.h` | Renamed internal `mshr` field to `pending_reqs` (it tracks pending response routing, has nothing to do with cache misses). |
| `sim/simx/mem/lsu_mem_adapter.cpp` | Computes `mr.hart_id = make_hart_id(req.cid, req.wid, req.tids[i])` at the LsuReq → MemReq conversion site. Sets `mr.flags.io/local` from `get_addr_type(mr.addr)`. Both bypass and multi-lane paths. |
| `sim/simx/mem/mem_coalescer.cpp` | Propagates `flags`. Forwards per-lane `tids` for the AMO seed lane. Merges `data`/`byteen` for stores **and AMOs** (was the cause of an early AMO RMW data mismatch). |
| `sim/simx/mem/local_mem_switch.cpp` | Propagates `op`, `flags`, `tids` on the dc_req path. |
| `sim/simx/dxa/dxa_core.cpp` | Sets `req.flags.dxa_notify_done = 1` and `req.flags.dxa_notify_bar_id = …` on the last LMEM-DMA write (replaces the bare `req.notify_done` / `req.notify_bar_id` fields). Stale `req.cid` → `req.hart_id` rename applied. |
| `sim/simx/cluster.cpp` | Barrier-event release reads from `flags`: `if (req.is_write() && req.flags.dxa_notify_done) core->barrier_event_release(req.flags.dxa_notify_bar_id);` |
| `sim/simx/tcu/tcu_tbuf.cpp` | TBUF reads use `MemReq m(MemOp::LD, addr, nullptr, 0, tag, 0, 0); m.flags.local = 1;` (TBUF reads from LMEM). |

#### C. Consumers — **changed** (read the new fields)

| File | What changed |
|------|--------------|
| `sim/simx/mem/cache.cpp` | `bank_req_t` carries typed `op`, `hart_id`, `flags`. `commitAmo` derives width from `byteen` popcount, `rhs` from `data` via `amo_load_word`, `unsigned_minmax` from `flags`. `bank_req.write = (core_req.op == MemOp::ST)` — path-oriented (AMO uses dedicated path). `need_core_rsp = !bank_req.write \|\| config_.write_reponse \|\| bank_req.flags.strsp`. IO test reads `core_req.flags.io`. |
| `sim/simx/mem/local_mem.cpp` | Bank rsp gate uses `bank_req.flags.strsp \|\| …`. Byte-enabled writes still applied from the TLM payload. |
| `sim/simx/amo/amo_unit.{h,cpp}` | `compute()` takes `MemOp` + explicit `width` + `unsigned_minmax`. |
| `sim/simx/amo/amo_ops.h` | `amo_compute` switches on `MemOp`; `AMO_MIN`/`AMO_MAX` cases branch on `unsigned_minmax` for signed/unsigned compare. |

#### D. Opaque pass-through — **no source change**

Scheduler, scoreboard, dispatcher, sequencer, etc. propagate
`MemReq` by value or reference. They recompile against the new
layout without source edits.

#### E. Build files — **already done**

| File | Status |
|------|--------|
| `sim/simx/Makefile` | `amo/amo_unit.cpp` conditional via `EXT_A_ENABLE`. `cache.cpp`'s `#include "amo_unit.h"` guarded by `#if EXT_A_ENABLED`. |

---

## 15. SimX migration plan — completed

Executed as a 4-stage rollout, validated against AMO ISA + regression
AMO + non-AMO sanity (`dxa_copy`, `draw3d`) at each stage:

### Stage 1 — `op` as primary dispatch ✅
- Repopulated `req.op` at every producer (LSU, DXA, TEX, RASTER, OM,
  TCU). cache.cpp dispatches on `op` exclusively.

### Stage 2 — `cid` → `hart_id`, computed at the LsuMemAdapter ✅
- Renamed across `sim/simx/`.
- Per-lane `hart_id` is computed at the **LsuMemAdapter conversion
  site** via `make_hart_id(cid, wid, tids[i])`, not pre-baked in the
  LsuReq. This required adding per-lane `tids[]` to LsuReq because
  the LSU compaction step renumbers lanes (compact-by-tmask), so
  the lane index ≠ original tid for divergent warps. **Discovered
  during AMO test 7 (lrsc_counter) failure** — the reservation table
  is keyed on the per-lane hart_id, so collapsing to scalar `cid`
  broke LR/SC pairing across SIMD lanes.

### Stage 3 — Delete `amo_req_t`, keep `AmoType` ✅
- Replaced all `req.amo.{valid,op,hart_id,width,rhs}` reads with
  `req.op` / `req.hart_id` / popcount-derived width / `amo_load_word`.
- Deleted `amo_req_t`, `MemReq::amo`, `amo_to_memop`, `memop_to_amo`.
- **Kept `AmoType`** (decode-side type — `IntrAmoArgs::type`).
- Updated `AmoUnit::compute()` to take `MemOp` + `width` +
  `unsigned_minmax`.

### Stage 4 — `MemFlags` bitfield, drop `write`/`type` fields ✅
- Added `MemFlags` bitfield struct to `MemReq`.
- Promoted DXA `notify_done`/`notify_bar_id` into `flags` (DXA-gated).
- Dropped `MemReq::write`, `MemReq::type`, `MemReq::thread_id`,
  `MemRsp::thread_id`, `LsuReq::write` — all replaced by accessors.

### Bonus — bank_req routing semantics ✅
- Made the distinction between `MemReq::is_write()` (semantic — true
  for ST and AMO RMW) and `bank_req.write` (path — true only for ST,
  since AMO uses a dedicated path). Documented at the assignment
  site in `cache.cpp`.

After Stage 4, SimX's `MemReq` shape matches the *intended* RTL
`req_data_t` layout (Part A still pending). The on-wire shape
isn't bit-for-bit yet because Part A hasn't landed; trace-parity
asserts will be added when both halves meet.

---

## 16. SimX–RTL parity

The whole point of mirroring is so SimX traces validate RTL traces
bit-for-bit. SimX (this part) is now in its destination shape; RTL
(Part A) still uses the legacy `amo_req_t` sideband. Bit-level trace
parity will land when Part A is implemented; at that point a shared
header should pin the contract:

```cpp
// sw/common/mem_ops.h (proposed)
static_assert((int)MemOp::LD       == 0);
static_assert((int)MemOp::ST       == 1);
static_assert((int)MemOp::FLUSH    == 2);
static_assert((int)MemOp::AMO_LR   == 3);
…
static_assert((int)MemOp::AMO_MAX  == 11);

// Flag bit positions
static_assert(offsetof_bit(MemFlags, strsp)        == 0);
static_assert(offsetof_bit(MemFlags, io)           == 1);
static_assert(offsetof_bit(MemFlags, local)        == 2);
static_assert(offsetof_bit(MemFlags, amo_unsigned) == 3);
```

Until then, SimX validates against expected functional behavior
(AMO ISA tests + regression `amo`) rather than bit-trace against RTL.

---

## 17. Part B summary — implemented

| What                                | Status |
|-------------------------------------|--------|
| `op` is primary cache dispatch      | ✅ done |
| `cid` → `hart_id` rename            | ✅ done |
| Per-lane `hart_id` via `make_hart_id(cid,wid,tids[i])` | ✅ done |
| `MemFlags` bitfield struct          | ✅ done (typed accessors) |
| DXA notify sideband moved into flags| ✅ done (DXA-gated) |
| `amo_req_t` deleted                 | ✅ done |
| `MemReq::write`/`type`/`thread_id` deleted | ✅ done (derived) |
| `AmoType` retained (decode-side)    | ✅ kept (scoped re-evaluated) |
| `bank_req.write` distinct from `is_write()` | ✅ documented in cache.cpp |
| LSU `mshr` → `pending_reqs` rename  | ✅ done |
| AmoUnit takes `MemOp` + width + unsigned | ✅ done |
| AMO ISA tests pass                  | ✅ 10/10 |
| Regression `amo` passes             | ✅ 12/12 |
| Regression `dxa_copy` passes (simx) | ✅ |
| Regression `draw3d` passes (simx)   | ✅ |

Net code reduction: 3 typedefs, 2 conversion functions, 5 redundant
struct fields, and ~5 nested field accesses per cache.cpp branch
removed. The simx build stays `EXT_A`-conditional via the Makefile
guard; the guard now scopes only the `AmoUnit` class itself, not
field accesses sprinkled through cache.cpp.

---

# Combined summary

| Axis                                     | Part A (RTL — proposed) | Part B (SimX — landed)           |
|------------------------------------------|--------------------------|----------------------------------|
| Files touched                            | 15–19 (none in `libs/`)  | ~12                              |
| Per-instance bit savings                 | 149 (RV32-4) / 553 (RV64-8) | C++ struct shrunk by 6 fields  |
| Eliminates `amo_req_t`?                  | ✓                        | ✓                                |
| Adds typed `op` dispatch?                | ✓ (new `mem_op_e`)       | ✓                                |
| `hart_id` placement?                     | top-level `req_data_t.hart_id` (per-lane on LSU side) | top-level `MemReq::hart_id` (per-lane) |
| `tag_t` touched?                         | **no** (stays opaque)    | n/a                              |
| `flags` reorganized?                     | ✓ (4 bits, room to grow) | ✓ (`MemFlags` bitfield struct)  |
| Trace-level bit-compatible RTL↔SimX?     | future (parity asserts)  | future (waiting on Part A)      |
| Future-proof for §6 catalog?             | ✓                        | ✓                                |

Part B is landed and validated. Part A still pending — when it
lands, add the `static_assert` parity header in §16 to lock the
contract.
