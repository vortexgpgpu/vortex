# Lib-tree Feature-Agnostic Redesign + Memory-Interface Attr Refactor

**Date:** 2026-05-08
**Status:** v0.4 — Stage A landed; design refinements per user review.
**Scope:** Restore `hw/rtl/libs/*` to **pure general-purpose IP** (only
`VX_platform.vh`, no `VX_gpu_pkg.sv` / `VX_define.vh` / `EXT_*` macros).
Replace the AMO-specific typedef pollution introduced by commit
`0487e1e8` with **type-agnostic, parameter-driven** ports. Restructure
`flags` into a typed **`attr`** field with separate per-lane (LSU
internal) and scalar (mem-bus) layers.

---

## 1. Why this is needed

Commit `0487e1e8` (RVA atomic extension) added Vortex-specific
typedef ports + `EXT_A_ENABLED` arithmetic to two lib-tree files:

```
hw/rtl/libs/VX_mem_scheduler.sv
hw/rtl/libs/VX_mem_coalescer.sv
```

Direct symptoms:

1. **`hw/unittest/mem_scheduler/`** standalone harness fails to elaborate
   because its build doesn't include `VX_config.vh` and thus has no
   `EXT_A_ENABLED` macro defined. Pre-existing bug, surfaced by
   `ci/regression.sh --all`.
2. The lib-tree purity invariant: `hw/rtl/libs/*` must be reusable IPs
   that include only `VX_platform.vh` — no Vortex configuration tree
   dependency. The AMO commit broke this.

This proposal restores the invariant *and* tightens the bus-interface
typing as a side benefit.

---

## 2. Affected files

| File | Currently polluted? | Action |
|------|:-------------------:|--------|
| `hw/rtl/libs/VX_mem_scheduler.sv` | YES — imports `amo_req_t`, references `EXT_A_ENABLED` | Drop AMO ports + `EXT_A` refs; rename `flags`/`FLAGS_WIDTH` → `user`/`USER_WIDTH`. |
| `hw/rtl/libs/VX_mem_coalescer.sv` | YES — same shape | Same redesign. |
| `hw/rtl/libs/VX_mem_bank_adapter.sv` | NO (only `VX_platform.vh`) | No change. |
| `hw/rtl/libs/VX_mem_data_adapter.sv` | NO (only `VX_platform.vh`) | No change. |
| `hw/rtl/libs/VX_mem_arb.sv`, `VX_mem_xbar.sv`, `VX_mem_switch.sv`, `VX_lsu_mem_arb.sv`, `VX_stream_*` | NO (treat `req_data` opaquely via `$bits`) | No change. |
| `hw/rtl/cache/*` | YES — references `EXT_A_ENABLE`, `amo_req_t` | Replace `EXT_A_ENABLE` with `AMO_ENABLE` parameter. Use scalar `attr` access via offset parameters. |
| `hw/rtl/mem/VX_mem_bus_if.sv` | n/a | Rename `flags` → `attr`, parameterize `ATTR_WIDTH`. |
| `hw/rtl/mem/VX_lsu_mem_if.sv` | n/a | Rename `flags` → `attr` (typed `lsu_bus_attr_t`). |
| `hw/rtl/VX_gpu_pkg.sv` | n/a | New typedefs (`amo_op_e`, slim `amo_req_t`, `lsu_bus_attr_t`, `dcache_bus_attr_t`, etc.); offset localparams. |

---

## 3. New typedefs (in `VX_gpu_pkg.sv`)

### 3.1 AMO opcode + slim request struct

```sv
`ifdef EXT_A_ENABLE
typedef enum logic [3:0] {
    AMO_OP_LR    = 4'h0,  AMO_OP_SC    = 4'h1,
    AMO_OP_ADD   = 4'h2,  AMO_OP_SWAP  = 4'h3,
    AMO_OP_XOR   = 4'h4,  AMO_OP_OR    = 4'h5,
    AMO_OP_AND   = 4'h6,  AMO_OP_MIN   = 4'h7,
    AMO_OP_MAX   = 4'h8
    // MINU/MAXU collapse via amo_req_t.unsigned
} amo_op_e;

// Slim AMO sideband: 6 bits.
// width derives from byteen popcount; rhs from data; hart_id is separate.
typedef struct packed {
    logic     valid;
    amo_op_e  op;
    logic     unsigned;
} amo_req_t;
`endif
```

### 3.2 LSU-internal attr (per-lane fields)

```sv
typedef struct packed {
    // ─── load-bearing field order ───────────────────────────────────
    // (offset localparams below depend on this layout — DO NOT REORDER
    //  without updating LSU_BUS_ATTR_*_OFF localparams)
`ifdef EXT_A_ENABLE
    logic [`NUM_LSU_LANES-1:0][HART_ID_WIDTH-1:0]   hart_id;        // per-lane
    amo_req_t                                        amo;            // scalar
`endif
`ifdef LMEM_ENABLE
    logic [`NUM_LSU_LANES-1:0]                      is_addr_local;   // per-lane
`endif
    logic [`NUM_LSU_LANES-1:0]                      is_addr_io;      // per-lane
} lsu_bus_attr_t;
```

### 3.3 DCache-facing attr (scalar fields)

After the LSU adapter collapses per-lane → per-mem-request:

```sv
typedef struct packed {
    // ─── load-bearing field order ─── (DCACHE_ATTR_*_OFF below)
`ifdef EXT_A_ENABLE
    logic [HART_ID_WIDTH-1:0]   hart_id;        // scalar
    amo_req_t                   amo;            // scalar
`endif
`ifdef LMEM_ENABLE
    logic                       is_addr_local;  // scalar
`endif
    logic                       is_addr_io;     // scalar
} dcache_bus_attr_t;

// Offsets — derived from the layout above
localparam DCACHE_ATTR_IO_OFFS    = 0;
`ifdef LMEM_ENABLE
localparam DCACHE_ATTR_LOCAL_OFFS = 1;
localparam DCACHE_ATTR_AMO_OFF    = 2;
`else
localparam DCACHE_ATTR_AMO_OFF    = 1;
`endif
`ifdef EXT_A_ENABLE
localparam DCACHE_ATTR_HARTID_OFF = DCACHE_ATTR_AMO_OFF + $bits(amo_req_t);
`endif
```

### 3.4 L2 / L3 attrs (alias dcache for now)

```sv
typedef dcache_bus_attr_t l2_bus_attr_t;
typedef dcache_bus_attr_t l3_bus_attr_t;
```

(Aliases since the layout propagates unchanged through cache levels.
Distinct typedefs allow future divergence without API breakage.)

### 3.5 Other consumers — no attr

TEX / RASTER / OM / DXA / icache / tcache / ocache / rcache do not
need attr (no IO/LMEM/AMO concerns at their interfaces). They
instantiate `VX_mem_bus_if` with `ATTR_WIDTH = 0`. The struct field
becomes `[`UP(0)-1:0] = [0:0]` — 1 bit unused, tied off.

---

## 4. Bus interface changes

### 4.1 `VX_mem_bus_if.sv`

```sv
interface VX_mem_bus_if import VX_gpu_pkg::*; #(
    parameter DATA_SIZE     = 1,
    parameter ATTR_WIDTH    = 0,                  // ← renamed from FLAGS_WIDTH; default 0
    parameter TAG_WIDTH     = UUID_WIDTH + 1,
    parameter MEM_ADDR_WIDTH = `MEM_ADDR_WIDTH,
    parameter ADDR_WIDTH    = MEM_ADDR_WIDTH - `CLOG2(DATA_SIZE)
) ();

    typedef struct packed {
        logic [UUID_WIDTH-1:0]           uuid;
        logic [TAG_WIDTH-UUID_WIDTH-1:0] value;
    } tag_t;

    typedef struct packed {
        logic                       rw;
        logic [ADDR_WIDTH-1:0]      addr;
        logic [DATA_SIZE*8-1:0]     data;
        logic [DATA_SIZE-1:0]       byteen;
        logic [`UP(ATTR_WIDTH)-1:0] attr;            // ← renamed from flags; opaque bits
        tag_t                       tag;
    } req_data_t;
    // (no `op`, no `hart_id` field; no `amo` field — all in attr or removed)

    typedef struct packed {
        logic [DATA_SIZE*8-1:0] data;
        tag_t                   tag;
    } rsp_data_t;
    // ...
endinterface
```

### 4.2 `VX_lsu_mem_if.sv`

```sv
interface VX_lsu_mem_if import VX_gpu_pkg::*; #(
    parameter NUM_LANES = 1,
    parameter DATA_SIZE = 1,
    parameter TAG_WIDTH = 1,
    parameter MEM_ADDR_WIDTH = `MEM_ADDR_WIDTH,
    parameter ADDR_WIDTH = MEM_ADDR_WIDTH - `CLOG2(DATA_SIZE)
) ();

    typedef struct packed {
        logic [UUID_WIDTH-1:0]           uuid;
        logic [TAG_WIDTH-UUID_WIDTH-1:0] value;
    } tag_t;

    typedef struct packed {
        logic [NUM_LANES-1:0]                   mask;
        logic                                   rw;
        logic [NUM_LANES-1:0][ADDR_WIDTH-1:0]   addr;
        logic [NUM_LANES-1:0][DATA_SIZE*8-1:0]  data;
        logic [NUM_LANES-1:0][DATA_SIZE-1:0]    byteen;
        lsu_bus_attr_t                          attr;        // typed struct
        tag_t                                   tag;
    } req_data_t;
    // (no `flags`, no `amo`, no `op`, no `hart_id` — all in attr)
    // ...
endinterface
```

### 4.3 LSU adapter — per-lane → scalar conversion

The LSU adapter (or wherever per-warp lsu_mem_if becomes per-channel
mem_bus_if) collapses per-lane attr fields to scalar:

```sv
// For each output mem-channel (ch_idx = lane_idx for non-coalesced;
//                              lane_idx = representative lane for coalesced)
dcache_bus_attr_t out_attr;
assign out_attr.is_addr_io    = lsu_attr.is_addr_io   [lane_idx];
assign out_attr.is_addr_local = lsu_attr.is_addr_local[lane_idx];
assign out_attr.hart_id       = lsu_attr.hart_id      [lane_idx];
assign out_attr.amo           = lsu_attr.amo;   // already scalar
assign mem_bus_if.req_data.attr = out_attr;     // implicit cast struct→flat
```

---

## 5. Cache module changes

### 5.1 New parameters

```sv
module VX_cache #(
    parameter NUM_REQS         = 1,
    ...
    // Attr-related (NEW)
    parameter ATTR_WIDTH       = 0,
    parameter ATTR_IO_OFFS     = 0,    // bit position of is_addr_io within attr
    parameter AMO_ENABLE       = 0,    // 1 = instantiate AMO unit + reservation table
    parameter ATTR_AMO_OFF     = 0,    // bit position of amo_req_t within attr (used iff AMO_ENABLE=1)
    parameter ATTR_HARTID_OFF  = 0,    // bit position of hart_id within attr (used iff AMO_ENABLE=1)
    ...
);
```

### 5.2 Internal extraction

```sv
// Base IO bypass (always, when ATTR_WIDTH > 0)
wire [NUM_REQS-1:0] req_io;
for (genvar i = 0; i < NUM_REQS; ++i) begin
    assign req_io[i] = (ATTR_WIDTH > 0) ? core_req_attr[i][ATTR_IO_OFFS] : 1'b0;
end

// AMO logic — gated on AMO_ENABLE
if (AMO_ENABLE) begin : g_amo
    for (genvar i = 0; i < NUM_REQS; ++i) begin
        amo_req_t amo;
        wire [HART_ID_WIDTH-1:0] hart_id;
        assign amo     = core_req_attr[i][ATTR_AMO_OFF +: $bits(amo_req_t)];   // typed cast
        assign hart_id = core_req_attr[i][ATTR_HARTID_OFF +: HART_ID_WIDTH];
        // ... AMO ALU + reservation table
    end
end
```

### 5.3 `EXT_A_ENABLE` removal from cache modules

All `` `ifdef EXT_A_ENABLE `` blocks in `hw/rtl/cache/*` get replaced
by `if (AMO_ENABLE)` parameter-gated generate blocks. The
`VX_amo_unit.sv` / `VX_amo_alu.sv` files remove their internal
`` `ifdef EXT_A_ENABLE `` (always-compile, only-instantiate-when-enabled).

`EXT_A_ENABLE` stays everywhere outside `hw/rtl/cache` and `hw/rtl/libs`
(LSU instruction decode, gpu_pkg typedef gating, etc.).

### 5.4 Instantiation example

LSU's dcache:
```sv
.ATTR_WIDTH       ($bits(dcache_bus_attr_t)),
.ATTR_IO_OFFS     (DCACHE_ATTR_IO_OFFS),
.AMO_ENABLE       (`EXT_A_ENABLED),
.ATTR_AMO_OFF     (DCACHE_ATTR_AMO_OFF),
.ATTR_HARTID_OFF  (DCACHE_ATTR_HARTID_OFF),
```

TEX/RASTER/OM caches:
```sv
.ATTR_WIDTH (0),
.AMO_ENABLE (0),
// ATTR_*_OFFS use defaults (unused)
```

---

## 6. Lib changes (`VX_mem_scheduler.sv`, `VX_mem_coalescer.sv`)

### 6.1 Removals

- `import VX_gpu_pkg::amo_req_t / AMO_REQ_BITS;`
- `core_req_amo` input port
- `mem_req_amo` output port
- Internal `reqq_amo`, `reqq_amo_s`, `mem_req_amo_s`, `mem_req_amo_b`
- `` `EXT_A_ENABLED * AMO_REQ_BITS `` arithmetic in queue widths
- All `` `ifdef EXT_A_ENABLE `` blocks pack/unpacking amo

### 6.2 Renames

- Parameter `FLAGS_WIDTH` → `USER_WIDTH`
- Port `core_req_flags` → `core_req_user`
- Port `mem_req_flags` → `mem_req_user`
- Internal `reqq_flags` → `reqq_user`, etc.

### 6.3 Result

After redesign, libs files contain only:

```sv
`include "VX_platform.vh"
```

No `import VX_gpu_pkg`, no `` `EXT_A_ENABLED ``, no `amo_req_t` —
pure general-purpose IP again.

`req_queue_rw_notify` output **stays** (existing semantic — fires on
write requests being queued). Not renamed in this round (`posted`
not introduced).

---

## 7. Migration plan

### Stage A — Add attr typedefs + offsets (additive, no behavior change)

- Add typedefs to `VX_gpu_pkg.sv`: `amo_op_e`, slim `amo_req_t`,
  `lsu_bus_attr_t`, `dcache_bus_attr_t`, `l2_bus_attr_t`,
  `l3_bus_attr_t`, offset localparams.
- Build still uses old `flags` field; new typedefs unused.

### Stage B — Switch interfaces to `attr` (renames + struct typing)

- Rename `flags` → `attr` in `VX_mem_bus_if.sv` and `VX_lsu_mem_if.sv`.
- Update consumers' field references (`req_data.flags` → `req_data.attr`).
- LSU adapter: introduce per-lane → scalar conversion.
- Cache reads `attr[ATTR_IO_OFFS]` for IO bypass instead of `flags[MEM_REQ_FLAG_IO]`.

### Stage C — Migrate cache to `AMO_ENABLE` parameter

- Add `AMO_ENABLE` + `ATTR_*_OFF` parameters to `VX_cache`,
  `VX_cache_cluster`, `VX_cache_wrap`, `VX_cache_bank`.
- Replace `` `ifdef EXT_A_ENABLE `` in cache modules with
  `if (AMO_ENABLE)` generate blocks.
- Cache bank reads AMO via typed cast at `ATTR_AMO_OFF`.
- Drop `core_req_amo` / `mem_req_amo` port plumbing in cache.

### Stage D — Lib redesign (drop AMO pollution)

- Rename `flags`/`FLAGS_WIDTH` → `user`/`USER_WIDTH` in
  `VX_mem_scheduler.sv` + `VX_mem_coalescer.sv`.
- Remove `core_req_amo`/`mem_req_amo` ports.
- Remove `import VX_gpu_pkg::amo_req_t / AMO_REQ_BITS`.
- Remove `` `EXT_A_ENABLED * AMO_REQ_BITS `` arithmetic.
- Verify libs files reference only `VX_platform.vh` (grep clean).

### Stage E — Cleanup

- Delete bloated `amo_req_t` (the wide one with `width`, `rhs`,
  `hart_id`, `valid`, `op` — replaced by slim 6-bit version).
- Delete `INST_AMO_*` localparams (replaced by `amo_op_e` enum).
- Delete `MEM_REQ_FLAG_*` localparams (subsumed by typed `attr` fields).
- Drop the legacy AMO-via-`rw=0` trick if cleanly replaceable.

### Stage F — Validation

- Smoke (`sgemm` + `amo` + `dxa_copy` on `rtlsim`, 60s each).
- Full `ci/regression.sh --all` end-to-end.
- Standalone `hw/unittest/mem_scheduler/` builds without
  `EXT_A_ENABLED` defined.

---

## 8. Out of scope

- `posted` / STRSP — `rw` stays. Future-work.
- `MEM_REQ_FLAG_FLUSH` → `MEM_OP_FLUSH` migration. Independent of this
  proposal; the FLUSH flag stays a flag bit (now a bit in attr if used,
  or in flush-bus's own typedef).
- L2/L3 having distinct attrs (currently aliased to `dcache_bus_attr_t`).
- AXI converter (`VX_axi_adapter.sv`) updates: maps `attr` → `AxUSER`
  1:1 for AMO-aware paths; for non-AMO paths AxUSER width can be 0.

---

## 9. Final consensus signal/typedef table

| Layer | Type | Where | Notes |
|-------|------|-------|-------|
| `amo_op_e` | enum 4-bit | `VX_gpu_pkg.sv` | Replaces `INST_AMO_*` localparams |
| `amo_req_t` | struct {valid, op, unsigned} 6-bit | `VX_gpu_pkg.sv` | Slim — width/rhs derivable, hart_id separate |
| `lsu_bus_attr_t` | per-lane struct | `VX_gpu_pkg.sv` | LSU pipeline internal |
| `dcache_bus_attr_t` | scalar struct | `VX_gpu_pkg.sv` | Post-LSU-adapter, mem-bus-facing |
| `l2_bus_attr_t` | alias of dcache | `VX_gpu_pkg.sv` | LLC commit |
| `l3_bus_attr_t` | alias of dcache | `VX_gpu_pkg.sv` | (typically same as L2) |
| Bus `attr` field | flat `[`UP(ATTR_WIDTH)-1:0]` | `VX_mem_bus_if.sv` | Renamed from `flags`. ATTR_WIDTH=0 for non-LSU consumers. |
| LSU bus `attr` field | typed `lsu_bus_attr_t` | `VX_lsu_mem_if.sv` | Per-lane fields inside |
| Lib `user` port | flat `[`UP(USER_WIDTH)-1:0]` | `VX_mem_scheduler.sv`, `VX_mem_coalescer.sv` | Renamed from `flags` |
| Cache `ATTR_*_OFFS` params | int | `VX_cache.sv` etc. | `ATTR_IO_OFFS`, `ATTR_AMO_OFF`, `ATTR_HARTID_OFF` |
| Cache `AMO_ENABLE` param | bit | `VX_cache.sv` etc. | Replaces `` `ifdef EXT_A_ENABLE `` in cache |
