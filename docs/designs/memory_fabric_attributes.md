# Memory-Fabric Attributes & Feature-Agnostic Library IP — Design

**Scope:** the opaque per-request **memory-bus attribute** system
(`mem_bus_attr_t`) that lets feature-specific sidebands (AMO, flush,
IO/LMEM routing) ride the shared memory fabric without contaminating the
reusable library IP. This is what keeps
[`hw/rtl/libs/VX_mem_scheduler.sv`](../../hw/rtl/libs/VX_mem_scheduler.sv)
and [`VX_mem_coalescer.sv`](../../hw/rtl/libs/VX_mem_coalescer.sv)
feature-agnostic.

---

## 1. The attribute model

Memory requests carry an opaque `attr`/`user` field of fixed width
`MEM_ATTR_WIDTH`. The typed layout lives in
[`hw/rtl/VX_gpu_pkg.sv`](../../hw/rtl/VX_gpu_pkg.sv):
`mem_bus_attr_t` ([`:195-200`](../../hw/rtl/VX_gpu_pkg.sv#L195)) with fixed
bit offsets `MEM_ATTR_FLUSH_OFFS=0`, `IO_OFFS=1`, `LOCAL_OFFS=2`,
`AMO_OFFS=3` ([`:207-210`](../../hw/rtl/VX_gpu_pkg.sv#L207)). The AMO
sideband is a slim `amo_req_t` (`{amo_valid, amo_op, amo_unsigned,
hart_id}`, [`:184-189`](../../hw/rtl/VX_gpu_pkg.sv#L184)) packed at
`MEM_ATTR_AMO_OFFS`.

The field is per-lane on
[`VX_lsu_mem_if.sv`](../../hw/rtl/mem/VX_lsu_mem_if.sv) (`user`,
`USER_WIDTH = MEM_ATTR_WIDTH`) and scalar on
[`VX_mem_bus_if.sv`](../../hw/rtl/mem/VX_mem_bus_if.sv) (`attr`).

---

## 2. Feature-agnostic library IP

The shared scheduler and coalescer carry the attribute as opaque
`USER_WIDTH` bits and include only `VX_platform.vh` — they reference no
feature: a grep for `amo|dxa|tcu|EXT_A|VX_gpu_pkg` across `hw/rtl/libs/`
returns nothing. `VX_mem_scheduler` threads `core_req_user`/`mem_req_user`
through its request queue (retaining only the generic `req_queue_rw_notify`);
`VX_mem_coalescer` threads `in_req_user`/`out_req_user`. Graphics caches
(TEX/RASTER/OM) instantiate with `USER_WIDTH(0)`.

The cache tree carries **no** `` `ifdef EXT_A_ENABLE ``: AMO is a
parameter `AMO_ENABLE`
([`VX_cache.sv:72`](../../hw/rtl/cache/VX_cache.sv#L72)) gating generate
blocks, and the cache bank recovers the typed sideband via an offset cast
`amo_req_t'(core_req_attr[MEM_ATTR_AMO_OFFS +: AMO_REQ_BITS])` under
`AMO_ENABLE` ([`VX_cache_bank.sv:136-139`](../../hw/rtl/cache/VX_cache_bank.sv#L136)).
AMO `width`/`rhs` are **derived** at the bank (from `byteen` popcount +
the store word), not carried in the attribute. See
[`atomic_memory_operations.md`](atomic_memory_operations.md) for the AMO consumer.

This is the mechanism that lets one set of memory-fabric IP serve plain
loads/stores, atomics, cache flushes, and IO/LMEM routing without
per-feature edits.

---

## 3. Proposed but not yet implemented

1. **Posted writes / store responses** (out of scope): `rw` stays a bit
   today; a `MEM_OP`-style posted-write encoding is future.
2. **Per-level (L2/L3) distinct attribute typedefs** — the as-built uses
   one unified `mem_bus_attr_t` everywhere; the proposal's separate
   `l2_bus_attr_t`/`l3_bus_attr_t` aliases were not needed.
3. **`VX_axi_adapter` attr→AxUSER 1:1 mapping** — propagating the
   attribute onto AXI `AxUSER` at the platform boundary is unverified.
4. **Standalone validation matrix** (Stage F): a `VX_mem_scheduler`
   unit test building without `EXT_A_ENABLED`, plus `amo`/`dxa_copy`
   smoke — a test checklist not encoded in source.

**Superseded directions** (recorded to avoid revival): the two-layer split
into separate per-lane `lsu_bus_attr_t` + scalar `dcache_bus_attr_t`
typedefs with `DCACHE_ATTR_*_OFF` (collapsed to a single unified
`mem_bus_attr_t` with `MEM_ATTR_*_OFFS`); and a hart_id-less 6-bit
`amo_req_t` (the as-built slim form still carries `hart_id`). The as-built
also folded cache-flush into `mem_bus_attr_t.is_flush` rather than a
separate `MEM_REQ_FLAG_FLUSH`.

This redesign delivered the same end as the abandoned
`amo_packing_optimization` Part-A `mem_op_e` route (AMO-clean library IP),
by the attribute-passthrough mechanism instead — see
[`atomic_memory_operations.md`](atomic_memory_operations.md) §6.

---

## 4. Source proposal

This design consolidates and supersedes `libs_feature_agnostic_redesign.md`
(now removed from `docs/proposals/`).
