# VM Hierarchy Redesign — True-GPU-Aligned TLB & Centralized PTW

**Scope:** [hw/rtl/mem/VX_mmu.sv](../../hw/rtl/mem/VX_mmu.sv), [VX_mmu_tlb.sv](../../hw/rtl/mem/VX_mmu_tlb.sv), [VX_mmu_ptw.sv](../../hw/rtl/mem/VX_mmu_ptw.sv), [hw/rtl/Vortex.sv](../../hw/rtl/Vortex.sv), [sim/simx/mem/mmu.{h,cpp}](../../sim/simx/mem/), [sim/simx/core.cpp](../../sim/simx/core.cpp), [VX_config.toml](../../VX_config.toml)
**Reference:** gem5 `GPU-VIPER` (`X86GPUTLB`, `GPUCoalescer`, walker layout)
**Branch:** `feature_vm_v2` (off `feature_vm`)
**Status:** Proposal — supersedes the prior `mmu_perf_optimization_proposal.md`
**Related:** [vm_migration_proposal.md](vm_migration_proposal.md), [../vm.md](../vm.md), [command_processor_proposal.md](command_processor_proposal.md)

---

## 1. Motivation

The current per-core MMU that landed in `feature_vm` (`a4699d8f`) is
functionally correct but does not match how mainstream GPUs structure
address translation. Specifically:

- **Placement:** a single per-core MMU with a 32-entry CAM TLB and an
  embedded single-walk PTW FSM. No L2/L3 TLBs. No sharing across cores.
- **Hierarchy:** translation collapses into one level. Capacity, hit
  rate, walk concurrency, and PTE-locality are all bound by that one
  level.
- **Concurrency:** lanes serialize through a single CAM port; misses
  block the pipeline; one walk in flight at a time.
- **CP interaction:** the CP's DMA engine sits inside the CP RTL and
  bypasses Vortex's MMU entirely. There is no path for translated bulk
  data movement.
- **Configurability:** TLB sizing is a single `VX_CFG_TLB_SIZE` knob; there is
  no notion of policy (inclusion, banking, walker count).

Mainstream GPUs (AMD CDNA / GPU-VIPER, NVIDIA Volta+ GMMU) structure
this differently: small per-CU L1 TLBs after coalescing, a shared L2
TLB per cluster, a chip-wide L3 TLB (or system-level MMU TLB) co-located
with the LLC, and a centralized multi-walker PTW with a PTE walk-cache.

This proposal restructures Vortex's VM subsystem to that model.

---

## 2. Reference architecture

Patterned after gem5 `GPU-VIPER`'s `X86GPUTLB` arrangement:

- **L1 TLB:** small (16 entries), fully-associative, PLRU, one per core.
  Both data-side (post-coalescer) and instruction-side (per fetch unit)
  instances.
- **L2 TLB:** larger (2048 entries), 8-way set-associative SRAM, PLRU,
  one per cluster, shared across the cluster's cores.
- **L3 TLB:** chip-wide (8192 entries), 16-way set-associative SRAM,
  PLRU, co-located with the LLC.
- **PTW:** a single chip-wide engine co-located with the IS_LLC TLB
  (the deepest enabled level). Configurable walker count (default 4),
  with a PTE walk-cache.

Sizing rationale comes directly from `GPU-VIPER`'s defaults; Vortex
keeps the same numbers as the v1 baseline so cross-comparison is clean.

---

## 3. Scope (v1)

In scope:

- 3-level TLB hierarchy (L1 / L2 / L3) with the gem5 sizing above.
- Non-blocking miss handling at every level (MSHR, max 16 entries).
- Multi-banked L1 DTLB sitting **after** the memory coalescer.
- Per-core L1 ITLB on the fetch path.
- Centralized multi-walker PTW at IS_LLC with PTE walk-cache.
- `IS_LLC` semantics for TLBs (same contract as caches): the deepest
  enabled level holds the PTW; anything beyond it is passthru.
- New configuration knobs (§6) covering sizes, ways, banks, walker
  count, inclusion policy.
- Single global VA space — SATP programmed once at GPU init, never
  changes per dispatch (§11).
- Broadcast TLB invalidation on `SFENCE.VMA` (§10).
- CP/DMA split: CP retains command orchestration; data movement moves
  to a new pre-LLC DMA block (`VX_dma`) with its own TLB feeding the
  centralized PTW (§12).
- SV32 implementation. PTW FSM and walk-cache generalized over walk
  depth so SV39 follows in v1.1.
- Configuration knobs declared for cache inclusion policy too (§9),
  defaulting to `NINE` (current de-facto behavior).

Out of scope:

- ASIDs (single-process model justifies absence — see §11).
- PMP / page protection enforcement.
- A/D bit writeback (runtime continues to pre-set `A=D=1` on PTE
  install, TLBs are read-only).
- SV48 / SV57 addressing modes.
- Cache inclusion policy implementations other than `NINE` (the knob is
  declared with all three values, but `STATIC_ASSERT` restricts to
  `NINE` for v1; INCL/EXCL cache machinery is a separate proposal).
- Multi-process / multi-context isolation.
- Cycle-accurate per-stage SimX modeling of the TLB hierarchy (SimX
  remains functional + perf counters; see §13).

---

## 4. Architecture overview

```
   ┌─────────────────────────────  Core 0 ──────────────────────────────┐
   │                                                                    │
   │  fetch ─► L1 ITLB (8 entries, FA)                                  │
   │            │  miss                                                 │
   │            ▼                                                       │
   │  LSU lanes ─► coalescer ─► L1 DTLB (16 entries, FA, multi-banked)  │
   │                              │ miss                                │
   └──────────────────────────────┼─────────────────────────────────────┘
                                  │
                  ┌───────────────┴──────────────────┐
                  │       L2 TLB (per cluster)       │  2048 entries, 8-way
                  │       — shared across cores —    │  non-blocking, MSHR=16
                  └───────────────┬──────────────────┘
                                  │ miss
                                  ▼
                  ┌──────────────────────────────────┐
                  │       L3 TLB (chip-wide)         │  8192 entries, 16-way
                  │       co-located with LLC        │  non-blocking, MSHR=16
                  └───────────────┬──────────────────┘
                                  │ miss
                                  ▼
                  ┌──────────────────────────────────┐
                  │  PTW (chip-wide, N walkers,      │
                  │  PTE walk-cache)                 │
                  │  - issues PTE fetches via LLC    │
                  └──────────────────────────────────┘

   Separate translated data path:

   host AXI ────► VX_dma ─► L1 DTLB-equivalent ─► (L2 ─► L3) ─► LLC ─► dev mem
                    ▲
                    │ commands from CP (control plane only)
                    │
                  ┌─┴───┐
                  │ CP  │ (orchestration, FSM, completion)
                  └─────┘
```

Key invariants:

1. **`IS_LLC` floats with enabled levels.** If L3 is disabled, IS_LLC =
   L2 (and PTW lives there). If both L2 and L3 are disabled, IS_LLC =
   L1 (and PTW collapses to per-core, the legacy mode). The contract is
   identical to `VX_cache_wrap.sv:192` for caches.
2. **TLB placement after coalescer.** L1 DTLB sees coalesced (and
   thus fewer) requests, reducing TLB port count. Coalescing operates
   on virtual addresses — same as AMD/NVIDIA. A coalesced bundle that
   straddles two pages is split back out at the L1 DTLB stage.
3. **Translated DMA.** `VX_dma` issues device-side accesses through the
   same TLB hierarchy as compute; host-side accesses go through the
   platform shim (XRT/OPAE) untranslated.
4. **Single global page table.** SATP is fixed for the device's
   lifetime; no per-dispatch CSR changes.

---

## 5. Configuration parameters

Parameters split between [VX_types.toml](../../VX_types.toml) and
[VX_config.toml](../../VX_config.toml) following the existing
convention:

- **`VX_types.toml`** — HW↔SW contract constants (page size, addressing
  mode, PT layout, page-table base address) consumed by both the
  hardware and the SW VM manager / runtime / loader. Identifiers carry
  the `VX_` prefix with a section sub-prefix: `VX_VM_*` in `[vm]`,
  `VX_MEM_*` in `[memmap]`.
- **`VX_config.toml`** — HW-only microarchitecture knobs (TLB sizes,
  ways, banks, MSHR depth, walker count, inclusion policy). All
  identifiers carry the `VX_CFG_` prefix and follow the cache-side
  convention (`VX_CFG_DCACHE_*`, `VX_CFG_ICACHE_*`, `VX_CFG_L2_*`,
  `VX_CFG_L3_*`, with `_ENABLE` / `_ENABLED`, `_NUM_WAYS`,
  `_NUM_BANKS`, `_MSHR_SIZE`, `_REPL_POLICY`).

### 5.1 SW-visible VM constants — `VX_types.toml` (already in place)

The VM page-table format constants already live in `VX_types.toml`
`[vm]` and the page-table base address in `VX_types.toml` `[memmap]`
(relocated from `VX_config.toml` in a prior pass — see
[config_hw_sw_layering_proposal.md](config_hw_sw_layering_proposal.md)).
This proposal makes **no further changes** to them; they are listed
here so the design is self-contained:

```toml
# VX_types.toml
[vm]
VX_VM_PAGE_LOG2_SIZE = 12
VX_VM_PAGE_SIZE      = "expr: 1 << $VX_VM_PAGE_LOG2_SIZE"
VX_VM_ADDR_MODE      = "expr: 'SV39' if ($XLEN == 64) else 'SV32'"
VX_VM_PT_LEVEL       = "expr: 3 if ($XLEN == 64) else 2"
VX_VM_PTE_SIZE       = "expr: 8 if ($XLEN == 64) else 4"
VX_VM_PT_SIZE        = "expr: $VX_VM_PAGE_SIZE"
VX_VM_PT_SIZE_LIMIT  = "expr: (1 << 25) if ($XLEN == 64) else (1 << 23)"

[memmap]
VX_MEM_PAGE_TABLE_BASE_ADDR = "expr: 0x00000000F0000000 if ($XLEN == 64) else 0xF0000000"
```

Notes: `$XLEN` is the `[[builtin]]` axis declared in `VX_types.toml`
(not `$XLEN_64`); the prior pass also dropped the dead
`VX_NUM_PTE_ENTRY` entry and it is not reintroduced.

### 5.2 L1 DTLB / ITLB (per core) — `VX_config.toml`, `[vm]`

```toml
# VX_config.toml
[vm]
# L1 DTLB (per core; placed after the memory coalescer)
VX_CFG_DTLB_SIZE         = 16                              # entries; fully-associative
VX_CFG_DTLB_NUM_BANKS    = "expr: min(4, $VX_CFG_DCACHE_NUM_REQS)"
VX_CFG_DTLB_BANK_HASH    = 0                               # 0 = VPN low bits, 1 = VPN XOR hash
VX_CFG_DTLB_MSHR_SIZE    = 16
VX_CFG_DTLB_REPL_POLICY  = "expr: $__cache_repl_plru"

# L1 ITLB (per core; on the fetch path)
VX_CFG_ITLB_SIZE         = 8                               # entries; fully-associative
VX_CFG_ITLB_MSHR_SIZE    = 16
VX_CFG_ITLB_REPL_POLICY  = "expr: $__cache_repl_plru"
```

`VX_CFG_TLB_SIZE = 32` (the old single knob) is removed.

### 5.3 L2 / L3 TLB — `VX_config.toml`, `[vm]`

Mirroring the cache convention exactly:

```toml
# VX_config.toml
[vm]
# L2 TLB (per cluster)
VX_CFG_L2_TLB_ENABLE      = true
VX_CFG_L2_TLB_SIZE        = 2048
VX_CFG_L2_TLB_NUM_WAYS    = 8
VX_CFG_L2_TLB_MSHR_SIZE   = 16
VX_CFG_L2_TLB_REPL_POLICY = "expr: $__cache_repl_plru"

# L3 TLB (chip-wide; co-located with LLC)
VX_CFG_L3_TLB_ENABLE      = true
VX_CFG_L3_TLB_SIZE        = 8192
VX_CFG_L3_TLB_NUM_WAYS    = 16
VX_CFG_L3_TLB_MSHR_SIZE   = 16
VX_CFG_L3_TLB_REPL_POLICY = "expr: $__cache_repl_plru"

# Derived enable flags (mirror cache convention: VX_CFG_L2_ENABLE → VX_CFG_L2_ENABLED)
VX_CFG_L2_TLB_ENABLED     = "expr: 1 if $VX_CFG_L2_TLB_ENABLE else 0"
VX_CFG_L3_TLB_ENABLED     = "expr: 1 if $VX_CFG_L3_TLB_ENABLE else 0"

# IS_LLC for TLBs: deepest enabled level (matches cache convention)
VX_CFG_TLB_IS_LLC_LEVEL   = "expr: 3 if $VX_CFG_L3_TLB_ENABLE else (2 if $VX_CFG_L2_TLB_ENABLE else 1)"
```

### 5.4 PTW — `VX_config.toml`, `[vm]`

```toml
# VX_config.toml
[vm]
VX_CFG_PTW_NUM_WALKERS     = 4          # fixed default; bounded by mem BW
VX_CFG_PTW_WALK_CACHE_SIZE = 8          # direct-mapped, per non-leaf level
```

### 5.5 Inclusion policy — `VX_config.toml`

Per-section knobs (cache hierarchy vs. TLB hierarchy), independently
controllable. String-enum values follow the `VX_CFG_FPU_TYPE` /
`VX_CFG_TCU_TYPE` precedent:

```toml
# VX_config.toml
[l2cache]
VX_CFG_L2_INCL_POLICY  = "NINE"         # NINE | INCL | EXCL — v1 = NINE only

[vm]
VX_CFG_TLB_INCL_POLICY = "NINE"         # NINE | INCL | EXCL — v1 supports all three

# Enum declarations (added to the existing VX_config.toml [[enum]] table)
[[enum]]
VX_CFG_L2_INCL_POLICY  = ["NINE", "INCL", "EXCL"]
VX_CFG_TLB_INCL_POLICY = ["NINE", "INCL", "EXCL"]
```

`STATIC_ASSERT(VX_CFG_L2_INCL_POLICY_NINE)` enforces NINE-only for the
cache side in v1 (matches the `VX_CFG_FPU_TYPE_<value>` pattern emitted
by the config generator for string enums).

### 5.6 DMA address translation — `VX_config.toml`, `[vm]`

```toml
# VX_config.toml
[vm]
VX_CFG_DMA_TLB_ENABLE    = true         # VX_dma translates device-side via TLB hierarchy
VX_CFG_DMA_TLB_SIZE      = 32           # private L1-style TLB inside VX_dma
VX_CFG_DMA_TLB_NUM_WAYS  = 0            # 0 = fully-associative (cache convention)
VX_CFG_DMA_TLB_MSHR_SIZE = 16
```

---

## 6. TLB design (per level)

### 6.1 L1 DTLB

- 16 entries, fully-associative CAM, PLRU.
- Multi-banked. Default `min(4, VX_CFG_DCACHE_NUM_REQS)`.
- Bank hash: VPN low bits in v1 (`VX_CFG_DTLB_BANK_HASH=0`). Optional
  VPN[low] XOR VPN[mid] mode if bank-conflict regression shows it.
- **Placed after the memory coalescer**, so input port count =
  `VX_CFG_DCACHE_NUM_REQS / coalescing_factor` (one per coalesced request).
- Coalesced bundles spanning two pages are split at the DTLB stage:
  the coalescer guarantees within-cacheline coalescing, which for the
  4 KB page case means same-page (cacheline is much smaller than a
  page), so this split is a corner-case path, not the common path.
- Non-blocking with 16-entry MSHR: a missing VPN parks in the MSHR,
  other lanes/banks continue translating hits. Multiple lanes missing
  on the *same* VPN coalesce on the MSHR (no duplicate L2 query).

### 6.2 L1 ITLB

- 8 entries, fully-associative, PLRU.
- One port (fetch is single-stream).
- Non-blocking with 16-entry MSHR for the same coalescing benefit
  when fetch crosses pages.
- Misses go to the same shared L2 TLB as the DTLB.

### 6.3 L2 TLB (per cluster)

- 2048 entries, 8-way set-associative SRAM, PLRU.
- One per cluster; shared across the cluster's cores (and across
  D-side / I-side traffic).
- Hit latency target: 3-4 cycles (SRAM tag check + way mux).
- Non-blocking with 16-entry MSHR. Misses query L3 (if enabled) or PTW.
- Input arbitration: round-robin across upstream L1 TLBs.

### 6.4 L3 TLB (chip-wide)

- 8192 entries, 16-way set-associative SRAM, PLRU.
- Co-located with the LLC; shared across all clusters.
- Hit latency target: 8-10 cycles + interconnect.
- Non-blocking with 16-entry MSHR. Misses query the PTW.

### 6.5 Common properties

- All levels are **read-only TLBs** (no A/D writeback; runtime
  pre-sets `A=D=1`).
- All levels participate in `SFENCE.VMA` invalidation broadcast (§10).
- `IS_LLC` flag is statically computed from enable bits; the level
  with `IS_LLC=1` is the one hosting the PTW interface.

---

## 7. PTW design

### 7.1 Placement

The PTW lives at the IS_LLC TLB. With both L2 and L3 enabled, that's
L3. With only L2 enabled, the PTW lives at L2. With neither, the PTW
falls back to per-core inside the L1 TLB (legacy mode).

### 7.2 Walkers

- `VX_CFG_PTW_NUM_WALKERS = 4` by default. Range 2-16.
- Each walker holds independent state: `{state, vaddr, level_ppns[],
  pending_tag, originating_tlb_id}`.
- PTW is non-blocking: `miss_ready = (|free_walker_mask)`.
- PTE fetches go through the LLC (not direct memory) — same
  TLM-correct path the current per-core MMU uses.

### 7.3 Walk-cache (PTE cache)

- Direct-mapped, 8 entries per non-leaf level.
- For SV32: caches L1 (top-level) PTEs.
- For SV39: caches L2 and L1 PTEs.
- Indexed by VPN[level] low bits.
- On walk start, look up the walk-cache; if hit at level N, skip
  fetches for levels N+1..top. Expected ~50% latency reduction for
  spatially-local workloads. Larger gain for SV39 (more levels).

### 7.4 Walk depth generalization

The PTW FSM is parameterized on `VX_VM_PT_LEVEL` (the HW↔SW constant
in `VX_types.toml` `[vm]`). SV32 → 2 levels; SV39 → 3 levels. The state
machine is a counted loop over `VX_VM_PT_LEVEL-1` indirect fetches
followed by a leaf load. Walk-cache covers the top `VX_VM_PT_LEVEL-1`
levels.

---

## 8. Inclusion policy

### 8.1 TLB inclusion (v1: all three policies supported)

`VX_CFG_TLB_INCL_POLICY` ∈ {`NINE`, `INCL`, `EXCL`}, default `NINE`.

- **NINE (default):** L2 fill brings entry into L2 only. L1 miss
  queries L2; on L2 hit, copy into L1 (now in both). L2 evictions do
  not propagate to L1.
- **INCL:** L2 ⊇ L1. On L2 eviction, broadcast back-invalidate to all
  L1 TLBs.
- **EXCL:** L1 ∩ L2 = ∅. On L1 miss + L2 hit, move (not copy) from L2
  to L1. On L1 eviction, demote to L2.

All three are cheap for TLBs (no dirty data, no coherence protocol).
Implementation effort is bounded; making the knob real in v1 is justified
because it enables policy-comparison experiments.

### 8.2 Cache inclusion (v1: NINE-only)

`VX_CFG_L2_INCL_POLICY` ∈ {`NINE`, `INCL`, `EXCL`}, default `NINE`. v1
enforces `NINE` via `STATIC_ASSERT`. The knob exists so the config
surface is stable from day one; INCL/EXCL machinery (back-invalidate
paths, dirty-line handling, MSHR adjustments) is a separate proposal
because the implementation depth is much larger than for TLBs.

---

## 9. Invalidation / `SFENCE.VMA`

v1 strategy: **broadcast invalidation**.

- `SFENCE.VMA` with no VA: full-flush every TLB level (L1 of every
  core, L2 of every cluster, L3). Walk-cache also flushed.
- `SFENCE.VMA vaddr`: range-scoped — broadcast a single-VPN invalidate
  to all L1s, invalidate matching set at L2/L3.
- PTW walkers are not flushed mid-walk (the originating TLB will
  re-issue if needed).

Broadcast is the simplest correct mechanism. It costs more than ASID-
scoped invalidation, but `SFENCE.VMA` is rare in the single-global-page-
table model (§11): only the host runtime issues it, only on actual
mapping mutations (alloc, free, mprotect). The kernel boundary is *not*
a flush event.

---

## 10. Single global VA space

The device runs with one page table shared across all queues, cores,
and concurrent dispatches.

- Host driver builds one page table at GPU init, programs SATP via DCR
  once.
- KMU never reprograms SATP on dispatch.
- All concurrent kernels — across all CP queues — see the same VA→PA
  mapping. The runtime allocates VA space globally per device.
- `SFENCE.VMA` fires only on host-driven mapping mutations.
- No ASIDs needed; no per-queue context.

This matches the model used by HSA shared virtual memory and CUDA
unified memory at the driver level. The architectural cost is
"no multi-process on the same device" — not a current Vortex requirement.

If multi-process ever becomes a requirement, ASIDs follow as a clean
tag-width extension to TLB entries and a SATP-per-queue CSR addition,
with no impact on the v1 structural choices.

---

## 11. CP / DMA split

### 11.1 Motivation

Today `VX_cp_dma` (per [command_processor_proposal.md §6.6](command_processor_proposal.md))
lives inside the CP and runs its data path on a borrowed Vortex memory
port or a shared host AXI fabric. It bypasses the MMU entirely, so:

- Bulk transfers don't exercise the translation hierarchy (no perf
  signal from DMA traffic on TLB pressure).
- VA-to-VA copies between regions of the shared VA space aren't
  expressible.
- The CP's RTL carries data-movement state that is logically separate
  from command sequencing.

### 11.2 Design

A new top-level block `VX_dma` is split out of `rtl/cp/`:

```
CP (control)                  VX_dma (data)
  ┌────────────┐                ┌──────────────────┐
  │ CPE        │   command      │ DMA engine       │
  │ decodes    │───────────────►│ src/dst/size FSM │
  │ CMD_MEM_*  │                │ AXI initiator    │
  │            │◄───────────────│ completion       │
  │            │   done/err     │                  │
  └────────────┘                └──────────────────┘
                                       │
                                       ▼ device side
                              ┌──────────────────┐
                              │ DMA L1 TLB       │ 32 entries
                              │ (private)        │
                              └────────┬─────────┘
                                       │ miss
                                       ▼
                              ┌──────────────────┐
                              │ L2 TLB (cluster) │
                              │  → L3 → PTW      │
                              └──────────────────┘
                                       │ hit
                                       ▼
                              ┌──────────────────┐
                              │      LLC         │
                              └──────────────────┘
```

- CP retains the command FSM, arbitration, completion writeback, and
  `CMD_MEM_*` decode. The CPE dispatches a transfer request to
  `VX_dma` and blocks on its done signal before retiring the command.
- `VX_dma` owns the data path: source/dest addresses, burst sizing,
  AXI initiator, completion to CP.
- Device-side endpoints translate through the shared TLB hierarchy.
- Host-side endpoints go through the platform shim untranslated (XRT
  BAR / OPAE). This is the same as today — host-side memory was never
  translated.
- `VX_CFG_CP_DMA_DEV_PORT_MODE = SHARED | DEDICATED` is preserved;
  SHARED is still v1 default for single-bank XRT shells.

### 11.3 What the "IO region bypass" idea was — and why it's dropped

A prior design sketch suggested marking the host command queue's
address range as an IO region in the dcache MMU to skip translation.

This is dropped because:

1. The CP's command fetch path **already** bypasses the MMU — the CP
   runs its own AXI master through the platform shim, never touching
   any Vortex core's dcache port. There is no MMU to "skip."
2. Carving an IO region into the GPU VA to let kernels peek at host
   memory is not how real GPUs do SVM — that's the IOMMU/ATC's job. If
   Vortex ever needs SVM, propose an IOMMU model; don't hack the IO
   region.

The IO region in `VX_config.toml` continues to mean what it means
today: MMIO-style device registers (cout, exit code), identity-mapped,
not translatable from kernel context.

---

## 12. SimX modeling strategy

SimX remains **functional + perf counters** for translation, not
cycle-accurate per stage:

- Functional path: each TLB level is a SimObject that performs lookup
  on every request. Hits return immediately with the configured
  latency credited to the request's commit time. Misses recurse to the
  next level.
- PTW is a SimObject with N walker slots and the same walk-cache
  behavior as RTL.
- Perf counters per level: reads, hits, misses, evictions, MSHR
  occupancy, walker occupancy, walk-cache hit rate.

This matches how the current cache hierarchy is modeled in SimX
(latency credit, no per-stage pipelining). Cycle-accurate per-stage
TLB modeling is a separate proposal if ever justified.

The new SimX objects live in [sim/simx/mem/](../../sim/simx/mem/):

- `tlb_l1.{h,cpp}` — per-core L1 TLBs (D + I)
- `tlb_l2.{h,cpp}` — per-cluster L2 TLB
- `tlb_l3.{h,cpp}` — chip-wide L3 TLB
- `ptw.{h,cpp}` — centralized PTW (replaces the embedded PTW in
  `mmu.cpp`)
- `dma.{h,cpp}` — `VX_dma` model

The existing `mmu.{h,cpp}` becomes a thin wrapper that wires
per-core L1 DTLB + L1 ITLB to the cluster-level L2 TLB binding; the
embedded PTW FSM is gone.

---

## 13. Latency budget (targets, not contract)

| Level         | RTL hit | SimX credit | Miss penalty (next-level hit) |
|---------------|---------|-------------|-------------------------------|
| L1 DTLB / ITLB| 1 cyc   | 1           | +3-4 to L2                    |
| L2 TLB        | 3-4 cyc | 4           | +8-10 to L3                   |
| L3 TLB        | 8-10 cyc| 10          | +PTW                          |
| PTW (walk-cache hit, 2-level walk) | ~10 cyc | 10 | LLC PTE latency |
| PTW (walk-cache miss, 2-level walk) | ~25 cyc| 25 | LLC PTE latency × 2 |

Numbers anchor the design's timing-feasibility argument; final values
are confirmed in synthesis (RTL) and tuned in [VX_config.toml](../../VX_config.toml)
(SimX).

---

## 14. Migration plan — phased commits on `feature_vm_v2`

Per the no-PR / direct-commit + commit-only-at-feature-completion
project rules, each phase below lands as one substantial commit on
`feature_vm_v2` with end-to-end regression passing.

### Phase 1: L1 + L2 TLB hierarchy

- Implement multi-banked L1 DTLB + L1 ITLB (per-core).
- Implement per-cluster L2 TLB.
- Keep PTW per-core for now (lives at L2 if enabled, else L1).
- TLB invalidation broadcast across L1 (per-core) + L2 (per-cluster).
- New VX_config knobs for L1/L2 TLB.
- SimX models for L1 / L2 TLB.
- Regression: existing VM tests pass; new TLB stress microbench passes
  (small-stride / medium-stride / large-stride).

### Phase 2: L3 TLB + centralized PTW

- Implement chip-wide L3 TLB at the LLC.
- Move PTW from per-core (or per-cluster) to chip-wide, co-located
  with L3 (or with IS_LLC if L3 disabled).
- Implement multi-walker PTW with PTE walk-cache.
- Broadcast invalidation extended to L3.
- New VX_config knobs for L3 TLB + walker count + walk-cache.
- SimX models for L3 TLB + centralized PTW.
- Regression: VM tests pass; large-stride TLB stress now shows
  expected L3 hit rate; multi-core concurrent-translation test passes.

### Phase 3: CP / DMA split

- Extract `VX_dma` from `rtl/cp/`.
- Wire DMA L1 TLB + L2/L3/PTW connection.
- CP retains command FSM and completion; data path moves.
- Preserve `VX_CFG_CP_DMA_DEV_PORT_MODE=SHARED|DEDICATED`.
- Update [command_processor_proposal.md](command_processor_proposal.md)
  §6.6 with the new block.
- Regression: all CP tests pass; new translated-DMA microbench passes;
  DMA SHARED mode still builds on single-bank XRT shell.

### Phase 4 (v1.1): SV39

- Extend PTW FSM to 3-level walk (already parameterized on
  `VX_VM_PT_LEVEL`).
- Walk-cache extended to cover 2 non-leaf levels.
- Runtime updates to build SV39 page tables (driver-side).
- Regression: SV39 VM tests pass on RV64 builds.

---

## 15. Test plan

Microbenchmarks (new, add to `tests/regression/vm/`):

- `tlb_stress_l1`: 100 KB strided buffer; fits in L1. Verifies L1 hit
  rate ≥ 95%.
- `tlb_stress_l2`: 8 MB strided buffer; fits in L2, exceeds L1.
  Verifies L1 miss + L2 hit dominant; L2 hit rate ≥ 95%.
- `tlb_stress_l3`: 64 MB strided buffer; exceeds L2, fits in L3.
  Verifies L3 hit rate ≥ 90%, PTW invocation rate matches L3 misses.
- `tlb_stress_walk`: pseudo-random VA pattern across 256 MB. Verifies
  walk-cache hit rate ≥ 50%, multi-walker concurrency observable.
- `tlb_concurrent`: multi-core concurrent translation of disjoint VAs;
  verifies L1 per-core isolation and L2/L3 sharing.
- `dma_translated`: `CMD_MEM_WRITE` of 64 MB; verifies DMA traffic
  flows through the TLB hierarchy and `dma_tlb_*` perf counters update.

Regression integration:

- `ci/regression.sh.in` gains a `--vm-hierarchy` flag that runs the new
  microbenchmarks under SimX (functional + perf counters check).
- RTL regression runs the same microbenchmarks under `xrt` (the
  primary RTL test path per project convention).

Performance signals:

- Perf counter dump (VM counters in `VX_DCR_MPM_CLASS_MEM`) reports per-level
  reads/hits/misses, MSHR occupancy, walker occupancy, walk-cache hit
  rate. Compare against baseline (current per-core MMU) and document
  the gain.

---

## 16. Open questions

1. **DMA `dev_port` placement under PTW contention.** With centralized
   PTW and `VX_CFG_DMA_TLB_ENABLE=true`, bulk DMA may starve kernel
   walks under sustained concurrent dispatch. Mitigation if measured:
   walker reservation per source (kernel vs DMA), or a
   `VX_CFG_DMA_WALKER_PRIORITY` knob. Defer to Phase 3 perf
   measurement.

2. **L2 TLB sharing across D/I.** Combining D-side and I-side traffic
   into one L2 simplifies routing but increases conflict pressure on
   2048 entries. Split L2 (one for D, one for I) is an alternative.
   Defer to Phase 1 perf measurement.

3. **Walk-cache replacement policy.** Direct-mapped is simple; 4-way
   might be worth the modest area. Defer to Phase 2 perf measurement.

4. **Coalescing of duplicate VPN misses across cores at L2.** The
   per-level MSHR collapses same-VPN misses within a level. With
   multiple L1 TLBs from different cores feeding L2, two cores missing
   on the same VPN can hit the L2 MSHR and share a single PTW invocation
   — confirm this works as expected with the chosen MSHR arbitration.
