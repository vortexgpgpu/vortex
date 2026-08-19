# Virtual Memory (MMU / TLB / PTW) — Design

**Scope:** the Vortex virtual-memory subsystem — the per-core MMU
([`hw/rtl/mem/VX_mmu.sv`](../../hw/rtl/mem/VX_mmu.sv),
[`VX_mmu_tlb.sv`](../../hw/rtl/mem/VX_mmu_tlb.sv),
[`VX_mmu_ptw.sv`](../../hw/rtl/mem/VX_mmu_ptw.sv)), the SimX model
([`sim/simx/mem/mmu.{cpp,h}`](../../sim/simx/mem/mmu.cpp),
[`mmu_tlb.{cpp,h}`](../../sim/simx/mem/mmu_tlb.cpp)), and the runtime VM
software stack ([`sw/runtime/common/vm.{cpp,h}`](../../sw/runtime/common/vm.cpp),
[`sw/common/vm_types.h`](../../sw/common/vm_types.h)).

This document is the architectural reference; the existing
[`docs/vm.md`](../vm.md) covers usage and perf counters (and has some
stale file-path references that this document corrects — the RTL MMU lives
in `hw/rtl/mem/`, and SimX uses a dedicated `sim/simx/mem/mmu.cpp`
SimObject, not the old `sim/common/mem.cpp` `MemoryUnit`).

VM is gated by `VX_CFG_VM_ENABLE` (default off,
[`VX_config.toml:24`](../../VX_config.toml#L24)).

---

## 1. The v3 VM model

VM in v3 lives in two places:

1. The **compute-core MMU** (RTL + SimX) translates VA→PA for kernel
   LSU/fetch traffic.
2. The **CP DMA software walker** translates VA operands of `CMD_MEM_*`
   commands (see [`command_processor.md`](command_processor.md) §8).

There is **no shared device-side MMU** and **no RTL CP MMU** yet. The host
runtime API is VA-only; the host never translates at transfer time — the
CP does.

```
  kernel LSU/fetch VA                         runtime (host)
        │                                     ──────────────
        ▼                                     VMManager: PA alloc, mint VA,
   per-core MMU (after coalescer)             build host-shadow page tables,
        │  satp[31]? BARE → bypass            batched flush of dirty PT pages
        ▼                                            │
   TLB CAM ── hit ─► PA ─► cache               CP_SATP_LO/HI ──► CP cp_translate()
        │ miss                                 (Sv32/Sv39 SW walk of device PTs)
        ▼
   PTW walk (PTE loads via same cache port) ─► fill TLB ─► replay as PA
```

---

## 2. ISA, CSRs, and configuration

- **SATP**: the compute cores use the RISC-V `satp` CSR `0x180`
  ([`VX_types.toml:322`](../../VX_types.toml#L322)), programmed once at
  boot by [`vx_start.S`](../../sw/kernel/src/vx_start.S) (`csrw satp` with
  PT-base PPN + mode, per-core, under `#ifdef VX_CFG_VM_ENABLE`) and
  surfaced as `sched_csr_if.csr_satp`
  ([`VX_csr_data.sv:142-150`](../../hw/rtl/core/VX_csr_data.sv#L142)). The
  **CP DMA** carries its own separate `CP_SATP_LO/HI` at CP regfile offsets
  `0x028/0x02C` — same packed value, distinct mechanism (dual-SATP
  plumbing).
- **Page-table format** ([`VX_types.toml:47-54`](../../VX_types.toml#L47)):
  `VX_VM_ADDR_MODE = SV39 if XLEN==64 else SV32`, `VX_VM_PT_LEVEL = 3/2`,
  `VX_VM_PTE_SIZE = 8/4`, `VX_VM_PAGE_LOG2_SIZE = 12`. PT base region at
  `VX_MEM_PAGE_TABLE_BASE_ADDR`. `VM_ADDR_MODE` enum lists
  `[BARE,SV32,SV39,SV48,SV57]` ([`:646`](../../VX_types.toml#L646)).
- **TLB sizing**: a single flat `VX_CFG_TLB_SIZE`-entry (32) fully-
  associative CAM, one per dcache MMU + one per icache MMU per core
  ([`VX_config.toml:160`](../../VX_config.toml#L160)). No L2/L3.
- **Perf**: 6 VM perf CSRs in the memory-subsystem class
  `VX_DCR_MPM_CLASS_MEM = 7` at 0xB0B–0xB10 (+_H mirrors), alongside
  off-chip memory / lmem / coalescer (`[csr_mpm_mem]` in
  [VX_types.toml](../../VX_types.toml)).
- **Runtime caps**: `VX_CAPS_VM_SUPPORT`, `VX_MEM_PHYS = 0x8`
  ([`vortex2.h:74,121`](../../sw/runtime/include/vortex2.h#L74)).

**Sv32 vs Sv39:** the SW stack, SimX MMU, and CP walker support **both**;
the **RTL PTW is Sv32-only** (hardcoded 2-level), so on FPGA only Sv32/RV32
VM is real — consistent with the project's 32-bit-only RTL policy.

---

## 3. RTL components

- [`VX_mmu.sv`](../../hw/rtl/mem/VX_mmu.sv) (top) — merges an
  elastic-buffered TLB path and a bypass path through `VX_mem_bus_arb`.
  Takes `satp[XLEN-1:0]`; `needs_translation()` bypasses only on the SATP
  mode field (BARE) — there is no address-range bypass. TLB misses leave
  the core on a `VX_ptw_bus_if` toward the shared walker.
- [`VX_mmu_tlb.sv`](../../hw/rtl/mem/VX_mmu_tlb.sv) — banked TLB wrapper:
  `VX_CFG_TLB_SIZE` entries split over `VX_CFG_TLB_NUM_BANKS` banks
  selected by the low VPN bits (the iTLB is always single-banked). Lanes
  are distributed and gathered by `VX_stream_xbar`s so up to a bank-count
  of translations proceed per cycle; each bank owns one outstanding walk,
  identified on the walker bus by its bank index.
- [`VX_mmu_tlb_bank.sv`](../../hw/rtl/mem/VX_mmu_tlb_bank.sv) —
  fully-associative CAM bank, MRU victim select, 3-state FSM
  (`READY/PTW_WAIT/REPLAY`). Entries carry their page level: superpage
  leaves (Sv32 megapages; Sv39 mega/gigapages) match on the VPN bits above
  their level and translate with the matching offset width. A faulted walk
  replays the access untranslated (identity), mirroring the CP walker's
  defensive pass-through.
- [`VX_ptw_bus_if.sv`](../../hw/rtl/mem/VX_ptw_bus_if.sv) /
  [`VX_ptw_arb.sv`](../../hw/rtl/mem/VX_ptw_arb.sv) — the walk
  request/fill bus (`{vpn, root_ppn, tag}` / `{ppn, level, flags, fault,
  tag}`) and its N→1 arbiter; one arb per level folds the source index
  into the tag: banks → core (i/d) → socket → cluster → device.
- [`VX_mmu_ptw.sv`](../../hw/rtl/mem/VX_mmu_ptw.sv) — **one instance per
  device**: a generic Sv32/Sv39 walker with `VX_CFG_PTW_NUM_WALKERS`
  concurrent walk slots, fetching PTEs on a **dedicated L3 requestor
  port** (`L3_PTW_IDX`). Leaf/validity checks per the privileged spec
  (V, R/W combos, superpage alignment, no-leaf-at-level-0) report `fault`
  on the fill; under simulation a fault also raises an error. Two
  direct-mapped page-walk caches (`VX_mmu_pwc.sv`,
  `VX_CFG_PTW_WALK_CACHE_SIZE` entries) cache the non-leaf entries of the
  two upper levels so a warm walk starts one (Sv32) or two (Sv39) levels
  below the root.

The per-core MMUs are instantiated in
[`VX_core.sv`](../../hw/rtl/core/VX_core.sv) (dcache MMU with
`DCACHE_NUM_REQS` ports and `VX_CFG_TLB_NUM_BANKS` banks; icache MMU with
1 port), both under `#ifdef VX_CFG_VM_ENABLE`, **after** the coalescer /
LSU adapter. The TLBs are invalidated by a one-cycle pulse at the start of
the DCR cache flush (the flush request itself is translated, so the
invalidate cannot be held at the level of the pending flush), and the
device walker drops its walk caches on the same DCR event.

---

## 4. SimX model

[`sim/simx/mem/mmu.{cpp,h}`](../../sim/simx/mem/mmu.cpp) is a per-core
`Mmu` SimObject with 4 channels. Unlike the RTL, its PTW is **generalized
over `VX_VM_PT_LEVEL`** via a level-counter FSM
([`mmu.cpp:56-167`](../../sim/simx/mem/mmu.cpp#L56)), so it models **both
Sv32 and Sv39**; it performs real page-fault checks (PTE `V`, `R=0 & W=1`)
and aborts on fault, and reconstructs superpage offsets correctly (caching
megapages as single 4 KB TLB entries).
[`mmu_tlb.{cpp,h}`](../../sim/simx/mem/mmu_tlb.cpp) is the matching
fully-associative TLB with reads/hits/misses/evictions counters. Wired in
[`core.cpp:191-213`](../../sim/simx/core.cpp#L191); `set_satp` fans out to
both MMUs.

---

## 5. Runtime VM software stack

[`sw/runtime/common/vm.{cpp,h}`](../../sw/runtime/common/vm.cpp) provides
`VMManager`: a page-table builder + VA allocator owned by `vx::Device`.
Page tables are **host-shadowed** (`shadow_pt_`, `dirty_pt_pages_`) and
flushed to the device in one bulk `CMD_MEM_WRITE(physical)` per dirty PT
page. `mem_alloc` allocates a PA from `global_mem_`, then either
`phy_to_virt_map` (mint a VA, install PTEs) or `install_identity_map` for
`VX_MEM_PHYS` buffers. The runtime API is VA-only.

The stack is **VM-discovery-driven, not compile-time gated**: `vm.{h,cpp}`
and `vm_types.h` carry no `#ifdef VM_ENABLE`, and the runtime discovers VM
at runtime from the CP `DEV_CAPS.VM_ENABLED` bit
([`device.cpp:277`](../../sw/runtime/common/device.cpp#L277)), branching on
`vm_enabled_`. (One compile-time constant remains:
`VX_CFG_VM_PINNED_REGION_SIZE` for pinned-slab sizing,
[`device.cpp:34`](../../sw/runtime/common/device.cpp#L34).) Randomized-VA
testing is available via `VORTEX_RANDOMIZE_VA`/`VA_SEED`.

[`sw/common/vm_types.h`](../../sw/common/vm_types.h) holds the pure
host-side `SATP_t`, `PTE_t`, `vAddr_t`, and `Page_Fault_Exception`,
Sv32/Sv39-split, including only the SW-facing `VX_types.h`.

### 5.1 CP composition

The CP DMA is MMU-aware: `cmd_processor.cpp:cp_translate()` performs the
identical Sv32/Sv39 walk as `VMManager::page_table_walk`, reading PTEs from
device RAM. Every `CMD_MEM_*` operand is a VA and is translated unless the
`MEM_FLAG_PHYSICAL` flag is set. This is the only device-side translation
path today.

---

## 6. End-to-end flow

1. LSU lanes → coalescer/adapter → per-core dcache MMU. If `satp` MSB is
   clear (BARE) the request bypasses translation; otherwise the per-lane
   VA hits the TLB CAM.
2. TLB hit → forward to cache as PA. TLB miss → kick the PTW.
3. PTW walks the page table by issuing PTE-fetch loads through the **same
   downstream cache port** (RTL: `ptw_mem_if` merged via `VX_mem_arb`;
   SimX: `ReqOut[0]` with a PTW tag marker). RTL = fixed 2-level Sv32;
   SimX = `VX_VM_PT_LEVEL`-deep loop.
4. On fill, the TLB caches `{vpn→ppn, flags}` and the request replays as a
   PA to the cache. (SimX delivers page faults; RTL does not yet.)
5. SATP is programmed once at boot by `vx_start.S`; the CP's `CP_SATP` is
   programmed by the host at `cp_init`.

---

## 7. Deliberate simplifications (current model)

These are architectural choices in the as-built v3 VM, recorded so they
are not mistaken for bugs:

- **No ASID.** `SATP_t` parses an `asid` field but it is unused; the RTL
  PTW drops the `satp` ASID bits. Single global VA space, no per-dispatch
  SATP reprogramming.
- **No A/D-bit writeback.** The runtime pre-sets `A=D=1`; the TLBs are
  read-only.
- **Page-fault delivery is partial.** SimX aborts on a fault; the RTL PTW
  checks `V/R/W/X` and superpage alignment and reports `fault` on the TLB
  fill (the access then replays untranslated), but no fault is routed to
  the LSU as an exception yet.
- **No PMP / page protection enforcement.**
- **SV48/SV57** enum values exist but are unimplemented.

---

## 8. Proposed but not yet implemented

1. **GPU-aligned multi-level TLB hierarchy** — this is the subject of a
   **retained** proposal, `mmu_perf_optimization_proposal.md` (kept in
   `docs/proposals/` because **none** of it is implemented yet). It
   specifies a 3-level L1/L2/L3 TLB hierarchy (multi-banked post-coalescer
   L1 DTLB, per-core L1 ITLB, per-cluster L2, chip-wide L3), a centralized
   multi-walker PTW with a PTE walk-cache, non-blocking TLB MSHRs, TLB
   inclusion-policy knobs (NINE/INCL/EXCL), broadcast `SFENCE.VMA`
   invalidation, and a `VX_dma` block split out of the CP with its own TLB
   into the shared hierarchy. It is the forward roadmap (`feature_vm_v2`);
   the current MMU is a single flat 32-entry FA TLB per cache port.
2. **RTL page-fault delivery to the LSU** — faults are detected and
   reported on the TLB fill, but not yet routed to the LSU as an
   exception (`U` and PMP checks also remain open).
3. **RTL CP shared device-side MMU** — Phase 2 of `vm_sw_stack_redesign`,
   deferred past v3: add the SATP regfile decode + a hardware walker so the
   CP DMA honors VM in RTL, matching the SimX/CP-software path (see
   `command_processor.md` §10 item 2).
4. **`configure --vm` first-class flag** — VM is still forced per build via
   `CONFIGS=-DVX_CFG_VM_ENABLE`.

Delivered since the v3 baseline (previously items on this list): a
generic Sv32/Sv39 RTL walker with superpage fills, shared per device with
concurrent walk slots and page-walk caches on a dedicated L3 port; banked
per-core TLBs honoring `VX_CFG_TLB_SIZE`; rtlsim VM in CI on both XLENs
(`ci/testcases/vm.yaml`), including the `vm_stress` TLB-pressure test.

**Superseded directions** (recorded to avoid revival): the per-LSU-slice
MMU placement of `vm_migration` (replaced by a single per-core MMU after
the coalescer); wiring the orphaned `sim/common/mem.cpp` `MemoryUnit`
(replaced by the dedicated `sim/simx/mem/mmu.cpp` SimObject); and the
original compile-time `VM_ENABLE` + per-transfer host-side translation
model (replaced by runtime `DEV_CAPS.VM_ENABLED` discovery + MMU-aware CP
DMA — the host no longer translates per transfer).

---

## 9. Source proposals

This design consolidates and supersedes `vm_migration_proposal.md` and
`vm_sw_stack_redesign_proposal.md` (now removed from `docs/proposals/`).
`mmu_perf_optimization_proposal.md` is **retained** in `docs/proposals/`
as the unimplemented forward roadmap (§8 item 1).
