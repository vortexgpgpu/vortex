# Virtual Memory

Vortex supports virtual-to-physical address translation using the RISC-V
Sv32 (XLEN=32) and Sv39 (XLEN=64) page-table formats. VM is enabled per
build with `CONFIGS="-DVX_CFG_VM_ENABLE"`; the default in
[VX_config.toml](../VX_config.toml) is off. The authoritative architecture
description is
[docs/designs/virtual_memory_subsystem.md](designs/virtual_memory_subsystem.md);
this page covers usage, configuration knobs, and perf reporting.

## Components

| Layer | Where | Role |
|---|---|---|
| Page table | RAM at `VX_MEM_PAGE_TABLE_BASE_ADDR` (0xF0000000) | Multi-level table installed by the runtime; consumed by every walker |
| Runtime `VMManager` | [sw/runtime/common/vm.{h,cpp}](../sw/runtime/common/vm.cpp) | Builds the page table (host-shadow + batched flush), mints VAs on `vx_buffer_create`, identity-maps system regions and `VX_MEM_PHYS` buffers |
| Kernel SATP write | [sw/kernel/src/vx_start.S](../sw/kernel/src/vx_start.S) | Each core writes the SATP CSR with the PT base and addressing mode at boot |
| Per-core MMU (RTL) | [hw/rtl/mem/VX_mmu.sv](../hw/rtl/mem/VX_mmu.sv) + [VX_mmu_tlb.sv](../hw/rtl/mem/VX_mmu_tlb.sv) + [VX_mmu_tlb_bank.sv](../hw/rtl/mem/VX_mmu_tlb_bank.sv) | Two instances per core (dcache and icache side): a banked CAM TLB translating inline on hits, superpage-aware |
| Shared walker (RTL) | [hw/rtl/mem/VX_mmu_ptw.sv](../hw/rtl/mem/VX_mmu_ptw.sv) + [VX_mmu_pwc.sv](../hw/rtl/mem/VX_mmu_pwc.sv) | One per device: generic Sv32/Sv39 walker, `VX_CFG_PTW_NUM_WALKERS` concurrent walks, page-walk caches, PTE fetches on a dedicated L3 port |
| SimX model | [sim/simx/mem/mmu.{h,cpp}](../sim/simx/mem/mmu.cpp) + [ptw.{h,cpp}](../sim/simx/mem/ptw.cpp) | Timing twin of the RTL: banked per-core TLBs, one shared `Ptw` SimObject on the L3 port |
| CP DMA translation | [sim/common/cmd_processor.cpp](../sim/common/cmd_processor.cpp) | The command processor walks the table for every `CMD_MEM_*` operand, so the host API is VA-only |

Translation is gated only by the SATP mode (BARE bypasses); there is no
address-range bypass. The runtime identity-maps the IO region, kernel
image, page-table region, and `VX_MEM_PHYS` allocations (using superpage
leaves where alignment allows), so PA-addressed traffic still resolves
correctly through the table.

## Configuration

| Knob | Default | Meaning |
|---|---|---|
| `VX_CFG_VM_ENABLE` | off | Enables the MMU/walker hardware and VM runtime |
| `VX_CFG_TLB_SIZE` | 32 | TLB entries per MMU (power of two) |
| `VX_CFG_TLB_NUM_BANKS` | 4 | dTLB lookup banks (power of two dividing `TLB_SIZE`); the iTLB is always single-banked |
| `VX_CFG_PTW_NUM_WALKERS` | 8 | Concurrent walk slots in the shared walker |
| `VX_CFG_PTW_WALK_CACHE_SIZE` | 64 | Entries per page-walk cache (direct-mapped, power of two) |
| `VX_CFG_VM_PINNED_REGION_SIZE` | 256 MB | Identity-mapped slab for `VX_MEM_PHYS` allocations |

## Environment variables

`VORTEX_RANDOMIZE_VA` and `VORTEX_VA_SEED` are read by `VMManager`'s
constructor — see [vm.cpp](../sw/runtime/common/vm.cpp).

- `VORTEX_RANDOMIZE_VA=0` (default) — sequential VA allocation.
- `VORTEX_RANDOMIZE_VA=1` — for each allocation, mint a random
  page-aligned contiguous VA range. The user receives the random VA; the
  PA stays wherever `global_mem_` placed it.
- `VORTEX_VA_SEED=N` — RNG seed (default `0x12345678`). Same seed → same
  VA stream across runs.

## Perf counters

The MMU counters live in the memory-subsystem MPM class
(`VX_DCR_MPM_CLASS_MEM`). The TLB counters are per core (icache + dcache
MMU summed); the walker and walk-cache counters belong to the shared
device-level walker and read the same value on every core.

| CSR | Meaning |
|---|---|
| `VX_CSR_MPM_TLB_READS` | Total TLB lookups (icache + dcache MMU) |
| `VX_CSR_MPM_TLB_HITS` | TLB hits |
| `VX_CSR_MPM_TLB_MISSES` | TLB misses (each triggers a walk) |
| `VX_CSR_MPM_TLB_EVICTS` | TLB evictions on fill |
| `VX_CSR_MPM_PTW_WALKS` | Walks started |
| `VX_CSR_MPM_PTW_LATENCY` | Sum of per-walk latencies (avg = LATENCY / WALKS) |
| `VX_CSR_MPM_PWC1_HITS` / `_MISSES` | Walks that skipped the top level via the walk cache |
| `VX_CSR_MPM_PWC2_HITS` / `_MISSES` | Sv39 only: walks that also skipped the middle level |

[sw/runtime/common/perf.cpp](../sw/runtime/common/perf.cpp) prints these
with `--perf=7` (MEM class):

```
PERF: core0: tlb: reads=2086, hit=75%, misses=522, evicts=487
PERF: ptw: walks=522, avg_lat=27.15 cyc, pwc1_hit=99%, pwc2_hit=0%
```

## Testing

The CI catalog is [ci/testcases/vm.yaml](../ci/testcases/vm.yaml): compute
regressions on simx **and rtlsim** at both XLENs, the
[tests/regression/vm_stress](../tests/regression/vm_stress) TLB-pressure
test (strided page touches + a `VX_MEM_PHYS` buffer), and full-tier
configuration variants (single-banked TLB, multi-cluster with L2/L3).

```bash
./ci/regression.sh --test vm
# or a single case:
CONFIGS="-DVX_CFG_VM_ENABLE" ./ci/blackbox.sh --driver=rtlsim --app=vm_stress --perf=7
```

## Disabling VM

Leave `VX_CFG_VM_ENABLE` unset (the default). The per-core MMU paths in
[VX_core.sv](../hw/rtl/core/VX_core.sv) compile out (the dcache and icache
buses connect straight through), the shared walker and its L3 port are not
instantiated, and the runtime `VMManager` is never constructed.
