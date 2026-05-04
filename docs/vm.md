# Virtual Memory

Vortex supports per-core virtual-to-physical address translation using the
RISC-V SV32 (XLEN=32) and SV39 (XLEN=64) page-table formats. VM is enabled
by the TOML setting `VM_ENABLE = true` in [VX_config.toml](../VX_config.toml).

This document covers the runtime model, the environment-variable knobs
exposed for testing, and the perf counters surfaced through the
`vx_dump_perf` reporting.

## Components

| Layer | Where | Role |
|---|---|---|
| Page table | RAM at `PAGE_TABLE_BASE_ADDR` (0xF0000000 on 32‑bit, 0x0F0000000 on 64‑bit) | Multi-level table installed by the runtime; consumed by every PTW |
| Runtime `VMManager` | [sw/runtime/common/vm.{h,cpp}](../sw/runtime/common/vm.cpp) | Allocates the page table, mints VAs on `vx_mem_alloc`, walks the table on `mem_free` / `vx_copy_to_dev` / `vx_copy_from_dev` |
| Kernel SATP write | [sw/kernel/src/vx_start.S](../sw/kernel/src/vx_start.S) | Each core writes the SATP CSR with the PT base and addressing mode at boot |
| Per-core MMU (RTL) | [hw/rtl/core/VX_mmu.sv](../hw/rtl/core/VX_mmu.sv) + [VX_mmu_tlb.sv](../hw/rtl/core/VX_mmu_tlb.sv) + [VX_mmu_ptw.sv](../hw/rtl/core/VX_mmu_ptw.sv) | Two instances per core: dcache MMU and icache MMU. Each owns a 32-entry CAM TLB and an SV32/SV39 page-table walker that fetches PTEs through its own bus-level shim sitting between [VX_mem_unit](../hw/rtl/core/VX_mem_unit.sv) and the cache cluster |
| Per-core MMU (SimX) | [sim/simx/core.cpp](../sim/simx/core.cpp), uses `MemoryUnit` from [sim/common/mem.h](../sim/common/mem.h) | Functional translator. `LsuUnit::process_request_step` calls `core_->translate(va, type)` for each lane before stuffing the PA into `LsuReq.addrs`; the icache fetch path does the same for `trace->PC` |

## Address layout (XLEN=32, SV32)

```
0x00000000 ─┬── IO region (no translation)        bypass
            │
0x00010000 ─┴── USER_BASE_ADDR
            │
            │   Translated user VA range
            │
0x80000000 ─┬── STARTUP_ADDR                       bypass
            │   (kernel code at boot)             (40000 bytes)
0x80040000 ─┴──
            │
            │   Translated user VA range (cont'd)
            │
0xF0000000 ─┬── PAGE_TABLE_BASE_ADDR              bypass
            │   (page tables themselves)
            │
0xFFFF0000 ─┴── STACK / LMEM (above PT base)      bypass
```

Anything in the bypass ranges flows through the MMU's bypass path with
zero translation overhead. Only addresses in the translated ranges incur
TLB lookups and (on miss) PTW walks.

## Environment variables

`VORTEX_RANDOMIZE_VA` and `VORTEX_VA_SEED` are read by `VMManager`'s
constructor — see [vm.cpp](../sw/runtime/common/vm.cpp).

- `VORTEX_RANDOMIZE_VA=0` (default) — identity mapping. `vx_mem_alloc`
  returns a VA equal to the underlying PA. Useful as the baseline; verifies
  the translation pipeline does not corrupt addresses.
- `VORTEX_RANDOMIZE_VA=1` — for each `vx_mem_alloc`, mint a random
  page-aligned base VA in `[ALLOC_BASE_ADDR, PAGE_TABLE_BASE_ADDR)` (32-bit
  bounded), reserve the contiguous range, and install per-page PTEs. The
  user receives the random VA; PA stays in `global_mem_`.
- `VORTEX_VA_SEED=N` — seed for the `std::mt19937_64` RNG. Default
  `0x12345678`. Same seed → same VA stream across runs.

### Randomization algorithm

The runtime allocates an entire contiguous VA range upfront, then maps each
page sequentially so multi-page buffers stay contiguous in VA space:

```cpp
// 1. Find a random contiguous VA range
uint64_t candidate_va = random_address_in_range();
if (virtual_mem_->reserve(candidate_va, size) == 0) {
  base_vpn = candidate_va >> MEM_PAGE_LOG2_SIZE;
}
// 2. Map each PPN to a sequential VPN
for (uint64_t i = 0; i < num_pages; i++) {
  update_page_table(base_ppn + i, base_vpn + i, flags);
}
```

After 1000 failed reservation attempts (heavily fragmented VA space), it
falls back to sequential allocation so progress is guaranteed.

## Perf counters

Six MMU-related counters live in their own MPM class
(`VX_DCR_MPM_CLASS_VM`). The hardware sums the icache and dcache MMU
counters into one bank exposed via `pipeline_perf.mmu` in
[VX_gpu_pkg.sv](../hw/rtl/VX_gpu_pkg.sv).

| CSR | Meaning |
|---|---|
| `VX_CSR_MPM_TLB_READS` | Total TLB lookups (icache + dcache MMU) |
| `VX_CSR_MPM_TLB_HITS` | TLB hits |
| `VX_CSR_MPM_TLB_MISSES` | TLB misses (each triggers a PTW) |
| `VX_CSR_MPM_TLB_EVICTS` | TLB evictions on fill |
| `VX_CSR_MPM_PTW_WALKS` | Completed PTW walks |
| `VX_CSR_MPM_PTW_LATENCY` | Total PTW latency in cycles (avg = LATENCY / WALKS) |

[stub/perf.cpp](../sw/runtime/stub/perf.cpp) reads these and prints a
`vm:` line in the per-core report when `--perf=1` (CORE class) is passed
to `blackbox.sh`. Example:

```
PERF: vm: tlb_reads=96, hit=96%, evicts=0, ptw_walks=4, ptw_avg_lat=84.75
```

## Testing

The project ships a regression script at the repo root:

```bash
./run_vm_regression.sh --driver=simx              # identity mapping
./run_vm_regression.sh --driver=rtlsim --perf
./run_vm_regression.sh --driver=simx --randomize  # randomized VAs
./run_vm_regression.sh --driver=simx --randomize --seed=1   # reproducible
```

The script runs a 24-test common subset on the chosen driver. With
`--randomize` and a fixed `--seed`, two runs produce identical VA
sequences across all tests — useful for triaging VM bugs.

## Disabling VM

Set `VM_ENABLE = false` in [VX_config.toml](../VX_config.toml) and
re-`./configure`. With VM disabled the per-core MMU paths in
[VX_core.sv](../hw/rtl/core/VX_core.sv) compile out (the dcache and
icache buses connect straight through), the SimX `Core::translate` path
becomes a no-op, and the runtime `VMManager` is never constructed.
