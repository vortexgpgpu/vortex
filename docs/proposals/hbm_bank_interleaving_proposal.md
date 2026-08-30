# HBM Bank Interleaving for the V80 (VRT/AVED) Memory Path

**Status:** proposal — not implemented
**Scope:** `hw/syn/xilinx/aved`, `hw/rtl/afu/common`, `hw/rtl/libs/VX_mem_bank_adapter.sv`

---

## 1. Summary

The VRT/AVED path does **not** interleave across HBM banks today. It presents a
single 512-bit AXI master, and the bank adapter that would do the interleaving
is bypassed entirely — measurably so: it costs 4 LUTs in the shipped bitstream.

Interleaving *is* implementable on this platform, and more cheaply than the
current platform notes suggest: the V80 compute shell exposes **eight VNOC
ingress ports under one logical `MEM` tag**, all sharing a single unified HBM
address space. The per-bank base-address problem that blocks the `HBM[i]`
channel tags does not arise there.

The benefit is entirely a function of core count:

| Cores | Best speedup from interleaving |
|------:|-------------------------------:|
| 1     | 1.15× |
| 4     | 2.50× |
| 16    | 2.94× |

**Recommendation: do not implement this now.** The V80 bitstream is a 1-core
configuration where the measured gain is 1.15%, and the memory path is running
at **0.6% of a single port's capacity** — memory is nowhere near the bottleneck.
Revisit when the core count rises above ~4. At that point 8 banks is the right
target; the benefit saturates there exactly, and 8 is also the number of VNOC
ports available.

---

## 2. Current state

### 2.1 What the V80 build actually does

`hw/syn/xilinx/aved/platforms.mk`:

```makefile
override CONFIGS += -DVX_CFG_PLATFORM_MEMORY_NUM_BANKS=1
override CONFIGS += -DVX_CFG_PLATFORM_MEMORY_ADDR_WIDTH=34   # 16 GB
override CONFIGS += -DPLATFORM_MERGED_MEMORY_INTERFACE
override CONFIGS += -DPLATFORM_MEMORY_OFFSET=40'h4000000000
MEM_TAG = MEM
```

`config.cfg.tmpl` binds one port: `sp=vortex_afu_0.m_axi_mem_0:MEM`.

The datapath consequence chains through three files:

```
PLATFORM_MERGED_MEMORY_INTERFACE
  → VX_afu_wrap.sv:46      C_M_AXI_MEM_NUM_BANKS = 1
  → VX_afu_wrap.sv:416     Vortex_axi AXI_NUM_BANKS = 1
  → Vortex_axi.sv:206      NUM_BANKS_OUT = 1
  → VX_mem_bank_adapter.sv:88   `if (NUM_BANKS_OUT > 1)` is false
                                → g_no_bank_sel, no interleaving logic at all
```

`VX_CFG_PLATFORM_MEMORY_INTERLEAVE = 1` (the `VX_config.toml` default) is
therefore **inert** on this platform. It only has meaning when
`NUM_BANKS_OUT > 1`.

### 2.1.1 What the interleave would do, precisely

`ADDR_WIDTH_IN` is word-addressable where the word is `DATA_WIDTH/8` = 64 B, so
the bank select is taken from the low bits of the **64-byte block index**:

```systemverilog
req_bank_sel  = mem_req_addr_dst[BANK_SEL_BITS-1:0]
req_bank_addr = mem_req_addr_dst[BANK_SEL_BITS +: BANK_ADDR_WIDTH]
```

That is block-cyclic placement at cache-line granularity: consecutive 64 B
blocks land in consecutive banks. `g_no_interleave` is the alternative —
high-order bits select, giving each bank one contiguous region.

The placement is derived in hardware from the address. The host writes a linear
address range and the adapter scatters it; **no driver or copy-path change is
needed** to get interleaving.

### 2.1.2 `NUM_BANKS` also throttles the L1 memory interface

This parameter is not only an AXI port count:

```c
#define VX_CFG_L1_MEM_PORTS __MIN(VX_CFG_DCACHE_NUM_BANKS, VX_CFG_PLATFORM_MEMORY_NUM_BANKS)
```

At `PLATFORM_MEMORY_NUM_BANKS=1` the L1 data cache is clamped to a **single**
memory port regardless of `DCACHE_NUM_BANKS`. So the current V80 setting
narrows the cache-to-memory path inside the core, upstream of any AXI concern.

This means the speedups in §4.2 are **not** attributable to AXI masters alone —
raising `NUM_BANKS` simultaneously widens the L1 memory interface. The two
effects are coupled by this expression and were not separated by the sweep.
Separating them (by sweeping `DCACHE_NUM_BANKS` independently) would sharpen
the estimate and is listed in §8.

### 2.2 Where the bandwidth actually goes

`MEM_TAG=MEM` routes into the HBM VNOC, which *does* spread traffic across HBM
pseudo-channels. So the current design is not stuck on one HBM channel. The
ceiling is the single AXI master:

```
512 bits × 200 MHz = 64 B/cycle = 12.8 GB/s
```

against the 460 GB/s the platform config advertises.

### 2.3 A correction worth recording

It is natural to read `-DVX_CFG_PLATFORM_MEMORY_NUM_BANKS=32` on the U55C/U50
XRT targets as "those platforms interleave across 32 banks". They do not.
Both also set `PLATFORM_MERGED_MEMORY_INTERFACE`, which forces
`AXI_NUM_BANKS = 1` by the chain above, so their bank adapters are bypassed
too. On a merged platform `VX_CFG_PLATFORM_MEMORY_NUM_BANKS` affects only the
capability word the CP reports to software
(`VX_cp_axil_regfile.sv:115-120`) — it is cosmetic in the datapath, and it
makes `VX_CAPS_NUM_MEM_BANKS` misreport.

**U280 is the only target that emits multiple masters** (32 banks, no merge).
Its connectivity binds only `m_axi_mem_0:HBM[0:31]`, leaving `m_axi_mem_1..31`
unbound — that looks like a latent defect and should be checked independently
of this proposal.

---

## 3. Why the documented blocker does not apply

`platforms.mk` explains the single-master choice:

> A single wide master fanned across the channels avoids the per-bank base
> address problem: the linker assigns each memory range independently, and
> those bases are not known at synthesis time.

That reasoning is correct **for the `HBM[i]` channel tags**, where each channel
is an independently mapped region. It does not hold for the `MEM` tag.

`linker/slashkit/core/bd_ports.py` documents the tag as one logical name with
many RTL endpoints:

```
MEM:HBM_VNOC_INI_00 AXI4FULL
MEM:HBM_VNOC_INI_01 AXI4FULL
...
MEM:HBM_VNOC_INI_07 AXI4FULL
```

and `emit/hw/tcl_gen.py:320` confirms the pool size:

```python
terms_mem_noc = build_mem_noc_terminators(
    used_targets, num_mem=8, noc_pin_fmt="/hbm_vnoc_0{index}/S00_AXI")
```

`BlockDesignPorts.resolve()` accepts `MEM` with an index and returns
`mems[index]`, so ports can be assigned individually.

All eight ingress ports front the **same** VNOC address space. Therefore:

* `PLATFORM_MEMORY_OFFSET = 0x40_0000_0000` stays correct for every port,
  unchanged.
* No per-bank base address has to be known at synthesis time.
* Vortex's own interleave (low address bits select the bank) is free to
  distribute traffic however it likes, because every bank reaches the same
  memory.

This is the crux: **the blocker is a property of the `HBM[i]` tags, not of the
platform.** Multi-master on `MEM` sidesteps it.

The upper bound also changes. Eight ports at 512 bits and 200 MHz gives
102.4 GB/s — better than 12.8, still well short of the platform's advertised
460 GB/s. Closing the rest would need a higher kernel clock or wider ports,
which are separate changes.

---

## 4. Measured data

### 4.1 Hardware baseline (V80, 1 core / 4 warps / 4 threads, 200 MHz)

`vecadd`, `TARGET=hw`, current single-master build:

| N | instructions | cycles | IPC |
|---:|---:|---:|---:|
| 65,536 | 781,940 | 2,724,805 | 0.287 |
| 262,144 | 2,354,820 | 8,687,545 | 0.271 |
| 1,048,576 | 8,646,292 | 32,538,101 | 0.266 |

At N = 1,048,576 the kernel moves 3 arrays × 4 MB = 12,582,912 B in
32,538,101 cycles:

```
0.387 bytes/cycle  =  77.3 MB/s at 200 MHz
```

A single 512-bit port sustains 64 B/cycle. **The current design uses 0.60% of
one port** — and roughly 0.02% of the platform's advertised bandwidth.

This is the single most important number in this document. Memory bandwidth is
not the bottleneck on the V80 today; core throughput is (IPC 0.27 on a 1-core,
4-warp, 4-thread configuration).

### 4.2 Where interleaving starts to matter (SimX, `vecadd -n65536`)

Cycles, sweeping cores × `VX_CFG_PLATFORM_MEMORY_NUM_BANKS`:

| cores | banks=1 | banks=2 | banks=4 | banks=8 | banks=16 | best gain |
|------:|--------:|--------:|--------:|--------:|---------:|----------:|
| 1  | 1,322,173 | 1,223,825 | 1,231,091 | 1,153,281 | — | **1.15×** |
| 4  |   721,847 |   425,456 |   310,492 |   288,977 | — | **2.50×** |
| 16 |   481,127 |   259,156 |   192,258 |   163,917 | 163,917 | **2.94×** |

Two clear results:

1. **The gain saturates at 8 banks.** 16 banks is bit-identical to 8
   (163,917 cycles). Eight is also exactly the number of VNOC ports available,
   which is a convenient coincidence.

2. **Interleaving unlocks core scaling.** Scaling from 1 to 16 cores:

   | banks | 1 core | 16 cores | scaling | efficiency |
   |------:|-------:|---------:|--------:|-----------:|
   | 1 | 1,322,173 | 481,127 | 2.75× | 17% |
   | 8 | 1,153,281 | 163,917 | **7.04×** | **44%** |

   Bank count more than doubles multi-core scaling efficiency. Without it,
   memory serialization caps a 16-core part at under 3× a single core.

**Caveat:** these are SimX numbers. SimX is the RTL's timing model and the two
are kept in lockstep by the `model_parity` gate, but a bank-count sweep is not
something that gate covers. The figures should be confirmed on `rtlsim` before
any synthesis effort is committed.

### 4.3 Area baseline (measured, shipped V80 bitstream)

From `report_utilization_vortex_afu.txt` (`xcv80-lsva4737-2MHP-e-S`, routed):

| Instance | LUTs | FFs | RAMB36 |
|---|---:|---:|---:|
| `vortex_afu_0` (whole AFU) | 47,867 (2.93%) | 55,761 (1.71%) | 93 (3.85%) |
| `vortex_axi` | 42,338 (2.59%) | 46,076 (1.41%) | 93 |
| `vortex` (core) | 42,334 (2.59%) | 46,074 (1.41%) | 93 |
| `dcache` | 7,895 (0.48%) | 8,998 (0.28%) | 43 |

`vortex_axi − vortex = 4 LUTs`. That is the entire memory adapter layer today —
direct confirmation that the crossbar is compiled out at `NUM_BANKS=1`.

The AFU occupies under 3% of the device, so area is not a constraint at this
scale.

---

## 5. Proposed implementation

Three changes, all in the platform layer. No RTL modification is required —
the interleaving logic already exists and is merely disabled.

**1. `hw/syn/xilinx/aved/platforms.mk`** — drop the merge, declare 8 banks:

```makefile
override CONFIGS += -DVX_CFG_PLATFORM_MEMORY_NUM_BANKS=8
override CONFIGS += -DVX_CFG_PLATFORM_MEMORY_ADDR_WIDTH=34
# PLATFORM_MERGED_MEMORY_INTERFACE deliberately NOT set: the merge collapses
# Vortex's bank adapter to a single AXI master and bypasses the interleave.
override CONFIGS += -DPLATFORM_MEMORY_OFFSET=40'h4000000000
```

`PLATFORM_MEMORY_OFFSET` is unchanged and applies to every port, because all
eight VNOC ingress ports share one address space.

**2. `hw/syn/xilinx/aved/config.cfg.tmpl`** — bind all eight masters:

```ini
sp=vortex_afu_0.m_axi_mem_0:MEM
sp=vortex_afu_0.m_axi_mem_1:MEM
...
sp=vortex_afu_0.m_axi_mem_7:MEM
sp=vortex_afu_0.m_axi_host:@HOST_TAG@
```

Requires confirming that slashkit assigns repeated `MEM` bindings to distinct
`HBM_VNOC_INI_0N` endpoints rather than rejecting the duplicate or collapsing
them. `BlockDesignPorts.resolve()` supports indexing; whether the `sp=` parser
exposes that is the one open question in this plan and should be settled first
(see §8).

**3. Host runtime configuration.** The runtime is built from `VX_config.toml`
defaults, not from the platform's `override CONFIGS`. The self-test added in
`205160014` shows the resulting disagreement on the current build:

```
device reports : 1 bank(s) x 17179869184 bytes   (vbin: NUM_BANKS=1, ADDR_WIDTH=34)
host built for : 4294967296 bytes                (runtime: toml, NUM_BANKS=2, ADDR_WIDTH=32)
```

This is benign today — the allocator stays under 4 GB, which fits inside the
16 GB aperture — but `VX_CAPS_NUM_MEM_BANKS` is already wrong, and any future
bank-aware host logic would inherit the error. Worth fixing independently of
this proposal.

---

## 6. Expected benefit

| Configuration | Expected gain | Worth doing? |
|---|---|---|
| 1 core (today's V80) | 1.15× | **No** |
| 4 cores | 2.50× | Marginal |
| 8–16 cores | 2.9× and rising scaling efficiency | **Yes** |

The honest framing: this is not a fix for anything currently broken or slow.
It is a prerequisite for multi-core scaling on the V80. Applied to the present
1-core bitstream it would consume a resynthesis cycle and buy ~1%.

---

## 7. Area cost — estimated, not measured

**This section is an estimate.** No synthesis run was performed at
`NUM_BANKS>1`, and the figure should not be quoted as measured.

Structurally, moving from 1 to 8 banks adds:

* **7 additional 512-bit AXI-4 master ports.** Interface wiring only; the
  device has ample I/O to the VNOC.
* **A crossbar in `VX_mem_bank_adapter`,** sized `NUM_PORTS_IN × NUM_BANKS_OUT`.
  With `VX_CFG_L1_MEM_PORTS=1` this is 1×8 — a fan-out with per-bank
  arbitration, not a full N×M crossbar. This is the dominant new cost.
* **Per-bank tag buffers,** `TAG_BUFFER_SIZE=32` entries each.
* **Response arbitration** across 8 banks back to one input port.

Given the adapter is 4 LUTs today and the whole AFU is 47,867 LUTs, even a
generous estimate keeps the increment in the low thousands of LUTs — a few
percent of the AFU, well under 1% of the device. Area is very unlikely to be
the deciding factor.

**To measure it properly:** build the AFU IP out-of-context at
`NUM_BANKS ∈ {1,2,4,8}` and diff `report_utilization`. That is much cheaper
than a full place-and-route and should precede any commitment.

The more likely physical risk is **timing, not area** — eight 512-bit masters
routed to eight separate VNOC ingress points is a placement constraint, and the
kernel clock is already only 200 MHz.

---

## 8. Open questions and risks

1. **Does `sp=` accept repeated `MEM` bindings?** The plan depends on it.
   `BlockDesignPorts` supports many endpoints per logical name, but the `sp=`
   parser path was not traced end-to-end. **Settle this before anything else** —
   it is cheap to check and it gates the whole proposal.
2. **SimX-only benefit data.** Confirm the bank sweep on `rtlsim` before
   spending synthesis time.
3. **Timing closure at 200 MHz** with eight wide masters. Unknown.
4. **`VX_CAPS_NUM_MEM_BANKS` is already wrong** on merged platforms; fixing the
   runtime/platform config disagreement should come first so the reported
   capability matches the hardware.
5. **Benefit is workload-shaped.** `vecadd` is pure streaming and close to a
   best case for interleaving. Compute-bound kernels will gain less.
6. **The §4.2 speedups conflate two effects.** `NUM_BANKS` sets both the AXI
   master count and (via `L1_MEM_PORTS`) the width of the L1 cache's memory
   interface. Sweep `DCACHE_NUM_BANKS` and `PLATFORM_MEMORY_NUM_BANKS`
   independently to attribute the gain. If most of it comes from the L1 port
   width, a cheaper change than eight AXI masters may capture much of it.

---

## 9. Recommendation

**Do not implement now.** Sequenced instead:

1. **Now (free):** fix the host/platform config disagreement so
   `VX_CAPS_NUM_MEM_BANKS` and the memory size reported to applications match
   the bitstream.
2. **Now (cheap):** confirm the `sp=` repeated-`MEM` question in §8.1, and
   check the U280 unbound-ports observation in §2.3.
3. **Before any multi-core V80 bitstream:** reproduce the bank sweep on
   `rtlsim`, then do an out-of-context area/timing check at
   `NUM_BANKS ∈ {1,2,4,8}`.
4. **When core count exceeds ~4:** implement §5 with `NUM_BANKS=8`. The
   measured saturation point and the VNOC port count agree on 8.

The controlling fact is §4.1: the V80 memory path currently runs at 0.6% of a
single port. Until core throughput rises enough to change that, interleaving
optimizes something that is not limiting anything.
