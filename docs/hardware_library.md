# Vortex Hardware IP Library Reference

## Overview

The Vortex hardware IP library (`hw/rtl/libs/`) provides a comprehensive collection of 75 reusable, parameterized SystemVerilog IP modules used throughout the Vortex GPGPU processor. These modules cover arbitration, buffering, arithmetic, memory, interconnect, and debug infrastructure, forming the foundational building blocks of the Vortex microarchitecture.

All modules are parameterized for data width, depth, and implementation strategy, with support for both FPGA (Xilinx/Intel) and ASIC synthesis flows.

**License:** Apache 2.0  
**Copyright:** 2019-2023 The Vortex Authors

---

## Table of Contents

1. [Buffers and Flow Control](#1-buffers-and-flow-control)
2. [Arbiters and Schedulers](#2-arbiters-and-schedulers)
3. [Multiplexing and Demultiplexing](#3-multiplexing-and-demultiplexing)
4. [Stream Processing Utilities](#4-stream-processing-utilities)
5. [Crossbar and Interconnect Networks](#5-crossbar-and-interconnect-networks)
6. [Encoders and Decoders](#6-encoders-and-decoders)
7. [Arithmetic and Computation](#7-arithmetic-and-computation)
8. [Carry-Save Adders and Compression Trees](#8-carry-save-adders-and-compression-trees)
9. [Memory Elements](#9-memory-elements)
10. [Memory Controllers and Adapters](#10-memory-controllers-and-adapters)
11. [Bit Manipulation Utilities](#11-bit-manipulation-utilities)
12. [Synchronization and Timing](#12-synchronization-and-timing)
13. [Allocators and Counters](#13-allocators-and-counters)
14. [Debug and Tracing](#14-debug-and-tracing)
15. [Utility and Placeholder](#15-utility-and-placeholder)

---

## 1. Buffers and Flow Control

These modules implement elastic pipeline buffering with valid/ready handshake protocols. They are the core flow-control building blocks for all streaming datapaths.

### VX_stream_buffer

**File:** `VX_stream_buffer.sv`  
**Purpose:** Full-bandwidth elastic buffer with decoupled ready signals.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATAW` | - | Data width in bits |
| `OUT_REG` | 0 | Enable output register |
| `PASSTHRU` | 0 | Bypass mode (wire-through) |

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `clk` | input | 1 | Clock |
| `reset` | input | 1 | Reset |
| `valid_in` | input | 1 | Input valid |
| `ready_in` | output | 1 | Input ready |
| `data_in` | input | DATAW | Input data |
| `data_out` | output | DATAW | Output data |
| `valid_out` | output | 1 | Output valid |
| `ready_out` | input | 1 | Output ready |

**Features:**
- Two-stage internal storage enables full-bandwidth throughput (accepts data every cycle)
- `ready_in` and `ready_out` are fully decoupled -- upstream and downstream can stall independently
- `PASSTHRU` mode creates a zero-latency wire-through for timing-insensitive paths

---

### VX_toggle_buffer

**File:** `VX_toggle_buffer.sv`  
**Purpose:** Half-bandwidth elastic buffer with fully registered output.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATAW` | - | Data width in bits |
| `PASSTHRU` | 0 | Bypass mode |

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `clk` | input | 1 | Clock |
| `reset` | input | 1 | Reset |
| `valid_in` | input | 1 | Input valid |
| `ready_in` | output | 1 | Input ready |
| `data_in` | input | DATAW | Input data |
| `data_out` | output | DATAW | Output data |
| `valid_out` | output | 1 | Output valid |
| `ready_out` | input | 1 | Output ready |

**Features:**
- Single register storage -- accepts data every other cycle (half bandwidth)
- `data_out` is fully registered, providing clean timing on the output path
- Minimal area footprint -- useful where bandwidth is not critical

---

### VX_bypass_buffer

**File:** `VX_bypass_buffer.sv`  
**Purpose:** Full-bandwidth buffer with combinational bypass path.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATAW` | - | Data width in bits |
| `PASSTHRU` | 0 | Bypass mode |

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `clk` | input | 1 | Clock |
| `reset` | input | 1 | Reset |
| `valid_in` | input | 1 | Input valid |
| `ready_in` | output | 1 | Input ready |
| `data_in` | input | DATAW | Input data |
| `data_out` | output | DATAW | Output data |
| `valid_out` | output | 1 | Output valid |
| `ready_out` | input | 1 | Output ready |

**Features:**
- Single register with bypass multiplexer for full-bandwidth throughput
- `ready_in` and `ready_out` are **coupled** (not decoupled)
- `data_out` is **not** registered -- creates a combinational path from input to output
- Best used when downstream timing is not critical

---

### VX_skid_buffer

**File:** `VX_skid_buffer.sv`  
**Purpose:** Configurable buffer that selects between toggle and stream buffer modes.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATAW` | - | Data width in bits |
| `PASSTHRU` | 0 | Bypass mode |
| `HALF_BW` | 0 | Use half-bandwidth (toggle) mode |
| `OUT_REG` | 0 | Enable output register |

**Features:**
- When `HALF_BW=1`: instantiates `VX_toggle_buffer` (half bandwidth, minimal area)
- When `HALF_BW=0`: instantiates `VX_stream_buffer` (full bandwidth)
- Unified interface for design-time bandwidth/area trade-off selection

---

### VX_elastic_buffer

**File:** `VX_elastic_buffer.sv`  
**Purpose:** Elastic buffer with configurable depth and output registration.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATAW` | - | Data width in bits |
| `SIZE` | 1 | Buffer depth |
| `OUT_REG` | 0 | Enable output register |
| `LUTRAM` | 0 | Use LUTRAM for storage |

**Features:**
- Size-dependent implementation selection:
  - `SIZE=0`: passthrough (wire)
  - `SIZE=1`: pipe register
  - `SIZE=2`: stream buffer
  - `SIZE>=3`: FIFO queue
- Automatically picks the optimal implementation for the requested depth
- Optional LUTRAM hint for FPGA synthesis

---

### VX_pipe_buffer

**File:** `VX_pipe_buffer.sv`  
**Purpose:** Pipelined elastic buffer with configurable depth.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATAW` | - | Data width in bits |
| `RESETW` | 0 | Number of bits subject to reset |
| `DEPTH` | 1 | Number of pipeline stages |

**Features:**
- Full-bandwidth throughput with one register per pipeline stage
- `data_out` is fully registered at every stage
- Supports partial reset (only first `RESETW` bits are reset)

---

### VX_pipe_register

**File:** `VX_pipe_register.sv`  
**Purpose:** Configurable pipeline register with optional partial reset.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATAW` | - | Data width in bits |
| `RESETW` | 0 | Number of bits subject to reset |
| `DEPTH` | 1 | Number of pipeline stages |
| `INIT_VALUE` | 0 | Reset initialization value |

**Features:**
- Multi-stage pipeline with parameterized depth
- Partial reset: only the first `RESETW` bits are cleared on reset, saving reset fanout area
- Configurable reset value via `INIT_VALUE`

---

### VX_shift_register

**File:** `VX_shift_register.sv`  
**Purpose:** Multi-tap shift register with configurable tap positions.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATAW` | - | Data width in bits |
| `RESETW` | 0 | Number of bits subject to reset |
| `DEPTH` | 1 | Shift register depth |
| `NUM_TAPS` | 1 | Number of output taps |
| `TAP_START` | 0 | First tap position |
| `TAP_STRIDE` | 1 | Stride between taps |

**Features:**
- Multiple output taps from different stages of the shift chain
- Configurable tap positions via `TAP_START` and `TAP_STRIDE`
- Supports partial reset for power-efficient designs

---

### VX_elastic_adapter

**File:** `VX_elastic_adapter.sv`  
**Purpose:** Protocol adapter between valid/ready handshake and busy/strobe flow control.

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `clk` | input | 1 | Clock |
| `reset` | input | 1 | Reset |
| `valid_in` | input | 1 | Input valid (handshake side) |
| `ready_in` | output | 1 | Input ready (handshake side) |
| `ready_out` | input | 1 | Output ready |
| `valid_out` | output | 1 | Output valid |
| `busy` | input | 1 | Busy signal (strobe side) |
| `strobe` | output | 1 | Push strobe (strobe side) |

**Features:**
- Bridges between valid/ready elastic protocol and busy/strobe push protocol
- Internal `loaded` flag buffers one transaction
- Zero-area overhead in passthrough scenarios

---

## 2. Arbiters and Schedulers

A comprehensive suite of arbitration modules supporting different fairness policies. All arbiters share a common port interface with `requests`, `grant_index`, `grant_onehot`, and `grant_valid` signals.

### Common Arbiter Interface

All arbiters (except `VX_gto_arbiter`) share the following port structure:

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `clk` | input | 1 | Clock |
| `reset` | input | 1 | Reset |
| `requests` | input | NUM_REQS | Active request bitmap |
| `grant_ready` | input | 1 | Downstream ready to accept grant |
| `grant_index` | output | LOG_NUM_REQS | Binary index of granted request |
| `grant_onehot` | output | NUM_REQS | One-hot grant vector |
| `grant_valid` | output | 1 | Grant is valid |

---

### VX_priority_arbiter

**File:** `VX_priority_arbiter.sv`  
**Purpose:** Static priority arbiter -- lowest index has highest priority.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_REQS` | - | Number of requesters |
| `STICKY` | 0 | Hold grant until request deasserted |

**Features:**
- Lowest-numbered active request always wins
- Uses `VX_priority_encoder` for index computation
- `STICKY` mode retains grant to avoid unnecessary switching

---

### VX_rr_arbiter

**File:** `VX_rr_arbiter.sv`  
**Purpose:** Round-robin arbiter with multiple implementation models.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_REQS` | - | Number of requesters |
| `MODEL` | 1 | Implementation model (0-3) |
| `STICKY` | 0 | Hold grant until request deasserted |
| `LUT_OPT` | 0 | LUT optimization for small N |

**Features:**
- Rotates priority pointer after each successful grant for fairness
- **Model 0:** Basic circular scan
- **Model 1:** Default model with STICKY support
- **Model 2:** Alternate scan implementation
- **Model 3:** Parallel prefix style
- `LUT_OPT` provides hand-optimized logic for N=2..8

---

### VX_cyclic_arbiter

**File:** `VX_cyclic_arbiter.sv`  
**Purpose:** Cyclic (pointer-based) round-robin arbiter.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_REQS` | - | Number of requesters |
| `STICKY` | 0 | Hold grant until request deasserted |

**Features:**
- Maintains a grant pointer that increments after each grant (wraps around)
- Simpler than the `VX_rr_arbiter` -- single counter tracks priority
- Good for moderate requester counts where simplicity is preferred

---

### VX_matrix_arbiter

**File:** `VX_matrix_arbiter.sv`  
**Purpose:** Matrix-based fair arbiter using pairwise priority tracking.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_REQS` | - | Number of requesters |
| `STICKY` | 0 | Hold grant until request deasserted |

**Features:**
- Uses an N x N state matrix to track pairwise priority between all requesters
- Guarantees strict fairness: no requester can be granted twice before all active requesters are served once
- Higher area cost than round-robin, but provides stronger fairness guarantees

---

### VX_gto_arbiter

**File:** `VX_gto_arbiter.sv`  
**Purpose:** Greedy-Then-Oldest (GTO) arbiter with age-based priority.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_REQS` | - | Number of requesters |
| `AGE_W` | LOG2UP(NUM_REQS) | Age counter width (bits) |

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `clk` | input | 1 | Clock |
| `reset` | input | 1 | Reset |
| `requests` | input | NUM_REQS | Active request bitmap |
| `suppress` | input | NUM_REQS | Per-request suppression mask (default '0) |
| `grant_ready` | input | 1 | Downstream ready |
| `grant_index` | output | LOG_NUM_REQS | Granted index |
| `grant_onehot` | output | NUM_REQS | One-hot grant |
| `grant_valid` | output | 1 | Grant valid |

**Features:**
- Greedy phase: continues serving current requester while active
- Oldest phase: when current requester deasserts, selects the request with the longest wait time
- Per-request `suppress` input allows temporarily disabling specific requesters
- Saturating age counters with configurable width (`AGE_W`)
- Commonly used for warp scheduling in GPU architectures

---

### VX_generic_arbiter

**File:** `VX_generic_arbiter.sv`  
**Purpose:** Wrapper that instantiates different arbiter types based on a string parameter.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_REQS` | - | Number of requesters |
| `TYPE` | "P" | Arbiter type selector |
| `STICKY` | 0 | Hold grant until deasserted |

**Supported TYPE values:**
| TYPE | Arbiter |
|------|---------|
| `"P"` | `VX_priority_arbiter` |
| `"R"` | `VX_rr_arbiter` |
| `"M"` | `VX_matrix_arbiter` |
| `"C"` | `VX_cyclic_arbiter` |
| `"G"` | `VX_gto_arbiter` |

**Features:**
- Provides a single, unified interface for all arbiter types
- Allows changing arbitration policy by modifying a single parameter
- Used extensively throughout Vortex for design-time policy selection

---

### VX_stream_arb

**File:** `VX_stream_arb.sv`  
**Purpose:** Streaming arbiter with configurable input/output count and fanout management.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_INPUTS` | - | Number of input streams |
| `NUM_OUTPUTS` | - | Number of output streams |
| `DATAW` | - | Data width in bits |
| `ARBITER` | "R" | Arbitration policy (same as VX_generic_arbiter TYPE) |
| `STICKY` | 0 | Sticky grant mode |
| `MAX_FANOUT` | 0 | Maximum fanout before hierarchical splitting |
| `OUT_BUF` | 0 | Output buffer configuration |
| `PERF_CTR_BITS` | 0 | Performance counter width for collision tracking |

**Features:**
- Handles asymmetric input/output ratios with automatic multiplexing/demultiplexing
- Hierarchical fanout management for large multiplexer trees
- Optional output buffering per output port
- Collision performance counter for monitoring contention

---

### VX_mem_scheduler

**File:** `VX_mem_scheduler.sv`  
**Purpose:** Complex memory request scheduler with coalescing and batch management.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CORE_REQS` | - | Number of core request ports |
| `MEM_CHANNELS` | - | Number of memory channels |
| `WORD_SIZE` | - | Word size in bytes |
| `LINE_SIZE` | - | Cache line size in bytes |
| `CORE_TAG_WIDTH` | - | Core-side tag width |
| `MEM_TAG_WIDTH` | - | Memory-side tag width |
| `UUID_WIDTH` | 0 | UUID width for multi-core tracing |
| `QUEUE_SIZE` | - | Internal queue depth |

**Features:**
- Coalesces multiple small requests into cache-line-sized transactions
- Batch management for outstanding requests
- UUID tracking for multi-core/multi-thread environments
- Handles response reordering and demultiplexing

---

## 3. Multiplexing and Demultiplexing

Combinational data routing modules for selecting and distributing data.

### VX_mux

**File:** `VX_mux.sv`  
**Purpose:** Simple binary-indexed multiplexer.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATAW` | - | Data width in bits |
| `N` | - | Number of inputs |
| `LN` | LOG2UP(N) | Select signal width |

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `data_in` | input | N * DATAW | Input data array |
| `sel` | input | LN | Select index |
| `data_out` | output | DATAW | Selected output |

**Features:**
- Purely combinational -- direct array indexing
- Synthesizes to standard MUX trees

---

### VX_onehot_mux

**File:** `VX_onehot_mux.sv`  
**Purpose:** One-hot encoded multiplexer with multiple implementation models.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATAW` | - | Data width in bits |
| `N` | - | Number of inputs |
| `MODEL` | 1 | Implementation model (1-3) |
| `LUT_OPT` | 0 | LUT optimization for small N |

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `data_in` | input | N * DATAW | Input data array |
| `sel_in` | input | N | One-hot select |
| `data_out` | output | DATAW | Selected output |

**Features:**
- **Model 1:** AND-mask each input with its select bit, then OR-reduce
- **Model 2:** Convert one-hot to binary via `VX_find_first`, then use `VX_mux`
- **Model 3:** Priority-based selection
- `LUT_OPT` provides hand-optimized logic for N=2..8

---

### VX_demux

**File:** `VX_demux.sv`  
**Purpose:** One-hot demultiplexer distributing a single input to multiple outputs.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATAW` | - | Data width in bits |
| `N` | - | Number of outputs |
| `MODEL` | 0 | Implementation model |

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `data_in` | input | DATAW | Input data |
| `sel_in` | input | N | One-hot output select |
| `data_out` | output | N * DATAW | Output data array |

**Features:**
- **Model 0:** AND-mask input with each output select bit
- **Model 1:** Alternative implementation
- Purely combinational

---

### VX_transpose

**File:** `VX_transpose.sv`  
**Purpose:** Combinational 2D array transposition.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATAW` | 1 | Element data width |
| `N` | 1 | First dimension |
| `M` | 1 | Second dimension |

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `data_in` | input | N * M * DATAW | Input array [N][M] |
| `data_out` | output | M * N * DATAW | Transposed output [M][N] |

**Features:**
- Pure wire reassignment: `data_out[j][i] = data_in[i][j]`
- Zero latency, zero area (synthesis optimizes to routing only)

---

## 4. Stream Processing Utilities

Modules for splitting, merging, packing, and dispatching streaming data with valid/ready flow control.

### VX_stream_fork

**File:** `VX_stream_fork.sv`  
**Purpose:** Forks a single input stream to multiple output streams.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_OUTPUTS` | - | Number of output streams |
| `DATAW` | - | Data width in bits |
| `OUT_BUF` | 0 | Output buffer configuration |
| `EAGER` | 0 | Enable eager delivery mode |

**Features:**
- **Lockstep mode** (`EAGER=0`): waits until all outputs are ready before accepting input
- **Eager mode** (`EAGER=1`): delivers to ready outputs immediately, buffers for others
- Optional output buffering per output port

---

### VX_stream_join

**File:** `VX_stream_join.sv`  
**Purpose:** Joins multiple input streams into a single output stream.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_INPUTS` | - | Number of input streams |
| `DATAW` | - | Data width in bits |
| `OUT_BUF` | 0 | Output buffer configuration |
| `EAGER` | 0 | Enable eager acceptance mode |

**Features:**
- **Lockstep mode** (`EAGER=0`): waits until all inputs are valid before producing output
- **Eager mode** (`EAGER=1`): accepts partial arrivals, outputs when all have arrived
- Concatenates all input data into a single wide output

---

### VX_stream_pack

**File:** `VX_stream_pack.sv`  
**Purpose:** Packs multiple requests with matching tags into a single output.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_REQS` | - | Number of request ports |
| `DATA_WIDTH` | - | Data width per request |
| `TAG_WIDTH` | - | Tag width per request |
| `TAG_SEL_BITS` | - | Number of tag bits used for grouping |
| `ARBITER` | "R" | Arbiter type for tag selection |
| `OUT_BUF` | 0 | Output buffer configuration |

**Features:**
- Groups requests by matching `TAG_SEL_BITS` of their tag
- Outputs a mask indicating which requests are packed together
- Used for memory coalescing and batch processing

---

### VX_stream_unpack

**File:** `VX_stream_unpack.sv`  
**Purpose:** Unpacks a single packed request into multiple individual outputs.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_REQS` | - | Number of output request ports |
| `DATA_WIDTH` | - | Data width per request |
| `TAG_WIDTH` | - | Tag width per request |
| `OUT_BUF` | 0 | Output buffer configuration |

**Features:**
- Takes a masked packed input and delivers individual requests to outputs
- Tracks which outputs have been delivered; completes when all masked entries are sent
- Inverse operation of `VX_stream_pack`

---

### VX_stream_dispatch

**File:** `VX_stream_dispatch.sv`  
**Purpose:** Pull-based stream dispatcher that routes arbitrated input to the first ready output.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_INPUTS` | 1 | Number of input streams |
| `NUM_OUTPUTS` | 1 | Number of output streams |
| `DATAW` | 1 | Data width in bits |
| `ARBITER` | "R" | Arbiter type for input selection |
| `BUFFERED` | 0 | FIFO buffer depth between arbiter and dispatcher |
| `OUT_BUF` | 0 | Output buffer configuration |

**Features:**
- Arbitrates among multiple inputs using configurable arbiter
- Optional FIFO buffering between arbitration and dispatch stages
- Output selection determined by downstream readiness (pull-based)
- Priority encoder selects the first ready output

---

### VX_nz_iterator

**File:** `VX_nz_iterator.sv`  
**Purpose:** Non-zero element iterator for sparse data processing.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATAW` | - | Data width per element |
| `KEYW` | - | Key width |
| `N` | - | Number of elements |
| `OUT_REG` | 0 | Output register enable |
| `LPID_WIDTH` | - | Loop/partition ID width |

**Features:**
- Iterates over non-zero elements in an N-element array
- Outputs `pid` (current index), `sop` (start-of-packet), `eop` (end-of-packet)
- Identifies first and last non-zero elements for boundary detection
- Used in sparse computation paths (e.g., WMMA operations)

---

### VX_pe_serializer

**File:** `VX_pe_serializer.sv`  
**Purpose:** Serializer that batches lane-parallel data through a smaller number of processing elements.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_LANES` | - | Number of input lanes |
| `NUM_PES` | - | Number of processing elements |
| `LATENCY` | - | PE processing latency |
| `DATA_IN_WIDTH` | - | Input data width per lane |
| `DATA_OUT_WIDTH` | - | Output data width per lane |
| `SHARED_WIDTH` | 0 | Shared data width (broadcast to all PEs) |
| `TAG_WIDTH` | - | Tag width for tracking |

**Features:**
- Batches `NUM_LANES` inputs through `NUM_PES` processing elements over multiple cycles
- Collects results and reassembles into lane-parallel output
- Handles `NUM_LANES != NUM_PES` cases with eager or lockstep modes
- Tracks in-flight operations via tags

---

## 5. Crossbar and Interconnect Networks

High-performance switching fabrics for routing data between multiple sources and destinations.

### VX_stream_xbar

**File:** `VX_stream_xbar.sv`  
**Purpose:** Full crossbar switch with arbitration and collision tracking.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_INPUTS` | - | Number of input ports |
| `NUM_OUTPUTS` | - | Number of output ports |
| `DATAW` | - | Data width in bits |
| `ARBITER` | "R" | Arbitration policy |
| `OUT_BUF` | 0 | Output buffer configuration |
| `MAX_FANOUT` | 0 | Maximum fanout before tree splitting |
| `PERF_CTR_BITS` | 0 | Performance counter width |

**Features:**
- Full N x M crossbar with per-output arbitration
- Reports which input each output selected (`sel_out`)
- Collision counter for performance monitoring
- Hierarchical fanout control for large port counts
- O(N*M) area complexity

---

### VX_stream_omega

**File:** `VX_stream_omega.sv`  
**Purpose:** Omega network (multi-stage logarithmic switching network).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_INPUTS` | - | Number of input ports |
| `NUM_OUTPUTS` | - | Number of output ports |
| `RADIX` | 2 | Switch radix per stage |
| `DATAW` | - | Data width in bits |
| `ARBITER` | "R" | Arbitration policy |
| `OUT_BUF` | 0 | Output buffer configuration |

**Features:**
- O(N log N) area complexity -- more scalable than full crossbar for large N
- Multi-stage switching with configurable radix
- Collision tracking for performance analysis
- Better suited for large port counts (>8) where crossbar area is prohibitive

---

### VX_stream_xpoint

**File:** `VX_stream_xpoint.sv`  
**Purpose:** Programmable crosspoint switch with explicit routing.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_INPUTS` | - | Number of input ports |
| `NUM_OUTPUTS` | - | Number of output ports |
| `DATAW` | - | Data width in bits |
| `OUT_DRIVEN` | 0 | Routing driven by output (1) or input (0) side |
| `OUT_BUF` | 0 | Output buffer configuration |

**Features:**
- Each input or output has a dedicated `sel` signal controlling its routing
- **Input-driven** (`OUT_DRIVEN=0`): each input specifies which output to target
- **Output-driven** (`OUT_DRIVEN=1`): each output specifies which input to read
- No arbitration -- assumes external conflict resolution

---

### VX_stream_switch

**File:** `VX_stream_switch.sv`  
**Purpose:** Configurable stream switch for asymmetric input/output ratios.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_INPUTS` | - | Number of input streams |
| `NUM_OUTPUTS` | - | Number of output streams |
| `DATAW` | - | Data width in bits |
| `OUT_BUF` | 0 | Output buffer configuration |

**Features:**
- Handles mismatched input/output counts gracefully
- Automatic multiplexing (many-to-one) or demultiplexing (one-to-many) as needed
- Optional output buffering

---

### VX_scope_switch

**File:** `VX_scope_switch.sv`  
**Purpose:** Broadcast-style switch for debug scope infrastructure.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N` | - | Number of outputs |

**Features:**
- Broadcasts a single request to N outputs
- Collects and merges responses from all outputs
- Used in the debug/scope tap infrastructure

---

## 6. Encoders and Decoders

Combinational logic for encoding, priority detection, and bit scanning.

### VX_priority_encoder

**File:** `VX_priority_encoder.sv`  
**Purpose:** Multi-bit input to priority index and one-hot conversion.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N` | - | Input width |
| `REVERSE` | 0 | 0=LSB priority, 1=MSB priority |
| `MODEL` | 1 | Implementation model (1-3) |

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `data_in` | input | N | Input bitmap |
| `onehot_out` | output | N | One-hot of highest-priority set bit |
| `index_out` | output | LOG2UP(N) | Binary index of highest-priority bit |
| `valid_out` | output | 1 | At least one bit is set |

**Features:**
- Multiple implementation models (parallel prefix, scan, loop)
- `REVERSE` selects priority direction (LSB-first or MSB-first)

---

### VX_onehot_encoder

**File:** `VX_onehot_encoder.sv`  
**Purpose:** Converts one-hot input to binary index.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N` | - | Input width |
| `REVERSE` | 0 | Scan direction |
| `MODEL` | 1 | Implementation model (1-2) |

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `data_in` | input | N | One-hot input |
| `data_out` | output | LOG2UP(N) | Binary index |
| `valid_out` | output | 1 | Input is valid (has exactly one bit set) |

**Features:**
- Parallel prefix computation for O(log N) latency
- Used internally by many arbiter and mux modules

---

### VX_lzc

**File:** `VX_lzc.sv`  
**Purpose:** Leading (or trailing) zero counter.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N` | - | Input width |
| `REVERSE` | 0 | 0=leading zeros, 1=trailing zeros |

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `data_in` | input | N | Input value |
| `data_out` | output | LOG2UP(N) | Zero count |
| `valid_out` | output | 1 | Input is non-zero |

**Features:**
- Built on `VX_find_first` for efficient tree-based computation
- Essential building block for normalization, floating-point, and shift operations

---

### VX_find_first

**File:** `VX_find_first.sv`  
**Purpose:** Finds the first valid entry in a data array.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N` | - | Number of entries |
| `DATAW` | - | Data width per entry |
| `REVERSE` | 0 | Search direction |

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `valid_in` | input | N | Valid flags per entry |
| `data_in` | input | N * DATAW | Data array |
| `data_out` | output | DATAW | Data from first valid entry |
| `valid_out` | output | 1 | At least one entry is valid |

**Features:**
- Tree-based binary selection for O(log N) latency
- Returns both the data value and validity

---

### VX_scan

**File:** `VX_scan.sv`  
**Purpose:** Parallel prefix scan (Kogge-Stone style).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N` | - | Input width |
| `OP` | "XOR" | Scan operation: "XOR", "AND", "OR" |
| `REVERSE` | 0 | Scan direction |

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `data_in` | input | N | Input vector |
| `data_out` | output | N | Prefix scan result |

**Features:**
- O(log N) latency parallel prefix computation
- Supports XOR (parity), AND (all-ones prefix), OR (any-one prefix)
- Used internally by round-robin arbiters and priority encoders

---

## 7. Arithmetic and Computation

Parameterized arithmetic units supporting various area/performance trade-offs.

### VX_multiplier

**File:** `VX_multiplier.sv`  
**Purpose:** Combinatorial or pipelined multiplier wrapper.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `A_WIDTH` | - | Operand A width |
| `B_WIDTH` | - | Operand B width |
| `R_WIDTH` | - | Result width |
| `SIGNED` | 0 | Signed multiplication |
| `LATENCY` | 0 | Pipeline stages (0=combinational) |

**Features:**
- Uses `VX_pipe_register` for optional pipelining
- Relies on synthesis tool for multiplier mapping (DSP blocks on FPGA)

---

### VX_serial_mul

**File:** `VX_serial_mul.sv`  
**Purpose:** Iterative shift-and-add multiplier (ZipCPU algorithm).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `A_WIDTH` | - | Operand A width |
| `B_WIDTH` | - | Operand B width |
| `R_WIDTH` | - | Result width |
| `SIGNED` | 0 | Signed multiplication |
| `LANES` | 1 | Number of parallel lanes |

**Features:**
- Iterative implementation -- minimal area, multi-cycle latency
- Multi-lane support for throughput scaling
- Signed and unsigned modes
- Useful when DSP blocks are unavailable or area-constrained

---

### VX_fold_mul

**File:** `VX_fold_mul.sv`  
**Purpose:** Folded multiplier for processing multiple operand pairs sequentially.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_INPUTS` | - | Number of operand pairs |
| `IN_WIDTH` | - | Input operand width |
| `OUT_WIDTH` | - | Output width |
| `SIGNED` | 0 | Signed multiplication |
| `LATENCY` | 0 | Pipeline stages |

**Features:**
- Shares a single multiplier across multiple operand pairs
- Cascaded multiplication for multi-input reduction
- Useful for dot-product and accumulation operations

---

### VX_wallace_mul

**File:** `VX_wallace_mul.sv`  
**Purpose:** Wallace tree multiplier for high-speed parallel multiplication.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N` | - | Operand width |
| `P` | 2*N | Product width |
| `CPA_KS` | 0 | Use Kogge-Stone carry-propagate adder |

**Features:**
- Partial product generation + CSA tree + final carry-propagate addition
- Wallace reduction tree minimizes critical path depth
- Optional Kogge-Stone CPA for the final addition stage

---

### VX_divider

**File:** `VX_divider.sv`  
**Purpose:** Integer divider with FPGA-vendor-specific support.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_WIDTH` | - | Numerator width |
| `D_WIDTH` | - | Denominator width |
| `Q_WIDTH` | - | Quotient width |
| `R_WIDTH` | - | Remainder width |
| `N_SIGNED` | 0 | Numerator signed |
| `D_SIGNED` | 0 | Denominator signed |
| `LATENCY` | 0 | Pipeline stages |

**Features:**
- QUARTUS (Intel FPGA): uses `lpm_divide` megafunction
- Other targets: uses SystemVerilog `/` and `%` operators with pipelining
- Configurable signed/unsigned for numerator and denominator independently

---

### VX_serial_div

**File:** `VX_serial_div.sv`  
**Purpose:** Iterative restoring divider with multi-lane support.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WIDTHN` | - | Numerator width |
| `WIDTHD` | - | Denominator width |
| `WIDTHQ` | - | Quotient width |
| `WIDTHR` | - | Remainder width |
| `LANES` | 1 | Number of parallel lanes |

**Features:**
- Restoring division algorithm -- minimal area, multi-cycle latency
- Multi-lane support for throughput
- Signed and unsigned modes
- Suitable for area-constrained designs

---

### VX_ks_adder

**File:** `VX_ks_adder.sv`  
**Purpose:** Kogge-Stone parallel prefix adder.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N` | - | Operand width |
| `BYPASS` | 0 | Use simple `+` operator instead |

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `a` | input | N | Operand A |
| `b` | input | N | Operand B |
| `cin` | input | 1 | Carry in |
| `result` | output | N | Sum result |
| `cout` | output | 1 | Carry out |

**Features:**
- O(log N) latency carry-lookahead style computation
- `BYPASS` mode uses the synthesis tool's `+` operator (for comparison/fallback)
- Used as the final CPA stage in Wallace tree multipliers

---

### VX_popcount

**File:** `VX_popcount.sv`  
**Purpose:** Population count (number of set bits).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N` | - | Input width |
| `MODEL` | 1 | Implementation model (1-2) |

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `data_in` | input | N | Input bitmap |
| `data_out` | output | LOG2UP(N+1) | Number of set bits |

**Features:**
- Optimized implementations for small N (1-18)
- Model selection for area/speed trade-off
- Handles both simulation and synthesis paths
- Used in mask processing and SIMT lane management

---

### VX_reduce_tree

**File:** `VX_reduce_tree.sv`  
**Purpose:** Tree-based reduction of multiple operands using a configurable operator.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IN_W` | - | Input element width |
| `OUT_W` | - | Output width |
| `N` | - | Number of input elements |
| `OP` | "+" | Reduction operator: "+", "^", "&", "\|" |

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `data_in` | input | N * IN_W | Input array |
| `data_out` | output | OUT_W | Reduced result |

**Features:**
- Recursive binary tree implementation
- Supports addition, XOR, AND, and OR reduction
- O(log N) depth for all operators

---

## 8. Carry-Save Adders and Compression Trees

Specialized arithmetic building blocks for high-speed multi-operand addition, primarily used in multiplier designs.

### VX_csa_32

**File:** `VX_csa_32.sv`  
**Purpose:** 3:2 carry-save compressor (full adder array).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N` | - | Input operand width |
| `WIDTH_O` | N+1 | Output width |

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `a`, `b`, `c` | input | N | Three input operands |
| `s` | output | WIDTH_O | Sum output |
| `co` | output | WIDTH_O | Carry output |

**Features:**
- Reduces 3 operands to 2 (sum + carry) in a single gate delay
- Fundamental building block for all CSA trees

---

### VX_csa_42

**File:** `VX_csa_42.sv`  
**Purpose:** 4:2 carry-save compressor.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N` | - | Input operand width |
| `WIDTH_O` | N+2 | Output width |

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `a`, `b`, `c`, `d` | input | N | Four input operands |
| `s` | output | WIDTH_O | Sum output |
| `co` | output | WIDTH_O | Carry output |

**Features:**
- Reduces 4 operands to 2 in two gate delays
- Cascaded 3:2 compressors internally
- Higher throughput than two sequential 3:2 stages

---

### VX_csa_mod4

**File:** `VX_csa_mod4.sv`  
**Purpose:** Modulo-4 CSA reduction tree for arbitrary operand counts.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N` | - | Number of input operands |
| `W` | - | Operand width |
| `S` | - | Output width |
| `CPA_KS` | 0 | Use Kogge-Stone for final CPA |
| `NO_CPA` | 0 | Skip final carry-propagate addition |

**Features:**
- Hierarchical 4:2 compressor tree
- Handles any operand count N >= 2
- Remainder handling for non-divisible-by-4 counts
- Optional final CPA stage (can output in carry-save form)

---

### VX_csa_tree

**File:** `VX_csa_tree.sv`  
**Purpose:** Configurable CSA reduction tree with clustering and balancing.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N` | - | Number of input operands |
| `W` | - | Operand width |
| `K` | 4 | Cluster size |
| `BAL` | 1 | Balanced (1) or ragged (0) tree |
| `S` | - | Output width |

**Features:**
- Input clustering: groups operands into clusters of size K before reduction
- **Balanced mode** (`BAL=1`): uniform tree depth across all paths
- **Ragged mode** (`BAL=0`): allows uneven path depths for smaller area
- Bit-width grows by 2 per reduction level
- Used in Wallace tree multipliers and multi-operand accumulators

---

## 9. Memory Elements

Parameterized memory primitives with support for FPGA (LUTRAM/BRAM) and ASIC synthesis.

### VX_dp_ram

**File:** `VX_dp_ram.sv`  
**Purpose:** Dual-port RAM with configurable read/write semantics.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATAW` | - | Data width in bits |
| `SIZE` | - | Number of entries |
| `WRENW` | 1 | Write enable granularity |
| `OUT_REG` | 0 | Output register enable |
| `LUTRAM` | 0 | Force LUTRAM implementation |
| `RDW_MODE` | "W" | Read-during-write mode: "W" (write-first), "R" (read-first) |
| `RADDR_REG` | 0 | Read address registered hint |
| `INIT_ENABLE` | 0 | Enable initialization |
| `INIT_FILE` | "" | Initialization memory file |

**Features:**
- Separate read and write ports with independent addresses
- Bit-level write enable (`WRENW` granularity)
- Write-first or read-first semantics for simultaneous read/write to same address
- FPGA: infers LUTRAM or BRAM based on `LUTRAM` parameter
- ASIC: uses register-based implementation
- Optional output register for timing improvement
- File-based initialization for preloaded content

---

### VX_sp_ram

**File:** `VX_sp_ram.sv`  
**Purpose:** Single-port RAM with configurable modes.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATAW` | - | Data width in bits |
| `SIZE` | - | Number of entries |
| `WRENW` | 1 | Write enable granularity |
| `OUT_REG` | 0 | Output register enable |
| `LUTRAM` | 0 | Force LUTRAM implementation |
| `RDW_MODE` | "W" | Read-during-write: "W" (write-first), "R" (read-first), "N" (no-change) |
| `INIT_ENABLE` | 0 | Enable initialization |
| `INIT_FILE` | "" | Initialization memory file |

**Features:**
- Single address shared between read and write
- Three read-during-write modes including "N" (no-change) which holds previous read data
- `ASYNC_BRAM_PATCH` support for asynchronous read BRAM workarounds
- Same FPGA/ASIC implementation paths as VX_dp_ram

---

### VX_async_ram_patch

**File:** `VX_async_ram_patch.sv`  
**Purpose:** Asynchronous RAM wrapper with patching support for synthesis tools.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATAW` | 1 | Data width |
| `SIZE` | 1 | Number of entries |
| `WRENW` | 1 | Write enable granularity |
| `DUAL_PORT` | 0 | Enable dual-port operation |
| `FORCE_BRAM` | 0 | Force block RAM inference |
| `RADDR_REG` | 0 | Read address registered |
| `RADDR_RESET` | 0 | Read address has reset |
| `WRITE_FIRST` | 0 | Write-first semantics |
| `INIT_ENABLE` | 0 | Enable initialization |
| `INIT_FILE` | "" | Initialization file |
| `INIT_VALUE` | 0 | Default init value |

**Features:**
- Wraps synchronous RAM with asynchronous read path emulation
- Uses placeholder modules for tool-specific register/memory patching
- Supports dual-port and single-port configurations
- Handles vendor-specific synthesis quirks (Xilinx BRAM async read workaround)

---

### VX_fifo_queue

**File:** `VX_fifo_queue.sv`  
**Purpose:** FIFO queue with configurable depth and status flags.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATAW` | - | Data width |
| `DEPTH` | - | Queue depth |
| `ALM_FULL` | DEPTH-1 | Almost-full threshold |
| `ALM_EMPTY` | 1 | Almost-empty threshold |
| `OUT_REG` | 0 | Output register enable |
| `LUTRAM` | 0 | Force LUTRAM |

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `clk` | input | 1 | Clock |
| `reset` | input | 1 | Reset |
| `push` | input | 1 | Enqueue |
| `pop` | input | 1 | Dequeue |
| `data_in` | input | DATAW | Enqueue data |
| `data_out` | output | DATAW | Dequeue data |
| `empty` | output | 1 | Queue is empty |
| `alm_empty` | output | 1 | Queue size <= ALM_EMPTY |
| `full` | output | 1 | Queue is full |
| `alm_full` | output | 1 | Queue size >= ALM_FULL |
| `size` | output | LOG2UP(DEPTH+1) | Current occupancy |

**Features:**
- Circular buffer with read/write pointers
- Configurable almost-full and almost-empty thresholds for flow control
- Backed by `VX_dp_ram` for efficient storage
- Optional output register for timing

---

### VX_index_buffer

**File:** `VX_index_buffer.sv`  
**Purpose:** Indexed buffer with acquire/release semantics.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATAW` | - | Data width |
| `SIZE` | - | Buffer capacity |
| `LUTRAM` | 0 | Force LUTRAM |

**Features:**
- Uses `VX_allocator` for free-slot management
- `VX_dp_ram` for data storage
- Acquire returns next free index; release frees a specific index
- Used for tag/transaction tracking in memory subsystems

---

### VX_index_queue

**File:** `VX_index_queue.sv`  
**Purpose:** Queue with automatic index allocation and per-entry validity.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATAW` | - | Data width |
| `SIZE` | - | Queue capacity |

**Features:**
- Per-entry valid flags for sparse occupancy
- Auto-advances head pointer when head entry becomes invalid
- Separate push (write to allocated index) and pop (read from head) interfaces

---

## 10. Memory Controllers and Adapters

Complex adapter modules for interfacing between different memory bus protocols and data widths.

### VX_mem_bank_adapter

**File:** `VX_mem_bank_adapter.sv`  
**Purpose:** Multi-port to multi-bank memory adapter with address banking and tag management.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATA_WIDTH` | - | Data width |
| `ADDR_WIDTH_IN` | - | Input address width |
| `ADDR_WIDTH_OUT` | - | Output address width |
| `TAG_WIDTH_IN` | - | Input tag width |
| `TAG_WIDTH_OUT` | - | Output tag width |
| `NUM_PORTS_IN` | - | Number of input ports |
| `NUM_BANKS_OUT` | - | Number of output banks |
| `INTERLEAVE` | 0 | Interleaved (1) or sequential (0) banking |
| `TAG_BUFFER_SIZE` | - | Tag buffer depth |
| `ARBITER` | "R" | Arbitration policy |

**Features:**
- Maps multiple input ports to multiple output banks
- Interleaved or sequential address-to-bank mapping
- Crossbar-based request routing
- Tag buffer for bank-to-port response routing
- Separate request and response crossbars

---

### VX_mem_data_adapter

**File:** `VX_mem_data_adapter.sv`  
**Purpose:** Adapter for bridging different data bus widths.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SRC_DATA_WIDTH` | - | Source data width |
| `SRC_ADDR_WIDTH` | - | Source address width |
| `DST_DATA_WIDTH` | - | Destination data width |
| `DST_ADDR_WIDTH` | - | Destination address width |
| `TAG_WIDTH` | - | Tag width |

**Features:**
- Handles both **width expansion** (narrow source to wide destination) and **width reduction** (wide to narrow)
- Byte-level granularity for write enables
- Maintains transaction ordering and correctness for partial transfers
- Automatic address translation between different word sizes

---

### VX_mem_coalescer

**File:** `VX_mem_coalescer.sv`  
**Purpose:** Memory request coalescer for merging small requests into cache-line transactions.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_REQS` | - | Number of request ports |
| `ADDR_WIDTH` | - | Address width |
| `DATA_IN_SIZE` | - | Input data size (bytes) |
| `DATA_OUT_SIZE` | - | Output data size (bytes, typically cache line) |
| `TAG_WIDTH` | - | Tag width |
| `QUEUE_SIZE` | - | Coalescing queue depth |
| `UUID_WIDTH` | 0 | UUID width for multi-core tracing |

**Features:**
- Merges multiple small (word-sized) requests targeting the same cache line
- Batch management for in-flight coalesced transactions
- Handles partial coalescing when queue is full
- UUID tracking for multi-core/multi-thread debug visibility
- Critical for memory bandwidth efficiency in GPGPU workloads

---

### VX_axi_adapter

**File:** `VX_axi_adapter.sv`  
**Purpose:** Full AXI4 bus protocol adapter.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATA_WIDTH` | - | AXI data width |
| `ADDR_WIDTH_IN` | - | Internal address width (word) |
| `ADDR_WIDTH_OUT` | - | AXI address width (byte) |
| `TAG_WIDTH_IN` | - | Internal tag width |
| `TAG_WIDTH_OUT` | - | AXI tag width |
| `NUM_CHANNELS` | - | Number of AXI channels |

**Features:**
- Full AXI4 protocol support:
  - Write address channel (AW)
  - Write data channel (W)
  - Write response channel (B)
  - Read address channel (AR)
  - Read data channel (R)
- Word-to-byte address translation
- Complex tag management for outstanding transaction tracking
- Multi-channel support

---

### VX_axi_write_ack

**File:** `VX_axi_write_ack.sv`  
**Purpose:** AXI write transaction handshake controller.

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `clk` | input | 1 | Clock |
| `reset` | input | 1 | Reset |
| `awvalid` | input | 1 | Write address valid |
| `awready` | input | 1 | Write address ready |
| `wvalid` | input | 1 | Write data valid |
| `wready` | input | 1 | Write data ready |
| `aw_ack` | output | 1 | Address channel acknowledged |
| `w_ack` | output | 1 | Data channel acknowledged |
| `tx_ack` | output | 1 | Full transaction acknowledged |
| `tx_rdy` | output | 1 | Ready for new transaction |

**Features:**
- Tracks separate AXI write address and data channel completions
- Produces unified transaction acknowledgment when both channels complete
- Handles out-of-order address/data channel completion

---

### VX_avs_adapter

**File:** `VX_avs_adapter.sv`  
**Purpose:** Avalon-ST (Intel/Altera) bus adapter.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATA_WIDTH` | - | Avalon data width |
| `ADDR_WIDTH_IN` | - | Internal address width |
| `ADDR_WIDTH_OUT` | - | Avalon address width |
| `BURST_WIDTH` | - | Burst count width |
| `NUM_BANKS` | - | Number of memory banks |
| `TAG_WIDTH` | - | Tag width |

**Features:**
- Avalon-ST address and burst count translation
- Single-cycle transaction support
- Multi-bank routing
- Tag buffer for response routing

---

## 11. Bit Manipulation Utilities

Simple combinational modules for bit-level data manipulation.

### VX_bits_insert

**File:** `VX_bits_insert.sv`  
**Purpose:** Inserts S bits at a specified position within an N-bit vector.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N` | - | Original data width |
| `S` | - | Number of bits to insert |
| `POS` | - | Insertion position |

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `data_in` | input | N | Original data |
| `ins_in` | input | S | Bits to insert |
| `data_out` | output | N+S | Result with inserted bits |

---

### VX_bits_remove

**File:** `VX_bits_remove.sv`  
**Purpose:** Removes S bits from a specified position within an (N+S)-bit vector.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N` | - | Remaining data width |
| `S` | - | Number of bits to remove |
| `POS` | - | Removal position |

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `data_in` | input | N+S | Input data |
| `rem_out` | output | S | Removed bits |
| `data_out` | output | N | Remaining data |

---

### VX_bits_concat

**File:** `VX_bits_concat.sv`  
**Purpose:** Concatenates two bit vectors with flexible widths.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `L` | - | Left operand width |
| `R` | - | Right operand width |

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `left` | input | L | Left operand |
| `right` | input | R | Right operand |
| `data_out` | output | L+R | Concatenated result |

---

### VX_onehot_shift

**File:** `VX_onehot_shift.sv`  
**Purpose:** Cross-product of two one-hot encoded signals.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N` | - | First input width |
| `M` | - | Second input width |

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `data_in` | input | N | First one-hot input |
| `shift_in` | input | M | Second one-hot input |
| `data_out` | output | N*M | Cross-product output |

---

## 12. Synchronization and Timing

Modules for clock management, reset distribution, and edge detection.

### VX_clockgate

**File:** `VX_clockgate.sv`  
**Purpose:** Integrated clock gating cell for power management.

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `clk` | input | 1 | Input clock |
| `enable` | input | 1 | Gate enable |
| `clk_out` | output | 1 | Gated clock |

**Features:**
- Glitch-free output using a transparent latch
- Latch is transparent while clock is low, capturing enable
- Gated clock output is AND of latched enable and input clock
- Essential for dynamic power reduction in large designs

---

### VX_edge_trigger

**File:** `VX_edge_trigger.sv`  
**Purpose:** Edge detector producing a single-cycle pulse.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `POS` | 1 | 1=positive edge, 0=negative edge |
| `INIT` | 0 | Initial state of internal register |

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `clk` | input | 1 | Clock |
| `reset` | input | 1 | Reset |
| `data_in` | input | 1 | Input signal |
| `data_out` | output | 1 | Edge pulse |

**Features:**
- Single flip-flop implementation
- Detects rising or falling edges based on `POS` parameter

---

### VX_reset_relay

**File:** `VX_reset_relay.sv`  
**Purpose:** Reset tree distribution with fanout control.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N` | - | Number of reset outputs |
| `MAX_FANOUT` | 0 | Maximum fanout per buffer (0=unlimited) |

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `clk` | input | 1 | Clock |
| `reset` | input | 1 | Input reset |
| `reset_o` | output | N | Distributed reset outputs |

**Features:**
- Reduces reset skew by distributing through a buffer tree
- Automatically inserts intermediate buffers when fanout exceeds `MAX_FANOUT`
- Ensures clean reset timing across large designs

---

## 13. Allocators and Counters

Resource management primitives for tracking allocations, pending operations, and mutual exclusion.

### VX_allocator

**File:** `VX_allocator.sv`  
**Purpose:** Free-list based resource allocator.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SIZE` | - | Number of allocatable slots |

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `clk` | input | 1 | Clock |
| `reset` | input | 1 | Reset |
| `acquire_en` | input | 1 | Acquire a slot |
| `acquire_addr` | output | LOG2UP(SIZE) | Acquired slot index |
| `release_en` | input | 1 | Release a slot |
| `release_addr` | input | LOG2UP(SIZE) | Slot to release |
| `empty` | output | 1 | No slots available |
| `full` | output | 1 | All slots available (none in use) |

**Features:**
- Priority encoder selects the lowest-numbered free slot
- Bitmap-based free list for O(1) allocation
- Used for tag allocation, buffer management, and transaction tracking

---

### VX_pending_size

**File:** `VX_pending_size.sv`  
**Purpose:** Counter tracking the number of pending (in-flight) operations.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SIZE` | - | Maximum count |
| `INCRW` | 1 | Increment width (multi-bit for bulk operations) |
| `DECRW` | 1 | Decrement width |
| `ALM_FULL` | SIZE-1 | Almost-full threshold |
| `ALM_EMPTY` | 1 | Almost-empty threshold |

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `clk` | input | 1 | Clock |
| `reset` | input | 1 | Reset |
| `incr` | input | INCRW | Increment amount |
| `decr` | input | DECRW | Decrement amount |
| `empty` | output | 1 | Count is zero |
| `alm_empty` | output | 1 | Count <= ALM_EMPTY |
| `full` | output | 1 | Count == SIZE |
| `alm_full` | output | 1 | Count >= ALM_FULL |
| `size` | output | LOG2UP(SIZE+1) | Current count |

**Features:**
- Multi-bit increment/decrement for bulk operations
- Configurable almost-full and almost-empty thresholds
- Efficient implementation for single-step (INCRW=1) increments
- Used for credit-based flow control and queue occupancy tracking

---

### VX_ticket_lock

**File:** `VX_ticket_lock.sv`  
**Purpose:** Ticket-based mutual exclusion lock guaranteeing FIFO fairness.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N` | - | Number of lock slots |

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `clk` | input | 1 | Clock |
| `reset` | input | 1 | Reset |
| `acquire` | input | 1 | Request lock acquisition |
| `acquire_id` | output | LOG2UP(N) | Ticket number assigned |
| `release` | input | 1 | Release lock |
| `release_id` | input | LOG2UP(N) | Ticket to release |
| `empty` | output | 1 | Lock is free |
| `full` | output | 1 | All tickets in use |

**Features:**
- Guarantees strict FIFO ordering of lock acquisitions
- Assigns incrementing ticket numbers
- Used for serializing access to shared resources

---

## 14. Debug and Tracing

Infrastructure for runtime debug observation and serial trace extraction.

### VX_scope_tap

**File:** `VX_scope_tap.sv`  
**Purpose:** Debug trace buffer with serial readout interface.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SCOPE_ID` | - | Unique scope identifier |
| `SCOPE_IDW` | - | Scope ID width |
| `XTRIGGERW` | 0 | Edge trigger probe width |
| `HTRIGGERW` | 0 | High trigger probe width |
| `PROBEW` | - | Total probe data width |
| `DEPTH` | - | Trace buffer depth |
| `IDLE_CTRW` | - | Idle counter width |
| `TX_DATAW` | - | Serial transmit data width |

**Features:**
- Captures state transitions into a circular trace buffer
- Two trigger types:
  - **Edge triggers** (`XTRIGGERW`): capture on signal transitions
  - **High triggers** (`HTRIGGERW`): capture while signal is high
- Timestamping of captured events
- Serial readout protocol for off-chip extraction
- Idle counter tracks gaps between events
- Used with `VX_scope_switch` for multi-tap debug networks

---

## 15. Utility and Placeholder

### VX_placeholder

**File:** `VX_placeholder.sv`  
**Purpose:** Empty black-box module for hierarchical design placeholder.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `I` | 1 | Input width |
| `O` | 1 | Output width |

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| `data_in` | input | I | Input (unconnected) |
| `data_out` | output | O | Output (unconnected) |

**Features:**
- Marked with `BLACKBOX_CELL` synthesis directive
- Used as a structural placeholder for tool-specific cell replacement
- Zero logic -- purely a synthesis anchor point

---

## Module Summary Table

| # | Module | Category | Area | Latency |
|---|--------|----------|------|---------|
| 1 | VX_allocator | Allocator | Small | 0 cycles |
| 2 | VX_async_ram_patch | Memory | Varies | 0-1 cycles |
| 3 | VX_avs_adapter | Memory Adapter | Large | Multi-cycle |
| 4 | VX_axi_adapter | Memory Adapter | Large | Multi-cycle |
| 5 | VX_axi_write_ack | Memory Adapter | Small | 0-1 cycles |
| 6 | VX_bits_concat | Bit Manip | Zero | 0 cycles |
| 7 | VX_bits_insert | Bit Manip | Zero | 0 cycles |
| 8 | VX_bits_remove | Bit Manip | Zero | 0 cycles |
| 9 | VX_bypass_buffer | Buffer | Small | 0-1 cycles |
| 10 | VX_clockgate | Sync | Minimal | 0 cycles |
| 11 | VX_csa_32 | CSA | Small | 1 gate |
| 12 | VX_csa_42 | CSA | Medium | 2 gates |
| 13 | VX_csa_mod4 | CSA | Varies | O(log N) |
| 14 | VX_csa_tree | CSA | Varies | O(log N) |
| 15 | VX_cyclic_arbiter | Arbiter | Small | 0 cycles |
| 16 | VX_demux | Mux/Demux | Small | 0 cycles |
| 17 | VX_divider | Arithmetic | Large | Configurable |
| 18 | VX_dp_ram | Memory | Varies | 0-1 cycles |
| 19 | VX_edge_trigger | Sync | Minimal | 1 cycle |
| 20 | VX_elastic_adapter | Buffer | Minimal | 0-1 cycles |
| 21 | VX_elastic_buffer | Buffer | Varies | Configurable |
| 22 | VX_fifo_queue | Memory | Varies | 0-1 cycles |
| 23 | VX_find_first | Encoder | Small | O(log N) |
| 24 | VX_fold_mul | Arithmetic | Medium | Configurable |
| 25 | VX_generic_arbiter | Arbiter | Varies | 0 cycles |
| 26 | VX_gto_arbiter | Arbiter | Medium | 0 cycles |
| 27 | VX_index_buffer | Memory | Medium | 0-1 cycles |
| 28 | VX_index_queue | Memory | Medium | 0-1 cycles |
| 29 | VX_ks_adder | Arithmetic | Medium | O(log N) |
| 30 | VX_lzc | Encoder | Small | O(log N) |
| 31 | VX_matrix_arbiter | Arbiter | Medium | 0 cycles |
| 32 | VX_mem_bank_adapter | Memory Adapter | Large | Multi-cycle |
| 33 | VX_mem_coalescer | Memory Adapter | Large | Multi-cycle |
| 34 | VX_mem_data_adapter | Memory Adapter | Medium | Multi-cycle |
| 35 | VX_mem_scheduler | Scheduler | Large | Multi-cycle |
| 36 | VX_multiplier | Arithmetic | Varies | Configurable |
| 37 | VX_mux | Mux/Demux | Small | 0 cycles |
| 38 | VX_nz_iterator | Stream Util | Small | 0-1 cycles |
| 39 | VX_onehot_encoder | Encoder | Small | O(log N) |
| 40 | VX_onehot_mux | Mux/Demux | Small | 0 cycles |
| 41 | VX_onehot_shift | Bit Manip | Small | 0 cycles |
| 42 | VX_pe_serializer | Stream Util | Medium | Multi-cycle |
| 43 | VX_pending_size | Counter | Small | 0-1 cycles |
| 44 | VX_pipe_buffer | Buffer | Varies | N cycles |
| 45 | VX_pipe_register | Buffer | Varies | N cycles |
| 46 | VX_placeholder | Utility | Zero | 0 cycles |
| 47 | VX_popcount | Arithmetic | Small | O(log N) |
| 48 | VX_priority_arbiter | Arbiter | Small | 0 cycles |
| 49 | VX_priority_encoder | Encoder | Small | O(log N) |
| 50 | VX_reduce_tree | Arithmetic | Small | O(log N) |
| 51 | VX_reset_relay | Sync | Small | 1 cycle |
| 52 | VX_rr_arbiter | Arbiter | Medium | 0 cycles |
| 53 | VX_scan | Encoder | Medium | O(log N) |
| 54 | VX_scope_switch | Debug | Small | 0 cycles |
| 55 | VX_scope_tap | Debug | Large | Multi-cycle |
| 56 | VX_serial_div | Arithmetic | Small | Multi-cycle |
| 57 | VX_serial_mul | Arithmetic | Small | Multi-cycle |
| 58 | VX_shift_register | Buffer | Varies | N cycles |
| 59 | VX_skid_buffer | Buffer | Small | 0-1 cycles |
| 60 | VX_sp_ram | Memory | Varies | 0-1 cycles |
| 61 | VX_stream_arb | Interconnect | Medium | 0-1 cycles |
| 62 | VX_stream_buffer | Buffer | Small | 0-1 cycles |
| 63 | VX_stream_dispatch | Stream Util | Medium | 0-1 cycles |
| 64 | VX_stream_fork | Stream Util | Small | 0 cycles |
| 65 | VX_stream_join | Stream Util | Small | 0 cycles |
| 66 | VX_stream_omega | Interconnect | Large | O(log N) |
| 67 | VX_stream_pack | Stream Util | Medium | 0-1 cycles |
| 68 | VX_stream_switch | Interconnect | Medium | 0-1 cycles |
| 69 | VX_stream_unpack | Stream Util | Medium | Multi-cycle |
| 70 | VX_stream_xbar | Interconnect | Large | 0-1 cycles |
| 71 | VX_stream_xpoint | Interconnect | Medium | 0-1 cycles |
| 72 | VX_ticket_lock | Allocator | Small | 0-1 cycles |
| 73 | VX_toggle_buffer | Buffer | Small | 1 cycle |
| 74 | VX_transpose | Bit Manip | Zero | 0 cycles |
| 75 | VX_wallace_mul | Arithmetic | Large | O(log N) |
