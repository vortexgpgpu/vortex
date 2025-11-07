# Vortex Cache System Architecture Analysis

## Overview
The Vortex cache system implements a sophisticated multi-level, multi-banked cache hierarchy with integrated TLB (Translation Lookaside Buffer) support and comprehensive arbitration mechanisms. This analysis covers the cache structure, hierarchies, TLB integration, and arbitration points.

## Cache Hierarchy Structure

### 1. Top-Level Hierarchy
```
VX_cache_top (Interface Layer)
    ↓
VX_cache_wrap (Wrapper with Bypass Logic)
    ↓
VX_cache (Core Cache Logic)
    ↓
VX_cache_bank[] (Individual Cache Banks)
```

### 2. Cache Cluster Architecture
The system supports cache clustering through `VX_cache_cluster`, which can instantiate multiple cache units with arbitration between them:

```
Multiple Input Sources → Core Request Arbiter → Multiple Cache Units → Memory Request Arbiter → Memory System
```

### 3. Individual Cache Bank Structure
Each `VX_cache_bank` contains:
- **VX_cache_tags**: Tag array for hit/miss detection
- **VX_cache_data**: Data array storage
- **VX_cache_repl**: Replacement policy logic (FIFO, Random, PLRU)
- **VX_cache_mshr**: Miss Status Holding Register for handling misses
- **VX_cache_flush**: Flush control logic

## Cache Parameters and Configuration

### Key Parameters:
- **CACHE_SIZE**: Total cache size in bytes (default: 32KB-65KB)
- **LINE_SIZE**: Cache line size in bytes (default: 64 bytes)
- **NUM_BANKS**: Number of cache banks (default: 4, must be power of 2)
- **NUM_WAYS**: Associativity (default: 4 ways)
- **WORD_SIZE**: Word size in bytes (default: 16 bytes)
- **NUM_REQS**: Number of concurrent requests per cycle (default: 4)
- **MEM_PORTS**: Number of memory ports (default: 1)

### Address Mapping:
The cache uses a specific address mapping scheme defined in `VX_cache_define.vh`:
- **Word Select Bits**: `[0:WORD_SEL_BITS-1]` - Selects word within cache line
- **Bank Select Bits**: `[WORD_SEL_BITS:BANK_SEL_BITS-1]` - Selects cache bank
- **Line Select Bits**: `[BANK_SEL_BITS:LINE_SEL_BITS-1]` - Selects cache line within bank
- **Tag Bits**: `[LINE_SEL_BITS:TAG_BITS-1]` - Tag for hit/miss detection

## TLB Integration Architecture

### 1. TLB Hierarchy
```
Core Requests (Virtual Addresses)
    ↓
VX_tlb_wrap (TLB Wrapper)
    ↓
VX_tlb (Main TLB Logic)
    ↓
VX_tlb_bank[] (TLB Banks - matches cache bank structure)
    ↓
Translated Physical Addresses → Cache System
```

### 2. TLB Design Features
- **Banked Structure**: TLB banks match cache banks for parallel access
- **Page Size**: 4KB pages (12-bit page offset)
- **Associativity**: Configurable ways (default: 4-way set associative)
- **Replacement Policy**: Supports FIFO, Random, and PLRU
- **PTW Integration**: Interface for Page Table Walker on TLB misses

### 3. TLB-Cache Integration Points
The TLB integrates with the cache system at multiple levels:

#### In VX_cache_cluster:
```systemverilog
`ifdef VM_ENABLE
    for (genvar i = 0; i < NUM_CACHES; i++) begin : g_tlb_wrap
        VX_tlb_wrap #(...) tlb_wrap (
            .core_bus_if (arb_core_bus_if[i * NUM_REQS +: NUM_REQS]),
            .mem_bus_if  (tlb_core_bus_if[i * MEM_PORTS +: MEM_PORTS])
        );
    end
`endif
```

#### Address Translation Flow:
1. **Virtual Address Input**: Core requests arrive with virtual addresses
2. **TLB Lookup**: TLB banks perform parallel VPN→PPN translation
3. **Physical Address Output**: Translated addresses sent to cache
4. **TLB Miss Handling**: Misses trigger PTW (Page Table Walker) requests

### 4. TLB Miss Handling
- **Miss Detection**: Each TLB bank detects misses independently
- **PTW Interface**: Dedicated interface for Page Table Walker communication
- **Update Mechanism**: PTW updates TLB entries with new translations
- **Sequential Processing**: Multiple TLB misses handled sequentially by PTW

## Arbitration Mechanisms

### 1. Core Request Arbitration
**Location**: `VX_cache.sv` lines 330-353
**Component**: `VX_stream_xbar` (Request Crossbar)
**Purpose**: Distributes core requests to appropriate cache banks
**Arbitration Policy**: Round-robin ("R")
**Configuration**:
```systemverilog
VX_stream_xbar #(
    .NUM_INPUTS  (NUM_REQS),      // 4 concurrent requests
    .NUM_OUTPUTS (NUM_BANKS),     // 4 cache banks
    .DATAW       (CORE_REQ_DATAW),
    .ARBITER     ("R"),           // Round-robin arbitration
    .OUT_BUF     (REQ_XBAR_BUF)
) req_xbar
```

### 2. Memory Request Arbitration
**Location**: `VX_cache.sv` lines 516-531
**Component**: `VX_stream_arb` (Memory Request Arbiter)
**Purpose**: Arbitrates memory requests from multiple cache banks to memory ports
**Arbitration Policy**: Round-robin ("R")
**Configuration**:
```systemverilog
VX_stream_arb #(
    .NUM_INPUTS (NUM_BANKS),      // 4 cache banks
    .NUM_OUTPUTS(MEM_PORTS),      // 1 memory port
    .DATAW      (MEM_REQ_DATAW),
    .ARBITER    ("R")             // Round-robin arbitration
) mem_req_arb
```

### 3. Memory Response Routing
**Location**: `VX_cache.sv` lines 207-225
**Component**: `VX_stream_omega` (Memory Response Crossbar)
**Purpose**: Routes memory responses back to originating cache banks
**Arbitration Policy**: Round-robin ("R")
**Configuration**:
```systemverilog
VX_stream_omega #(
    .NUM_INPUTS  (MEM_PORTS),     // 1 memory port
    .NUM_OUTPUTS (NUM_BANKS),     // 4 cache banks
    .DATAW       (MEM_RSP_DATAW-MEM_ARB_SEL_BITS),
    .ARBITER     ("R"),           // Round-robin arbitration
    .OUT_BUF     (3)
) mem_rsp_xbar
```

### 4. Core Response Arbitration
**Location**: `VX_cache.sv` lines 457-474
**Component**: `VX_stream_xbar` (Response Crossbar)
**Purpose**: Routes cache responses back to originating core requests
**Arbitration Policy**: Round-robin ("R")
**Configuration**:
```systemverilog
VX_stream_xbar #(
    .NUM_INPUTS  (NUM_BANKS),     // 4 cache banks
    .NUM_OUTPUTS (NUM_REQS),      // 4 core requests
    .DATAW       (CORE_RSP_DATAW),
    .ARBITER     ("R")            // Round-robin arbitration
) rsp_xbar
```

### 5. Cache Cluster Arbitration
**Location**: `VX_cache_cluster.sv`
**Components**: Multiple arbiters for cluster-level coordination

#### Core Request Arbitration (lines 131-145):
```systemverilog
VX_mem_arb #(
    .NUM_INPUTS   (NUM_INPUTS),   // Multiple input sources
    .NUM_OUTPUTS  (NUM_CACHES),   // Multiple cache units
    .DATA_SIZE    (WORD_SIZE),
    .TAG_WIDTH    (TAG_WIDTH),
    .ARBITER      ("R")           // Round-robin arbitration
) core_arb
```

#### Memory Request Arbitration (lines 233-247):
```systemverilog
VX_mem_arb #(
    .NUM_INPUTS  (NUM_CACHES),    // Multiple cache units
    .NUM_OUTPUTS (1),             // Single memory port
    .DATA_SIZE   (LINE_SIZE),
    .TAG_WIDTH   (MEM_TAG_WIDTH),
    .ARBITER     ("R")            // Round-robin arbitration
) mem_arb
```

### 6. TLB Arbitration
**Location**: `VX_tlb.sv` lines 206-224 and 251-268
**Components**: Request and Response Crossbars
**Purpose**: Distributes TLB requests across TLB banks and routes responses

#### TLB Request Arbitration:
```systemverilog
VX_stream_xbar #(
    .NUM_INPUTS  (NUM_REQS),      // 4 concurrent requests
    .NUM_OUTPUTS (NUM_BANKS),     // 4 TLB banks
    .DATAW       (CORE_REQ_DATAW),
    .ARBITER     ("R"),           // Round-robin arbitration
    .OUT_BUF     (2)
) req_xbar
```

## MSHR (Miss Status Holding Register) System

### Purpose
The MSHR system handles cache misses by:
1. **Allocation**: Allocating slots for pending requests
2. **Tracking**: Tracking multiple requests to the same cache line
3. **Replay**: Replaying requests when data arrives from memory
4. **Deallocation**: Releasing slots when requests complete

### Key Features
- **Linked List Structure**: Pending requests for same line are linked
- **Ordered Processing**: Requests processed in arrival order
- **Pipeline Integration**: Tightly coupled with cache bank pipeline
- **Memory Fill Coordination**: Coordinates with memory fill responses

## Performance Monitoring

### Cache Performance Counters
- **Reads/Writes**: Core request counts
- **Read/Write Misses**: Cache miss counts
- **Bank Stalls**: Crossbar collision counts
- **MSHR Stalls**: Miss handling stalls
- **Memory Stalls**: Memory system stalls
- **Response Stalls**: Core response stalls

### TLB Performance Counters
- **TLB Hits**: Successful translations
- **TLB Misses**: Translation misses
- **TLB Stalls**: TLB access stalls

## Key Design Principles

### 1. Parallelism
- **Bank Parallelism**: Multiple cache banks operate in parallel
- **Request Parallelism**: Multiple requests processed per cycle
- **TLB Parallelism**: TLB banks match cache bank structure

### 2. Arbitration Fairness
- **Round-Robin**: All arbiters use round-robin for fairness
- **Backpressure**: Proper flow control with ready/valid handshaking
- **Buffering**: Elastic buffers prevent stalls

### 3. Scalability
- **Configurable Parameters**: All key parameters are configurable
- **Hierarchical Design**: Supports multiple cache levels
- **Cluster Support**: Multiple cache units can be clustered

### 4. Integration
- **TLB Integration**: Seamless virtual-to-physical address translation
- **Memory System Integration**: Clean interface to memory hierarchy
- **Core Integration**: Standard memory bus interface

## Conclusion

The Vortex cache system represents a sophisticated, highly parallel cache architecture with:
- **Multi-banked structure** for high throughput
- **Integrated TLB** for virtual memory support
- **Comprehensive arbitration** for fair resource sharing
- **MSHR system** for efficient miss handling
- **Performance monitoring** for optimization
- **Scalable design** for various configurations

The system is designed to handle high-bandwidth, low-latency memory access patterns typical in GPU workloads while maintaining fairness and correctness across multiple concurrent memory requests.

