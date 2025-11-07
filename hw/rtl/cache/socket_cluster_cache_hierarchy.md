# Vortex Cache Hierarchy: Socket & Cluster Architecture

## Overview
This document provides a detailed analysis of the Vortex GPU cache hierarchy from the socket level through cluster level, with emphasis on cache organization and arbitration mechanisms.

---

## 1. Complete System Hierarchy

```
┌────────────────────────────────────────────────────────────────────────┐
│                         VX_cluster (Top Level)                          │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                        L2 Cache (Shared)                          │  │
│  │  VX_cache_wrap: 1MB, 16 banks, 8-way, writeback                 │  │
│  │  NUM_REQS = NUM_SOCKETS × L1_MEM_PORTS (e.g., 4 sockets × 4)    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                ▲                                         │
│                                │ L1_MEM_ARB_TAG_WIDTH                   │
│  ┌─────────────────────────────┴──────────────────────────────────┐   │
│  │  per_socket_mem_bus_if[NUM_SOCKETS * L1_MEM_PORTS]              │   │
│  │  (e.g., 4 sockets × 4 ports = 16 request channels to L2)       │   │
│  └─────────────────────────────┬──────────────────────────────────┘   │
│                                 │                                        │
│   ┌──────────────┬──────────────┼──────────────┬──────────────┐        │
│   │              │              │              │              │        │
│   ▼              ▼              ▼              ▼              │        │
│ Socket0        Socket1        Socket2        Socket3         ...       │
└───┬──────────────┬──────────────┬──────────────┬──────────────────────┘
    │              │              │              │
```

**Hierarchy levels:**
- **VX_cluster** → Contains N sockets + shared L2 cache
- **VX_socket** → Contains M cores + I-cache cluster + D-cache cluster
- **VX_cache_cluster** → Contains K cache units + arbiters
- **VX_cache** → Contains P banks + crossbars
- **VX_cache_bank** → Contains actual tag/data arrays

---

## 2. VX_cluster: Top-Level Organization

**File:** `hw/rtl/VX_cluster.sv`

### Key Parameters
```systemverilog
parameter NUM_SOCKETS = 4;           // Number of sockets in cluster
parameter L2_CACHE_SIZE = 1048576;   // 1MB L2 cache
parameter L2_NUM_BANKS = 16;         // 16-way banked
parameter L2_NUM_WAYS = 8;           // 8-way associative
parameter L1_MEM_PORTS = 4;          // Ports per socket to L2
```

### L2 Cache Configuration (lines 86-118)
```systemverilog
VX_cache_wrap #(
    .INSTANCE_ID    ("l2cache"),
    .CACHE_SIZE     (L2_CACHE_SIZE),      // 1MB
    .NUM_BANKS      (L2_NUM_BANKS),       // 16 banks
    .NUM_WAYS       (L2_NUM_WAYS),        // 8-way
    .NUM_REQS       (L2_NUM_REQS),        // NUM_SOCKETS × L1_MEM_PORTS
    .MEM_PORTS      (L2_MEM_PORTS),       // To main memory
    .WRITEBACK      (L2_WRITEBACK),       // Write policy
    .NC_ENABLE      (1),                  // Non-cacheable support
    .PASSTHRU       (!L2_ENABLED)         // Bypass if disabled
) l2cache (
    .core_bus_if    (per_socket_mem_bus_if[NUM_SOCKETS * L1_MEM_PORTS]),
    .mem_bus_if     (mem_bus_if[L2_MEM_PORTS])
);
```

### Socket Instantiation (lines 125-150)
```systemverilog
for (genvar socket_id = 0; socket_id < NUM_SOCKETS; ++socket_id) begin
    VX_socket #(
        .SOCKET_ID ((CLUSTER_ID * NUM_SOCKETS) + socket_id)
    ) socket (
        .mem_bus_if (per_socket_mem_bus_if[socket_id * L1_MEM_PORTS +: L1_MEM_PORTS])
    );
end
```

**Key insight:** L2 sees `NUM_SOCKETS × L1_MEM_PORTS` input channels (e.g., 4 sockets × 4 ports = 16 simultaneous requests possible).

---

## 3. VX_socket: Per-Socket Architecture

**File:** `hw/rtl/VX_socket.sv`

### Socket Contents
Each socket contains:
1. **SOCKET_SIZE cores** (e.g., 4 cores)
2. **1 I-cache cluster** (instruction cache, shared by all cores)
3. **1 D-cache cluster** (data cache, shared by all cores)
4. **L1 memory arbiters** (merge I-cache + D-cache traffic to L2)

### I-Cache Cluster (lines 89-121)
```systemverilog
VX_cache_cluster #(
    .INSTANCE_ID    ("icache"),
    .NUM_UNITS      (NUM_ICACHES),        // 1-2 cache units
    .NUM_INPUTS     (SOCKET_SIZE),        // 4 cores
    .NUM_REQS       (1),                  // 1 req/core (fetch)
    .NUM_BANKS      (1),                  // Single-banked
    .NUM_WAYS       (ICACHE_NUM_WAYS),    // e.g., 4-way
    .MEM_PORTS      (1),                  // 1 port to L2
    .WRITE_ENABLE   (0),                  // Read-only
    .CACHE_SIZE     (ICACHE_SIZE)         // e.g., 16KB
) icache (
    .core_bus_if    (per_core_icache_bus_if[SOCKET_SIZE]),
    .mem_bus_if     (icache_mem_bus_if[1])
);
```

### D-Cache Cluster (lines 137-171)
```systemverilog
VX_cache_cluster #(
    .INSTANCE_ID    ("dcache"),
    .NUM_UNITS      (NUM_DCACHES),        // 1-2 cache units
    .NUM_INPUTS     (SOCKET_SIZE),        // 4 cores
    .NUM_REQS       (DCACHE_NUM_REQS),    // 4 LSU lanes/core
    .NUM_BANKS      (DCACHE_NUM_BANKS),   // 4 banks
    .NUM_WAYS       (DCACHE_NUM_WAYS),    // 4-way
    .MEM_PORTS      (L1_MEM_PORTS),       // 4 ports to L2
    .WRITE_ENABLE   (1),                  // Read-write
    .WRITEBACK      (DCACHE_WRITEBACK),   // Write policy
    .CACHE_SIZE     (DCACHE_SIZE)         // e.g., 32KB
) dcache (
    .core_bus_if    (per_core_dcache_bus_if[SOCKET_SIZE * DCACHE_NUM_REQS]),
    .mem_bus_if     (dcache_mem_bus_if[L1_MEM_PORTS])
);
```

### 🔑 Critical: L1 Memory Port Arbitration (lines 175-216)

This is where I-cache and D-cache compete for access to L2.

```systemverilog
for (genvar i = 0; i < L1_MEM_PORTS; ++i) begin
    if (i == 0) begin  // Port 0: I-cache + D-cache arbiter
        
        VX_mem_arb #(
            .NUM_INPUTS (2),              // [0]=I-cache, [1]=D-cache
            .NUM_OUTPUTS(1),              // Merged to L2
            .ARBITER    ("P")             // 🔑 PRIORITY arbiter
        ) mem_arb (
            .bus_in_if[0]  (icache_mem_bus_if[0]),
            .bus_in_if[1]  (dcache_mem_bus_if[0]),
            .bus_out_if[0] (mem_bus_if[0])
        );
        
    end else begin  // Ports 1-3: D-cache only (pass-through)
        mem_bus_if[i] = dcache_mem_bus_if[i];
    end
end
```

**Visual diagram:**
```
Socket Memory Port Arbitration
─────────────────────────────────────────────────────
Port 0:  [I-cache] ──┐
                     ├─→ VX_mem_arb (Priority) → L2[0]
         [D-cache] ──┘   ⚠️ I-cache gets priority!

Port 1:  [D-cache] ────────────────────────────→ L2[1]

Port 2:  [D-cache] ────────────────────────────→ L2[2]

Port 3:  [D-cache] ────────────────────────────→ L2[3]
```

**Why priority arbitration?**
- Ensures instruction fetch never starves
- D-cache has 4 ports total (higher aggregate bandwidth)
- Critical for avoiding pipeline stalls

---

## 4. VX_cache_cluster: Multi-Unit Cache Organization

**File:** `hw/rtl/cache/VX_cache_cluster.sv`

### Purpose
Allows multiple cache units to share the load from multiple cores, reducing conflict misses and increasing effective capacity.

### Three-Stage Arbitration

```
┌─────────────────────────────────────────────────────────────────┐
│                      VX_cache_cluster                            │
│                                                                   │
│  Stage 1: Core Request Arbitration (per request channel)        │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  core_bus_if[0..15] (4 cores × 4 reqs = 16 inputs)    │    │
│  │               ↓                                         │    │
│  │  VX_mem_arb: 16 inputs → NUM_CACHES units (R)         │    │
│  │               ↓                                         │    │
│  │  arb_core_bus_if[0..7] (2 units × 4 reqs = 8)        │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                   │
│  Stage 2: Cache Units (parallel processing)                     │
│  ┌─────────────────┐         ┌─────────────────┐               │
│  │ VX_cache_wrap 0 │         │ VX_cache_wrap 1 │               │
│  │  CACHE_SIZE/2   │         │  CACHE_SIZE/2   │               │
│  │  NUM_BANKS      │         │  NUM_BANKS      │               │
│  │  MEM_PORTS      │         │  MEM_PORTS      │               │
│  └────────┬────────┘         └────────┬────────┘               │
│           │                            │                         │
│  Stage 3: Memory Port Arbitration (per port)                    │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  cache_mem_bus_if[0..7] (2 units × 4 ports = 8)       │    │
│  │               ↓                                         │    │
│  │  VX_mem_arb: NUM_CACHES × MEM_PORTS → MEM_PORTS (R)   │    │
│  │               ↓                                         │    │
│  │  mem_bus_if[0..3] (MEM_PORTS to L2/next level)        │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Stage 1: Core Request Arbiter (lines 116-150)
```systemverilog
for (genvar i = 0; i < NUM_REQS; ++i) begin : g_core_arb
    VX_mem_arb #(
        .NUM_INPUTS  (NUM_INPUTS),        // e.g., 4 cores
        .NUM_OUTPUTS (NUM_CACHES),        // e.g., 2 cache units
        .ARBITER     ("R"),               // Round-robin
        .TAG_SEL_IDX (TAG_SEL_IDX),       // For routing
        .REQ_OUT_BUF (2),                 // Buffering
        .RSP_OUT_BUF (CORE_OUT_BUF)
    ) core_arb (
        .bus_in_if  (core_bus_tmp_if[NUM_INPUTS]),
        .bus_out_if (arb_core_bus_tmp_if[NUM_CACHES])
    );
end
```

**What happens:**
- Each of NUM_REQS request channels gets its own arbiter
- Example: 4 cores × 4 LSU lanes = 16 input channels
- Each arbiter routes to NUM_CACHES units (e.g., 2 units)
- Result: 2 cache units × 4 reqs/unit = 8 request channels

### Stage 2: Cache Unit Instantiation (lines 177-216)
```systemverilog
for (genvar i = 0; i < NUM_CACHES; ++i) begin : g_cache_wrap
    VX_cache_wrap #(
        .CACHE_SIZE   (CACHE_SIZE),       // Full size (divided internally)
        .NUM_BANKS    (NUM_BANKS),        // e.g., 4 banks
        .NUM_WAYS     (NUM_WAYS),         // e.g., 4-way
        .NUM_REQS     (NUM_REQS),         // e.g., 4 req channels
        .MEM_PORTS    (MEM_PORTS),        // e.g., 4 ports
        .WRITEBACK    (WRITEBACK),
        .NC_ENABLE    (NC_ENABLE)
    ) cache_wrap (
        .core_bus_if (arb_core_bus_if[i * NUM_REQS +: NUM_REQS]),
        .mem_bus_if  (cache_mem_bus_if[i * MEM_PORTS +: MEM_PORTS])
    );
end
```

**Each cache unit:**
- Processes NUM_REQS request channels in parallel
- Has NUM_BANKS internal banks (address-based routing)
- Outputs MEM_PORTS memory request channels

### Stage 3: Memory Port Arbiter (lines 218-254)
```systemverilog
for (genvar i = 0; i < MEM_PORTS; ++i) begin : g_mem_bus_if
    VX_mem_arb #(
        .NUM_INPUTS  (NUM_CACHES),        // e.g., 2 cache units
        .NUM_OUTPUTS (1),                 // Merge to 1 port
        .ARBITER     ("R"),               // Round-robin
        .TAG_SEL_IDX (TAG_SEL_IDX),
        .REQ_OUT_BUF (MEM_OUT_BUF),
        .RSP_OUT_BUF (2)
    ) mem_arb (
        .bus_in_if  (arb_core_bus_tmp_if[NUM_CACHES]),
        .bus_out_if (mem_bus_tmp_if[1])
    );
end
```

**What happens:**
- Each memory port gets its own arbiter
- Example: MEM_PORTS=4, so 4 parallel arbiters
- Each arbiter selects between NUM_CACHES cache units
- Result: 4 merged memory ports to next level (socket/L2)

---

## 5. Complete Request Flow Example

### Scenario: Core 0, LSU Lane 2 issues D-cache read

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Core Level                                                    │
│    Core 0, LSU Lane 2 → dcache_bus_if[0*4 + 2]                  │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. VX_cache_cluster (D-cache)                                    │
│    Core arbiter (req channel 2):                                │
│      - Inputs: 4 cores → core_bus_if[0*4+2, 1*4+2, 2*4+2, ...]  │
│      - Round-robin selects cache unit (e.g., unit 0)            │
│    → arb_core_bus_if[0*4 + 2]                                   │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. VX_cache_wrap[0]                                              │
│    Bypass check (NC_ENABLE):                                    │
│      - Non-cacheable? → bypass path                             │
│      - Cacheable → VX_cache                                     │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. VX_cache                                                      │
│    Request crossbar (address-based):                            │
│      - Extract bank_id from addr[WORD_SEL +: BANK_SEL]          │
│      - Example: addr[5:4] = 2 → route to Bank 2                 │
│    → per_bank_core_req_valid[2]                                 │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. VX_cache_bank[2]                                              │
│    2-stage pipeline:                                             │
│      Stage 0: Tag lookup                                        │
│        - VX_cache_tags: compare tag, check valid               │
│        - VX_cache_repl: select victim way                      │
│      Stage 1: Data access                                       │
│        - Hit? VX_cache_data reads, pushes to core_rsp_queue    │
│        - Miss? Allocate MSHR, push fill req to mem_req_queue   │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6a. Hit Path (response)                                          │
│     VX_cache_bank[2].core_rsp_queue                              │
│       → VX_cache response xbar (uses core_rsp_idx)              │
│       → VX_cache_wrap[0].core_bus_if[2].rsp                     │
│       → VX_cache_cluster core arbiter (tag-based routing)       │
│       → Core 0, LSU Lane 2 receives data                        │
└─────────────────────────────────────────────────────────────────┘
                             
┌─────────────────────────────────────────────────────────────────┐
│ 6b. Miss Path (memory request)                                   │
│     VX_cache_bank[2].mem_req_queue                               │
│       → VX_cache mem arbiter (4 banks → MEM_PORTS)              │
│         - Round-robin, bank 2 wins slot on port_i               │
│       → VX_cache_cluster mem arbiter (NUM_CACHES → MEM_PORTS)   │
│         - Round-robin, cache unit 0 wins port_i                 │
│       → Socket L1 mem arbiter (I-cache + D-cache)               │
│         - Priority arbiter, D-cache on port_i                   │
│       → VX_cluster L2 cache                                     │
│         - Address-based routing to L2 banks                     │
│                                                                   │
│     Memory response flows back via tags:                        │
│       L2 → Socket → Cluster → Cache → Bank                      │
│       → MSHR dequeue → replay request → hit → response          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Arbiter Summary Table

| Level | Arbiter Location | Inputs | Outputs | Policy | Purpose |
|-------|------------------|--------|---------|--------|---------|
| **Cache Bank** | mem_req_arb | NUM_BANKS | MEM_PORTS | Round-robin | Banks compete for mem ports |
| **Cache** | req_xbar | NUM_REQS | NUM_BANKS | Address | Route requests to banks |
| **Cache** | rsp_xbar | NUM_BANKS | NUM_REQS | Tag-based | Route responses to requesters |
| **Cache** | mem_rsp_omega | MEM_PORTS | NUM_BANKS | Tag-based | Route mem responses to banks |
| **Cache Cluster** | core_arb (×NUM_REQS) | NUM_INPUTS | NUM_CACHES | Round-robin | Cores compete for cache units |
| **Cache Cluster** | mem_arb (×MEM_PORTS) | NUM_CACHES | 1 | Round-robin | Cache units compete for mem ports |
| **Socket** | L1 mem_arb[0] | 2 (I+D) | 1 | **Priority** | I-cache prioritized over D-cache |
| **Socket** | L1 mem_arb[1-3] | 1 (D) | 1 | Pass-through | D-cache only |
| **Cluster** | L2 internal | (sockets×ports) | NUM_BANKS | Address | Inside L2 cache crossbar |

---

## 7. Typical Configuration Example

### Small GPU Configuration
```
VX_cluster:
  - NUM_SOCKETS = 2
  - L2: 512KB, 8 banks, 4-way
  - L1_MEM_PORTS = 2

VX_socket (×2):
  - SOCKET_SIZE = 2 cores
  - I-cache: 16KB, 1 bank, 4-way, 1 port
  - D-cache: 16KB, 4 banks, 4-way, 2 ports

Total L2 inputs: 2 sockets × 2 ports = 4 channels
Total L1 capacity: 2 sockets × (16KB I + 16KB D) = 64KB
Total L2 capacity: 512KB
```

### Large GPU Configuration
```
VX_cluster:
  - NUM_SOCKETS = 4
  - L2: 2MB, 16 banks, 8-way
  - L1_MEM_PORTS = 4

VX_socket (×4):
  - SOCKET_SIZE = 4 cores
  - I-cache cluster: 2 units × 16KB = 32KB, 1 bank/unit, 1 port
  - D-cache cluster: 2 units × 32KB = 64KB, 4 banks/unit, 4 ports

Total L2 inputs: 4 sockets × 4 ports = 16 channels
Total L1 capacity: 4 sockets × (32KB I + 64KB D) = 384KB
Total L2 capacity: 2MB
Peak bandwidth: 16 parallel accesses to L2 per cycle
```

---

## 8. Key Design Principles

### Bandwidth Scaling
1. **Horizontal scaling**: More sockets/cores = more L2 request channels
2. **Vertical scaling**: More cache units in cluster = higher capacity, lower conflicts
3. **Bank-level parallelism**: Multiple banks allow concurrent accesses to different addresses

### Arbitration Philosophy
1. **Round-robin for fairness**: Most arbiters use "R" to prevent starvation
2. **Priority where critical**: I-cache gets priority (port 0) for performance
3. **Address-based for determinism**: Bank selection has no arbitration delay
4. **Tag-based for routing**: Responses find their way back via embedded tags

### Buffering Strategy
1. **Crossbar outputs**: 2-3 cycles of buffering for timing closure
2. **Arbiter outputs**: Configurable based on hierarchy level
3. **Response paths**: Deep buffering to prevent backpressure

### Scalability Considerations
1. **L2 input channels scale linearly**: O(NUM_SOCKETS × L1_MEM_PORTS)
2. **Cache cluster allows shared capacity**: Trade-off between area and miss rate
3. **Bank count limits conflicts**: More banks = less collision, but more area

---

## 9. Critical Code References

### VX_cluster.sv
- **Lines 79-118**: L2 cache instantiation and configuration
- **Lines 125-150**: Socket array generation

### VX_socket.sv
- **Lines 89-121**: I-cache cluster instantiation
- **Lines 137-171**: D-cache cluster instantiation
- **Lines 175-216**: L1 memory port arbitration (I+D merge)

### VX_cache_cluster.sv
- **Lines 116-150**: Core request arbiters (per-channel)
- **Lines 177-216**: Cache unit instantiation
- **Lines 218-254**: Memory port arbiters (per-port)

### VX_cache.sv
- **Lines 330-353**: Request crossbar (cores → banks)
- **Lines 543-560**: Response crossbar (banks → cores)
- **Lines 207-225**: Memory response router (ports → banks)
- **Lines 613-641**: Memory request arbiter (banks → ports)

---

## 10. Future Considerations

### When Modifying the Hierarchy
1. **Tag width calculations**: Must account for arbiter selection bits
2. **Buffering**: May need adjustment for longer paths
3. **Performance counters**: Add monitoring for new arbitration points
4. **Deadlock avoidance**: Ensure no circular dependencies in request/response paths

### Potential Enhancements
1. **Adaptive arbitration**: Switch between round-robin and priority based on load
2. **Quality-of-Service**: Per-core priority levels
3. **Prefetching integration**: Add prefetch channels through hierarchy
4. **Power gating**: Per-unit or per-bank sleep modes

---

## Document Revision History
- **v1.0** (2025-01-XX): Initial comprehensive analysis of socket/cluster cache hierarchy
- Focus: Arbitration mechanisms, request flow, typical configurations

---

## Related Documents
- `cache_architecture.md`: Cache cluster internal structure
- `VX_cache_bank_analysis.md`: Bank-level pipeline and MSHR details
- `TLB_integration.md`: Virtual memory and TLB path integration

