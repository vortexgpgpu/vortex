# I-Cache vs D-Cache Differences in Vortex

## Overview
The Vortex system implements separate Instruction Cache (I-cache) and Data Cache (D-cache) with distinct configurations optimized for their different access patterns and requirements.

## Key Architectural Differences

### 1. **Write Capability**
**I-Cache (Read-Only)**:
```systemverilog
.WRITE_ENABLE   (0),  // I-cache is read-only
```

**D-Cache (Read-Write)**:
```systemverilog
.WRITE_ENABLE   (1),  // D-cache supports writes
.WRITEBACK      (`DCACHE_WRITEBACK),  // Configurable writeback policy
.DIRTY_BYTES    (`DCACHE_DIRTYBYTES), // Dirty byte tracking
```

**Impact**: I-cache only needs to handle instruction fetches, while D-cache must handle both loads and stores with potential writeback to memory.

### 2. **Request Concurrency**
**I-Cache**:
```systemverilog
.NUM_REQS       (1),  // Single request per cycle
```

**D-cache**:
```systemverilog
.NUM_REQS       (DCACHE_NUM_REQS),  // Multiple concurrent requests
```

**Default Values**:
- I-cache: **1 request per cycle**
- D-cache: **Multiple requests per cycle** (typically 4-8 based on LSU configuration)

**Impact**: D-cache needs higher bandwidth to handle multiple concurrent memory operations from different warps/threads.

### 3. **Banking Structure**
**I-Cache**:
```systemverilog
.NUM_BANKS      (1),  // Single bank
```

**D-cache**:
```systemverilog
.NUM_BANKS      (`DCACHE_NUM_BANKS),  // Multiple banks (default: 4)
```

**Default Values**:
- I-cache: **1 bank** (simpler, lower latency)
- D-cache: **4 banks** (higher bandwidth, parallel access)

**Impact**: D-cache uses banking to support multiple concurrent requests, while I-cache uses a simpler single-bank design.

### 4. **Memory Ports**
**I-Cache**:
```systemverilog
.MEM_PORTS      (1),  // Single memory port
```

**D-cache**:
```systemverilog
.MEM_PORTS      (`L1_MEM_PORTS),  // Multiple memory ports (default: 4)
```

**Impact**: D-cache has more memory bandwidth to handle higher request rates.

### 5. **Non-Cacheable Support**
**I-Cache**:
```systemverilog
.NC_ENABLE      (0),  // No non-cacheable support
```

**D-cache**:
```systemverilog
.NC_ENABLE      (1),  // Supports non-cacheable addresses
```

**Impact**: D-cache can bypass cache for certain memory regions (e.g., memory-mapped I/O), while I-cache always caches instructions.

### 6. **Queue Sizes**
**I-Cache**:
```systemverilog
.CRSQ_SIZE      (`ICACHE_CRSQ_SIZE),  // Default: 2
.MSHR_SIZE      (`ICACHE_MSHR_SIZE),  // Default: 16
.MREQ_SIZE      (`ICACHE_MREQ_SIZE),  // Default: 4
.MRSQ_SIZE      (`ICACHE_MRSQ_SIZE),  // Default: 0
```

**D-cache**:
```systemverilog
.CRSQ_SIZE      (`DCACHE_CRSQ_SIZE),  // Default: 2
.MSHR_SIZE      (`DCACHE_MSHR_SIZE),  // Default: 16
.MREQ_SIZE      (`DCACHE_WRITEBACK ? `DCACHE_MSHR_SIZE : `DCACHE_MREQ_SIZE),
.MRSQ_SIZE      (`DCACHE_MRSQ_SIZE),  // Default: 4
```

**Key Difference**: I-cache has **MRSQ_SIZE = 0** (no memory response queue), while D-cache has **MRSQ_SIZE = 4**.

### 7. **Word Size**
**I-cache**:
```systemverilog
.WORD_SIZE      (ICACHE_WORD_SIZE),  // Default: 4 bytes (32-bit instructions)
```

**D-cache**:
```systemverilog
.WORD_SIZE      (DCACHE_WORD_SIZE),  // Default: 16 bytes (wider for data)
```

**Impact**: D-cache uses wider words to handle vectorized data operations efficiently.

### 8. **Flags Support**
**I-cache**:
```systemverilog
.FLAGS_WIDTH    (0),  // No flags needed
```

**D-cache**:
```systemverilog
.FLAGS_WIDTH    (`MEM_REQ_FLAGS_WIDTH),  // Memory request flags
```

**Impact**: D-cache supports memory request flags for different memory types and operations.

## Configuration Parameters

### I-Cache Default Configuration
```systemverilog
// From VX_config.vh
`define ICACHE_SIZE 16384        // 16KB
`define ICACHE_NUM_WAYS 4        // 4-way set associative
`define ICACHE_CRSQ_SIZE 2       // Core response queue
`define ICACHE_MSHR_SIZE 16      // Miss status holding register
`define ICACHE_MREQ_SIZE 4       // Memory request queue
`define ICACHE_MRSQ_SIZE 0       // No memory response queue
`define ICACHE_MEM_PORTS 1       // Single memory port
`define ICACHE_REPL_POLICY 1     // FIFO replacement
```

### D-Cache Default Configuration
```systemverilog
// From VX_config.vh
`define DCACHE_SIZE 16384        // 16KB
`define DCACHE_NUM_WAYS 4        // 4-way set associative
`define DCACHE_NUM_BANKS 4       // 4 banks (default)
`define DCACHE_CRSQ_SIZE 2       // Core response queue
`define DCACHE_MSHR_SIZE 16      // Miss status holding register
`define DCACHE_MREQ_SIZE 4       // Memory request queue
`define DCACHE_MRSQ_SIZE 4       // Memory response queue
`define DCACHE_WRITEBACK 1       // Writeback enabled
`define DCACHE_DIRTYBYTES 1      // Dirty byte tracking
`define DCACHE_REPL_POLICY 1     // FIFO replacement
```

## Access Pattern Differences

### I-Cache Access Patterns
- **Sequential Access**: Instructions typically accessed sequentially
- **Read-Only**: No write operations
- **Predictable**: Branch prediction helps with prefetching
- **Lower Bandwidth**: Single request per cycle sufficient
- **Lower Latency**: Simpler single-bank design

### D-Cache Access Patterns
- **Random Access**: Data accesses can be random and unpredictable
- **Read-Write**: Both loads and stores
- **High Bandwidth**: Multiple concurrent requests from different warps
- **Coalescing**: Memory coalescing for vectorized operations
- **Non-Cacheable**: Some memory regions bypass cache

## Performance Implications

### I-Cache Optimizations
1. **Single Bank**: Lower latency for sequential access
2. **No Write Logic**: Simpler pipeline, no writeback complexity
3. **No Non-Cacheable Logic**: Simpler control logic
4. **Smaller Queues**: Reduced buffering requirements

### D-Cache Optimizations
1. **Multiple Banks**: Higher bandwidth for concurrent access
2. **Write Support**: Full read-write capability with writeback
3. **Non-Cacheable Support**: Flexible memory access patterns
4. **Larger Queues**: Better handling of memory system stalls
5. **Dirty Byte Tracking**: Efficient writeback of modified data

## Memory System Integration

### I-Cache Memory Interface
```systemverilog
VX_mem_bus_if #(
    .DATA_SIZE (ICACHE_WORD_SIZE),    // 4 bytes
    .TAG_WIDTH (ICACHE_TAG_WIDTH)
) per_core_icache_bus_if[`SOCKET_SIZE]();
```

### D-Cache Memory Interface
```systemverilog
VX_mem_bus_if #(
    .DATA_SIZE (DCACHE_WORD_SIZE),    // 16 bytes
    .TAG_WIDTH (DCACHE_TAG_WIDTH)
) per_core_dcache_bus_if[`SOCKET_SIZE * DCACHE_NUM_REQS]();
```

## Summary

The key differences between I-cache and D-cache in Vortex are:

| Feature | I-Cache | D-Cache |
|---------|---------|---------|
| **Write Support** | Read-only (WRITE_ENABLE=0) | Read-write (WRITE_ENABLE=1) |
| **Concurrency** | 1 request/cycle | Multiple requests/cycle |
| **Banking** | 1 bank | 4 banks (default) |
| **Memory Ports** | 1 port | 4 ports (default) |
| **Word Size** | 4 bytes | 16 bytes |
| **Non-Cacheable** | No support | Supported |
| **Memory Response Queue** | 0 entries | 4 entries |
| **Flags Support** | None | Memory request flags |
| **Writeback** | N/A | Configurable |
| **Dirty Tracking** | N/A | Supported |

These differences reflect the distinct access patterns and requirements of instruction fetching versus data access in GPU workloads, with I-cache optimized for low-latency sequential access and D-cache optimized for high-bandwidth concurrent access.
