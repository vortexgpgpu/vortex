# TLB Wrapper Functional Requirements

## Current Status

### VX_tlb_wrapper.sv (Current - Pass-through)
✅ **What it has:**
- Single-entry elastic buffer
- Proper ready/valid handshaking
- Pass-through for all signals
- Infrastructure for future TLB integration

❌ **What it's missing:**
- No TLB storage (tag/data arrays)
- No address translation
- No hit/miss detection
- No replacement policy
- No TLB update mechanism

## Your Requirements

Based on your specifications:
1. ✅ **TLB Integration**: Insert TLB between arbiter and cache (DONE in VX_cache_cluster.sv)
2. ⚠️  **Address Translation**: TLB lookup and VA→PA translation (NEEDED)
3. ⚠️  **TLB Miss Handling**: Treat misses as cache misses, forward to memory (NEEDED)
4. ❌ **PTW**: Not needed for now (ignored as per your request)
5. ⚠️  **Priority**: TLB requests get priority in arbiter (NEEDS ARBITER MODIFICATION)

## What Needs to Be Added to VX_tlb_wrapper.sv

### Option 1: Enhance Current VX_tlb_wrapper.sv (Recommended)

Replace the current pass-through implementation with actual TLB functionality:

#### 1. **TLB Storage Arrays**
```systemverilog
// Tag array: stores VPN (Virtual Page Numbers)
reg [TLB_WAYS-1:0][VPN_WIDTH-1:0] tlb_tag_array [TLB_SETS];

// Data array: stores PPN (Physical Page Numbers)
reg [TLB_WAYS-1:0][PPN_WIDTH-1:0] tlb_data_array [TLB_SETS];

// Valid bits per way
reg [TLB_WAYS-1:0] tlb_valid_array [TLB_SETS];

// Replacement state (FIFO counter or LRU bits)
reg [TLB_WAY_WIDTH-1:0] tlb_repl_state [TLB_SETS];
```

**Parameters needed:**
- `TLB_ENTRIES` = 32 (total entries)
- `TLB_WAYS` = 4 (4-way set-associative)
- `TLB_SETS` = TLB_ENTRIES / TLB_WAYS = 8 sets
- `PAGE_OFFSET_BITS` = 12 (4KB pages)

#### 2. **Address Breakdown**
```
Virtual Address (ADDR_WIDTH bits):
[ADDR_WIDTH-1 : 12]  - VPN (Virtual Page Number)
[11 : 0]             - Page Offset (4KB = 2^12)

TLB Index: VPN[LOG2(TLB_SETS)-1 : 0]
TLB Tag:   VPN[VPN_WIDTH-1 : LOG2(TLB_SETS)]
```

#### 3. **TLB Lookup Logic (Combinational)**
```systemverilog
// Extract VPN from input address
wire [VPN_WIDTH-1:0] in_vpn = in_addr[ADDR_WIDTH-1:PAGE_OFFSET_BITS];
wire [PAGE_OFFSET_BITS-1:0] in_offset = in_addr[PAGE_OFFSET_BITS-1:0];

// Index into TLB
wire [TLB_INDEX_BITS-1:0] tlb_index = in_vpn[TLB_INDEX_BITS-1:0];

// Tag matching
wire [TLB_WAYS-1:0] way_matches;
for (genvar i = 0; i < TLB_WAYS; i++) begin
    assign way_matches[i] = tlb_valid_array[tlb_index][i] && 
                           (tlb_tag_array[tlb_index][i] == in_vpn);
end

// Hit/Miss detection
wire tlb_hit = |way_matches;
wire tlb_miss = in_valid && ~tlb_hit;
```

#### 4. **Address Translation**
```systemverilog
// Find which way hit
wire [TLB_WAY_BITS-1:0] hit_way;
VX_find_first #(.N(TLB_WAYS)) hit_selector (
    .valid_in(way_matches),
    .index_out(hit_way),
    ...
);

// Get PPN from hitting way
wire [PPN_WIDTH-1:0] hit_ppn = tlb_data_array[tlb_index][hit_way];

// Translated address: PPN + offset
wire [ADDR_WIDTH-1:0] translated_addr = tlb_hit ? {hit_ppn, in_offset} : in_addr;
```

#### 5. **Output with Translated Address**
```systemverilog
always @(posedge clk) begin
    if (accept_in) begin
        r_addr <= translated_addr;  // Use translated address!
        // ... rest of the fields
    end
end
```

### Option 2: Use Existing VX_tlb_bank.sv

The codebase already has `VX_tlb_bank.sv` with full TLB functionality. We could instantiate it inside VX_tlb_wrapper:

```systemverilog
module VX_tlb_wrapper (
    // ... existing ports ...
);

    // Instantiate actual TLB bank
    VX_tlb_bank #(
        .INSTANCE_ID   (INSTANCE_ID),
        .BANK_ID       (BANK_ID),
        .TLB_ENTRIES   (32),
        .TLB_WAYS      (4),
        .NUM_BANKS     (1),
        .WORD_SIZE     (BYTEEN_WIDTH),
        .UUID_WIDTH    (0),
        .TAG_WIDTH     (TAG_WIDTH)
    ) tlb_bank (
        .clk            (clk),
        .reset          (reset),
        
        // Core request
        .core_req_valid (in_valid),
        .core_req_addr  (in_addr),
        .core_req_rw    (in_rw),
        .core_req_tag   (in_tag),
        .core_req_ready (in_ready),
        
        // Core response (translated)
        .core_rsp_valid (out_valid),
        .core_rsp_addr  (out_addr),  // Translated!
        .core_rsp_rw    (out_rw),
        .core_rsp_tag   (out_tag),
        .core_rsp_ready (out_ready),
        
        // TLB miss (tie off for now)
        .tlb_miss_valid (),
        .tlb_miss_vpn   (),
        .tlb_miss_ready (1'b0),
        
        // TLB update (tie off for now)
        .tlb_update_valid (1'b0),
        .tlb_update_vpn   ('0),
        .tlb_update_ppn   ('0),
        .tlb_update_ready ()
    );
    
    // Pass through other fields not handled by TLB
    // ... wsel, byteen, data, idx, flags ...

endmodule
```

**Issue with Option 2:**
- VX_tlb_bank interface doesn't match VX_mem_bus_if structure
- Would need adaptation logic to convert between interfaces

## TLB Initialization Challenge

### Problem: How to Populate the TLB?

Without PTW, the TLB needs to be populated somehow. Options:

#### Option A: Identity Mapping (VA == PA)
```systemverilog
// On reset or initialization, populate TLB with identity mapping
initial begin
    for (int i = 0; i < TLB_SETS; i++) begin
        tlb_valid_array[i][0] = 1'b1;  // First way valid
        tlb_tag_array[i][0] = i;       // VPN = index
        tlb_data_array[i][0] = i;      // PPN = VPN (identity)
    end
end
```

**Pros:** Simple, works for initial testing
**Cons:** Not realistic, only works if VA==PA

#### Option B: Software-Driven TLB Fill
```systemverilog
// Add a software interface to populate TLB entries
input wire                          sw_tlb_write,
input wire [TLB_INDEX_BITS-1:0]     sw_tlb_index,
input wire [TLB_WAY_BITS-1:0]       sw_tlb_way,
input wire [VPN_WIDTH-1:0]          sw_tlb_vpn,
input wire [PPN_WIDTH-1:0]          sw_tlb_ppn,

always @(posedge clk) begin
    if (sw_tlb_write) begin
        tlb_valid_array[sw_tlb_index][sw_tlb_way] <= 1'b1;
        tlb_tag_array[sw_tlb_index][sw_tlb_way] <= sw_tlb_vpn;
        tlb_data_array[sw_tlb_index][sw_tlb_way] <= sw_tlb_ppn;
    end
end
```

**Pros:** Flexible, can set up any VA→PA mapping
**Cons:** Requires software support, more complex

#### Option C: Miss = Identity Translation
```systemverilog
// On TLB miss, treat as identity mapping and optionally install entry
wire [ADDR_WIDTH-1:0] translated_addr = tlb_hit ? {hit_ppn, in_offset} 
                                                : in_addr;  // Identity on miss

// Optional: Install identity mapping on miss
always @(posedge clk) begin
    if (accept_in && tlb_miss) begin
        wire [TLB_WAY_BITS-1:0] repl_way = tlb_repl_state[tlb_index];
        tlb_valid_array[tlb_index][repl_way] <= 1'b1;
        tlb_tag_array[tlb_index][repl_way] <= in_vpn;
        tlb_data_array[tlb_index][repl_way] <= in_vpn;  // PPN = VPN
        tlb_repl_state[tlb_index] <= repl_way + 1;
    end
end
```

**Pros:** Self-populating, works without external intervention
**Cons:** Assumes identity mapping is correct

## TLB Miss Handling (Your Requirement)

You specified: "All misses are treated as memory request to next cache level"

### Current Approach in VX_tlb_wrapper_functional.sv:
```systemverilog
// On miss, forward original address (no translation)
wire [ADDR_WIDTH-1:0] translated_addr = tlb_hit ? {hit_ppn, in_offset} : in_addr;

// Send to cache
assign out_addr = r_addr;  // Will be original addr on miss
```

**This means:**
- TLB Hit → Translated address goes to cache
- TLB Miss → Original (virtual) address goes to cache → Cache miss → Goes to next level

**Problem:** The cache will see a virtual address on TLB miss, which may not match anything!

### Better Approach: TLB Miss → Bypass Cache
```systemverilog
// Add a "bypass cache" flag on TLB miss
output wire out_bypass,  // Signal to cache to bypass

assign out_bypass = r_was_miss;  // Bypass on TLB miss
```

Then in VX_cache_cluster, route TLB misses directly to memory arbiter, bypassing cache.

**But this requires modifying the cache cluster architecture!**

## Priority in Arbiter (Your Requirement)

You said: "TLB requests getting priority in arbiter"

### Current Arbiter in VX_cache_cluster:
```systemverilog
VX_mem_arb #(
    .NUM_INPUTS  (NUM_CACHES),
    .NUM_OUTPUTS (1),
    .ARBITER     ("R"),  // Round-robin
    ...
) mem_arb (...)
```

### To Give TLB Priority:
Need to change arbiter to support priority. Options:

#### Option 1: Change ARBITER parameter
```systemverilog
.ARBITER ("P"),  // Priority arbiter
```

**But:** TLB requests come from within cache, not as separate inputs!

#### Option 2: Add TLB miss ports to mem_arb
```systemverilog
VX_mem_arb #(
    .NUM_INPUTS (NUM_CACHES * 2),  // Regular + TLB
    ...
) mem_arb (
    // First NUM_CACHES inputs: TLB misses (high priority)
    // Next NUM_CACHES inputs: Cache misses (low priority)
    ...
);
```

**This requires significant architecture changes!**

## Recommended Implementation Path

### Phase 1: Basic TLB with Identity Mapping (Simplest)
1. ✅ Use `VX_tlb_wrapper_functional.sv` (already created)
2. ✅ Initialize TLB with identity mapping (VA == PA)
3. ✅ TLB hit → translate, TLB miss → forward original address
4. ✅ No arbiter priority changes needed

**Testing:**
- All addresses initially miss → go to cache as VA
- After warmup, addresses hit → go to cache as PA
- If VA == PA in your system, this works correctly!

### Phase 2: Software-Populated TLB
1. Add software write interface to populate TLB entries
2. OS/bootloader sets up page tables in TLB
3. TLB hit → translate, TLB miss → fault/trap
4. More realistic behavior

### Phase 3: Full PTW Integration (Future)
1. Connect `mem_req_*` ports to memory arbiter
2. Implement page table walk state machine
3. Automatically populate TLB on misses
4. Add arbiter priority for PTW requests

## Files Created

### 1. VX_tlb_wrapper_functional.sv (New)
A fully functional TLB wrapper with:
- ✅ TLB storage arrays (tag, data, valid)
- ✅ Hit/miss detection
- ✅ Address translation (VA → PA)
- ✅ FIFO replacement policy
- ✅ Identity mapping initialization
- ✅ Debug tracing
- ✅ Performance counters

### 2. TLB_FUNCTIONAL_REQUIREMENTS.md (This file)
Complete documentation of what's needed for functional TLB.

## Decision Points

### Question 1: Which TLB implementation to use?

**A) Use VX_tlb_wrapper_functional.sv (New file)**
- ✅ Purpose-built for cache cluster integration
- ✅ Matches VX_mem_bus_if interface
- ✅ Simpler, more focused
- ❌ Needs initialization strategy

**B) Adapt VX_tlb_bank.sv (Existing)**
- ✅ Already tested TLB logic
- ✅ More complete feature set
- ❌ Interface mismatch with VX_mem_bus_if
- ❌ Needs wrapper/adaptation logic

**C) Keep pass-through VX_tlb_wrapper.sv (Current)**
- ✅ No changes needed
- ✅ Infrastructure in place
- ❌ No TLB functionality
- ❌ Doesn't meet your requirements

### Question 2: How to initialize TLB?

**A) Identity mapping (VA == PA)**
- Use if your system doesn't use virtual memory translation
- Simplest for testing

**B) Software write interface**
- Use if you want OS to control TLB
- Most flexible

**C) PTW (Page Table Walk)**
- Use for full virtual memory support
- Most complex, but you said skip for now

### Question 3: How to handle TLB miss priority?

**A) No special handling (Current)**
- TLB miss = cache miss
- Goes to next level like any miss
- Simplest, no arbiter changes

**B) Separate TLB miss path**
- Add dedicated ports for TLB misses
- Modify memory arbiter for priority
- More complex, needs architecture changes

## My Recommendation

**Start with VX_tlb_wrapper_functional.sv** with these settings:

1. **Use the functional TLB wrapper** (replaces current pass-through)
2. **Identity mapping initialization** (VA == PA on reset)
3. **TLB miss = forward original address** (treat as cache miss)
4. **No arbiter priority changes** (keep simple for now)

This gives you:
- ✅ Actual TLB functionality
- ✅ Address translation on hits
- ✅ Minimal architecture changes
- ✅ Easy to test and verify
- ✅ Foundation for future PTW

Then we can add:
- Software TLB write interface (if needed)
- PTW integration (when ready)
- Arbiter priority (if performance testing shows need)

## Next Steps

**What do you want to do?**

1. Replace `VX_tlb_wrapper.sv` with `VX_tlb_wrapper_functional.sv`?
2. Keep current pass-through and add TLB logic incrementally?
3. Use existing `VX_tlb_bank.sv` with adaptation layer?
4. Something else?

**Let me know your preference and I'll proceed with the implementation!**

