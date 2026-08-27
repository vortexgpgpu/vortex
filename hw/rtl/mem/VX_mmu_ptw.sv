// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "VX_define.vh"

// Shared page-table walker: one instance per device, fed by every core TLB
// through the VX_ptw_bus_if hierarchy. Up to NUM_WALKERS walks proceed
// concurrently, each identified by the bus tag of the requesting TLB bank.
// Page-table entries are fetched on a dedicated L3 port. Non-leaf entries of
// the upper levels are cached in page-walk caches so that a walk can start
// one (Sv32) or two (Sv39) levels below the root.
module VX_mmu_ptw import VX_gpu_pkg::*; #(
    parameter NUM_WALKERS   = `VX_CFG_PTW_NUM_WALKERS,
    parameter PWC_SIZE      = `VX_CFG_PTW_WALK_CACHE_SIZE,
    parameter TAG_WIDTH     = PTW_DEV_TAG_WIDTH,
    parameter MEM_DATA_SIZE = L3_WORD_SIZE,
    parameter MEM_TAG_WIDTH = L3_TAG_WIDTH
) (
    input wire clk,
    input wire reset,
    input wire flush,

`ifdef PERF_ENABLE
    output ptw_perf_t       ptw_perf,
`endif

    VX_ptw_bus_if.slave     ptw_bus_if,
    VX_mem_bus_if.master    mem_bus_if
);
    `STATIC_ASSERT(`IS_POW2(NUM_WALKERS), ("NUM_WALKERS must be a power of 2"))
    `STATIC_ASSERT((VM_PT_LEVELS == 2) || (VM_PT_LEVELS == 3), ("only Sv32 and Sv39 page tables are supported"))
    `STATIC_ASSERT(MEM_DATA_SIZE >= VM_PTE_SIZE, ("memory word must hold a PTE"))

    localparam SLOT_BITS      = `LOG2UP(NUM_WALKERS);
    localparam MEM_ADDR_WIDTH = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(MEM_DATA_SIZE);
    localparam PTE_SHIFT      = `CLOG2(VM_PTE_SIZE);
    localparam PTE_BITS       = VM_PTE_SIZE * 8;
    localparam PTES_PER_WORD  = MEM_DATA_SIZE / VM_PTE_SIZE;
    localparam PTE_SEL_BITS   = `LOG2UP(PTES_PER_WORD);
    localparam PWC_KEY_WIDTH  = VM_PPN_WIDTH + VM_VPN_LEVEL_BITS;
    localparam TOP_LEVEL      = VM_PT_LEVELS - 1;

    `STATIC_ASSERT(SLOT_BITS <= (MEM_TAG_WIDTH - UUID_WIDTH), ("walker id does not fit the L3 tag"))

    typedef enum logic [1:0] {
        SLOT_IDLE     = 2'd0,
        SLOT_MEM_REQ  = 2'd1,
        SLOT_MEM_RSP  = 2'd2,
        SLOT_DONE     = 2'd3
    } slot_state_t;

    typedef struct packed {
        logic [VM_VPN_WIDTH-1:0]       vpn;
        logic [VM_PPN_WIDTH-1:0]       root_ppn;
        logic [VM_PPN_WIDTH-1:0]       cur_ppn;    // table being walked, then the leaf PPN
        logic [VM_LEVEL_BITS-1:0]      level;
        logic [VM_PTE_FLAGS_WIDTH-1:0] flags;
        logic                          fault;
        logic [`UP(PTE_SEL_BITS)-1:0]  pte_sel;    // PTE index within the fetched word
        logic [TAG_WIDTH-1:0]          tag;
        logic                          stale;      // started before the last flush: no PWC fills
    } slot_t;

    slot_state_t slot_state [NUM_WALKERS];
    slot_t       slots      [NUM_WALKERS];

    wire [NUM_WALKERS-1:0] slot_idle, slot_mem_req, slot_done;
    for (genvar s = 0; s < NUM_WALKERS; ++s) begin : g_slot_flags
        assign slot_idle[s]    = (slot_state[s] == SLOT_IDLE);
        assign slot_mem_req[s] = (slot_state[s] == SLOT_MEM_REQ);
        assign slot_done[s]    = (slot_state[s] == SLOT_DONE);
    end

    function automatic logic [VM_VPN_LEVEL_BITS-1:0] vpn_slice(
        input logic [VM_VPN_WIDTH-1:0]  vpn,
        input logic [VM_LEVEL_BITS-1:0] level
    );
        return vpn[level * VM_VPN_LEVEL_BITS +: VM_VPN_LEVEL_BITS];
    endfunction

    // A superpage leaf must have its low PPN bits clear (the page offset
    // covers them); anything else is a misaligned superpage and faults.
    function automatic logic superpage_misaligned(
        input logic [VM_PPN_WIDTH-1:0]  ppn,
        input logic [VM_LEVEL_BITS-1:0] level
    );
        logic [VM_PPN_WIDTH-1:0] mask;
        mask = (VM_PPN_WIDTH'(1) << (level * VM_VPN_LEVEL_BITS)) - VM_PPN_WIDTH'(1);
        return |(ppn & mask);
    endfunction

    // -------------------------------------------------------------------------
    // Walk cache lookup on the incoming request
    // -------------------------------------------------------------------------

    wire [VM_VPN_WIDTH-1:0] req_vpn      = ptw_bus_if.req_data.vpn;
    wire [VM_PPN_WIDTH-1:0] req_root_ppn = ptw_bus_if.req_data.root_ppn;

    wire                    pwc1_hit;
    wire [VM_PPN_WIDTH-1:0] pwc1_data;
    wire                    pwc1_fill_valid;
    wire [PWC_KEY_WIDTH-1:0] pwc1_fill_key;
    wire [VM_PPN_WIDTH-1:0] pwc1_fill_data;

    VX_mmu_pwc #(
        .KEY_WIDTH   (PWC_KEY_WIDTH),
        .DATA_WIDTH  (VM_PPN_WIDTH),
        .NUM_ENTRIES (PWC_SIZE)
    ) pwc1 (
        .clk         (clk),
        .reset       (reset),
        .flush       (flush),
        .lookup_key  ({req_root_ppn, vpn_slice(req_vpn, VM_LEVEL_BITS'(TOP_LEVEL))}),
        .lookup_hit  (pwc1_hit),
        .lookup_data (pwc1_data),
        .fill_valid  (pwc1_fill_valid),
        .fill_key    (pwc1_fill_key),
        .fill_data   (pwc1_fill_data)
    );

    wire                    pwc2_hit;
    wire [VM_PPN_WIDTH-1:0] pwc2_data;
    wire                    pwc2_fill_valid;
    wire [PWC_KEY_WIDTH-1:0] pwc2_fill_key;
    wire [VM_PPN_WIDTH-1:0] pwc2_fill_data;

    if (VM_PT_LEVELS == 3) begin : g_pwc2
        // Level-1 tables are keyed by the level-2 table the first cache
        // returned, so a double hit skips both upper fetches.
        VX_mmu_pwc #(
            .KEY_WIDTH   (PWC_KEY_WIDTH),
            .DATA_WIDTH  (VM_PPN_WIDTH),
            .NUM_ENTRIES (PWC_SIZE)
        ) pwc2 (
            .clk         (clk),
            .reset       (reset),
            .flush       (flush),
            .lookup_key  ({pwc1_data, vpn_slice(req_vpn, VM_LEVEL_BITS'(1))}),
            .lookup_hit  (pwc2_hit),
            .lookup_data (pwc2_data),
            .fill_valid  (pwc2_fill_valid),
            .fill_key    (pwc2_fill_key),
            .fill_data   (pwc2_fill_data)
        );
    end else begin : g_no_pwc2
        `UNUSED_VAR (pwc2_fill_valid)
        `UNUSED_VAR (pwc2_fill_key)
        `UNUSED_VAR (pwc2_fill_data)
        assign pwc2_hit  = 1'b0;
        assign pwc2_data = '0;
    end

    wire start_skip2 = pwc1_hit && pwc2_hit;
    wire start_skip1 = pwc1_hit && !pwc2_hit;

    wire [VM_LEVEL_BITS-1:0] start_level = start_skip2 ? VM_LEVEL_BITS'(TOP_LEVEL - 2) :
                                           start_skip1 ? VM_LEVEL_BITS'(TOP_LEVEL - 1) :
                                                         VM_LEVEL_BITS'(TOP_LEVEL);
    wire [VM_PPN_WIDTH-1:0] start_ppn = start_skip2 ? pwc2_data :
                                        start_skip1 ? pwc1_data :
                                                      req_root_ppn;

    // -------------------------------------------------------------------------
    // Slot allocation
    // -------------------------------------------------------------------------

    wire [SLOT_BITS-1:0] free_slot;
    wire                 free_valid;

    VX_priority_encoder #(
        .N (NUM_WALKERS)
    ) free_slot_enc (
        .data_in   (slot_idle),
        .index_out (free_slot),
        .valid_out (free_valid),
        `UNUSED_PIN (onehot_out)
    );

    assign ptw_bus_if.req_ready = free_valid;
    wire req_fire = ptw_bus_if.req_valid && ptw_bus_if.req_ready;

    // -------------------------------------------------------------------------
    // Memory requests: one outstanding fetch per slot, round-robin issue
    // -------------------------------------------------------------------------

    wire [SLOT_BITS-1:0] mem_slot;
    wire                 mem_slot_valid;

    VX_rr_arbiter #(
        .NUM_REQS (NUM_WALKERS)
    ) mem_arb (
        .clk          (clk),
        .reset        (reset),
        .requests     (slot_mem_req),
        .grant_index  (mem_slot),
        .grant_valid  (mem_slot_valid),
        .grant_ready  (mem_bus_if.req_ready),
        `UNUSED_PIN (grant_onehot)
    );

    wire [VM_VPN_LEVEL_BITS-1:0] mem_vpn_slice = vpn_slice(slots[mem_slot].vpn, slots[mem_slot].level);
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] mem_pte_addr = {slots[mem_slot].cur_ppn, {VM_PAGE_OFFSET_BITS{1'b0}}}
                                                   | `VX_CFG_MEM_ADDR_WIDTH'({mem_vpn_slice, {PTE_SHIFT{1'b0}}});

    assign mem_bus_if.req_valid       = mem_slot_valid;
    assign mem_bus_if.req_data.rw     = 1'b0;
    assign mem_bus_if.req_data.addr   = mem_pte_addr[`VX_CFG_MEM_ADDR_WIDTH-1 -: MEM_ADDR_WIDTH];
    `UNUSED_VAR (mem_pte_addr)
    assign mem_bus_if.req_data.data   = '0;
    assign mem_bus_if.req_data.byteen = {MEM_DATA_SIZE{1'b1}};
    assign mem_bus_if.req_data.attr   = '0;
    assign mem_bus_if.req_data.tag    = MEM_TAG_WIDTH'(mem_slot);

    wire mem_req_fire = mem_bus_if.req_valid && mem_bus_if.req_ready;

    // -------------------------------------------------------------------------
    // Memory responses: decode the PTE addressed by the slot
    // -------------------------------------------------------------------------

    wire [SLOT_BITS-1:0] rsp_slot = SLOT_BITS'(mem_bus_if.rsp_data.tag);
    wire mem_rsp_fire = mem_bus_if.rsp_valid && mem_bus_if.rsp_ready;
    // A slot that issued a fetch is always waiting for it.
    assign mem_bus_if.rsp_ready = 1'b1;

    wire [PTE_BITS-1:0] rsp_pte;
    if (PTES_PER_WORD > 1) begin : g_pte_select
        assign rsp_pte = mem_bus_if.rsp_data.data[slots[rsp_slot].pte_sel * PTE_BITS +: PTE_BITS];
    end else begin : g_pte_single
        assign rsp_pte = mem_bus_if.rsp_data.data[PTE_BITS-1:0];
    end

    wire [VM_PTE_FLAGS_WIDTH-1:0] rsp_flags = rsp_pte[VM_PTE_FLAGS_WIDTH-1:0];
    wire [VM_PPN_WIDTH-1:0]       rsp_ppn   = rsp_pte[VM_PTE_PPN_LSB +: VM_PPN_WIDTH];
    `UNUSED_VAR (rsp_pte)

    wire [VM_LEVEL_BITS-1:0] rsp_level  = slots[rsp_slot].level;
    wire rsp_is_leaf  = vm_pte_is_leaf(rsp_flags);
    wire rsp_fault    = !vm_pte_valid(rsp_flags)
                     || (!rsp_is_leaf && (rsp_level == '0))
                     || (rsp_is_leaf && superpage_misaligned(rsp_ppn, rsp_level));
    wire rsp_descend  = !rsp_fault && !rsp_is_leaf;

    wire rsp_cacheable = !slots[rsp_slot].stale;
    assign pwc1_fill_valid = mem_rsp_fire && rsp_cacheable && rsp_descend && (rsp_level == VM_LEVEL_BITS'(TOP_LEVEL));
    assign pwc1_fill_key   = {slots[rsp_slot].root_ppn, vpn_slice(slots[rsp_slot].vpn, rsp_level)};
    assign pwc1_fill_data  = rsp_ppn;

    assign pwc2_fill_valid = mem_rsp_fire && rsp_cacheable && rsp_descend && (rsp_level == VM_LEVEL_BITS'(1)) && (VM_PT_LEVELS == 3);
    assign pwc2_fill_key   = {slots[rsp_slot].cur_ppn, vpn_slice(slots[rsp_slot].vpn, rsp_level)};
    assign pwc2_fill_data  = rsp_ppn;

    // -------------------------------------------------------------------------
    // Completion: hand finished walks back in round-robin order
    // -------------------------------------------------------------------------

    wire [SLOT_BITS-1:0] done_slot;
    wire                 done_valid;

    VX_rr_arbiter #(
        .NUM_REQS (NUM_WALKERS)
    ) done_arb (
        .clk          (clk),
        .reset        (reset),
        .requests     (slot_done),
        .grant_index  (done_slot),
        .grant_valid  (done_valid),
        .grant_ready  (ptw_bus_if.rsp_ready),
        `UNUSED_PIN (grant_onehot)
    );

    assign ptw_bus_if.rsp_valid      = done_valid;
    assign ptw_bus_if.rsp_data.ppn   = slots[done_slot].cur_ppn;
    assign ptw_bus_if.rsp_data.level = slots[done_slot].level;
    assign ptw_bus_if.rsp_data.flags = slots[done_slot].flags;
    assign ptw_bus_if.rsp_data.fault = slots[done_slot].fault;
    assign ptw_bus_if.rsp_data.tag   = slots[done_slot].tag;

    wire rsp_fire = ptw_bus_if.rsp_valid && ptw_bus_if.rsp_ready;

    // -------------------------------------------------------------------------
    // Slot state
    // -------------------------------------------------------------------------

    always @(posedge clk) begin
        if (reset) begin
            for (integer s = 0; s < NUM_WALKERS; ++s) begin
                slot_state[s] <= SLOT_IDLE;
            end
        end else begin
            if (req_fire) begin
                slot_state[free_slot]      <= SLOT_MEM_REQ;
                slots[free_slot].vpn       <= req_vpn;
                slots[free_slot].root_ppn  <= req_root_ppn;
                slots[free_slot].cur_ppn   <= start_ppn;
                slots[free_slot].level     <= start_level;
                slots[free_slot].flags     <= '0;
                slots[free_slot].fault     <= 1'b0;
                slots[free_slot].tag       <= ptw_bus_if.req_data.tag;
                slots[free_slot].stale     <= 1'b0;
            end
            if (flush) begin
                // PTEs fetched by walks already in flight may predate the
                // page-table update this flush publishes; keep them out of
                // the caches just emptied (the TLB bank drops the result).
                for (integer s = 0; s < NUM_WALKERS; ++s) begin
                    if (slot_state[s] != SLOT_IDLE) begin
                        slots[s].stale <= 1'b1;
                    end
                end
            end
            if (mem_req_fire) begin
                slot_state[mem_slot]       <= SLOT_MEM_RSP;
                slots[mem_slot].pte_sel    <= mem_pte_addr[PTE_SHIFT +: `UP(PTE_SEL_BITS)];
            end
            if (mem_rsp_fire) begin
                if (rsp_descend) begin
                    slot_state[rsp_slot]   <= SLOT_MEM_REQ;
                    slots[rsp_slot].cur_ppn <= rsp_ppn;
                    slots[rsp_slot].level  <= rsp_level - VM_LEVEL_BITS'(1);
                end else begin
                    slot_state[rsp_slot]   <= SLOT_DONE;
                    slots[rsp_slot].cur_ppn <= rsp_ppn;
                    slots[rsp_slot].flags  <= rsp_flags;
                    slots[rsp_slot].fault  <= rsp_fault;
                end
            end
            if (rsp_fire) begin
                slot_state[done_slot]      <= SLOT_IDLE;
            end
        end
    end

`ifdef SIMULATION
    always @(posedge clk) begin
        if (!reset && mem_rsp_fire && rsp_fault) begin
            `ERROR(("%t: *** %s page fault: vpn=0x%0h level=%0d pte=0x%0h", $time, "ptw", slots[rsp_slot].vpn, rsp_level, rsp_pte));
        end
    end
`endif

`ifdef DBG_TRACE_MMU
    always @(posedge clk) begin
        if (req_fire) begin
            `TRACE(2, ("%t: ptw-req: slot=%0d, vpn=0x%0h, root=0x%0h, level=%0d, tag=0x%0h\n", $time, free_slot, req_vpn, req_root_ppn, start_level, ptw_bus_if.req_data.tag))
        end
        if (mem_req_fire) begin
            `TRACE(2, ("%t: ptw-mem-req: slot=%0d, addr=0x%0h, level=%0d\n", $time, mem_slot, mem_pte_addr, slots[mem_slot].level))
        end
        if (mem_rsp_fire) begin
            `TRACE(2, ("%t: ptw-mem-rsp: slot=%0d, pte=0x%0h, leaf=%b, fault=%b\n", $time, rsp_slot, rsp_pte, rsp_is_leaf, rsp_fault))
        end
        if (rsp_fire) begin
            `TRACE(2, ("%t: ptw-rsp: slot=%0d, ppn=0x%0h, level=%0d, fault=%b, tag=0x%0h\n", $time, done_slot, slots[done_slot].cur_ppn, slots[done_slot].level, slots[done_slot].fault, slots[done_slot].tag))
        end
    end
`endif

    // -------------------------------------------------------------------------
    // Performance counters
    // -------------------------------------------------------------------------

`ifdef PERF_ENABLE
    wire [NUM_WALKERS-1:0] slot_active = ~slot_idle;
    wire [`CLOG2(NUM_WALKERS+1)-1:0] active_count;
    `POP_COUNT(active_count, slot_active);

    reg [PERF_CTR_BITS-1:0] perf_walks_r, perf_latency_r;
    reg [PERF_CTR_BITS-1:0] perf_pwc1_hits_r, perf_pwc1_misses_r;
    reg [PERF_CTR_BITS-1:0] perf_pwc2_hits_r, perf_pwc2_misses_r;

    always @(posedge clk) begin
        if (reset) begin
            perf_walks_r       <= '0;
            perf_latency_r     <= '0;
            perf_pwc1_hits_r   <= '0;
            perf_pwc1_misses_r <= '0;
            perf_pwc2_hits_r   <= '0;
            perf_pwc2_misses_r <= '0;
        end else begin
            perf_latency_r <= perf_latency_r + PERF_CTR_BITS'(active_count);
            if (req_fire) begin
                perf_walks_r <= perf_walks_r + PERF_CTR_BITS'(1);
                if (pwc1_hit) begin
                    perf_pwc1_hits_r <= perf_pwc1_hits_r + PERF_CTR_BITS'(1);
                    if (VM_PT_LEVELS == 3) begin
                        if (pwc2_hit) perf_pwc2_hits_r   <= perf_pwc2_hits_r + PERF_CTR_BITS'(1);
                        else          perf_pwc2_misses_r <= perf_pwc2_misses_r + PERF_CTR_BITS'(1);
                    end
                end else begin
                    perf_pwc1_misses_r <= perf_pwc1_misses_r + PERF_CTR_BITS'(1);
                end
            end
        end
    end

    assign ptw_perf.walks       = perf_walks_r;
    assign ptw_perf.latency     = perf_latency_r;
    assign ptw_perf.pwc1_hits   = perf_pwc1_hits_r;
    assign ptw_perf.pwc1_misses = perf_pwc1_misses_r;
    assign ptw_perf.pwc2_hits   = perf_pwc2_hits_r;
    assign ptw_perf.pwc2_misses = perf_pwc2_misses_r;
`endif

endmodule
