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

// Shared page-table walker complex. A pool of independent level-counted
// walkers drains the cluster TLB's miss station; a small direct-mapped walk
// cache lets a walk skip the interior fetches when a spatially-adjacent walk
// already resolved its last-level table. PTE fetches from every walker merge
// onto one cache client port (the walker index rides the request tag so
// responses demux back). Structural faults raise a one-cycle report the
// fault-latch surface records.
module VX_ptw import VX_gpu_pkg::*, VX_tlb_pkg::*; #(
    parameter NUM_WALKERS     = `VX_CFG_PTW_NUM_WALKERS,
    parameter WALK_CACHE_SIZE = `VX_CFG_PTW_WALK_CACHE_SIZE,
    parameter PT_LEVELS       = `VX_VM_PT_LEVEL,
    parameter DATA_SIZE       = `VX_CFG_L1_LINE_SIZE,
    parameter ID_WIDTH        = `CLOG2(`VX_CFG_L2_TLB_MSHR_SIZE),
    // Cache-facing PTE-port tag width (must hold the mem-bus {uuid,value}
    // struct); the walker index rides its top bits.
    parameter MEM_TAG_WIDTH   = DCACHE_TAG_WIDTH_BASE + `ARB_SEL_BITS(`VX_CFG_PTW_NUM_WALKERS, 1)
) (
    input wire clk,
    input wire reset,

    input wire [`VX_CFG_XLEN-1:0] satp,

    VX_tlb_bus_if.slave    miss_if,
    VX_mem_bus_if.master   mem_bus_if,
    VX_tlb_flush_if.slave  flush_if,
    VX_mmu_fault_if.master fault_if,

    output wire            empty
);
    localparam WID_W        = `UP(`CLOG2(NUM_WALKERS));
    // The arb appends the walker index on top of each walker's base tag.
    localparam WALKER_TAG_W = MEM_TAG_WIDTH - `ARB_SEL_BITS(NUM_WALKERS, 1);
    localparam PAGE_LOG2    = `VX_VM_PAGE_LOG2_SIZE;
    localparam RSP_ID_W     = `UP(ID_WIDTH);
    localparam RSP_DATAW    = RSP_ID_W + 1 + TLB_LEVEL_WIDTH + TLB_PPN_WIDTH + TLB_FLAGS_WIDTH;

    wire [TLB_PPN_WIDTH-1:0] satp_root_ppn = satp[TLB_PPN_WIDTH-1:0];
    `UNUSED_VAR (satp[`VX_CFG_XLEN-1:TLB_PPN_WIDTH])

    // ---------------------------------------------------------------------
    // Walk cache — probed on the incoming request, filled by descending walks
    // ---------------------------------------------------------------------
    wire                     wc_hit;
    wire [TLB_PPN_WIDTH-1:0] wc_ppn;

    wire                     wc_wr_valid;
    wire [TLB_VPN_WIDTH-1:0] wc_wr_vpn;
    wire [TLB_PPN_WIDTH-1:0] wc_wr_ppn;

    VX_ptw_cache #(
        .NUM_ENTRIES (WALK_CACHE_SIZE)
    ) walk_cache (
        .clk       (clk),
        .reset     (reset),
        .probe_vpn (miss_if.req_data.vpn),
        .probe_hit (wc_hit),
        .probe_ppn (wc_ppn),
        .wr_valid  (wc_wr_valid),
        .wr_vpn    (wc_wr_vpn),
        .wr_ppn    (wc_wr_ppn),
        .flush     (flush_if.req)
    );

    // A hit resolves the last-level table directly; a miss starts at the root.
    wire wc_use = wc_hit && (PT_LEVELS > 1);
    wire [TLB_PPN_WIDTH-1:0]   start_ppn   = wc_use ? wc_ppn : satp_root_ppn;
    wire [TLB_LEVEL_WIDTH-1:0] start_level = wc_use ? '0 : TLB_LEVEL_WIDTH'(PT_LEVELS - 1);

    // ---------------------------------------------------------------------
    // Walker pool
    // ---------------------------------------------------------------------
    wire [NUM_WALKERS-1:0]                      w_req_ready;
    wire [NUM_WALKERS-1:0]                      w_active;
    wire [NUM_WALKERS-1:0]                      w_rsp_valid;
    wire [NUM_WALKERS-1:0][RSP_ID_W-1:0]        w_rsp_id;
    wire [NUM_WALKERS-1:0]                      w_rsp_fault;
    wire [NUM_WALKERS-1:0][TLB_LEVEL_WIDTH-1:0] w_rsp_level;
    wire [NUM_WALKERS-1:0][TLB_PPN_WIDTH-1:0]   w_rsp_ppn;
    wire [NUM_WALKERS-1:0][TLB_FLAGS_WIDTH-1:0] w_rsp_flags;
    wire [NUM_WALKERS-1:0]                      w_rsp_ready;
    wire [NUM_WALKERS-1:0]                      w_wc_wr_valid;
    wire [NUM_WALKERS-1:0][TLB_VPN_WIDTH-1:0]   w_wc_wr_vpn;
    wire [NUM_WALKERS-1:0][TLB_PPN_WIDTH-1:0]   w_wc_wr_ppn;

    VX_mem_bus_if #(
        .DATA_SIZE (DATA_SIZE),
        .TAG_WIDTH (WALKER_TAG_W)
    ) walker_mem [NUM_WALKERS] ();

    // Dispatch: grant the lowest idle walker.
    wire [NUM_WALKERS-1:0] disp_onehot;
    reg  [NUM_WALKERS-1:0] disp_pick;
    always @(*) begin
        disp_pick = '0;
        for (int i = NUM_WALKERS-1; i >= 0; --i) begin
            if (w_req_ready[i]) begin
                disp_pick = '0;
                disp_pick[i] = 1'b1;
            end
        end
    end
    wire any_free = (| w_req_ready);
    assign disp_onehot = disp_pick;
    assign miss_if.req_ready = any_free;
    wire dispatch_fire = miss_if.req_valid && any_free;

    // Per-walker record of the dispatched request, for fault reporting.
    reg [TLB_VPN_WIDTH-1:0] disp_vpn_r [NUM_WALKERS];
    reg [1:0]               disp_acc_r [NUM_WALKERS];
    reg                     disp_amo_r [NUM_WALKERS];

    for (genvar i = 0; i < NUM_WALKERS; ++i) begin : g_walkers
        wire dispatch_i = dispatch_fire && disp_onehot[i];

        always @(posedge clk) begin
            if (dispatch_i) begin
                disp_vpn_r[i] <= miss_if.req_data.vpn;
                disp_acc_r[i] <= miss_if.req_data.access;
                disp_amo_r[i] <= miss_if.req_data.amo;
            end
        end

        VX_ptw_walker #(
            .PT_LEVELS (PT_LEVELS),
            .DATA_SIZE (DATA_SIZE),
            .TAG_WIDTH (WALKER_TAG_W),
            .ID_WIDTH  (ID_WIDTH)
        ) walker (
            .clk          (clk),
            .reset        (reset),
            .req_valid    (dispatch_i),
            .req_id       (miss_if.req_data.id),
            .req_access   (miss_if.req_data.access),
            .req_amo      (miss_if.req_data.amo),
            .req_vpn      (miss_if.req_data.vpn),
            .req_base_ppn (start_ppn),
            .req_level    (start_level),
            .req_ready    (w_req_ready[i]),
            .rsp_valid    (w_rsp_valid[i]),
            .rsp_id       (w_rsp_id[i]),
            .rsp_fault    (w_rsp_fault[i]),
            .rsp_level    (w_rsp_level[i]),
            .rsp_ppn      (w_rsp_ppn[i]),
            .rsp_flags    (w_rsp_flags[i]),
            .rsp_ready    (w_rsp_ready[i]),
            .wc_wr_valid  (w_wc_wr_valid[i]),
            .wc_wr_vpn    (w_wc_wr_vpn[i]),
            .wc_wr_ppn    (w_wc_wr_ppn[i]),
            .active       (w_active[i]),
            .mem_bus_if   (walker_mem[i])
        );
    end

    // ---------------------------------------------------------------------
    // PTE fetch: merge every walker's port onto the single cache client port
    // ---------------------------------------------------------------------
    VX_mem_bus_if #(
        .DATA_SIZE (DATA_SIZE),
        .TAG_WIDTH (MEM_TAG_WIDTH)
    ) merged_mem [1] ();

    VX_mem_bus_arb #(
        .NUM_INPUTS  (NUM_WALKERS),
        .NUM_OUTPUTS (1),
        .DATA_SIZE   (DATA_SIZE),
        .TAG_WIDTH   (WALKER_TAG_W),
        .TAG_SEL_IDX (WALKER_TAG_W),
        .ARBITER     ("R"),
        .REQ_OUT_BUF ((NUM_WALKERS > 1) ? 2 : 0),
        .RSP_OUT_BUF ((NUM_WALKERS > 1) ? 2 : 0)
    ) mem_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (walker_mem),
        .bus_out_if (merged_mem)
    );

    `ASSIGN_VX_MEM_BUS_IF (mem_bus_if, merged_mem[0]);

    // ---------------------------------------------------------------------
    // Walk-cache fill: at most one descending walk writes per cycle
    // ---------------------------------------------------------------------
    reg [WID_W-1:0] wc_wr_pick;
    always @(*) begin
        wc_wr_pick = '0;
        for (int i = NUM_WALKERS-1; i >= 0; --i) begin
            if (w_wc_wr_valid[i]) begin
                wc_wr_pick = WID_W'(i);
            end
        end
    end
    assign wc_wr_valid = (| w_wc_wr_valid);
    assign wc_wr_vpn   = w_wc_wr_vpn[wc_wr_pick];
    assign wc_wr_ppn   = w_wc_wr_ppn[wc_wr_pick];

    // ---------------------------------------------------------------------
    // Response arbitration back to the miss station
    // ---------------------------------------------------------------------
    wire [NUM_WALKERS-1:0][RSP_DATAW-1:0] rsp_din;
    for (genvar i = 0; i < NUM_WALKERS; ++i) begin : g_rsp_pack
        assign rsp_din[i] = {
            w_rsp_id[i], w_rsp_fault[i], w_rsp_level[i], w_rsp_ppn[i], w_rsp_flags[i]
        };
    end

    wire [0:0]                 rsp_valid_out;
    wire [0:0][RSP_DATAW-1:0]  rsp_data_out;
    wire [0:0]                 rsp_ready_out;
    assign rsp_ready_out[0] = miss_if.rsp_ready;

    VX_stream_arb #(
        .NUM_INPUTS  (NUM_WALKERS),
        .NUM_OUTPUTS (1),
        .DATAW       (RSP_DATAW),
        .ARBITER     ("R"),
        .OUT_BUF     ((NUM_WALKERS > 1) ? 2 : 0)
    ) rsp_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (w_rsp_valid),
        .ready_in  (w_rsp_ready),
        .data_in   (rsp_din),
        .valid_out (rsp_valid_out),
        .data_out  (rsp_data_out),
        .ready_out (rsp_ready_out),
        `UNUSED_PIN (sel_out)
    );

    assign miss_if.rsp_valid = rsp_valid_out[0];
    wire [RSP_DATAW-1:0] rsp_dout = rsp_data_out[0];

    assign miss_if.rsp_data = '{
        id:    rsp_dout[(1 + TLB_LEVEL_WIDTH + TLB_PPN_WIDTH + TLB_FLAGS_WIDTH) +: RSP_ID_W],
        fault: rsp_dout[(TLB_LEVEL_WIDTH + TLB_PPN_WIDTH + TLB_FLAGS_WIDTH)],
        level: rsp_dout[(TLB_PPN_WIDTH + TLB_FLAGS_WIDTH) +: TLB_LEVEL_WIDTH],
        ppn:   rsp_dout[TLB_FLAGS_WIDTH +: TLB_PPN_WIDTH],
        flags: rsp_dout[0 +: TLB_FLAGS_WIDTH]
    };

    // ---------------------------------------------------------------------
    // First-fault report: a structural fault leaves the walker with fault=1
    // ---------------------------------------------------------------------
    wire [NUM_WALKERS-1:0] w_fault_fire = w_rsp_valid & w_rsp_fault & w_rsp_ready;
    reg [WID_W-1:0] fault_pick;
    always @(*) begin
        fault_pick = '0;
        for (int i = NUM_WALKERS-1; i >= 0; --i) begin
            if (w_fault_fire[i]) begin
                fault_pick = WID_W'(i);
            end
        end
    end

    wire [`VX_CFG_XLEN-1:0] fault_va;
    assign fault_va = `VX_CFG_XLEN'({disp_vpn_r[fault_pick], {PAGE_LOG2{1'b0}}});

    assign fault_if.valid  = (| w_fault_fire);
    assign fault_if.va     = fault_va;
    assign fault_if.access = disp_acc_r[fault_pick];
    assign fault_if.amo    = disp_amo_r[fault_pick];

    // ---------------------------------------------------------------------
    // Drain / flush
    // ---------------------------------------------------------------------
    assign empty = ~(| w_active);
    assign flush_if.done = flush_if.req && ~(| w_active);

endmodule
