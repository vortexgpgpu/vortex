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

// Shared memory-subsystem boundary for FU-local AGUs.
//
// Owns the VX_mem_scheduler + dcache port driver. Multiple FU clients
// (LSU per-lane AGU, TCU_LD warp-level AGU, future RTX/TEX/OM
// preload AGUs) share a single scheduler and a single dcache port.
//
// P2b: round-robin arbiter for NUM_CLIENTS > 1. The chosen client's
// tag is extended with a client_id sideband (CLIENT_ID_BITS bits
// prepended) so responses can be demuxed back to the originator.
// For NUM_CLIENTS=1 the arbiter degenerates to pass-through with no
// tag extension (zero-cost path for the LSU-only case).

module VX_lsu_scheduler import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_CLIENTS    = 1,
    parameter NUM_LANES      = `VX_CFG_NUM_LSU_LANES,
    parameter CORE_QUEUE_SIZE= `VX_CFG_LSUQ_IN_SIZE,
    parameter MEM_QUEUE_SIZE = `VX_CFG_LSUQ_OUT_SIZE
) (
    input wire clk,
    input wire reset,

    VX_lsu_sched_if.slave  client_if [NUM_CLIENTS],

    output wire             subsystem_drained,

    VX_lsu_mem_if.master    lsu_mem_if
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam CLIENT_ID_BITS  = (NUM_CLIENTS > 1) ? $clog2(NUM_CLIENTS) : 0;
    localparam SCHED_TAG_WIDTH = LSU_CLIENT_TAG_WIDTH + CLIENT_ID_BITS;

    // -----------------------------------------------------------------------
    // Arbiter (round-robin) — feeds the single scheduler input from the
    // multi-client request channels.
    // -----------------------------------------------------------------------
    wire                            sched_req_valid;
    wire                            sched_req_rw;
    wire [NUM_LANES-1:0]            sched_req_mask;
    wire [NUM_LANES-1:0][LSU_WORD_SIZE-1:0]    sched_req_byteen;
    wire [NUM_LANES-1:0][LSU_ADDR_WIDTH-1:0]   sched_req_addr;
    wire [NUM_LANES-1:0][MEM_ATTR_WIDTH-1:0]   sched_req_attr;
    wire [NUM_LANES-1:0][(LSU_WORD_SIZE*8)-1:0] sched_req_data;
    wire [SCHED_TAG_WIDTH-1:0]      sched_req_tag;
    wire                            sched_req_ready;

    wire                            sched_rsp_valid;
    wire [NUM_LANES-1:0]            sched_rsp_mask;
    wire [NUM_LANES-1:0][(LSU_WORD_SIZE*8)-1:0] sched_rsp_data;
    wire [SCHED_TAG_WIDTH-1:0]      sched_rsp_tag;
    wire                            sched_rsp_sop;
    wire                            sched_rsp_eop;
    wire                            sched_rsp_ready;

    // Flatten interface array into per-signal vectors so the arbiter
    // logic stays index-friendly (SystemVerilog interface arrays can't
    // be indexed inside expressions).
    wire [NUM_CLIENTS-1:0]                                       cli_req_valid;
    lsu_client_req_data_t                                        cli_req_data  [NUM_CLIENTS];
    wire [NUM_CLIENTS-1:0]                                       cli_req_ready;
    wire [NUM_CLIENTS-1:0]                                       cli_rsp_valid;
    lsu_client_rsp_data_t                                        cli_rsp_data  [NUM_CLIENTS];
    wire [NUM_CLIENTS-1:0]                                       cli_rsp_ready;

    for (genvar i = 0; i < NUM_CLIENTS; ++i) begin : g_cli_flat
        assign cli_req_valid[i]    = client_if[i].req_valid;
        assign cli_req_data[i]     = client_if[i].req_data;
        assign client_if[i].req_ready = cli_req_ready[i];

        assign client_if[i].rsp_valid = cli_rsp_valid[i];
        assign client_if[i].rsp_data  = cli_rsp_data[i];
        assign cli_rsp_ready[i]    = client_if[i].rsp_ready;
    end

    if (NUM_CLIENTS == 1) begin : g_no_arb
        // Pure pass-through — no client_id, no rotation, no tag extension.
        assign sched_req_valid    = cli_req_valid[0];
        assign sched_req_rw       = cli_req_data[0].rw;
        assign sched_req_mask     = cli_req_data[0].mask;
        assign sched_req_byteen   = cli_req_data[0].byteen;
        assign sched_req_addr     = cli_req_data[0].addr;
        assign sched_req_attr     = cli_req_data[0].attr;
        assign sched_req_data     = cli_req_data[0].data;
        assign sched_req_tag      = cli_req_data[0].tag;
        assign cli_req_ready[0]   = sched_req_ready;

        assign cli_rsp_valid[0]   = sched_rsp_valid;
        assign cli_rsp_data[0].mask  = sched_rsp_mask;
        assign cli_rsp_data[0].data  = sched_rsp_data;
        assign cli_rsp_data[0].tag   = sched_rsp_tag;
        assign cli_rsp_data[0].sop   = sched_rsp_sop;
        assign cli_rsp_data[0].eop   = sched_rsp_eop;
        assign sched_rsp_ready    = cli_rsp_ready[0];
    end else begin : g_arb
        // Round-robin priority. rr_ptr rotates after every accepted request.
        // For NUM_CLIENTS=2 this trivially alternates; the generic loop
        // generalizes to wider clients later.
        reg  [CLIENT_ID_BITS-1:0] rr_ptr_r;

        // Priority chain starting at rr_ptr (rotated), pick lowest set bit.
        wire [(2*NUM_CLIENTS)-1:0] rotated_double = {cli_req_valid, cli_req_valid} >> rr_ptr_r;
        wire [NUM_CLIENTS-1:0]     rotated_valid  = rotated_double[NUM_CLIENTS-1:0];
        `UNUSED_VAR (rotated_double[(2*NUM_CLIENTS)-1:NUM_CLIENTS])

        logic [CLIENT_ID_BITS-1:0] rotated_grant_idx;
        logic                       any_valid;
        always_comb begin
            rotated_grant_idx = '0;
            any_valid         = 1'b0;
            for (int i = NUM_CLIENTS-1; i >= 0; i--) begin
                if (rotated_valid[i]) begin
                    rotated_grant_idx = CLIENT_ID_BITS'(i);
                    any_valid         = 1'b1;
                end
            end
        end

        wire [CLIENT_ID_BITS-1:0] grant_idx = rr_ptr_r + rotated_grant_idx;
        wire                       grant_fire = any_valid && sched_req_ready;

        always @(posedge clk) begin
            if (reset) begin
                rr_ptr_r <= '0;
            end else if (grant_fire) begin
                rr_ptr_r <= grant_idx + CLIENT_ID_BITS'(1);
            end
        end

        // Mux selected client's request to the scheduler. Tag is extended
        // with grant_idx so the response demux can route back.
        assign sched_req_valid  = any_valid;
        assign sched_req_rw     = cli_req_data[grant_idx].rw;
        assign sched_req_mask   = cli_req_data[grant_idx].mask;
        assign sched_req_byteen = cli_req_data[grant_idx].byteen;
        assign sched_req_addr   = cli_req_data[grant_idx].addr;
        assign sched_req_attr   = cli_req_data[grant_idx].attr;
        assign sched_req_data   = cli_req_data[grant_idx].data;
        assign sched_req_tag    = {grant_idx, cli_req_data[grant_idx].tag};

        // Drive per-client ready: only the granted client sees ready high.
        for (genvar i = 0; i < NUM_CLIENTS; ++i) begin : g_creq_rdy
            assign cli_req_ready[i] = sched_req_ready && (grant_idx == CLIENT_ID_BITS'(i));
        end

        // Demux response: extract client_id from extended tag.
        wire [CLIENT_ID_BITS-1:0] rsp_cid = sched_rsp_tag[SCHED_TAG_WIDTH-1 -: CLIENT_ID_BITS];
        for (genvar i = 0; i < NUM_CLIENTS; ++i) begin : g_crsp
            assign cli_rsp_valid[i]    = sched_rsp_valid && (rsp_cid == CLIENT_ID_BITS'(i));
            assign cli_rsp_data[i].mask = sched_rsp_mask;
            assign cli_rsp_data[i].data = sched_rsp_data;
            assign cli_rsp_data[i].tag  = sched_rsp_tag[LSU_CLIENT_TAG_WIDTH-1:0];
            assign cli_rsp_data[i].sop  = sched_rsp_sop;
            assign cli_rsp_data[i].eop  = sched_rsp_eop;
        end
        // sched_rsp_ready is driven by the granted client's ready.
        logic sched_rsp_ready_l;
        always_comb begin
            sched_rsp_ready_l = 1'b0;
            for (int i = 0; i < NUM_CLIENTS; i++) begin
                if (rsp_cid == CLIENT_ID_BITS'(i)) sched_rsp_ready_l = cli_rsp_ready[i];
            end
        end
        assign sched_rsp_ready = sched_rsp_ready_l;
    end

    // -----------------------------------------------------------------------
    // VX_mem_scheduler — single instance, shared across clients.
    // -----------------------------------------------------------------------
    wire                                       lsu_mem_req_valid;
    wire                                       lsu_mem_req_rw;
    wire [NUM_LANES-1:0]                       lsu_mem_req_mask;
    wire [NUM_LANES-1:0][LSU_WORD_SIZE-1:0]    lsu_mem_req_byteen;
    wire [NUM_LANES-1:0][LSU_ADDR_WIDTH-1:0]   lsu_mem_req_addr;
    wire [NUM_LANES-1:0][MEM_ATTR_WIDTH-1:0]   lsu_mem_req_attr;
    wire [NUM_LANES-1:0][(LSU_WORD_SIZE*8)-1:0] lsu_mem_req_data;
    wire [LSU_TAG_WIDTH-1:0]                   lsu_mem_req_tag;
    wire                                       lsu_mem_req_ready;

    wire                                       lsu_mem_rsp_valid;
    wire [NUM_LANES-1:0]                       lsu_mem_rsp_mask;
    wire [NUM_LANES-1:0][(LSU_WORD_SIZE*8)-1:0] lsu_mem_rsp_data;
    wire [LSU_TAG_WIDTH-1:0]                   lsu_mem_rsp_tag;
    wire                                       lsu_mem_rsp_ready;

    wire                                       sched_req_queue_empty;

    VX_mem_scheduler #(
        .INSTANCE_ID (`SFORMATF(("%s-memsched", INSTANCE_ID))),
        .CORE_REQS   (NUM_LANES),
        .MEM_CHANNELS(NUM_LANES),
        .WORD_SIZE   (LSU_WORD_SIZE),
        .LINE_SIZE   (LSU_WORD_SIZE),
        .ADDR_WIDTH  (LSU_ADDR_WIDTH),
        .USER_WIDTH  (MEM_ATTR_WIDTH),
        .TAG_WIDTH   (SCHED_TAG_WIDTH),
        .CORE_QUEUE_SIZE (CORE_QUEUE_SIZE),
        .MEM_QUEUE_SIZE (MEM_QUEUE_SIZE),
        .UUID_WIDTH  (UUID_WIDTH),
        .RSP_PARTIAL (1),
        .MEM_OUT_BUF (0),
        .CORE_OUT_BUF(0)
    ) mem_scheduler (
        .clk            (clk),
        .reset          (reset),

        .core_req_valid (sched_req_valid),
        .core_req_rw    (sched_req_rw),
        .core_req_mask  (sched_req_mask),
        .core_req_byteen(sched_req_byteen),
        .core_req_addr  (sched_req_addr),
        .core_req_user  (sched_req_attr),
        .core_req_data  (sched_req_data),
        .core_req_tag   (sched_req_tag),
        .core_req_ready (sched_req_ready),
        .req_queue_empty(sched_req_queue_empty),
        `UNUSED_PIN (req_queue_rw_notify),

        .core_rsp_valid (sched_rsp_valid),
        .core_rsp_mask  (sched_rsp_mask),
        .core_rsp_data  (sched_rsp_data),
        .core_rsp_tag   (sched_rsp_tag),
        .core_rsp_sop   (sched_rsp_sop),
        .core_rsp_eop   (sched_rsp_eop),
        .core_rsp_ready (sched_rsp_ready),

        .mem_req_valid  (lsu_mem_req_valid),
        .mem_req_rw     (lsu_mem_req_rw),
        .mem_req_mask   (lsu_mem_req_mask),
        .mem_req_byteen (lsu_mem_req_byteen),
        .mem_req_addr   (lsu_mem_req_addr),
        .mem_req_user   (lsu_mem_req_attr),
        .mem_req_data   (lsu_mem_req_data),
        .mem_req_tag    (lsu_mem_req_tag),
        .mem_req_ready  (lsu_mem_req_ready),

        .mem_rsp_valid  (lsu_mem_rsp_valid),
        .mem_rsp_mask   (lsu_mem_rsp_mask),
        .mem_rsp_data   (lsu_mem_rsp_data),
        .mem_rsp_tag    (lsu_mem_rsp_tag),
        .mem_rsp_ready  (lsu_mem_rsp_ready)
    );

    // -----------------------------------------------------------------------
    // Drive the dcache port.
    // -----------------------------------------------------------------------
    assign lsu_mem_if.req_valid       = lsu_mem_req_valid;
    assign lsu_mem_if.req_data.mask   = lsu_mem_req_mask;
    assign lsu_mem_if.req_data.rw     = lsu_mem_req_rw;
    assign lsu_mem_if.req_data.byteen = lsu_mem_req_byteen;
    assign lsu_mem_if.req_data.addr   = lsu_mem_req_addr;
    assign lsu_mem_if.req_data.user   = lsu_mem_req_attr;
    assign lsu_mem_if.req_data.data   = lsu_mem_req_data;
    assign lsu_mem_if.req_data.tag    = lsu_mem_req_tag;
    assign lsu_mem_req_ready          = lsu_mem_if.req_ready;

    assign lsu_mem_rsp_valid          = lsu_mem_if.rsp_valid;
    assign lsu_mem_rsp_mask           = lsu_mem_if.rsp_data.mask;
    assign lsu_mem_rsp_data           = lsu_mem_if.rsp_data.data;
    assign lsu_mem_rsp_tag            = lsu_mem_if.rsp_data.tag;
    assign lsu_mem_if.rsp_ready       = lsu_mem_rsp_ready;

    assign subsystem_drained = sched_req_queue_empty & ~lsu_mem_if.req_valid;

endmodule
