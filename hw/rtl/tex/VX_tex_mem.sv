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

`include "VX_tex_define.vh"

module VX_tex_mem import VX_gpu_pkg::*; import VX_tex_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter REQ_TAGW    = 1,
    parameter NUM_LANES   = 1,
    parameter W_ADDR_BITS = `TEX_ADDR_BITS + 6
) (
    input wire clk,
    input wire reset,

   // memory interface
    VX_mem_bus_if.master                cache_bus_if [TCACHE_NUM_REQS],

    // inputs
    input wire                          req_valid,
    input wire [NUM_LANES-1:0]          req_mask,
    input wire [TEX_FILTER_BITS-1:0]   req_filter,
    input wire [`TEX_LGSTRIDE_BITS-1:0] req_lgstride,
    input wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][W_ADDR_BITS-1:0] req_baseaddr,
    input wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][3:0][31:0] req_addr,
    input wire [REQ_TAGW-1:0]           req_tag,
    output wire                         req_ready,

    // outputs
    output wire                         rsp_valid,
    output wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][3:0][31:0] rsp_data,
    output wire [REQ_TAGW-1:0]          rsp_tag,
    input wire                          rsp_ready
);

    // Taps run level-major: tap (k * 4 + j) is the bilinear corner j of level k.
    localparam NUM_TAPS  = TEX_NUM_LEVELS * 4;
    localparam TAP_SEL_BITS = `CLOG2(NUM_TAPS);
    localparam TAG_WIDTH = REQ_TAGW + TEX_FILTER_BITS + `TEX_LGSTRIDE_BITS + (NUM_LANES * NUM_TAPS * 2) + NUM_TAPS;

    wire                           mem_req_valid;
    wire [NUM_TAPS-1:0][NUM_LANES-1:0] mem_req_mask;
    wire [NUM_TAPS-1:0][NUM_LANES-1:0][TCACHE_ADDR_WIDTH-1:0] mem_req_addr;
    wire [NUM_TAPS-1:0][NUM_LANES-1:0][3:0] mem_req_byteen;
    wire [NUM_TAPS-1:0][NUM_LANES-1:0][1:0] mem_req_align;
    wire [TAG_WIDTH-1:0]           mem_req_tag;
    wire [NUM_TAPS-1:0]            mem_req_dups;
    wire                           mem_req_ready;

    wire                           mem_rsp_valid;
    wire [NUM_TAPS-1:0][NUM_LANES-1:0][31:0] mem_rsp_data;
    wire [TAG_WIDTH-1:0]           mem_rsp_tag;
    wire                           mem_rsp_ready;

    // A sample reads the level above only when the two are to be blended.
    wire mip_linear = req_filter[TEX_FILTER_BITS-1];

    // full address calculation

    wire [NUM_LANES-1:0][NUM_TAPS-1:0][W_ADDR_BITS-1:0] full_addr;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_full_addr
        for (genvar k = 0; k < TEX_NUM_LEVELS; ++k) begin : g_k
            for (genvar j = 0; j < 4; ++j) begin  : g_j
                assign full_addr[i][k*4+j] = req_baseaddr[i][k] + W_ADDR_BITS'(req_addr[i][k][j]);
            end
        end
    end

    // reorder addresses into per-quad requests

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_mem_req_align
        for (genvar t = 0; t < NUM_TAPS; ++t) begin : g_t
            assign mem_req_addr[t][i]   = TCACHE_ADDR_WIDTH'(full_addr[i][t][W_ADDR_BITS-1:2]);
            assign mem_req_align[t][i]  = full_addr[i][t][1:0];
            assign mem_req_byteen[t][i] = 4'b1111;
        end
    end

    // detect duplicate addresses

    // Lanes fetch one texel between them when they address the same one. The
    // level base has to be compared alongside the offset: lanes carry their own
    // lod, so two lanes reading corner 0 of different levels share an offset of
    // zero while addressing different texels. Both operands are pre-adder, so
    // the comparison does not sit behind the full-address sum.
    for (genvar t = 0; t < NUM_TAPS; ++t) begin : g_mem_req_dups
        if (NUM_LANES > 1) begin : g_lanes
            wire [NUM_LANES-2:0] addr_matches;
            for (genvar j = 0; j < (NUM_LANES-1); ++j) begin : g_j
                assign addr_matches[j] = ((req_addr[j+1][t/4][t%4] == req_addr[0][t/4][t%4])
                                       && (req_baseaddr[j+1][t/4] == req_baseaddr[0][t/4]))
                                       || ~req_mask[j+1];
            end
            assign mem_req_dups[t] = req_mask[0] && (& addr_matches);
        end else begin : g_1lane
            assign mem_req_dups[t] = 0;
        end
    end

    for (genvar t = 0; t < NUM_TAPS; ++t) begin : g_mem_req_mask
        wire texel_valid = (req_filter[0] || ((t % 4) == 0)) && (mip_linear || (t < 4));
        for (genvar j = 0; j < NUM_LANES; ++j) begin : g_j
            assign mem_req_mask[t][j] = req_mask[j] && texel_valid && (~mem_req_dups[t] || (j == 0));
        end
    end

    // submit request to memory

    assign mem_req_valid = req_valid;
    assign mem_req_tag   = {req_tag, req_filter, req_lgstride, mem_req_align, mem_req_dups};
    assign req_ready     = mem_req_ready;

    // schedule memory request

    VX_lsu_mem_if #(
        .NUM_LANES (TCACHE_NUM_REQS),
        .DATA_SIZE (4),
        .TAG_WIDTH (TCACHE_TAG_WIDTH)
    ) lsu_mem_if();

    VX_mem_scheduler #(
        .INSTANCE_ID (`SFORMATF(("%s-memsched", INSTANCE_ID))),
        .CORE_REQS   (TEX_MEM_REQS),
        .MEM_CHANNELS(TCACHE_NUM_REQS),
        .WORD_SIZE   (4),
        .ADDR_WIDTH  (TCACHE_ADDR_WIDTH),
        .USER_WIDTH  (0),
        .TAG_WIDTH   (TAG_WIDTH),
        .CORE_QUEUE_SIZE(`VX_CFG_TEX_MEM_QUEUE_SIZE),
        .UUID_WIDTH  (UUID_WIDTH),
        .RSP_PARTIAL (0),
        .RW_ENABLE   (0),
        .LUTRAM      (1),
        .REQQ_LUTRAM (1),
        .MEM_OUT_BUF (3), // fully register cache-request output (SLR-crossing skid)
        // A second skid entry on the 128-word response bus costs more logic
        // than the whole address path, and the bus does not leave the unit.
        .CORE_OUT_BUF(1)
    ) mem_scheduler (
        .clk            (clk),
        .reset          (reset),

        // Input request
        .core_req_valid (mem_req_valid),
        .core_req_rw    (1'b0),
        .core_req_mask  (mem_req_mask),
        .core_req_byteen(mem_req_byteen),
        .core_req_addr  (mem_req_addr),
        .core_req_user  ('0),
        .core_req_data  ('0),
        .core_req_tag   (mem_req_tag),
        .core_req_ready (mem_req_ready),
        `UNUSED_PIN (core_req_ready_rd),
        `UNUSED_PIN (core_req_ready_wr),
        `UNUSED_PIN (req_queue_empty),
        `UNUSED_PIN (req_queue_rw_notify),

        // Output response
        .core_rsp_valid (mem_rsp_valid),
        `UNUSED_PIN (core_rsp_mask),
        .core_rsp_data  (mem_rsp_data),
        .core_rsp_tag   (mem_rsp_tag),
        `UNUSED_PIN (core_rsp_sop),
        `UNUSED_PIN (core_rsp_eop),
        .core_rsp_ready (mem_rsp_ready),

        // Memory request
        .mem_req_valid  (lsu_mem_if.req_valid),
        .mem_req_rw     (lsu_mem_if.req_data.rw),
        .mem_req_mask   (lsu_mem_if.req_data.mask),
        .mem_req_byteen (lsu_mem_if.req_data.byteen),
        .mem_req_addr   (lsu_mem_if.req_data.addr),
        `UNUSED_PIN (mem_req_user),
        .mem_req_data   (lsu_mem_if.req_data.data),
        .mem_req_tag    (lsu_mem_if.req_data.tag),
        .mem_req_ready  (lsu_mem_if.req_ready),

        // Memory response
        .mem_rsp_valid  (lsu_mem_if.rsp_valid),
        .mem_rsp_mask   (lsu_mem_if.rsp_data.mask),
        .mem_rsp_data   (lsu_mem_if.rsp_data.data),
        .mem_rsp_tag    (lsu_mem_if.rsp_data.tag),
        .mem_rsp_ready  (lsu_mem_if.rsp_ready)
    );

    // Tex never sets any memory attr; tie off the scheduler-driven LSU bus.
    assign lsu_mem_if.req_data.user = '0;

    VX_lsu_adapter #(
        .NUM_LANES    (TCACHE_NUM_REQS),
        .DATA_SIZE    (4),
        .TAG_WIDTH    (TCACHE_TAG_WIDTH),
        .TAG_SEL_BITS (TCACHE_TAG_WIDTH - UUID_WIDTH),
        .REQ_OUT_BUF  (0),
        .RSP_OUT_BUF  (0)
    ) lsu_adapter (
        .clk        (clk),
        .reset      (reset),
        .lsu_mem_if (lsu_mem_if),
        .mem_bus_if (cache_bus_if)
    );

    // handle memory response

    wire [REQ_TAGW-1:0]           rsp_tag_s;
    wire [TEX_FILTER_BITS-1:0]   rsp_filter;
    wire [`TEX_LGSTRIDE_BITS-1:0] rsp_lgstride;
    wire [NUM_TAPS-1:0][NUM_LANES-1:0][1:0] mem_rsp_align;
    wire [NUM_TAPS-1:0]           mem_rsp_dups;

    assign {rsp_tag_s, rsp_filter, rsp_lgstride, mem_rsp_align, mem_rsp_dups} = mem_rsp_tag;

    wire [NUM_LANES-1:0][TEX_NUM_LEVELS-1:0][3:0][31:0] mem_rsp_data_qual;

    wire rsp_mip_linear = rsp_filter[TEX_FILTER_BITS-1];

    // A tap the request masked off carries no data, so it reads the tap that is
    // always fetched in its place: corner 0 of its level under point filtering,
    // and level 0 when the levels are not blended. The weights that select
    // between them downstream are zero in exactly those cases, so which tap the
    // fallback lands on only has to be a defined value. A source tap all lanes
    // shared was fetched once, by lane 0. The selection is the same for every
    // lane, so it is made once here.
    wire [NUM_TAPS-1:0][TAP_SEL_BITS-1:0] src_tap;
    wire [NUM_TAPS-1:0]                   src_dup;
    wire [NUM_TAPS-1:0][31:0]             lane0_data;
    wire [NUM_TAPS-1:0][1:0]              lane0_align;
    for (genvar k = 0; k < TEX_NUM_LEVELS; ++k) begin : g_lane0
        for (genvar j = 0; j < 4; ++j) begin : g_j
            assign src_tap[k*4+j] = TAP_SEL_BITS'(
                (rsp_mip_linear ? (k * 4) : 0) + (rsp_filter[0] ? j : 0));
            assign src_dup[k*4+j]     = mem_rsp_dups[src_tap[k*4+j]];
            assign lane0_data[k*4+j]  = mem_rsp_data[src_tap[k*4+j]][0];
            assign lane0_align[k*4+j] = mem_rsp_align[src_tap[k*4+j]][0];
        end
    end

    // Lane 0 always takes the shared word, which is its own fetch.
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_mem_rsp_data_qual
        wire [NUM_TAPS-1:0][31:0] lane_data;
        wire [NUM_TAPS-1:0][1:0]  lane_align;
        for (genvar t = 0; t < NUM_TAPS; ++t) begin : g_t
            assign lane_data[t]  = mem_rsp_data[t][i];
            assign lane_align[t] = mem_rsp_align[t][i];
        end

        VX_tex_rsp_qual #(
            .NUM_TAPS     (NUM_TAPS),
            .TAP_SEL_BITS (TAP_SEL_BITS)
        ) tex_rsp_qual (
            .lgstride     (rsp_lgstride),
            .src_tap      (src_tap),
            .take_shared  ((i == 0) ? {NUM_TAPS{1'b1}} : src_dup),
            .shared_data  (lane0_data),
            .shared_align (lane0_align),
            .lane_data    (lane_data),
            .lane_align   (lane_align),
            .data         (mem_rsp_data_qual[i])
        );
    end

    VX_pipe_buffer #(
        .DATAW (REQ_TAGW + (NUM_TAPS * NUM_LANES * 32))
    ) rsp_pipe_reg (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (mem_rsp_valid),
        .ready_in  (mem_rsp_ready),
        .data_in   ({rsp_tag_s, mem_rsp_data_qual}),
        .data_out  ({rsp_tag,   rsp_data}),
        .valid_out (rsp_valid),
        .ready_out (rsp_ready)
    );

`ifdef DBG_TRACE_TEX
    // The trace macros reach two dimensions, so a level's taps are dumped as
    // their own array rather than one three-dimensional view.
    wire [NUM_LANES-1:0][3:0][31:0] trace_req_addr [TEX_NUM_LEVELS];
    wire [NUM_LANES-1:0][3:0][31:0] trace_rsp_data [TEX_NUM_LEVELS];
    wire [NUM_LANES-1:0][W_ADDR_BITS-1:0] trace_baseaddr [TEX_NUM_LEVELS];
    for (genvar k = 0; k < TEX_NUM_LEVELS; ++k) begin : g_trace
        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_i
            assign trace_req_addr[k][i] = req_addr[i][k];
            assign trace_rsp_data[k][i] = rsp_data[i][k];
            assign trace_baseaddr[k][i] = req_baseaddr[i][k];
        end
    end

    always @(posedge clk) begin
        if (req_valid && req_ready) begin
            `TRACE(2, ("%d: %s-mem-req: valid=%b, filter=%0d, lgstride=%0d, baseaddr=", $time, INSTANCE_ID, req_mask, req_filter, req_lgstride))
            `TRACE_ARRAY1D(2, "0x%0h", trace_baseaddr[0], NUM_LANES)
            `TRACE(2, (", baseaddr_up="))
            `TRACE_ARRAY1D(2, "0x%0h", trace_baseaddr[1], NUM_LANES)
            `TRACE(2, (", addr="))
            `TRACE_ARRAY2D(2, "0x%0h", trace_req_addr[0], 4, NUM_LANES)
            `TRACE(2, (", addr_up="))
            `TRACE_ARRAY2D(2, "0x%0h", trace_req_addr[1], 4, NUM_LANES)
            `TRACE(2, (" (#%0d)\n", req_tag[REQ_TAGW-1 -: UUID_WIDTH]))
        end
        if (rsp_valid && rsp_ready) begin
            `TRACE(2, ("%d: %s-mem-rsp: data=", $time, INSTANCE_ID))
            `TRACE_ARRAY2D(2, "0x%0h", trace_rsp_data[0], 4, NUM_LANES)
            `TRACE(2, (", data_up="))
            `TRACE_ARRAY2D(2, "0x%0h", trace_rsp_data[1], 4, NUM_LANES)
            `TRACE(2, (" (#%0d)\n", rsp_tag[REQ_TAGW-1 -: UUID_WIDTH]))
        end
    end
`endif

endmodule
