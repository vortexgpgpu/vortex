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

`include "VX_cache_define.vh"

module VX_cache_bypass #(
    parameter NUM_REQS          = 1,
    parameter MEM_PORTS         = 1,
    parameter TAG_SEL_IDX       = 0,

    parameter CACHE_ENABLE      = 0,

    parameter WORD_SIZE         = 1,
    parameter LINE_SIZE         = 1,

    parameter CORE_ADDR_WIDTH   = 1,

    parameter CORE_TAG_WIDTH    = 1,

    parameter MEM_ADDR_WIDTH    = 1,
    parameter MEM_TAG_IN_WIDTH  = 1,

    parameter UUID_WIDTH        = 0,

    parameter CORE_OUT_BUF      = 0,
    parameter MEM_OUT_BUF       = 0
 ) (
    input wire clk,
    input wire reset,

    // Core request in
    VX_mem_bus_if.slave     core_bus_in_if [NUM_REQS],

    // Core request out
    VX_mem_bus_if.master    core_bus_out_if [NUM_REQS],

    // Memory request in
    VX_mem_bus_if.slave     mem_bus_in_if [MEM_PORTS],

    // Memory request out
    VX_mem_bus_if.master    mem_bus_out_if [MEM_PORTS]
);
    localparam DIRECT_PASSTHRU   = !CACHE_ENABLE && (`CS_WORD_SEL_BITS == 0) && (NUM_REQS == MEM_PORTS);
    localparam CORE_DATA_WIDTH   = WORD_SIZE * 8;
    localparam WORDS_PER_LINE    = LINE_SIZE / WORD_SIZE;
    localparam WSEL_BITS         = `CLOG2(WORDS_PER_LINE);

    localparam CORE_TAG_ID_WIDTH = CORE_TAG_WIDTH - UUID_WIDTH;
    localparam MEM_TAG_ID_WIDTH  = `CLOG2(`CDIV(NUM_REQS, MEM_PORTS)) + CORE_TAG_ID_WIDTH;
    localparam MEM_TAG_NC1_WIDTH = UUID_WIDTH + MEM_TAG_ID_WIDTH;
    localparam MEM_TAG_NC2_WIDTH = MEM_TAG_NC1_WIDTH + WSEL_BITS;
    localparam MEM_TAG_OUT_WIDTH = CACHE_ENABLE ? `MAX(MEM_TAG_IN_WIDTH, MEM_TAG_NC2_WIDTH) : MEM_TAG_NC2_WIDTH;

    `STATIC_ASSERT(0 == (`IO_BASE_ADDR % `MEM_BLOCK_SIZE), ("invalid parameter"))

    // hanlde non-cacheable core request switch ///////////////////////////////

    VX_mem_bus_if #(
        .DATA_SIZE (WORD_SIZE),
        .TAG_WIDTH (CORE_TAG_WIDTH)
    ) core_bus_nc_switch_if[(CACHE_ENABLE ? 2 : 1) * NUM_REQS]();

    wire [NUM_REQS-1:0] core_req_nc_sel;

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_core_req_is_nc
        if (CACHE_ENABLE) begin : g_cache
            assign core_req_nc_sel[i] = ~core_bus_in_if[i].req_data.flags[`MEM_REQ_FLAG_IO];
        end else begin : g_no_cache
            assign core_req_nc_sel[i] = 1'b0;
        end
    end

    VX_mem_switch #(
        .NUM_INPUTS  (NUM_REQS),
        .NUM_OUTPUTS ((CACHE_ENABLE ? 2 : 1) * NUM_REQS),
        .DATA_SIZE   (WORD_SIZE),
        .TAG_WIDTH   (CORE_TAG_WIDTH),
        .ARBITER     ("R"),
        .REQ_OUT_BUF (0),
        .RSP_OUT_BUF (DIRECT_PASSTHRU ? 0 : `TO_OUT_BUF_SIZE(CORE_OUT_BUF))
    ) core_bus_nc_switch (
        .clk       (clk),
        .reset     (reset),
        .bus_sel   (core_req_nc_sel),
        .bus_in_if (core_bus_in_if),
        .bus_out_if(core_bus_nc_switch_if)
    );

    VX_mem_bus_if #(
        .DATA_SIZE (WORD_SIZE),
        .TAG_WIDTH (CORE_TAG_WIDTH)
    ) core_bus_in_nc_if[NUM_REQS]();

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_core_bus_nc_switch_if

        assign core_bus_in_nc_if[i].req_valid = core_bus_nc_switch_if[0 * NUM_REQS + i].req_valid;
        assign core_bus_in_nc_if[i].req_data  = core_bus_nc_switch_if[0 * NUM_REQS + i].req_data;
        assign core_bus_nc_switch_if[0 * NUM_REQS + i].req_ready = core_bus_in_nc_if[i].req_ready;

        assign core_bus_nc_switch_if[0 * NUM_REQS + i].rsp_valid = core_bus_in_nc_if[i].rsp_valid;
        assign core_bus_nc_switch_if[0 * NUM_REQS + i].rsp_data  = core_bus_in_nc_if[i].rsp_data;
        assign core_bus_in_nc_if[i].rsp_ready = core_bus_nc_switch_if[0 * NUM_REQS + i].rsp_ready;

        if (CACHE_ENABLE) begin : g_cache
            assign core_bus_out_if[i].req_valid = core_bus_nc_switch_if[1 * NUM_REQS + i].req_valid;
            assign core_bus_out_if[i].req_data  = core_bus_nc_switch_if[1 * NUM_REQS + i].req_data;
            assign core_bus_nc_switch_if[1 * NUM_REQS + i].req_ready = core_bus_out_if[i].req_ready;

            assign core_bus_nc_switch_if[1 * NUM_REQS + i].rsp_valid = core_bus_out_if[i].rsp_valid;
            assign core_bus_nc_switch_if[1 * NUM_REQS + i].rsp_data  = core_bus_out_if[i].rsp_data;
            assign core_bus_out_if[i].rsp_ready = core_bus_nc_switch_if[1 * NUM_REQS + i].rsp_ready;
        end else begin : g_no_cache
            `INIT_VX_MEM_BUS_IF (core_bus_out_if[i])
        end
    end

    // handle memory requests /////////////////////////////////////////////////

    VX_mem_bus_if #(
        .DATA_SIZE (WORD_SIZE),
        .TAG_WIDTH (MEM_TAG_NC1_WIDTH)
    ) core_bus_nc_arb_if[MEM_PORTS]();

    VX_mem_arb #(
        .NUM_INPUTS (NUM_REQS),
        .NUM_OUTPUTS(MEM_PORTS),
        .DATA_SIZE  (WORD_SIZE),
        .TAG_WIDTH  (CORE_TAG_WIDTH),
        .TAG_SEL_IDX(TAG_SEL_IDX),
        .ARBITER    (CACHE_ENABLE ? "P" : "R"),
        .REQ_OUT_BUF(0),
        .RSP_OUT_BUF(0)
    ) core_bus_nc_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (core_bus_in_nc_if),
        .bus_out_if (core_bus_nc_arb_if)
    );

    VX_mem_bus_if #(
        .DATA_SIZE (LINE_SIZE),
        .TAG_WIDTH (MEM_TAG_NC2_WIDTH)
    ) mem_bus_out_nc_if[MEM_PORTS]();

    for (genvar i = 0; i < MEM_PORTS; ++i) begin : g_mem_bus_out_nc
        wire                        core_req_nc_arb_rw;
        wire [WORD_SIZE-1:0]        core_req_nc_arb_byteen;
        wire [CORE_ADDR_WIDTH-1:0]  core_req_nc_arb_addr;
        wire [`MEM_REQ_FLAGS_WIDTH-1:0] core_req_nc_arb_flags;
        wire [CORE_DATA_WIDTH-1:0]  core_req_nc_arb_data;
        wire [MEM_TAG_NC1_WIDTH-1:0] core_req_nc_arb_tag;

        assign {
            core_req_nc_arb_rw,
            core_req_nc_arb_addr,
            core_req_nc_arb_data,
            core_req_nc_arb_byteen,
            core_req_nc_arb_flags,
            core_req_nc_arb_tag
        } = core_bus_nc_arb_if[i].req_data;

        logic [MEM_ADDR_WIDTH-1:0] core_req_nc_arb_addr_w;
        logic [WORDS_PER_LINE-1:0][WORD_SIZE-1:0] core_req_nc_arb_byteen_w;
        logic [WORDS_PER_LINE-1:0][CORE_DATA_WIDTH-1:0] core_req_nc_arb_data_w;
        logic [CORE_DATA_WIDTH-1:0] core_rsp_nc_arb_data_w;
        wire [MEM_TAG_NC2_WIDTH-1:0] core_req_nc_arb_tag_w;
        wire [MEM_TAG_NC1_WIDTH-1:0] core_rsp_nc_arb_tag_w;

        if (WORDS_PER_LINE > 1) begin : g_multi_word_line
            wire [WSEL_BITS-1:0] rsp_wsel;
            wire [WSEL_BITS-1:0] req_wsel = core_req_nc_arb_addr[WSEL_BITS-1:0];
            always @(*) begin
                core_req_nc_arb_byteen_w = '0;
                core_req_nc_arb_byteen_w[req_wsel] = core_req_nc_arb_byteen;
                core_req_nc_arb_data_w = 'x;
                core_req_nc_arb_data_w[req_wsel] = core_req_nc_arb_data;
            end
            VX_bits_insert #(
                .N   (MEM_TAG_NC1_WIDTH),
                .S   (WSEL_BITS),
                .POS (TAG_SEL_IDX)
            ) wsel_insert (
                .data_in  (core_req_nc_arb_tag),
                .ins_in   (req_wsel),
                .data_out (core_req_nc_arb_tag_w)
            );
            VX_bits_remove #(
                .N   (MEM_TAG_NC2_WIDTH),
                .S   (WSEL_BITS),
                .POS (TAG_SEL_IDX)
            ) wsel_remove (
                .data_in  (mem_bus_out_nc_if[i].rsp_data.tag),
                .sel_out  (rsp_wsel),
                .data_out (core_rsp_nc_arb_tag_w)
            );
            assign core_req_nc_arb_addr_w   = core_req_nc_arb_addr[WSEL_BITS +: MEM_ADDR_WIDTH];
            assign core_rsp_nc_arb_data_w   = mem_bus_out_nc_if[i].rsp_data.data[rsp_wsel * CORE_DATA_WIDTH +: CORE_DATA_WIDTH];
        end else begin : g_single_word_line
            assign core_req_nc_arb_addr_w   = core_req_nc_arb_addr;
            assign core_req_nc_arb_byteen_w = core_req_nc_arb_byteen;
            assign core_req_nc_arb_data_w   = core_req_nc_arb_data;
            assign core_req_nc_arb_tag_w    = MEM_TAG_NC2_WIDTH'(core_req_nc_arb_tag);

            assign core_rsp_nc_arb_data_w   = mem_bus_out_nc_if[i].rsp_data.data;
            assign core_rsp_nc_arb_tag_w    = MEM_TAG_NC1_WIDTH'(mem_bus_out_nc_if[i].rsp_data.tag);
        end

        assign mem_bus_out_nc_if[i].req_valid = core_bus_nc_arb_if[i].req_valid;
        assign mem_bus_out_nc_if[i].req_data = {
            core_req_nc_arb_rw,
            core_req_nc_arb_addr_w,
            core_req_nc_arb_data_w,
            core_req_nc_arb_byteen_w,
            core_req_nc_arb_flags,
            core_req_nc_arb_tag_w
        };
        assign core_bus_nc_arb_if[i].req_ready = mem_bus_out_nc_if[i].req_ready;

        assign core_bus_nc_arb_if[i].rsp_valid = mem_bus_out_nc_if[i].rsp_valid;
        assign core_bus_nc_arb_if[i].rsp_data = {
            core_rsp_nc_arb_data_w,
            core_rsp_nc_arb_tag_w
        };
        assign mem_bus_out_nc_if[i].rsp_ready = core_bus_nc_arb_if[i].rsp_ready;
    end

    VX_mem_bus_if #(
        .DATA_SIZE (LINE_SIZE),
        .TAG_WIDTH (MEM_TAG_OUT_WIDTH)
    ) mem_bus_out_src_if[(CACHE_ENABLE ? 2 : 1) * MEM_PORTS]();

    for (genvar i = 0; i < MEM_PORTS; ++i) begin : g_mem_bus_out_src
        `ASSIGN_VX_MEM_BUS_IF_EX(mem_bus_out_src_if[0 * MEM_PORTS + i], mem_bus_out_nc_if[i], MEM_TAG_OUT_WIDTH, MEM_TAG_NC2_WIDTH, UUID_WIDTH);
        if (CACHE_ENABLE) begin : g_cache
            `ASSIGN_VX_MEM_BUS_IF_EX(mem_bus_out_src_if[1 * MEM_PORTS + i], mem_bus_in_if[i], MEM_TAG_OUT_WIDTH, MEM_TAG_IN_WIDTH, UUID_WIDTH);
        end else begin : g_no_cache
            `UNUSED_VX_MEM_BUS_IF(mem_bus_in_if[i])
        end
    end

    VX_mem_arb #(
        .NUM_INPUTS ((CACHE_ENABLE ? 2 : 1) * MEM_PORTS),
        .NUM_OUTPUTS(MEM_PORTS),
        .DATA_SIZE  (LINE_SIZE),
        .TAG_WIDTH  (MEM_TAG_OUT_WIDTH),
        .ARBITER    ("R"),
        .REQ_OUT_BUF(DIRECT_PASSTHRU ? 0 : `TO_OUT_BUF_SIZE(MEM_OUT_BUF)),
        .RSP_OUT_BUF(0)
    ) mem_bus_out_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (mem_bus_out_src_if),
        .bus_out_if (mem_bus_out_if)
    );

endmodule
