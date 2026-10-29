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

// Synthesis wrapper of the warp-wide local memory at the configured core
// shape: one warp request port per LSU block, and the row-wide DMA port. The
// requests arrive through a register each, as from the LSU blocks' local-path
// buffers and the DMA arbiter in the core; out-of-context synthesis leaves
// paths from the ports untimed, so this is what puts the unit's request
// decoding on the timing report.
module VX_lmem_unit_top import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter SIZE         = (1 << `VX_CFG_LMEM_LOG_SIZE),
    parameter NUM_BLOCKS   = `VX_CFG_NUM_LSU_BLOCKS,
    parameter NUM_LANES    = `VX_CFG_NUM_LSU_LANES,
    parameter NUM_BANKS    = `VX_CFG_LMEM_NUM_BANKS,
    parameter WORD_SIZE    = LSU_WORD_SIZE,
    parameter TAG_WIDTH    = LSU_TAG_WIDTH,
    parameter ADDR_WIDTH   = `CLOG2(SIZE / WORD_SIZE),
    parameter WORD_WIDTH   = WORD_SIZE * 8,
    parameter DMA_ADDR_WIDTH = ADDR_WIDTH - `CLOG2(NUM_BANKS),
    parameter DMA_DATA_SIZE  = NUM_BANKS * WORD_SIZE,
    parameter DMA_TAG_WIDTH  = LMEM_DMA_TAG_WIDTH
) (
    input wire clk,
    input wire reset,

    // warp requests
    input  wire [NUM_BLOCKS-1:0]                                   req_valid,
    input  wire [NUM_BLOCKS-1:0]                                   req_rw,
    input  wire [NUM_BLOCKS-1:0][NUM_LANES-1:0]                    req_mask,
    input  wire [NUM_BLOCKS-1:0][NUM_LANES-1:0][WORD_SIZE-1:0]     req_byteen,
    input  wire [NUM_BLOCKS-1:0][NUM_LANES-1:0][ADDR_WIDTH-1:0]    req_addr,
    input  wire [NUM_BLOCKS-1:0][NUM_LANES-1:0][MEM_ATTR_WIDTH-1:0] req_attr,
    input  wire [NUM_BLOCKS-1:0][NUM_LANES-1:0][WORD_WIDTH-1:0]    req_data,
    input  wire [NUM_BLOCKS-1:0][TAG_WIDTH-1:0]                    req_tag,
    output wire [NUM_BLOCKS-1:0]                                   req_ready,

    // responses
    output wire [NUM_BLOCKS-1:0]                                   rsp_valid,
    output wire [NUM_BLOCKS-1:0][NUM_LANES-1:0]                    rsp_mask,
    output wire [NUM_BLOCKS-1:0][NUM_LANES-1:0][WORD_WIDTH-1:0]    rsp_data,
    output wire [NUM_BLOCKS-1:0][TAG_WIDTH-1:0]                    rsp_tag,
    input  wire [NUM_BLOCKS-1:0]                                   rsp_ready,

    // DMA requests
    input  wire                                                    dma_req_valid,
    input  wire                                                    dma_req_rw,
    input  wire [DMA_ADDR_WIDTH-1:0]                               dma_req_addr,
    input  wire [DMA_DATA_SIZE*8-1:0]                              dma_req_data,
    input  wire [DMA_DATA_SIZE-1:0]                                dma_req_byteen,
    input  wire [DMA_TAG_WIDTH-1:0]                                dma_req_tag,
    output wire                                                    dma_req_ready,

    // DMA responses
    output wire                                                    dma_rsp_valid,
    output wire [DMA_DATA_SIZE*8-1:0]                              dma_rsp_data,
    output wire [DMA_TAG_WIDTH-1:0]                                dma_rsp_tag,
    input  wire                                                    dma_rsp_ready
);
    localparam REQ_DATAW = NUM_LANES + 1 + NUM_LANES * (LSU_ADDR_WIDTH + WORD_WIDTH + WORD_SIZE + MEM_ATTR_WIDTH) + TAG_WIDTH;
    localparam DMA_REQ_DATAW = 1 + DMA_ADDR_WIDTH + DMA_DATA_SIZE * 8 + DMA_DATA_SIZE + MEM_ATTR_WIDTH + DMA_TAG_WIDTH;

    VX_lsu_mem_if #(
        .NUM_LANES (NUM_LANES),
        .DATA_SIZE (WORD_SIZE),
        .TAG_WIDTH (TAG_WIDTH)
    ) lsu_mem_if[NUM_BLOCKS]();

    VX_mem_bus_if #(
        .DATA_SIZE  (NUM_BANKS * WORD_SIZE),
        .TAG_WIDTH  (DMA_TAG_WIDTH),
        .ADDR_WIDTH (DMA_ADDR_WIDTH)
    ) dma_bus_if();

    for (genvar k = 0; k < NUM_BLOCKS; ++k) begin : g_blocks
        wire [NUM_LANES-1:0][LSU_ADDR_WIDTH-1:0] lsu_addr;
        for (genvar i = 0; i < NUM_LANES; ++i) begin : g_addr
            assign lsu_addr[i] = LSU_ADDR_WIDTH'(req_addr[k][i]);
        end

        VX_elastic_buffer #(
            .DATAW   (REQ_DATAW),
            .SIZE    (1)
        ) req_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (req_valid[k]),
            .ready_in  (req_ready[k]),
            .data_in   ({req_mask[k], req_rw[k], lsu_addr, req_data[k], req_byteen[k], req_attr[k], req_tag[k]}),
            .data_out  (lsu_mem_if[k].req_data),
            .valid_out (lsu_mem_if[k].req_valid),
            .ready_out (lsu_mem_if[k].req_ready)
        );

        assign rsp_valid[k] = lsu_mem_if[k].rsp_valid;
        assign rsp_mask[k]  = lsu_mem_if[k].rsp_data.mask;
        assign rsp_data[k]  = lsu_mem_if[k].rsp_data.data;
        assign rsp_tag[k]   = lsu_mem_if[k].rsp_data.tag;
        assign lsu_mem_if[k].rsp_ready = rsp_ready[k];
    end

    VX_elastic_buffer #(
        .DATAW   (DMA_REQ_DATAW),
        .SIZE    (1)
    ) dma_req_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (dma_req_valid),
        .ready_in  (dma_req_ready),
        .data_in   ({dma_req_rw, dma_req_addr, dma_req_data, dma_req_byteen, MEM_ATTR_WIDTH'(0), dma_req_tag}),
        .data_out  (dma_bus_if.req_data),
        .valid_out (dma_bus_if.req_valid),
        .ready_out (dma_bus_if.req_ready)
    );

    assign dma_rsp_valid              = dma_bus_if.rsp_valid;
    assign dma_rsp_data               = dma_bus_if.rsp_data.data;
    assign dma_rsp_tag                = dma_bus_if.rsp_data.tag;
    assign dma_bus_if.rsp_ready       = dma_rsp_ready;

    VX_lmem_unit #(
        .INSTANCE_ID (INSTANCE_ID),
        .SIZE        (SIZE),
        .NUM_BLOCKS  (NUM_BLOCKS),
        .NUM_LANES   (NUM_LANES),
        .NUM_BANKS   (NUM_BANKS),
        .WORD_SIZE   (WORD_SIZE),
        .ADDR_WIDTH  (ADDR_WIDTH),
        .TAG_WIDTH   (TAG_WIDTH),
        .DMA_ENABLE  (1),
        .DMA_TAG_WIDTH (DMA_TAG_WIDTH),
        .AMO_ENABLE  (`VX_CFG_EXT_A_ENABLED)
    ) lmem_unit (
        .clk        (clk),
        .reset      (reset),
    `ifdef PERF_ENABLE
        `UNUSED_PIN (lmem_perf),
    `endif
        .lsu_mem_if (lsu_mem_if),
        .dma_bus_if (dma_bus_if)
    );

endmodule
