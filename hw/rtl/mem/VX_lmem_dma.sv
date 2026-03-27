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

// DMA mux for local memory.
// Routes DXA bank-parallel writes and/or TCU tile-buffer reads to the
// single bank-parallel DMA port of VX_local_mem.  Also instantiates the
// DXA completion detector when EXT_DXA_ENABLE is set.

module VX_lmem_dma import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input  wire  clk,
    input  wire  reset,

`ifdef EXT_DXA_ENABLE
    VX_dxa_bank_wr_if.slave  dxa_bank_wr_if,
    VX_txbar_bus_if.master   dxa_txbar_bus_if,
`endif

`ifdef TCU_WGMMA_ENABLE
    VX_tcu_lmem_if.slave     tcu_lmem_if,
`endif

    VX_mem_bus_if.master     lmem_dma_if
);

`ifdef EXT_DXA_ENABLE
    `ifdef TCU_WGMMA_ENABLE
    // -------------------------------------------------------------------
    // DXA + TCU: DXA writes take priority; TCU reads get the bus when DXA idle.
    // -------------------------------------------------------------------
    wire dxa_grant = dxa_bank_wr_if.wr_valid;

    assign lmem_dma_if.req_valid       = dxa_bank_wr_if.wr_valid || tcu_lmem_if.req_valid;
    assign lmem_dma_if.req_data.rw     = dxa_grant;
    assign lmem_dma_if.req_data.addr   = dxa_grant ? dxa_bank_wr_if.wr_addr : tcu_lmem_if.req_data.addr;
    assign lmem_dma_if.req_data.data   = dxa_grant ? dxa_bank_wr_if.wr_data : '0;
    assign lmem_dma_if.req_data.byteen = dxa_grant ? dxa_bank_wr_if.wr_byteen : '0;
    assign lmem_dma_if.req_data.flags  = '0;
    assign lmem_dma_if.req_data.tag    = dxa_grant ? dxa_bank_wr_if.wr_tag : '0;

    assign dxa_bank_wr_if.wr_ready = lmem_dma_if.req_ready && dxa_grant;
    assign tcu_lmem_if.req_ready   = lmem_dma_if.req_ready && !dxa_grant;

    // Track whether the last accepted DMA request was a write (DXA) or read (TCU)
    // to gate the TCU data_valid response.
    logic dma_rsp_is_write_r;
    always_ff @(posedge clk) begin
        if (reset)
            dma_rsp_is_write_r <= 1'b1;
        else if (lmem_dma_if.req_valid && lmem_dma_if.req_ready)
            dma_rsp_is_write_r <= lmem_dma_if.req_data.rw;
    end

    assign tcu_lmem_if.rsp_data.data = lmem_dma_if.rsp_data.data;
    assign tcu_lmem_if.rsp_valid     = lmem_dma_if.rsp_valid && !dma_rsp_is_write_r;
    assign lmem_dma_if.rsp_ready  = 1'b1;
    `UNUSED_VAR (lmem_dma_if.rsp_data.tag)

    `else
    // -------------------------------------------------------------------
    // DXA only
    // -------------------------------------------------------------------
    assign lmem_dma_if.req_valid       = dxa_bank_wr_if.wr_valid;
    assign lmem_dma_if.req_data.rw     = 1'b1;
    assign lmem_dma_if.req_data.addr   = dxa_bank_wr_if.wr_addr;
    assign lmem_dma_if.req_data.data   = dxa_bank_wr_if.wr_data;
    assign lmem_dma_if.req_data.byteen = dxa_bank_wr_if.wr_byteen;
    assign lmem_dma_if.req_data.flags  = '0;
    assign lmem_dma_if.req_data.tag    = dxa_bank_wr_if.wr_tag;
    assign dxa_bank_wr_if.wr_ready     = lmem_dma_if.req_ready;
    `UNUSED_VAR (lmem_dma_if.rsp_valid)
    `UNUSED_VAR (lmem_dma_if.rsp_data)
    assign lmem_dma_if.rsp_ready = 1'b0;
    `endif

    wire [`LMEM_NUM_BANKS-1:0] dxa_bank_wr_fire;
    for (genvar i = 0; i < `LMEM_NUM_BANKS; ++i) begin : g_dxa_bank_wr_fire
        assign dxa_bank_wr_fire[i] = lmem_dma_if.req_valid
                                  && lmem_dma_if.req_data.rw
                                  && (|lmem_dma_if.req_data.byteen[i*LSU_WORD_SIZE +: LSU_WORD_SIZE]);
    end

    VX_dxa_completion_detect #(
        .INSTANCE_ID (`SFORMATF(("%s-compl_det", INSTANCE_ID))),
        .NUM_BANKS   (`LMEM_NUM_BANKS),
        .TAG_WIDTH   (DXA_BANK_WR_TAG_WIDTH)
    ) dxa_completion_detect (
        .clk          (clk),
        .reset        (reset),
        .bank_wr_fire (dxa_bank_wr_fire),
        .bank_wr_tag  (DXA_BANK_WR_TAG_WIDTH'(lmem_dma_if.req_data.tag)),
        .txbar_bus_if (dxa_txbar_bus_if)
    );

    `ifdef DBG_TRACE_DXA
    always @(posedge clk) begin
        if (dxa_txbar_bus_if.valid && !reset) begin
            `TRACE(2, ("%t: %s: dxa_txbar valid=1 bar_addr=0x%0h ready=%b\n",
                $time, INSTANCE_ID, dxa_txbar_bus_if.data.addr, dxa_txbar_bus_if.ready))
        end
    end
    `endif

`elsif TCU_WGMMA_ENABLE
    // -------------------------------------------------------------------
    // TCU only
    // -------------------------------------------------------------------
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)
    assign lmem_dma_if.req_valid       = tcu_lmem_if.req_valid;
    assign lmem_dma_if.req_data.rw     = 1'b0;
    assign lmem_dma_if.req_data.addr   = tcu_lmem_if.req_data.addr;
    assign lmem_dma_if.req_data.data   = '0;
    assign lmem_dma_if.req_data.byteen = '0;
    assign lmem_dma_if.req_data.flags  = '0;
    assign lmem_dma_if.req_data.tag    = '0;
    assign tcu_lmem_if.req_ready       = lmem_dma_if.req_ready;
    assign tcu_lmem_if.rsp_data.data   = lmem_dma_if.rsp_data.data;
    assign tcu_lmem_if.rsp_valid       = lmem_dma_if.rsp_valid;
    assign lmem_dma_if.rsp_ready       = 1'b1;
    `UNUSED_VAR (lmem_dma_if.rsp_data.tag)

`else
    // -------------------------------------------------------------------
    // Neither DXA nor TCU: tie DMA port idle
    // -------------------------------------------------------------------
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)
    assign lmem_dma_if.req_valid = 1'b0;
    assign lmem_dma_if.req_data  = '0;
    assign lmem_dma_if.rsp_ready = 1'b0;
    `UNUSED_VAR (lmem_dma_if.req_ready)
    `UNUSED_VAR (lmem_dma_if.rsp_valid)
    `UNUSED_VAR (lmem_dma_if.rsp_data)
`endif

endmodule
