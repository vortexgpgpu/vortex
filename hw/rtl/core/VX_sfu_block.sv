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

`ifdef VX_CFG_EXT_F_ENABLE
`include "VX_fpu_define.vh"
`endif

// One SFU block: the PEs serving the warps with wid mod NUM_SFU_BLOCKS == block_idx.
module VX_sfu_block import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire              clk,
    input wire              reset,

    input wire [CORE_ID_WIDTH-1:0] core_id,
    input wire [SFU_BLOCK_W-1:0]   block_idx,

    VX_execute_if.slave     execute_if,
    VX_result_if.master     result_if,

`ifdef VX_CFG_EXT_F_ENABLE
    output wire [NW_WIDTH-1:0] fcsr_wid,
    input wire [INST_FRM_BITS+`FP_FLAGS_BITS-1:0]  fcsr_rd_data,
    output wire [2:0]       fcsr_wr_sel,
    output wire [INST_FRM_BITS+`FP_FLAGS_BITS-1:0] fcsr_wr_data,
`endif

`ifdef VX_CFG_EXT_DXA_ENABLE
    VX_dxa_req_bus_if.master dxa_req_bus_if,
`endif

`ifdef VX_CFG_EXT_TEX_ENABLE
    VX_tex_bus_if.master    tex_bus_if,
`endif

`ifdef VX_CFG_EXT_RTU_ENABLE
    VX_rtu_bus_if.master    rtu_bus_if,
    VX_sched_unlock_if.master sched_unlock_if,
`endif

    VX_sched_csr_if.slave   sched_csr_if,
    VX_warp_shared_if.master warp_shared_if,
    VX_warp_ctl_if.master   warp_ctl_if,

    // CTA dispatch's context writes
    input cta_bcast_t       cta_bcast
);
    localparam NUM_LANES    = `VX_CFG_NUM_SFU_LANES;
`ifndef VX_CFG_EXT_RTU_ENABLE
    `UNUSED_VAR (block_idx)
`endif
    localparam PE_COUNT     = 2 + `VX_CFG_EXT_DXA_ENABLED + `VX_CFG_EXT_TEX_ENABLED + `VX_CFG_EXT_OM_ENABLED + `VX_CFG_EXT_RASTER_ENABLED + `VX_CFG_EXT_RTU_ENABLED;
    localparam PE_SEL_BITS  = `CLOG2(PE_COUNT);
    localparam PE_IDX_WCTL  = 0;
    localparam PE_IDX_CSRS  = 1;
`ifdef VX_CFG_EXT_DXA_ENABLE
    localparam PE_IDX_DXA   = 2;
`endif
`ifdef VX_CFG_EXT_TEX_ENABLE
    localparam PE_IDX_TEX   = 2 + `VX_CFG_EXT_DXA_ENABLED;
`endif
`ifdef VX_CFG_EXT_OM_ENABLE
    localparam PE_IDX_OM    = 2 + `VX_CFG_EXT_DXA_ENABLED + `VX_CFG_EXT_TEX_ENABLED;
`endif
`ifdef VX_CFG_EXT_RASTER_ENABLE
    localparam PE_IDX_RASTER = 2 + `VX_CFG_EXT_DXA_ENABLED + `VX_CFG_EXT_TEX_ENABLED + `VX_CFG_EXT_OM_ENABLED;
`endif
`ifdef VX_CFG_EXT_RTU_ENABLE
    localparam PE_IDX_RTUW  = 2 + `VX_CFG_EXT_DXA_ENABLED + `VX_CFG_EXT_TEX_ENABLED + `VX_CFG_EXT_OM_ENABLED + `VX_CFG_EXT_RASTER_ENABLED;
`endif

    VX_execute_if #(
        .data_t (sfu_execute_t)
    ) pe_execute_if[PE_COUNT]();

    VX_result_if#(
        .data_t (sfu_result_t)
    ) pe_result_if[PE_COUNT]();

    reg [PE_SEL_BITS-1:0] pe_select;
    always @(*) begin
        pe_select = PE_SEL_BITS'(PE_IDX_WCTL);
        if (inst_sfu_is_csr(execute_if.data.op_type)) begin
            pe_select = PE_SEL_BITS'(PE_IDX_CSRS);
        end
    `ifdef VX_CFG_EXT_DXA_ENABLE
        if (execute_if.data.op_type == INST_SFU_DXA) begin
            pe_select = PE_SEL_BITS'(PE_IDX_DXA);
        end
    `endif
    `ifdef VX_CFG_EXT_TEX_ENABLE
        if (execute_if.data.op_type == INST_SFU_TEX) begin
            pe_select = PE_SEL_BITS'(PE_IDX_TEX);
        end
    `endif
    `ifdef VX_CFG_EXT_OM_ENABLE
        if (execute_if.data.op_type == INST_SFU_OM) begin
            pe_select = PE_SEL_BITS'(PE_IDX_OM);
        end
    `endif
    `ifdef VX_CFG_EXT_RASTER_ENABLE
        if (execute_if.data.op_type == INST_SFU_RASTER) begin
            pe_select = PE_SEL_BITS'(PE_IDX_RASTER);
        end
    `endif
    `ifdef VX_CFG_EXT_RTU_ENABLE
        if (execute_if.data.op_type == INST_SFU_RTUW) begin
            pe_select = PE_SEL_BITS'(PE_IDX_RTUW);
        end
    `endif
    end

    VX_pe_switch #(
        .PE_COUNT   (PE_COUNT),
        .NUM_LANES  (NUM_LANES),
        .ARBITER    ("R"),
        .REQ_OUT_BUF(0),
        .RSP_OUT_BUF(3)
    ) pe_switch (
        .clk            (clk),
        .reset          (reset),
        .pe_sel         (pe_select),
        .execute_in_if  (execute_if),
        .result_out_if  (result_if),
        .execute_out_if (pe_execute_if),
        .result_in_if   (pe_result_if)
    );

    VX_wctl_unit #(
        .INSTANCE_ID (`SFORMATF(("%s-wctl", INSTANCE_ID))),
        .NUM_LANES   (NUM_LANES)
    ) wctl_unit (
        .clk            (clk),
        .reset          (reset),
        .execute_if     (pe_execute_if[PE_IDX_WCTL]),
        .warp_shared_if (warp_shared_if),
        .warp_ctl_if    (warp_ctl_if),
        .result_if      (pe_result_if[PE_IDX_WCTL])
    );

    // The block's copy of the CTA contexts, read at the CSR request's CTA and
    // valid a cycle later.
    cta_ctx_t cta_ctx;

    VX_dp_ram #(
        .DATAW     ($bits(cta_ctx_t)),
        .SIZE      (NUM_CTA_MAX),
        .RDW_MODE  ("R"),
        .OUT_REG   (1)
    ) cta_ctx_ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (cta_bcast.ctx_write),
        .wren  (1'b1),
        .waddr (cta_bcast.ctx_waddr),
        .wdata (cta_bcast.ctx_wdata),
        .raddr (pe_execute_if[PE_IDX_CSRS].data.header.cta_id),
        .rdata (cta_ctx)
    );

    VX_csr_unit #(
        .INSTANCE_ID (`SFORMATF(("%s-csr", INSTANCE_ID))),
        .NUM_LANES (NUM_LANES)
    ) csr_unit (
        .clk            (clk),
        .reset          (reset),
        .core_id        (core_id),
        .execute_if     (pe_execute_if[PE_IDX_CSRS]),
    `ifdef VX_CFG_EXT_F_ENABLE
        .fcsr_wid       (fcsr_wid),
        .fcsr_rd_data   (fcsr_rd_data),
        .fcsr_wr_sel    (fcsr_wr_sel),
        .fcsr_wr_data   (fcsr_wr_data),
    `endif
        .sched_csr_if   (sched_csr_if),
        .cta_ctx        (cta_ctx),
        .result_if      (pe_result_if[PE_IDX_CSRS])
    );

`ifdef VX_CFG_EXT_DXA_ENABLE
    VX_dxa_unit #(
        .INSTANCE_ID (`SFORMATF(("%s-dxa", INSTANCE_ID)))
    ) dxa_unit (
        .clk        (clk),
        .reset      (reset),
        .core_id    (core_id),
        .execute_if (pe_execute_if[PE_IDX_DXA]),
        .result_if  (pe_result_if[PE_IDX_DXA]),
        .dxa_req_bus_if (dxa_req_bus_if)
    );
`endif

`ifdef VX_CFG_EXT_TEX_ENABLE
    VX_tex_unit #(
        .INSTANCE_ID (`SFORMATF(("%s-tex", INSTANCE_ID))),
        .NUM_LANES   (NUM_LANES)
    ) tex_unit (
        .clk        (clk),
        .reset      (reset),
        .execute_if (pe_execute_if[PE_IDX_TEX]),
        .result_if  (pe_result_if[PE_IDX_TEX]),
        .tex_bus_if (tex_bus_if)
    );
`endif

`ifdef VX_CFG_EXT_OM_ENABLE
    // A fragment leaves the shader as a store to the OM aperture through the LSU,
    // so the SFU services no OM op; the PE slot is kept for index stability.
    assign pe_execute_if[PE_IDX_OM].ready = 1'b1;
    assign pe_result_if[PE_IDX_OM].valid  = 1'b0;
    assign pe_result_if[PE_IDX_OM].data   = '0;
    `UNUSED_VAR (pe_execute_if[PE_IDX_OM].valid)
    `UNUSED_VAR (pe_execute_if[PE_IDX_OM].data)
`endif

`ifdef VX_CFG_EXT_RASTER_ENABLE
    // The fragment payload reaches the warp at launch, so the SFU services no
    // raster op; the PE slot is kept for index stability.
    assign pe_execute_if[PE_IDX_RASTER].ready = 1'b1;
    assign pe_result_if[PE_IDX_RASTER].valid  = 1'b0;
    assign pe_result_if[PE_IDX_RASTER].data   = '0;
    `UNUSED_VAR (pe_execute_if[PE_IDX_RASTER].valid)
    `UNUSED_VAR (pe_execute_if[PE_IDX_RASTER].data)
    `UNUSED_VAR (pe_result_if[PE_IDX_RASTER].ready)
`endif

`ifdef VX_CFG_EXT_RTU_ENABLE
    VX_rtu_unit #(
        .INSTANCE_ID   (`SFORMATF(("%s-rtuw", INSTANCE_ID))),
        .RTU_TAG_WIDTH (RTU_REQ_TAG_WIDTH),
        .NUM_LANES     (NUM_LANES)
    ) rtu_unit (
        .clk        (clk),
        .reset      (reset),
        .block_idx  (block_idx),
        .execute_if (pe_execute_if[PE_IDX_RTUW]),
        .result_if  (pe_result_if[PE_IDX_RTUW]),
        .rtu_bus_if (rtu_bus_if),
        .sched_unlock_if (sched_unlock_if)
    );
`endif

endmodule
