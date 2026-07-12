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

module VX_sfu_unit import VX_gpu_pkg::*;
`ifdef VX_CFG_EXT_RASTER_ENABLE
import VX_raster_pkg::*;
`endif
#(
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_ID = 0
) (
    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    input sysmem_perf_t     sysmem_perf,
    input pipeline_perf_t   pipeline_perf,
`endif

    // Inputs
    VX_dispatch_if.slave    dispatch_if [`VX_CFG_ISSUE_WIDTH],

`ifdef VX_CFG_EXT_F_ENABLE
    VX_fpu_csr_if.slave     fpu_csr_if [`VX_CFG_NUM_FPU_BLOCKS],
`endif

`ifdef VX_CFG_EXT_DXA_ENABLE
    VX_dxa_req_bus_if.master dxa_req_bus_if,
    VX_txbar_bus_if.slave   dxa_txbar_bus_if,
`endif

`ifdef VX_CFG_EXT_TEX_ENABLE
    VX_tex_bus_if.master    tex_bus_if,
`endif

`ifdef VX_CFG_EXT_OM_ENABLE
`endif

    // RASTER has no SFU port at all now: the fragment shader is launched by the
    // raster engine (push, not pull) and its stamp arrives in the launch, so
    // nothing about a fragment passes through this unit.

`ifdef VX_CFG_EXT_RTU_ENABLE
    VX_rtu_bus_if.master    rtu_bus_if,
    VX_sched_unlock_if.master sched_unlock_if,
`endif

    VX_sched_csr_if.slave   sched_csr_if,

    VX_dcr_csr_if           dcr_csr_if,

    // Outputs
    VX_commit_if.master     commit_if [`VX_CFG_ISSUE_WIDTH],
    VX_warp_ctl_if.master   warp_ctl_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    localparam BLOCK_SIZE   = 1;
    localparam NUM_LANES    = `VX_CFG_NUM_SFU_LANES;
    localparam PE_COUNT     = 2 + `VX_CFG_EXT_DXA_ENABLED + `VX_CFG_EXT_TEX_ENABLED + `VX_CFG_EXT_OM_ENABLED + `VX_CFG_EXT_RASTER_ENABLED + `EXT_GFX_ANY_ENABLED;
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
`ifdef EXT_GFX_ANY_ENABLE
    localparam PE_IDX_GFXW  = 2 + `VX_CFG_EXT_DXA_ENABLED + `VX_CFG_EXT_TEX_ENABLED + `VX_CFG_EXT_OM_ENABLED + `VX_CFG_EXT_RASTER_ENABLED;
`endif

    VX_execute_if #(
        .data_t (sfu_execute_t)
    ) per_block_execute_if[BLOCK_SIZE]();

    VX_result_if #(
        .data_t (sfu_result_t)
    ) per_block_result_if[BLOCK_SIZE]();

    VX_lane_dispatch #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_BUF    (3)
    ) lane_dispatch (
        .clk        (clk),
        .reset      (reset),
        .dispatch_if(dispatch_if),
        .execute_if (per_block_execute_if)
    );

    VX_execute_if #(
        .data_t (sfu_execute_t)
    ) pe_execute_if[PE_COUNT]();

    VX_result_if#(
        .data_t (sfu_result_t)
    ) pe_result_if[PE_COUNT]();

    reg [PE_SEL_BITS-1:0] pe_select;
    always @(*) begin
        pe_select = PE_SEL_BITS'(PE_IDX_WCTL);
        if (inst_sfu_is_csr(per_block_execute_if[0].data.op_type)) begin
            pe_select = PE_SEL_BITS'(PE_IDX_CSRS);
        end
    `ifdef VX_CFG_EXT_DXA_ENABLE
        if (per_block_execute_if[0].data.op_type == INST_SFU_DXA) begin
            pe_select = PE_SEL_BITS'(PE_IDX_DXA);
        end
    `endif
    `ifdef VX_CFG_EXT_TEX_ENABLE
        if (per_block_execute_if[0].data.op_type == INST_SFU_TEX) begin
            pe_select = PE_SEL_BITS'(PE_IDX_TEX);
        end
    `endif
    `ifdef VX_CFG_EXT_OM_ENABLE
        if (per_block_execute_if[0].data.op_type == INST_SFU_OM) begin
            pe_select = PE_SEL_BITS'(PE_IDX_OM);
        end
    `endif
    `ifdef VX_CFG_EXT_RASTER_ENABLE
        if (per_block_execute_if[0].data.op_type == INST_SFU_RASTER) begin
            pe_select = PE_SEL_BITS'(PE_IDX_RASTER);
        end
    `endif
    `ifdef EXT_GFX_ANY_ENABLE
        if (per_block_execute_if[0].data.op_type == INST_SFU_GFXW) begin
            pe_select = PE_SEL_BITS'(PE_IDX_GFXW);
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
        .clk        (clk),
        .reset      (reset),
        .pe_sel     (pe_select),
        .execute_in_if (per_block_execute_if[0]),
        .result_out_if (per_block_result_if[0]),
        .execute_out_if (pe_execute_if),
        .result_in_if (pe_result_if)
    );

    VX_txbar_bus_if txbar_bus_if();

    VX_wctl_unit #(
        .INSTANCE_ID (`SFORMATF(("%s-wctl", INSTANCE_ID))),
        .NUM_LANES (NUM_LANES)
    ) wctl_unit (
        .clk        (clk),
        .reset      (reset),
        .execute_if (pe_execute_if[PE_IDX_WCTL]),
        .warp_ctl_if(warp_ctl_if),
        .txbar_bus_if(txbar_bus_if),
        .result_if  (pe_result_if[PE_IDX_WCTL])
    );

    VX_csr_unit #(
        .INSTANCE_ID (`SFORMATF(("%s-csr", INSTANCE_ID))),
        .CORE_ID   (CORE_ID),
        .NUM_LANES (NUM_LANES)
    ) csr_unit (
        .clk            (clk),
        .reset          (reset),

        .execute_if     (pe_execute_if[PE_IDX_CSRS]),

    `ifdef PERF_ENABLE
        .sysmem_perf    (sysmem_perf),
        .pipeline_perf  (pipeline_perf),
    `endif

    `ifdef VX_CFG_EXT_F_ENABLE
        .fpu_csr_if     (fpu_csr_if),
    `endif

        .sched_csr_if   (sched_csr_if),
        .result_if      (pe_result_if[PE_IDX_CSRS]),
        .dcr_csr_if     (dcr_csr_if)
    );

`ifdef VX_CFG_EXT_DXA_ENABLE
    VX_dxa_unit #(
        .INSTANCE_ID (`SFORMATF(("%s-dxa", INSTANCE_ID))),
        .CORE_ID (CORE_ID)
    ) dxa_unit (
        .clk        (clk),
        .reset      (reset),
        .execute_if (pe_execute_if[PE_IDX_DXA]),
        .result_if  (pe_result_if[PE_IDX_DXA]),
        .dxa_req_bus_if (dxa_req_bus_if)
    );

    // The only txbar producer is the SMEM-completion path through
    // dxa_txbar_bus_if (DXA release); no arbitration is needed.
    assign txbar_bus_if.valid     = dxa_txbar_bus_if.valid;
    assign txbar_bus_if.data      = dxa_txbar_bus_if.data;
    assign dxa_txbar_bus_if.ready = txbar_bus_if.ready;
`else
    assign txbar_bus_if.valid = 1'b0;
    assign txbar_bus_if.data = 'x;
    `UNUSED_VAR (txbar_bus_if.ready)
`endif

// The graphics window has no FF consumers left. TEX was the last one — it spilled
// its 2x2 quad's eight (u,v) operands into window slots because no RISC-V encoding
// holds eight inputs, and that cost the window two full RAM mirrors. It now takes
// u/v/lod in registers (vx_tex, R4-type) and its shader computes the mip LOD with
// vx_tex_quad_lod(), so the window is down to a single mirror and its only tenant
// is the RTU.

`ifdef VX_CFG_EXT_TEX_ENABLE
    VX_tex_unit #(
        .INSTANCE_ID (`SFORMATF(("%s-tex", INSTANCE_ID))),
        .CORE_ID     (CORE_ID),
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
    // OM export (v2): a fragment leaves the shader as a STORE to the OM aperture
    // (vx_om_export -> VX_gfx_uops -> the LSU), so the SFU services no OM op. The
    // PE slot is retained for index stability and tied off, as RASTER's is.
    assign pe_execute_if[PE_IDX_OM].ready = 1'b1;
    assign pe_result_if[PE_IDX_OM].valid  = 1'b0;
    assign pe_result_if[PE_IDX_OM].data   = '0;
    `UNUSED_VAR (pe_execute_if[PE_IDX_OM].valid)
    `UNUSED_VAR (pe_execute_if[PE_IDX_OM].data)
`endif

`ifdef VX_CFG_EXT_RASTER_ENABLE
    // RASTER push (v2): the fragment payload reaches the warp at launch via the
    // core-level distributor's window seed (rast_win_wr_*), so the SFU services
    // no raster op. The PE slot is retained for index stability and tied off —
    // no kernel issues INST_SFU_RASTER under the push model.
    assign pe_execute_if[PE_IDX_RASTER].ready = 1'b1;
    assign pe_result_if[PE_IDX_RASTER].valid  = 1'b0;
    assign pe_result_if[PE_IDX_RASTER].data   = '0;
    `UNUSED_VAR (pe_execute_if[PE_IDX_RASTER].valid)
    `UNUSED_VAR (pe_execute_if[PE_IDX_RASTER].data)
    `UNUSED_VAR (pe_result_if[PE_IDX_RASTER].ready)
`endif

`ifdef EXT_GFX_ANY_ENABLE
    VX_gfx_window #(
        .INSTANCE_ID (`SFORMATF(("%s-gfxw", INSTANCE_ID))),
        .CORE_ID     (CORE_ID),
    `ifdef VX_CFG_EXT_RTU_ENABLE
        .RTU_TAG_WIDTH (RTU_REQ_TAG_WIDTH),
    `endif
        .NUM_LANES   (NUM_LANES)
    ) gfx_window (
        .clk        (clk),
        .reset      (reset),
        .execute_if (pe_execute_if[PE_IDX_GFXW]),
        .result_if  (pe_result_if[PE_IDX_GFXW])
    `ifdef VX_CFG_EXT_RTU_ENABLE
        ,
        .rtu_bus_if (rtu_bus_if),
        .sched_unlock_if (sched_unlock_if)
    `endif
    );
`endif

    VX_lane_gather #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_BUF    (3)
    ) lane_gather (
        .clk       (clk),
        .reset     (reset),
        .result_if (per_block_result_if),
        .commit_if (commit_if)
    );

endmodule
