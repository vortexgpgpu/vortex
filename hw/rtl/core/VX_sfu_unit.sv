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
    VX_om_bus_if.master     om_bus_if,
`endif

`ifdef VX_CFG_EXT_RASTER_ENABLE
    VX_raster_bus_if.slave  raster_bus_if,
`endif

`ifdef VX_CFG_EXT_RTU_ENABLE
    VX_rtu_bus_if.master    rtu_bus_if,
    VX_async_trap_if.master async_trap_if,
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

`ifdef EXT_GFX_ANY_ENABLE
    // Shared-window FF-consumer wires: the TEX/OM datapath PEs read their input
    // payload from VX_gfx_window's slot RF (8 read ports: vx_tex4 single uses
    // 0-1 (u,v) and quad uses 0-7 (u[0..3],v[0..3]); vx_om4 uses 0-7
    // (colour[0..3],depth[0..3])), and TEX additionally writes its texel back.
    // The SFU issues one op to one PE per cycle (VX_pe_switch demux + the
    // multi-cycle macro-op holds execute_if), so TEX and OM never read the
    // window in the same cycle — a select mux drives the single read-port set,
    // avoiding duplicated RF read ports (area/timing). Only TEX writes.
    localparam GFXW_CONS_RD_PORTS = 8;
    wire [NW_WIDTH-1:0]                                          gfxw_cons_rd_wid;
    wire [`CLOG2(`VX_CFG_NUM_THREADS)-1:0]                       gfxw_cons_rd_tbase;
    wire [GFXW_CONS_RD_PORTS-1:0][`CLOG2(`VX_RT_SLOT_COUNT)-1:0] gfxw_cons_rd_slot;
    wire [GFXW_CONS_RD_PORTS-1:0][NUM_LANES-1:0][31:0]           gfxw_cons_rd_data;
    wire                                                        gfxw_cons_wr_en;
    wire [NW_WIDTH-1:0]                                          gfxw_cons_wr_wid;
    wire [`CLOG2(`VX_CFG_NUM_THREADS)-1:0]                       gfxw_cons_wr_tbase;
    wire [NUM_LANES-1:0]                                         gfxw_cons_wr_mask;
    wire [`CLOG2(`VX_RT_SLOT_COUNT)-1:0]                         gfxw_cons_wr_slot;
    wire [NUM_LANES-1:0][31:0]                                   gfxw_cons_wr_data;

    // FWD raster payload → window write port (2nd consumer write port; driven by
    // VX_raster_unit below, tied off when RASTER is absent).
    wire                                                        rast_win_wr_en;
    wire [NW_WIDTH-1:0]                                          rast_win_wr_wid;
    wire [`CLOG2(`VX_CFG_NUM_THREADS)-1:0]                       rast_win_wr_tbase;
    wire [NUM_LANES-1:0]                                         rast_win_wr_mask;
    wire [`CLOG2(`VX_RT_SLOT_COUNT)-1:0]                         rast_win_wr_slot;
    wire [NUM_LANES-1:0][31:0]                                   rast_win_wr_data;
  `ifndef VX_CFG_EXT_RASTER_ENABLE
    assign rast_win_wr_en    = 1'b0;
    assign rast_win_wr_wid   = '0;
    assign rast_win_wr_tbase = '0;
    assign rast_win_wr_mask  = '0;
    assign rast_win_wr_slot  = '0;
    assign rast_win_wr_data  = '0;
  `endif

  `ifdef VX_CFG_EXT_TEX_ENABLE
    wire [NW_WIDTH-1:0]                                          tex_cons_rd_wid;
    wire [`CLOG2(`VX_CFG_NUM_THREADS)-1:0]                       tex_cons_rd_tbase;
    wire [GFXW_CONS_RD_PORTS-1:0][`CLOG2(`VX_RT_SLOT_COUNT)-1:0] tex_cons_rd_slot;
  `endif
  `ifdef VX_CFG_EXT_OM_ENABLE
    wire [NW_WIDTH-1:0]                                          om_cons_rd_wid;
    wire [`CLOG2(`VX_CFG_NUM_THREADS)-1:0]                       om_cons_rd_tbase;
    wire [GFXW_CONS_RD_PORTS-1:0][`CLOG2(`VX_RT_SLOT_COUNT)-1:0] om_cons_rd_slot;
  `endif

    // ── read-port select mux (TEX vs OM) ──────────────────────────────────
  `ifdef VX_CFG_EXT_TEX_ENABLE
   `ifdef VX_CFG_EXT_OM_ENABLE
    wire om_rd_active = pe_execute_if[PE_IDX_OM].valid;  // serialized w.r.t. TEX
    assign gfxw_cons_rd_wid   = om_rd_active ? om_cons_rd_wid   : tex_cons_rd_wid;
    assign gfxw_cons_rd_tbase = om_rd_active ? om_cons_rd_tbase : tex_cons_rd_tbase;
    assign gfxw_cons_rd_slot  = om_rd_active ? om_cons_rd_slot  : tex_cons_rd_slot;
   `else
    assign gfxw_cons_rd_wid   = tex_cons_rd_wid;
    assign gfxw_cons_rd_tbase = tex_cons_rd_tbase;
    assign gfxw_cons_rd_slot  = tex_cons_rd_slot;
   `endif
    // write port driven directly by TEX (gfxw_cons_wr_* below).
  `else
   `ifdef VX_CFG_EXT_OM_ENABLE
    assign gfxw_cons_rd_wid   = om_cons_rd_wid;
    assign gfxw_cons_rd_tbase = om_cons_rd_tbase;
    assign gfxw_cons_rd_slot  = om_cons_rd_slot;
   `else
    // No FF consumer (e.g. RTU-only): tie off the read ports.
    assign gfxw_cons_rd_wid   = '0;
    assign gfxw_cons_rd_tbase = '0;
    assign gfxw_cons_rd_slot  = '0;
    `UNUSED_VAR (gfxw_cons_rd_data)
   `endif
    // No TEX → no window writeback.
    assign gfxw_cons_wr_en    = 1'b0;
    assign gfxw_cons_wr_wid   = '0;
    assign gfxw_cons_wr_tbase = '0;
    assign gfxw_cons_wr_mask  = '0;
    assign gfxw_cons_wr_slot  = '0;
    assign gfxw_cons_wr_data  = '0;
  `endif
`endif

`ifdef VX_CFG_EXT_TEX_ENABLE
    VX_tex_unit #(
        .INSTANCE_ID (`SFORMATF(("%s-tex", INSTANCE_ID))),
        .CORE_ID     (CORE_ID),
        .NUM_LANES   (NUM_LANES),
        .CONS_RD_PORTS (GFXW_CONS_RD_PORTS)
    ) tex_unit (
        .clk        (clk),
        .reset      (reset),
        .execute_if (pe_execute_if[PE_IDX_TEX]),
        .result_if  (pe_result_if[PE_IDX_TEX]),
        .tex_bus_if (tex_bus_if),
        .cons_rd_wid   (tex_cons_rd_wid),
        .cons_rd_tbase (tex_cons_rd_tbase),
        .cons_rd_slot  (tex_cons_rd_slot),
        .cons_rd_data  (gfxw_cons_rd_data),
        .cons_wr_en    (gfxw_cons_wr_en),
        .cons_wr_wid   (gfxw_cons_wr_wid),
        .cons_wr_tbase (gfxw_cons_wr_tbase),
        .cons_wr_mask  (gfxw_cons_wr_mask),
        .cons_wr_slot  (gfxw_cons_wr_slot),
        .cons_wr_data  (gfxw_cons_wr_data)
    );
`endif

`ifdef VX_CFG_EXT_OM_ENABLE
    VX_om_unit #(
        .INSTANCE_ID (`SFORMATF(("%s-om", INSTANCE_ID))),
        .CORE_ID     (CORE_ID),
        .NUM_LANES   (NUM_LANES),
        .CONS_RD_PORTS (GFXW_CONS_RD_PORTS)
    ) om_unit (
        .clk        (clk),
        .reset      (reset),
        .execute_if (pe_execute_if[PE_IDX_OM]),
        .result_if  (pe_result_if[PE_IDX_OM]),
        .cons_rd_wid   (om_cons_rd_wid),
        .cons_rd_tbase (om_cons_rd_tbase),
        .cons_rd_slot  (om_cons_rd_slot),
        .cons_rd_data  (gfxw_cons_rd_data),
        .om_bus_if  (om_bus_if)
    );
`endif

`ifdef VX_CFG_EXT_RASTER_ENABLE
    VX_raster_unit #(
        .INSTANCE_ID (`SFORMATF(("%s-raster", INSTANCE_ID))),
        .CORE_ID     (CORE_ID),
        .NUM_LANES   (NUM_LANES)
    ) raster_unit (
        .clk          (clk),
        .reset        (reset),
        .execute_if   (pe_execute_if[PE_IDX_RASTER]),
        .result_if    (pe_result_if[PE_IDX_RASTER]),
        .raster_bus_if(raster_bus_if),
        .win_wr_en    (rast_win_wr_en),
        .win_wr_wid   (rast_win_wr_wid),
        .win_wr_tbase (rast_win_wr_tbase),
        .win_wr_mask  (rast_win_wr_mask),
        .win_wr_slot  (rast_win_wr_slot),
        .win_wr_data  (rast_win_wr_data)
    );
`endif

`ifdef EXT_GFX_ANY_ENABLE
    VX_gfx_window #(
        .INSTANCE_ID (`SFORMATF(("%s-gfxw", INSTANCE_ID))),
        .CORE_ID     (CORE_ID),
        .NUM_LANES   (NUM_LANES),
        .CONS_RD_PORTS (GFXW_CONS_RD_PORTS)
    ) gfx_window (
        .clk        (clk),
        .reset      (reset),
        .execute_if (pe_execute_if[PE_IDX_GFXW]),
        .result_if  (pe_result_if[PE_IDX_GFXW]),
        // FF-consumer window access (driven by the TEX/OM PEs, or tied off above).
        .cons_rd_wid   (gfxw_cons_rd_wid),
        .cons_rd_tbase (gfxw_cons_rd_tbase),
        .cons_rd_slot  (gfxw_cons_rd_slot),
        .cons_rd_data  (gfxw_cons_rd_data),
        .cons_wr_en    (gfxw_cons_wr_en),
        .cons_wr_wid   (gfxw_cons_wr_wid),
        .cons_wr_tbase (gfxw_cons_wr_tbase),
        .cons_wr_mask  (gfxw_cons_wr_mask),
        .cons_wr_slot  (gfxw_cons_wr_slot),
        .cons_wr_data  (gfxw_cons_wr_data),
        // FWD raster payload write port (2nd consumer write port)
        .rast_wr_en    (rast_win_wr_en),
        .rast_wr_wid   (rast_win_wr_wid),
        .rast_wr_tbase (rast_win_wr_tbase),
        .rast_wr_mask  (rast_win_wr_mask),
        .rast_wr_slot  (rast_win_wr_slot),
        .rast_wr_data  (rast_win_wr_data)
    `ifdef VX_CFG_EXT_RTU_ENABLE
        ,
        .rtu_bus_if (rtu_bus_if),
        .async_trap_if (async_trap_if)
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
