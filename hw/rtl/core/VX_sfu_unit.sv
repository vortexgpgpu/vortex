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
`ifdef EXT_RASTER_ENABLE
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
    VX_dispatch_if.slave    dispatch_if [`ISSUE_WIDTH],

`ifdef EXT_F_ENABLE
    VX_fpu_csr_if.slave     fpu_csr_if [`NUM_FPU_BLOCKS],
`endif

`ifdef EXT_DXA_ENABLE
    VX_dxa_req_bus_if.master dxa_req_bus_if,
    VX_txbar_bus_if.slave   dxa_txbar_bus_if,
`endif

`ifdef EXT_TEX_ENABLE
    VX_tex_bus_if.master    tex_bus_if,
`endif

`ifdef EXT_OM_ENABLE
    VX_om_bus_if.master     om_bus_if,
`endif

`ifdef EXT_RASTER_ENABLE
    VX_raster_bus_if.slave  raster_bus_if,
`endif

    VX_sched_csr_if.slave   sched_csr_if,

    VX_dcr_csr_if           dcr_csr_if,

    // Outputs
    VX_commit_if.master     commit_if [`ISSUE_WIDTH],
    VX_warp_ctl_if.master   warp_ctl_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    localparam BLOCK_SIZE   = 1;
    localparam NUM_LANES    = `NUM_SFU_LANES;
    localparam PE_COUNT     = 2 + `EXT_DXA_ENABLED + `EXT_TEX_ENABLED + `EXT_OM_ENABLED + `EXT_RASTER_ENABLED;
    localparam PE_SEL_BITS  = `CLOG2(PE_COUNT);
    localparam PE_IDX_WCTL  = 0;
    localparam PE_IDX_CSRS  = 1;
`ifdef EXT_DXA_ENABLE
    localparam PE_IDX_DXA   = 2;
`endif
`ifdef EXT_TEX_ENABLE
    localparam PE_IDX_TEX   = 2 + `EXT_DXA_ENABLED;
`endif
`ifdef EXT_OM_ENABLE
    localparam PE_IDX_OM    = 2 + `EXT_DXA_ENABLED + `EXT_TEX_ENABLED;
`endif
`ifdef EXT_RASTER_ENABLE
    localparam PE_IDX_RASTER = 2 + `EXT_DXA_ENABLED + `EXT_TEX_ENABLED + `EXT_OM_ENABLED;
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
    `ifdef EXT_DXA_ENABLE
        if (per_block_execute_if[0].data.op_type == INST_SFU_DXA) begin
            pe_select = PE_SEL_BITS'(PE_IDX_DXA);
        end
    `endif
    `ifdef EXT_TEX_ENABLE
        if (per_block_execute_if[0].data.op_type == INST_SFU_TEX) begin
            pe_select = PE_SEL_BITS'(PE_IDX_TEX);
        end
    `endif
    `ifdef EXT_OM_ENABLE
        if (per_block_execute_if[0].data.op_type == INST_SFU_OM) begin
            pe_select = PE_SEL_BITS'(PE_IDX_OM);
        end
    `endif
    `ifdef EXT_RASTER_ENABLE
        if (per_block_execute_if[0].data.op_type == INST_SFU_RASTER) begin
            pe_select = PE_SEL_BITS'(PE_IDX_RASTER);
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

`ifdef EXT_RASTER_ENABLE
    // Per-warp + per-pid raster CSR storage. Latched on each vx_rast pop
    // by VX_raster_unit; consumed by VX_csr_unit on raster CSR reads.
    VX_sfu_csr_if #(
        .NUM_LANES (NUM_LANES)
    ) raster_csr_if();
`endif

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

    `ifdef EXT_F_ENABLE
        .fpu_csr_if     (fpu_csr_if),
    `endif

        .sched_csr_if   (sched_csr_if),
    `ifdef EXT_RASTER_ENABLE
        .raster_csr_if  (raster_csr_if),
    `endif
        .result_if      (pe_result_if[PE_IDX_CSRS]),
        .dcr_csr_if     (dcr_csr_if)
    );

`ifdef EXT_DXA_ENABLE
    VX_txbar_bus_if dxa_txbar_attach_if();

    VX_dxa_unit #(
        .INSTANCE_ID (`SFORMATF(("%s-dxa", INSTANCE_ID))),
        .CORE_ID (CORE_ID)
    ) dxa_unit (
        .clk        (clk),
        .reset      (reset),
        .execute_if (pe_execute_if[PE_IDX_DXA]),
        .result_if  (pe_result_if[PE_IDX_DXA]),
        .dxa_req_bus_if (dxa_req_bus_if),
        .txbar_bus_if(dxa_txbar_attach_if)
    );

    // Arbitrate between DXA agent and DXA core (prioritizing the DXA agent)
    VX_txbar_bus_if txbar_arb_if[2]();
    assign txbar_arb_if[0].valid = dxa_txbar_bus_if.valid;
    assign txbar_arb_if[0].data  = dxa_txbar_bus_if.data;
    assign dxa_txbar_bus_if.ready = txbar_arb_if[0].ready;
    assign txbar_arb_if[1].valid = dxa_txbar_attach_if.valid;
    assign txbar_arb_if[1].data  = dxa_txbar_attach_if.data;
    assign dxa_txbar_attach_if.ready = txbar_arb_if[1].ready;

    VX_txbar_arb #(
        .NUM_REQS (2),
        .ARBITER  ("P"),
        .OUT_BUF  (0)
    ) txbar_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (txbar_arb_if),
        .bus_out_if (txbar_bus_if)
    );
`else
    assign txbar_bus_if.valid = 1'b0;
    assign txbar_bus_if.data = 'x;
    `UNUSED_VAR (txbar_bus_if.ready)
`endif

`ifdef EXT_TEX_ENABLE
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

`ifdef EXT_OM_ENABLE
    VX_om_unit #(
        .INSTANCE_ID (`SFORMATF(("%s-om", INSTANCE_ID))),
        .CORE_ID     (CORE_ID),
        .NUM_LANES   (NUM_LANES)
    ) om_unit (
        .clk        (clk),
        .reset      (reset),
        .execute_if (pe_execute_if[PE_IDX_OM]),
        .result_if  (pe_result_if[PE_IDX_OM]),
        .om_bus_if  (om_bus_if)
    );
`endif

`ifdef EXT_RASTER_ENABLE
    // Side-band CSR write port from VX_raster_unit → VX_raster_csr.
    localparam RASTER_PID_W = `UP(`LOG2UP(`NUM_THREADS / NUM_LANES));
    wire                              raster_csr_write_enable;
    wire [UUID_WIDTH-1:0]             raster_csr_write_uuid;
    wire [NW_WIDTH-1:0]              raster_csr_write_wid;
    wire [NUM_LANES-1:0]              raster_csr_write_tmask;
    wire [RASTER_PID_W-1:0]           raster_csr_write_pid;
    raster_stamp_t [NUM_LANES-1:0]    raster_csr_write_data;

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

        .csr_write_enable(raster_csr_write_enable),
        .csr_write_uuid  (raster_csr_write_uuid),
        .csr_write_wid   (raster_csr_write_wid),
        .csr_write_tmask (raster_csr_write_tmask),
        .csr_write_pid   (raster_csr_write_pid),
        .csr_write_data  (raster_csr_write_data)
    );

    VX_raster_csr #(
        .CORE_ID   (CORE_ID),
        .NUM_LANES (NUM_LANES)
    ) raster_csr (
        .clk            (clk),
        .reset          (reset),

        .write_enable   (raster_csr_write_enable),
        .write_uuid     (raster_csr_write_uuid),
        .write_wid      (raster_csr_write_wid),
        .write_tmask    (raster_csr_write_tmask),
        .write_pid      (raster_csr_write_pid),
        .write_data     (raster_csr_write_data),

        .raster_csr_if  (raster_csr_if)
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
