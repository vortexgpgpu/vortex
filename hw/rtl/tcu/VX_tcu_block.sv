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

// One TCU block: the MMA datapath and the TCU_LD unit that fills its metadata
// RAM, with WGMMA operands supplied by the core-wide tile buffers
// (VX_tcu_shared).
module VX_tcu_block import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter `STRING BLOCK_STR   = ""
) (
    `SCOPE_IO_DECL

    input wire              clk,
    input wire              reset,

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    // MMA execute handshake observed by the WGMMA tile buffers
    VX_execute_if.tap       exec_if,
    input wire [TCU_WG_A_DATA_SIZE-1:0][`VX_CFG_XLEN-1:0] tbuf_rs1_data,
    input wire [TCU_WG_RS2_WIDTH-1:0][`VX_CFG_XLEN-1:0]   tbuf_rs2_data,
    input wire              tbuf_ready,
`endif

`ifdef TCU_META_ENABLE
    // TCU_LD memory client of an LSU scheduler
    VX_lsu_sched_if.master  tcu_mem_if,
`endif

    VX_execute_if.slave     execute_if,
    VX_result_if.master     result_if
);
    `STATIC_ASSERT (`VX_CFG_NUM_TCU_BLOCKS == `VX_CFG_ISSUE_WIDTH, ("must be full issue execution"));
    `STATIC_ASSERT (`VX_CFG_NUM_TCU_LANES == `VX_CFG_NUM_THREADS, ("must be full warp execution"));

    VX_execute_if #(
        .NUM_LANES (`VX_CFG_NUM_TCU_LANES)
    ) core_execute_if();

    VX_result_if #(
        .NUM_LANES (`VX_CFG_NUM_TCU_LANES)
    ) core_result_if();

`ifdef TCU_META_ENABLE
    VX_execute_if #(
        .NUM_LANES (`VX_CFG_NUM_TCU_LANES)
    ) agu_execute_if();

    VX_result_if #(
        .NUM_LANES (`VX_CFG_NUM_TCU_LANES)
    ) agu_result_if();

    wire                    meta_wr_en;
    wire [NW_WIDTH-1:0]     meta_wr_wid;
    wire [4:0]              meta_wr_idx;
    wire [TCU_BLOCK_CAP-1:0] meta_wr_mask;
    wire [TCU_BLOCK_CAP-1:0][31:0] meta_wr_data;

    VX_tcu_agu #(
        .INSTANCE_ID (`SFORMATF(("%s-agu%s", INSTANCE_ID, BLOCK_STR)))
    ) agu (
        .clk          (clk),
        .reset        (reset),
        .ld_if        (agu_execute_if),
        .client_if    (tcu_mem_if),
        .meta_wr_en   (meta_wr_en),
        .meta_wr_wid  (meta_wr_wid),
        .meta_wr_idx  (meta_wr_idx),
        .meta_wr_mask (meta_wr_mask),
        .meta_wr_data (meta_wr_data),
        .result_if    (agu_result_if)
    );

    // TCU_LD goes to the AGU, every MMA op to the core
    wire is_tcu_ld = (execute_if.data.op_type == INST_TCU_LD);

    assign agu_execute_if.valid  = execute_if.valid && is_tcu_ld;
    assign agu_execute_if.data   = execute_if.data;
    assign core_execute_if.valid = execute_if.valid && !is_tcu_ld;
    assign core_execute_if.data  = execute_if.data;
    assign execute_if.ready      = is_tcu_ld ? agu_execute_if.ready : core_execute_if.ready;

    // AGU wins same-cycle conflicts; the core stalls and retries next cycle.
    assign result_if.valid       = agu_result_if.valid || core_result_if.valid;
    assign result_if.data        = agu_result_if.valid ? agu_result_if.data : core_result_if.data;
    assign agu_result_if.ready   = result_if.ready;
    assign core_result_if.ready  = result_if.ready && !agu_result_if.valid;

    // A TCU_LD writes its response fragments straight into the metadata RAM,
    // which an MMA reads while it is presented to the core. That is safe only
    // if no MMA of the loading warp that reads the region being loaded is
    // presented while the load is in flight: an older one has left the
    // in-order execute stream before the TCU_LD enters the AGU, and a younger
    // one waits at issue on the scoreboard until the TCU_LD retires.
`ifndef SYNTHESIS
`ifdef VX_CFG_TCU_SPARSE_ENABLE
    wire mma_reads_sp = (execute_if.data.op_type == INST_TCU_WMMA_SP)
                    `ifdef VX_CFG_TCU_WGMMA_ENABLE
                     || (execute_if.data.op_type == INST_TCU_WGMMA_SP)
                    `endif
                     ;
`else
    wire mma_reads_sp = 1'b0;
`endif
`ifdef VX_CFG_TCU_MX_ENABLE
    wire mma_reads_mx = tcu_fmt_is_mx(execute_if.data.op_args.tcu.fmt_s);
`else
    wire mma_reads_mx = 1'b0;
`endif
    // ~ld_ready: the AGU holds a TCU_LD, so its wid and slot are registered
    wire mma_reads_ld_region = meta_wr_idx[4] ? mma_reads_mx : mma_reads_sp;
    `RUNTIME_ASSERT(~(core_execute_if.valid && ~agu_execute_if.ready
                   && (execute_if.data.header.wid == meta_wr_wid) && mma_reads_ld_region),
        ("%t: *** %s MMA reads metadata a TCU_LD is loading: wid=%0d, slot=%0d, op_type=%0d (#%0d)",
            $time, INSTANCE_ID, meta_wr_wid, meta_wr_idx, execute_if.data.op_type, execute_if.data.header.uuid))
`endif
`else
    assign core_execute_if.valid = execute_if.valid;
    assign core_execute_if.data  = execute_if.data;
    assign execute_if.ready      = core_execute_if.ready;

    assign result_if.valid       = core_result_if.valid;
    assign result_if.data        = core_result_if.data;
    assign core_result_if.ready  = result_if.ready;
`endif

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    assign exec_if.valid = core_execute_if.valid;
    assign exec_if.ready = core_execute_if.ready;
    assign exec_if.data  = core_execute_if.data;
`endif

    VX_tcu_core #(
        .INSTANCE_ID (`SFORMATF(("%s-fused%s", INSTANCE_ID, BLOCK_STR)))
    ) tcu_core (
    `ifdef SCOPE
        .scope_reset   (scope_reset),
        .scope_bus_in  (scope_bus_in),
        .scope_bus_out (scope_bus_out),
    `endif
        .clk        (clk),
        .reset      (reset),
    `ifdef VX_CFG_TCU_WGMMA_ENABLE
        .tbuf_rs1_data (tbuf_rs1_data),
        .tbuf_rs2_data (tbuf_rs2_data),
        .tbuf_ready    (tbuf_ready),
    `endif
    `ifdef TCU_META_ENABLE
        .ext_meta_wr_en   (meta_wr_en),
        .ext_meta_wr_wid  (meta_wr_wid),
        .ext_meta_wr_idx  (meta_wr_idx),
        .ext_meta_wr_mask (meta_wr_mask),
        .ext_meta_wr_data (meta_wr_data),
    `endif
        .execute_if (core_execute_if),
        .result_if  (core_result_if)
    );

endmodule
