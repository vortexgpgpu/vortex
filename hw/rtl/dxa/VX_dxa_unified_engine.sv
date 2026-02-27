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

module VX_dxa_unified_engine import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_DXA_UNITS = 1,
    parameter ENABLE = 1
) (
    input wire clk,
    input wire reset,

    VX_dcr_bus_if.slave dcr_bus_if,
    VX_dxa_req_bus_if.slave cluster_dxa_bus_if[NUM_DXA_UNITS],
    VX_mem_bus_if.master dxa_gmem_bus_if[NUM_DXA_UNITS],
    VX_dxa_bank_wr_if.master dxa_smem_bank_wr_if[NUM_DXA_UNITS],
    output wire [NUM_DXA_UNITS-1:0][NC_WIDTH-1:0] dxa_smem_core_id
);
`ifdef EXT_DXA_UNIFIED_DISPATCH_ENABLE
    localparam DXA_UNIFIED_DISPATCH_ENABLE = 1;
`elsif EXT_DXA_UNIFIED_DISPATCH_DISABLE
    localparam DXA_UNIFIED_DISPATCH_ENABLE = 0;
`else
    localparam DXA_UNIFIED_DISPATCH_ENABLE = 1;
`endif
    if (ENABLE) begin : g_dxa_unified
        // Shared descriptor table (single copy) for all workers.
        wire [NUM_DXA_UNITS-1:0][DXA_DESC_SLOT_W-1:0] issue_desc_slot;
        wire [NUM_DXA_UNITS-1:0][`MEM_ADDR_WIDTH-1:0] issue_base_addr;
        wire [NUM_DXA_UNITS-1:0][31:0] issue_desc_meta;
        wire [NUM_DXA_UNITS-1:0][31:0] issue_desc_tile01;
        wire [NUM_DXA_UNITS-1:0][31:0] issue_desc_tile23;
        wire [NUM_DXA_UNITS-1:0][31:0] issue_desc_tile4;
        wire [NUM_DXA_UNITS-1:0][31:0] issue_desc_cfill;
        wire [NUM_DXA_UNITS-1:0][31:0] issue_size0_raw;
        wire [NUM_DXA_UNITS-1:0][31:0] issue_size1_raw;
        wire [NUM_DXA_UNITS-1:0][31:0] issue_stride0_raw;
        wire [NUM_DXA_UNITS-1:0] worker_idle;
        VX_dxa_req_bus_if worker_dxa_bus_if[NUM_DXA_UNITS]();

        VX_dxa_desc_table #(
            .NUM_READ_PORTS(NUM_DXA_UNITS)
        ) desc_table (
            .clk            (clk),
            .reset          (reset),
            .dcr_bus_if     (dcr_bus_if),
            .read_desc_slot (issue_desc_slot),
            .read_base_addr (issue_base_addr),
            .read_desc_meta (issue_desc_meta),
            .read_desc_tile01(issue_desc_tile01),
            .read_desc_tile23(issue_desc_tile23),
            .read_desc_tile4(issue_desc_tile4),
            .read_desc_cfill(issue_desc_cfill),
            .read_size0     (issue_size0_raw),
            .read_size1     (issue_size1_raw),
            .read_stride0   (issue_stride0_raw)
        );

        for (genvar i = 0; i < NUM_DXA_UNITS; ++i) begin : g_workers
            VX_dxa_worker #(
                .INSTANCE_ID(`SFORMATF(("%s-worker%0d", INSTANCE_ID, i))),
                .WORKER_ID  (i)
            ) worker (
                .clk               (clk),
                .reset             (reset),
                .dxa_req_bus_if    (worker_dxa_bus_if[i]),
                .issue_desc_slot_out(issue_desc_slot[i]),
                .issue_base_addr   (issue_base_addr[i]),
                .issue_desc_meta   (issue_desc_meta[i]),
                .issue_desc_tile01 (issue_desc_tile01[i]),
                .issue_desc_tile23 (issue_desc_tile23[i]),
                .issue_desc_tile4  (issue_desc_tile4[i]),
                .issue_desc_cfill  (issue_desc_cfill[i]),
                .issue_size0_raw   (issue_size0_raw[i]),
                .issue_size1_raw   (issue_size1_raw[i]),
                .issue_stride0_raw (issue_stride0_raw[i]),
                .gmem_bus_if       (dxa_gmem_bus_if[i]),
                .smem_bank_wr_if   (dxa_smem_bank_wr_if[i]),
                .smem_core_id      (dxa_smem_core_id[i]),
                .worker_idle       (worker_idle[i])
            );
        end

        if (DXA_UNIFIED_DISPATCH_ENABLE) begin : g_unified_dispatch
            localparam WORKER_BITS = `CLOG2(NUM_DXA_UNITS);
            localparam WORKER_W = `UP(WORKER_BITS);

            wire [NUM_DXA_UNITS-1:0] req_valid_in;
            wire [NUM_DXA_UNITS-1:0][DXA_REQ_DATAW-1:0] req_data_in;
            wire [NUM_DXA_UNITS-1:0] req_ready_in;
            wire [NUM_DXA_UNITS-1:0][WORKER_W-1:0] req_sel_in;

            function automatic [WORKER_W-1:0] dxa_affinity_worker;
                input [NC_WIDTH-1:0] core_id;
                input [NW_WIDTH-1:0] wid;
                reg [WORKER_W-1:0] worker_idx;
            begin
                if (NUM_DXA_UNITS > 1) begin
                    worker_idx = WORKER_W'((integer'(core_id) * `NUM_WARPS + integer'(wid)) % NUM_DXA_UNITS);
                end else begin
                    worker_idx = 0;
                end
                dxa_affinity_worker = worker_idx;
            end
            endfunction

            for (genvar i = 0; i < NUM_DXA_UNITS; ++i) begin : g_req_in
                wire [NC_WIDTH-1:0] req_core_id;
                wire [UUID_WIDTH-1:0] req_uuid;
                wire [NW_WIDTH-1:0] req_wid;
                wire [2:0] req_op;
                wire [`XLEN-1:0] req_rs1;
                wire [`XLEN-1:0] req_rs2;
                assign req_valid_in[i] = ~reset && (cluster_dxa_bus_if[i].req_valid === 1'b1);
                assign req_data_in[i]  = cluster_dxa_bus_if[i].req_data;
                assign cluster_dxa_bus_if[i].req_ready = req_ready_in[i];
                assign {req_core_id, req_uuid, req_wid, req_op, req_rs1, req_rs2} = req_data_in[i];
                assign req_sel_in[i] = dxa_affinity_worker(req_core_id, req_wid);
                `UNUSED_VAR (req_uuid)
                `UNUSED_VAR (req_op)
                `UNUSED_VAR (req_rs1)
                `UNUSED_VAR (req_rs2)
            end

            wire [NUM_DXA_UNITS-1:0] req_valid_out;
            wire [NUM_DXA_UNITS-1:0][DXA_REQ_DATAW-1:0] req_data_out;
            wire [NUM_DXA_UNITS-1:0][WORKER_W-1:0] req_sel_out;
            wire [NUM_DXA_UNITS-1:0] req_ready_out;

            VX_stream_xbar #(
                .NUM_INPUTS (NUM_DXA_UNITS),
                .NUM_OUTPUTS(NUM_DXA_UNITS),
                .DATAW      (DXA_REQ_DATAW),
                .ARBITER    ("R"),
                .OUT_BUF    ((NUM_DXA_UNITS > 1) ? 2 : 0)
            ) req_xbar (
                .clk      (clk),
                .reset    (reset),
                `UNUSED_PIN (collisions),
                .valid_in (req_valid_in),
                .data_in  (req_data_in),
                .sel_in   (req_sel_in),
                .ready_in (req_ready_in),
                .valid_out(req_valid_out),
                .data_out (req_data_out),
                .sel_out  (req_sel_out),
                .ready_out(req_ready_out)
            );
            `UNUSED_VAR (req_sel_out)

            for (genvar i = 0; i < NUM_DXA_UNITS; ++i) begin : g_req_to_worker
                assign worker_dxa_bus_if[i].req_valid = req_valid_out[i];
                assign worker_dxa_bus_if[i].req_data  = req_data_out[i];
                assign req_ready_out[i] = worker_dxa_bus_if[i].req_ready;
            end

            for (genvar i = 0; i < NUM_DXA_UNITS; ++i) begin : g_rsp_stub_cluster
                assign worker_dxa_bus_if[i].rsp_ready = 1'b1;
                assign cluster_dxa_bus_if[i].rsp_valid = 1'b0;
                assign cluster_dxa_bus_if[i].rsp_data  = '0;
                `UNUSED_VAR (cluster_dxa_bus_if[i].rsp_ready)
                `UNUSED_VAR (worker_dxa_bus_if[i].rsp_valid)
                `UNUSED_VAR (worker_dxa_bus_if[i].rsp_data)
            end
            `UNUSED_VAR (worker_idle)
        end else begin : g_unified_direct
            for (genvar i = 0; i < NUM_DXA_UNITS; ++i) begin : g_direct
                assign worker_dxa_bus_if[i].req_valid = cluster_dxa_bus_if[i].req_valid;
                assign worker_dxa_bus_if[i].req_data  = cluster_dxa_bus_if[i].req_data;
                assign cluster_dxa_bus_if[i].req_ready = worker_dxa_bus_if[i].req_ready;

                assign worker_dxa_bus_if[i].rsp_ready = 1'b1;
                assign cluster_dxa_bus_if[i].rsp_valid = 1'b0;
                assign cluster_dxa_bus_if[i].rsp_data  = '0;
                `UNUSED_VAR (cluster_dxa_bus_if[i].rsp_ready)
                `UNUSED_VAR (worker_dxa_bus_if[i].rsp_valid)
                `UNUSED_VAR (worker_dxa_bus_if[i].rsp_data)
            end
            `UNUSED_VAR (worker_idle)
        end
    end else begin : g_dxa_unified_off
        for (genvar i = 0; i < NUM_DXA_UNITS; ++i) begin : g_dxa_off
            assign cluster_dxa_bus_if[i].req_ready = 1'b1;
            assign cluster_dxa_bus_if[i].rsp_valid = 1'b0;
            assign cluster_dxa_bus_if[i].rsp_data  = '0;
            `UNUSED_VAR (cluster_dxa_bus_if[i].req_valid)
            `UNUSED_VAR (cluster_dxa_bus_if[i].req_data)

            assign dxa_gmem_bus_if[i].req_valid = 1'b0;
            assign dxa_gmem_bus_if[i].req_data  = '0;
            assign dxa_gmem_bus_if[i].rsp_ready = 1'b1;

            assign dxa_smem_bank_wr_if[i].wr_valid  = '0;
            assign dxa_smem_bank_wr_if[i].wr_addr   = '0;
            assign dxa_smem_bank_wr_if[i].wr_data   = '0;
            assign dxa_smem_bank_wr_if[i].wr_byteen = '0;
            assign dxa_smem_bank_wr_if[i].wr_tag    = '0;
            assign dxa_smem_core_id[i] = '0;
        end
    end

endmodule
