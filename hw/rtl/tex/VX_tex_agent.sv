//!/bin/bash

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

module VX_tex_agent import VX_tex_pkg::*; #(
    parameter CORE_ID = 0,
    parameter NUM_LANES = 1
) (
    input wire clk,
    input wire reset,

    // Inputs
    VX_execute_if.slave     execute_if,
    VX_sfu_csr_if.slave     tex_csr_if,

    // Outputs
    VX_tex_bus_if.master    tex_bus_if,
    VX_commit_if.master     commit_if
);
    `UNUSED_PARAM (CORE_ID)
    localparam PID_BITS   = `CLOG2(`NUM_THREADS / NUM_LANES);
    localparam PID_WIDTH  = `UP(PID_BITS);
    localparam REQ_QUEUE_BITS = `LOG2UP(`TEX_REQ_QUEUE_SIZE);

    // CSRs access

    tex_csrs_t tex_csrs;

    VX_tex_csr #(
        .CORE_ID   (CORE_ID),
        .NUM_LANES (NUM_LANES)
    ) tex_csr (
        .clk        (clk),
        .reset      (reset),

        // inputs
        .tex_csr_if (tex_csr_if),

        // outputs
        .tex_csrs   (tex_csrs)
    );

    `UNUSED_VAR (tex_csrs)

    // Store request info

    wire [1:0][NUM_LANES-1:0][31:0] sfu_exe_coords;
    wire [NUM_LANES-1:0][`VX_TEX_LOD_BITS-1:0] sfu_exe_lod;
    wire [`VX_TEX_STAGE_BITS-1:0] sfu_exe_stage;

    wire [`UUID_WIDTH-1:0]  rsp_uuid;
    wire [`NW_WIDTH-1:0]    rsp_wid;
    wire [NUM_LANES-1:0]    rsp_tmask;
    wire [`PC_BITS-1:0]     rsp_PC;
    wire [`NR_BITS-1:0]     rsp_rd;
    wire [PID_WIDTH-1:0]    rsp_pid;
    wire                    rsp_sop;
    wire                    rsp_eop;

    wire [REQ_QUEUE_BITS-1:0] mdata_waddr, mdata_raddr;
    wire mdata_full;

    assign sfu_exe_stage = execute_if.data.op_args.tex.stage;
    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign sfu_exe_coords[0][i] = execute_if.data.rs1_data[i][31:0];
        assign sfu_exe_coords[1][i] = execute_if.data.rs2_data[i][31:0];
        assign sfu_exe_lod[i]       = execute_if.data.rs3_data[i][0 +: `VX_TEX_LOD_BITS];
    end

    wire mdata_push = execute_if.valid && execute_if.ready;
    wire mdata_pop  = tex_bus_if.rsp_valid && tex_bus_if.rsp_ready;

    VX_index_buffer #(
        .DATAW (`NW_WIDTH + NUM_LANES + `PC_BITS + `NR_BITS + PID_WIDTH + 1 + 1),
        .SIZE  (`TEX_REQ_QUEUE_SIZE)
    ) tag_store (
        .clk          (clk),
        .reset        (reset),
        .acquire_en   (mdata_push),
        .write_addr   (mdata_waddr),
        .write_data   ({execute_if.data.wid, execute_if.data.tmask, execute_if.data.PC, execute_if.data.rd,execute_if.data.pid, execute_if.data.sop, execute_if.data.eop}),
        .read_data    ({rsp_wid, rsp_tmask, rsp_PC, rsp_rd, rsp_pid, rsp_sop, rsp_eop}),
        .read_addr    (mdata_raddr),
        .release_en   (mdata_pop),
        .full         (mdata_full),
        `UNUSED_PIN (empty)
    );

    // submit texture request

    wire valid_in, ready_in;
    assign valid_in = execute_if.valid && ~mdata_full;
    assign execute_if.ready = ready_in && ~mdata_full;

    wire [`TEX_REQ_TAG_WIDTH-1:0] req_tag = {execute_if.data.uuid, mdata_waddr};

    VX_elastic_buffer #(
        .DATAW   (NUM_LANES * (1 + 2 * 32 + `VX_TEX_LOD_BITS) + `VX_TEX_STAGE_BITS + `TEX_REQ_TAG_WIDTH),
        .SIZE    (2),
        .OUT_REG (2) // external bus should be registered
    ) req_sbuf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (valid_in),
        .ready_in  (ready_in),
        .data_in   ({execute_if.data.tmask, sfu_exe_coords, sfu_exe_lod, sfu_exe_stage, req_tag}),
        .data_out  ({tex_bus_if.req_data.mask, tex_bus_if.req_data.coords, tex_bus_if.req_data.lod, tex_bus_if.req_data.stage, tex_bus_if.req_data.tag}),
        .valid_out (tex_bus_if.req_valid),
        .ready_out (tex_bus_if.req_ready)
    );

    // handle texture response

    assign mdata_raddr = tex_bus_if.rsp_data.tag[0 +: REQ_QUEUE_BITS];
    assign rsp_uuid    = tex_bus_if.rsp_data.tag[REQ_QUEUE_BITS +: `UUID_WIDTH];

    wire [NUM_LANES-1:0][31:0] commit_data;

    VX_elastic_buffer #(
        .DATAW (`UUID_WIDTH + `NW_WIDTH + NUM_LANES + `PC_BITS + `NR_BITS + (NUM_LANES * 32) + PID_WIDTH + 1 + 1),
        .SIZE  (2)
    ) rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (tex_bus_if.rsp_valid),
        .ready_in  (tex_bus_if.rsp_ready),
        .data_in   ({rsp_uuid, rsp_wid, rsp_tmask, rsp_PC, rsp_rd, tex_bus_if.rsp_data.texels, rsp_pid, rsp_sop, rsp_eop}),
        .data_out  ({commit_if.data.uuid, commit_if.data.wid, commit_if.data.tmask, commit_if.data.PC, commit_if.data.rd, commit_data, commit_if.data.pid, commit_if.data.sop, commit_if.data.eop}),
        .valid_out (commit_if.valid),
        .ready_out (commit_if.ready)
    );

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign commit_if.data.data[i] = `XLEN'(commit_data[i]);
    end

    assign commit_if.data.wb = 1'b1;

`ifdef DBG_TRACE_TEX
    always @(posedge clk) begin
        if (execute_if.valid && execute_if.ready) begin
            `TRACE(1, ("%d: core%0d-tex-req: wid=%0d, PC=0x%0h, tmask=%b, u=", $time, CORE_ID, execute_if.data.wid, execute_if.data.PC, execute_if.data.tmask));
            `TRACE_ARRAY1D(1, "0x%0h", sfu_exe_coords[0], NUM_LANES);
            `TRACE(1, (", v="));
            `TRACE_ARRAY1D(1, "0x%0h", sfu_exe_coords[1], NUM_LANES);
            `TRACE(1, (", lod="));
            `TRACE_ARRAY1D(1, "0x%0h", sfu_exe_lod, NUM_LANES);
            `TRACE(1, (", stage=%0d, ibuf_idx=%0d (#%0d)\n", sfu_exe_stage, mdata_waddr, execute_if.data.uuid));
        end
        if (tex_bus_if.rsp_valid && tex_bus_if.rsp_ready) begin
            `TRACE(1, ("%d: core%0d-tex-rsp: wid=%0d, PC=0x%0h, tmask=%b, rd=%0d, texels=", $time, CORE_ID,
                rsp_wid, rsp_PC, rsp_tmask, rsp_rd));
            `TRACE_ARRAY1D(1, "0x%0h", tex_bus_if.rsp_data.texels, NUM_LANES);
            `TRACE(1, (" ibuf_idx=%0d (#%0d)\n", mdata_raddr, rsp_uuid));
        end
    end
`endif

endmodule
