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

`include "VX_fpu_define.vh"

module VX_fpu_unit import VX_fpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire clk,
    input wire reset,

    // Inputs
    VX_dispatch_if.slave    dispatch_if [`ISSUE_WIDTH],

    // Outputs
    VX_commit_if.master     commit_if [`ISSUE_WIDTH],
    VX_fpu_csr_if.master    fpu_csr_if[`NUM_FPU_BLOCKS]
);
    `UNUSED_SPARAM (INSTANCE_ID)
    localparam BLOCK_SIZE = `NUM_FPU_BLOCKS;
    localparam NUM_LANES  = `NUM_FPU_LANES;
    localparam PID_BITS   = `CLOG2(`NUM_THREADS / NUM_LANES);
    localparam PID_WIDTH  = `UP(PID_BITS);
    localparam TAG_WIDTH  = `LOG2UP(`FPUQ_SIZE);
    localparam PARTIAL_BW = (BLOCK_SIZE != `ISSUE_WIDTH) || (NUM_LANES != `NUM_THREADS);

    VX_execute_if #(
        .NUM_LANES (NUM_LANES)
    ) per_block_execute_if[BLOCK_SIZE]();

    VX_dispatch_unit #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_BUF    (PARTIAL_BW ? 1 : 0)
    ) dispatch_unit (
        .clk        (clk),
        .reset      (reset),
        .dispatch_if(dispatch_if),
        .execute_if (per_block_execute_if)
    );

    VX_commit_if #(
        .NUM_LANES (NUM_LANES)
    ) per_block_commit_if[BLOCK_SIZE]();

    for (genvar block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin
        `UNUSED_VAR (per_block_execute_if[block_idx].data.tid)
        `UNUSED_VAR (per_block_execute_if[block_idx].data.wb)

        `RESET_RELAY_EN (block_reset, reset, (BLOCK_SIZE > 1));

        // Store request info
        wire fpu_req_valid, fpu_req_ready;
        wire fpu_rsp_valid, fpu_rsp_ready;
        wire [NUM_LANES-1:0][`XLEN-1:0] fpu_rsp_result;
        fflags_t fpu_rsp_fflags;
        wire fpu_rsp_has_fflags;

        wire [`UUID_WIDTH-1:0]  fpu_rsp_uuid;
        wire [`NW_WIDTH-1:0]    fpu_rsp_wid;
        wire [NUM_LANES-1:0]    fpu_rsp_tmask;
        wire [`PC_BITS-1:0]     fpu_rsp_PC;
        wire [`NR_BITS-1:0]     fpu_rsp_rd;
        wire [PID_WIDTH-1:0]    fpu_rsp_pid;
        wire                    fpu_rsp_sop;
        wire                    fpu_rsp_eop;

        wire [TAG_WIDTH-1:0] fpu_req_tag, fpu_rsp_tag;
        wire mdata_full;

        wire [`INST_FMT_BITS-1:0] fpu_fmt = per_block_execute_if[block_idx].data.op_args.fpu.fmt;
        wire [`INST_FRM_BITS-1:0] fpu_frm = per_block_execute_if[block_idx].data.op_args.fpu.frm;

        wire execute_fire = per_block_execute_if[block_idx].valid && per_block_execute_if[block_idx].ready;
        wire fpu_rsp_fire = fpu_rsp_valid && fpu_rsp_ready;

        VX_index_buffer #(
            .DATAW  (`UUID_WIDTH + `NW_WIDTH + NUM_LANES + `PC_BITS + `NR_BITS + PID_WIDTH + 1 + 1),
            .SIZE   (`FPUQ_SIZE)
        ) tag_store (
            .clk          (clk),
            .reset        (block_reset),
            .acquire_en   (execute_fire),
            .write_addr   (fpu_req_tag),
            .write_data   ({per_block_execute_if[block_idx].data.uuid, per_block_execute_if[block_idx].data.wid, per_block_execute_if[block_idx].data.tmask, per_block_execute_if[block_idx].data.PC, per_block_execute_if[block_idx].data.rd, per_block_execute_if[block_idx].data.pid, per_block_execute_if[block_idx].data.sop, per_block_execute_if[block_idx].data.eop}),
            .read_data    ({fpu_rsp_uuid, fpu_rsp_wid, fpu_rsp_tmask, fpu_rsp_PC, fpu_rsp_rd, fpu_rsp_pid, fpu_rsp_sop, fpu_rsp_eop}),
            .read_addr    (fpu_rsp_tag),
            .release_en   (fpu_rsp_fire),
            .full         (mdata_full),
            `UNUSED_PIN (empty)
        );

        // resolve dynamic FRM from CSR
        wire [`INST_FRM_BITS-1:0] fpu_req_frm;
        `ASSIGN_BLOCKED_WID (fpu_csr_if[block_idx].read_wid, per_block_execute_if[block_idx].data.wid, block_idx, `NUM_FPU_BLOCKS)
        assign fpu_req_frm = (per_block_execute_if[block_idx].data.op_type != `INST_FPU_MISC
                           && fpu_frm == `INST_FRM_DYN) ? fpu_csr_if[block_idx].read_frm : fpu_frm;

        // submit FPU request

        assign fpu_req_valid = per_block_execute_if[block_idx].valid && ~mdata_full;
        assign per_block_execute_if[block_idx].ready = fpu_req_ready && ~mdata_full;

    `ifdef FPU_DPI

        VX_fpu_dpi #(
            .NUM_LANES  (NUM_LANES),
            .TAG_WIDTH  (TAG_WIDTH),
            .OUT_BUF    (PARTIAL_BW ? 1 : 3)
        ) fpu_dpi (
            .clk        (clk),
            .reset      (block_reset),

            .valid_in   (fpu_req_valid),
            .mask_in    (per_block_execute_if[block_idx].data.tmask),
            .op_type    (per_block_execute_if[block_idx].data.op_type),
            .fmt        (fpu_fmt),
            .frm        (fpu_req_frm),
            .dataa      (per_block_execute_if[block_idx].data.rs1_data),
            .datab      (per_block_execute_if[block_idx].data.rs2_data),
            .datac      (per_block_execute_if[block_idx].data.rs3_data),
            .tag_in     (fpu_req_tag),
            .ready_in   (fpu_req_ready),

            .valid_out  (fpu_rsp_valid),
            .result     (fpu_rsp_result),
            .has_fflags (fpu_rsp_has_fflags),
            .fflags     (fpu_rsp_fflags),
            .tag_out    (fpu_rsp_tag),
            .ready_out  (fpu_rsp_ready)
        );

    `elsif FPU_FPNEW

        VX_fpu_fpnew #(
            .NUM_LANES  (NUM_LANES),
            .TAG_WIDTH  (TAG_WIDTH),
            .OUT_BUF    (PARTIAL_BW ? 1 : 3)
        ) fpu_fpnew (
            .clk        (clk),
            .reset      (block_reset),

            .valid_in   (fpu_req_valid),
            .mask_in    (per_block_execute_if[block_idx].data.tmask),
            .op_type    (per_block_execute_if[block_idx].data.op_type),
            .fmt        (fpu_fmt),
            .frm        (fpu_req_frm),
            .dataa      (per_block_execute_if[block_idx].data.rs1_data),
            .datab      (per_block_execute_if[block_idx].data.rs2_data),
            .datac      (per_block_execute_if[block_idx].data.rs3_data),
            .tag_in     (fpu_req_tag),
            .ready_in   (fpu_req_ready),

            .valid_out  (fpu_rsp_valid),
            .result     (fpu_rsp_result),
            .has_fflags (fpu_rsp_has_fflags),
            .fflags     (fpu_rsp_fflags),
            .tag_out    (fpu_rsp_tag),
            .ready_out  (fpu_rsp_ready)
        );

    `elsif FPU_DSP

        VX_fpu_dsp #(
            .NUM_LANES  (NUM_LANES),
            .TAG_WIDTH  (TAG_WIDTH),
            .OUT_BUF    (PARTIAL_BW ? 1 : 3)
        ) fpu_dsp (
            .clk        (clk),
            .reset      (block_reset),

            .valid_in   (fpu_req_valid),
            .mask_in    (per_block_execute_if[block_idx].data.tmask),
            .op_type    (per_block_execute_if[block_idx].data.op_type),
            .fmt        (fpu_fmt),
            .frm        (fpu_req_frm),
            .dataa      (per_block_execute_if[block_idx].data.rs1_data),
            .datab      (per_block_execute_if[block_idx].data.rs2_data),
            .datac      (per_block_execute_if[block_idx].data.rs3_data),
            .tag_in     (fpu_req_tag),
            .ready_in   (fpu_req_ready),

            .valid_out  (fpu_rsp_valid),
            .result     (fpu_rsp_result),
            .has_fflags (fpu_rsp_has_fflags),
            .fflags     (fpu_rsp_fflags),
            .tag_out    (fpu_rsp_tag),
            .ready_out  (fpu_rsp_ready)
        );

    `endif

        // handle FPU response

        fflags_t fpu_rsp_fflags_q;

        if (PID_BITS != 0) begin
            fflags_t fpu_rsp_fflags_r;
            always @(posedge clk) begin
                if (block_reset) begin
                    fpu_rsp_fflags_r <= '0;
                end else if (fpu_rsp_fire) begin
                    fpu_rsp_fflags_r <= fpu_rsp_eop ? '0 : (fpu_rsp_fflags_r | fpu_rsp_fflags);
                end
            end
            assign fpu_rsp_fflags_q = fpu_rsp_fflags_r | fpu_rsp_fflags;
        end else begin
            assign fpu_rsp_fflags_q = fpu_rsp_fflags;
        end

        assign fpu_csr_if[block_idx].write_enable = fpu_rsp_fire && fpu_rsp_eop && fpu_rsp_has_fflags;
        `ASSIGN_BLOCKED_WID (fpu_csr_if[block_idx].write_wid, fpu_rsp_wid, block_idx, `NUM_FPU_BLOCKS)
        assign fpu_csr_if[block_idx].write_fflags = fpu_rsp_fflags_q;

        // send response

        VX_elastic_buffer #(
            .DATAW (`UUID_WIDTH + `NW_WIDTH + NUM_LANES + `PC_BITS + `NR_BITS + (NUM_LANES * `XLEN) + PID_WIDTH + 1 + 1),
            .SIZE  (0)
        ) rsp_buf (
            .clk       (clk),
            .reset     (block_reset),
            .valid_in  (fpu_rsp_valid),
            .ready_in  (fpu_rsp_ready),
            .data_in   ({fpu_rsp_uuid, fpu_rsp_wid, fpu_rsp_tmask, fpu_rsp_PC, fpu_rsp_rd, fpu_rsp_result, fpu_rsp_pid, fpu_rsp_sop, fpu_rsp_eop}),
            .data_out  ({per_block_commit_if[block_idx].data.uuid, per_block_commit_if[block_idx].data.wid, per_block_commit_if[block_idx].data.tmask, per_block_commit_if[block_idx].data.PC, per_block_commit_if[block_idx].data.rd, per_block_commit_if[block_idx].data.data, per_block_commit_if[block_idx].data.pid, per_block_commit_if[block_idx].data.sop, per_block_commit_if[block_idx].data.eop}),
            .valid_out (per_block_commit_if[block_idx].valid),
            .ready_out (per_block_commit_if[block_idx].ready)
        );
        assign per_block_commit_if[block_idx].data.wb = 1'b1;
    end

    VX_gather_unit #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_BUF    (PARTIAL_BW ? 3 : 0)
    ) gather_unit (
        .clk           (clk),
        .reset         (reset),
        .commit_in_if  (per_block_commit_if),
        .commit_out_if (commit_if)
    );

endmodule
