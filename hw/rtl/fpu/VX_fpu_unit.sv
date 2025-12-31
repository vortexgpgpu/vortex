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

module VX_fpu_unit import VX_gpu_pkg::*, VX_fpu_pkg::*; #(
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
    localparam TAG_WIDTH  = `LOG2UP(`FPUQ_SIZE);
    localparam PARTIAL_BW = (BLOCK_SIZE != `ISSUE_WIDTH) || (NUM_LANES != `SIMD_WIDTH);

    VX_execute_if #(
        .data_t (fpu_execute_t)
    ) per_block_execute_if[BLOCK_SIZE]();

    VX_lane_dispatch #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_BUF    (PARTIAL_BW ? 3 : 0)
    ) lane_dispatch (
        .clk        (clk),
        .reset      (reset),
        .dispatch_if(dispatch_if),
        .execute_if (per_block_execute_if)
    );

    VX_result_if #(
        .data_t (fpu_result_t)
    ) per_block_result_if[BLOCK_SIZE]();

    for (genvar block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin : g_blocks
        // Store request info
        wire fpu_req_valid, fpu_req_ready;
        wire fpu_rsp_valid, fpu_rsp_ready;
        wire [NUM_LANES-1:0][`XLEN-1:0] fpu_rsp_result;
        fflags_t fpu_rsp_fflags;
        wire fpu_rsp_has_fflags;

        fpu_header_t fpu_hdr;

        wire [TAG_WIDTH-1:0] fpu_req_tag, fpu_rsp_tag;
        wire mdata_full;

        wire [INST_FMT_BITS-1:0] fpu_fmt = per_block_execute_if[block_idx].data.op_args.fpu.fmt;
        wire [INST_FRM_BITS-1:0] fpu_frm = per_block_execute_if[block_idx].data.op_args.fpu.frm;

        wire execute_fire = per_block_execute_if[block_idx].valid && per_block_execute_if[block_idx].ready;
        wire fpu_rsp_fire = fpu_rsp_valid && fpu_rsp_ready;

        VX_index_buffer #(
            .DATAW  ($bits(fpu_header_t)),
            .SIZE   (`FPUQ_SIZE)
        ) tag_store (
            .clk          (clk),
            .reset        (reset),
            .acquire_en   (execute_fire),
            .write_addr   (fpu_req_tag),
            .write_data   (per_block_execute_if[block_idx].data.header),
            .read_data    (fpu_hdr),
            .read_addr    (fpu_rsp_tag),
            .release_en   (fpu_rsp_fire),
            .full         (mdata_full),
            `UNUSED_PIN (empty)
        );

        // resolve dynamic FRM from CSR
        wire [INST_FRM_BITS-1:0] fpu_req_frm;
        `ASSIGN_BLOCKED_WID (fpu_csr_if[block_idx].read_wid, per_block_execute_if[block_idx].data.header.wid, block_idx, `NUM_FPU_BLOCKS)
        assign fpu_req_frm = (per_block_execute_if[block_idx].data.op_type != INST_FPU_MISC
                           && fpu_frm == INST_FRM_DYN) ? fpu_csr_if[block_idx].read_frm : fpu_frm;

        // submit FPU request

        assign fpu_req_valid = per_block_execute_if[block_idx].valid && ~mdata_full;
        assign per_block_execute_if[block_idx].ready = fpu_req_ready && ~mdata_full;

    `ifdef FPU_TYPE_DPI

        VX_fpu_dpi #(
            .NUM_LANES  (NUM_LANES),
            .TAG_WIDTH  (TAG_WIDTH),
            .OUT_BUF    (PARTIAL_BW ? 1 : 3)
        ) fpu_dpi (
            .clk        (clk),
            .reset      (reset),

            .valid_in   (fpu_req_valid),
            .mask_in    (per_block_execute_if[block_idx].data.header.tmask),
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

    `elsif FPU_TYPE_FPNEW

        VX_fpu_fpnew #(
            .NUM_LANES  (NUM_LANES),
            .TAG_WIDTH  (TAG_WIDTH),
            .OUT_BUF    (PARTIAL_BW ? 1 : 3)
        ) fpu_fpnew (
            .clk        (clk),
            .reset      (reset),

            .valid_in   (fpu_req_valid),
            .mask_in    (per_block_execute_if[block_idx].data.header.tmask),
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

    `elsif FPU_TYPE_DSP

        VX_fpu_dsp #(
            .NUM_LANES  (NUM_LANES),
            .TAG_WIDTH  (TAG_WIDTH),
            .OUT_BUF    (PARTIAL_BW ? 1 : 3)
        ) fpu_dsp (
            .clk        (clk),
            .reset      (reset),

            .valid_in   (fpu_req_valid),
            .mask_in    (per_block_execute_if[block_idx].data.header.tmask),
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

        // handle CSR update
        fflags_t fpu_rsp_fflags_q;

        if (PID_BITS != 0) begin : g_fflags_pid
            fflags_t fpu_rsp_fflags_r;
            always @(posedge clk) begin
                if (reset) begin
                    fpu_rsp_fflags_r <= '0;
                end else if (fpu_rsp_fire) begin
                    fpu_rsp_fflags_r <= fpu_hdr.eop ? '0 : (fpu_rsp_fflags_r | fpu_rsp_fflags);
                end
            end
            assign fpu_rsp_fflags_q = fpu_rsp_fflags_r | fpu_rsp_fflags;
        end else begin : g_fflags_no_pid
            assign fpu_rsp_fflags_q = fpu_rsp_fflags;
        end

        VX_fpu_csr_if fpu_csr_tmp_if();
        assign fpu_csr_tmp_if.write_enable = fpu_rsp_fire && fpu_hdr.eop && fpu_rsp_has_fflags;
        `ASSIGN_BLOCKED_WID (fpu_csr_tmp_if.write_wid, fpu_hdr.wid, block_idx, `NUM_FPU_BLOCKS)
        assign fpu_csr_tmp_if.write_fflags = fpu_rsp_fflags_q;

         VX_pipe_register #(
            .DATAW  (1 + NW_WIDTH + $bits(fflags_t)),
            .RESETW (1)
        ) fpu_csr_reg (
            .clk      (clk),
            .reset    (reset),
            .enable   (1'b1),
            .data_in  ({fpu_csr_tmp_if.write_enable,        fpu_csr_tmp_if.write_wid,        fpu_csr_tmp_if.write_fflags}),
            .data_out ({fpu_csr_if[block_idx].write_enable, fpu_csr_if[block_idx].write_wid, fpu_csr_if[block_idx].write_fflags})
        );

        // send response

        VX_elastic_buffer #(
            .DATAW ($bits(fpu_header_t) + (NUM_LANES * `XLEN)),
            .SIZE  (0)
        ) rsp_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (fpu_rsp_valid),
            .ready_in  (fpu_rsp_ready),
            .data_in   ({fpu_hdr, fpu_rsp_result}),
            .data_out  (per_block_result_if[block_idx].data),
            .valid_out (per_block_result_if[block_idx].valid),
            .ready_out (per_block_result_if[block_idx].ready)
        );
        assign per_block_result_if[block_idx].data.header.wb = 1'b1;
    end

    VX_lane_gather #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_BUF    (PARTIAL_BW ? 3 : 0)
    ) lane_gather (
        .clk       (clk),
        .reset     (reset),
        .result_if (per_block_result_if),
        .commit_if (commit_if)
    );

endmodule
