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

// One FPU block: the FP datapath with its tag store and fflags accumulation.
module VX_fpu_block import VX_gpu_pkg::*, VX_fpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES  = 1,
    parameter PARTIAL_BW = 0
) (
    input wire clk,
    input wire reset,

    input wire [`LOG2UP(`VX_CFG_NUM_FPU_BLOCKS)-1:0] block_idx,

    VX_execute_if.slave     execute_if,
    VX_result_if.master     result_if,
    VX_fpu_csr_if.master    fpu_csr_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    localparam PID_BITS   = `CLOG2(`VX_CFG_NUM_THREADS / NUM_LANES);
    localparam TAG_WIDTH  = `LOG2UP(`VX_CFG_FPU_QUEUE_SIZE);

    wire fpu_req_valid, fpu_req_ready;
    wire fpu_rsp_valid, fpu_rsp_ready;
    wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] fpu_rsp_result;
    fflags_t fpu_rsp_fflags;
    wire fpu_rsp_has_fflags;

    fpu_header_t fpu_hdr, fpu_hdr_wb, fpu_hdr_store;

    // Carry the decoded wb through the tag store: FPU compares may target
    // integer x0 (e.g. feq.s x0, ...), which decode does not reserve in the
    // scoreboard, so forcing wb=1 on readback would trip the invalid
    // writeback assertion.
    always_comb begin
        fpu_hdr_store = execute_if.data.header;
    end

    always_comb begin
        fpu_hdr_wb = fpu_hdr;
    end

    wire [TAG_WIDTH-1:0] fpu_req_tag, fpu_rsp_tag;
    wire mdata_full;

    wire [INST_FMT_BITS-1:0] fpu_fmt = execute_if.data.op_args.fpu.fmt;
    wire [INST_FRM_BITS-1:0] fpu_frm = execute_if.data.op_args.fpu.frm;

    wire execute_fire = execute_if.valid && execute_if.ready;
    wire fpu_rsp_fire = fpu_rsp_valid && fpu_rsp_ready;

    VX_index_buffer #(
        .DATAW  ($bits(fpu_header_t)),
        .SIZE   (`VX_CFG_FPU_QUEUE_SIZE)
    ) tag_store (
        .clk          (clk),
        .reset        (reset),
        .acquire_en   (execute_fire),
        .write_addr   (fpu_req_tag),
        .write_data   (fpu_hdr_store),
        .read_data    (fpu_hdr),
        .read_addr    (fpu_rsp_tag),
        .release_en   (fpu_rsp_fire),
        .full         (mdata_full),
        `UNUSED_PIN (empty)
    );

    // resolve dynamic FRM from CSR
    wire [INST_FRM_BITS-1:0] fpu_req_frm;
    `ASSIGN_BLOCKED_WID (fpu_csr_if.read_wid, execute_if.data.header.wid, block_idx, `VX_CFG_NUM_FPU_BLOCKS)
    if (`VX_CFG_NUM_FPU_BLOCKS == 1) begin : g_block_idx_unused
        `UNUSED_VAR (block_idx)
    end
    assign fpu_req_frm = (execute_if.data.op_type != INST_FPU_MISC
                       && fpu_frm == INST_FRM_DYN) ? fpu_csr_if.read_frm : fpu_frm;

    // submit FPU request

    assign fpu_req_valid = execute_if.valid && ~mdata_full;
    assign execute_if.ready = fpu_req_ready && ~mdata_full;

`ifdef VX_CFG_FPU_TYPE_DPI

    VX_fpu_dpi #(
        .NUM_LANES  (NUM_LANES),
        .TAG_WIDTH  (TAG_WIDTH),
        .OUT_BUF    (PARTIAL_BW ? 1 : 3)
    ) fpu_dpi (
        .clk        (clk),
        .reset      (reset),

        .valid_in   (fpu_req_valid),
        .mask_in    (execute_if.data.header.tmask),
        .op_type    (execute_if.data.op_type),
        .fmt        (fpu_fmt),
        .frm        (fpu_req_frm),
        .dataa      (execute_if.data.rs1_data),
        .datab      (execute_if.data.rs2_data),
        .datac      (execute_if.data.rs3_data),
        .tag_in     (fpu_req_tag),
        .ready_in   (fpu_req_ready),

        .valid_out  (fpu_rsp_valid),
        .result     (fpu_rsp_result),
        .has_fflags (fpu_rsp_has_fflags),
        .fflags     (fpu_rsp_fflags),
        .tag_out    (fpu_rsp_tag),
        .ready_out  (fpu_rsp_ready)
    );

`elsif VX_CFG_FPU_TYPE_FPNEW

    VX_fpu_fpnew #(
        .NUM_LANES  (NUM_LANES),
        .TAG_WIDTH  (TAG_WIDTH),
        .OUT_BUF    (PARTIAL_BW ? 1 : 3)
    ) fpu_fpnew (
        .clk        (clk),
        .reset      (reset),

        .valid_in   (fpu_req_valid),
        .mask_in    (execute_if.data.header.tmask),
        .op_type    (execute_if.data.op_type),
        .fmt        (fpu_fmt),
        .frm        (fpu_req_frm),
        .dataa      (execute_if.data.rs1_data),
        .datab      (execute_if.data.rs2_data),
        .datac      (execute_if.data.rs3_data),
        .tag_in     (fpu_req_tag),
        .ready_in   (fpu_req_ready),

        .valid_out  (fpu_rsp_valid),
        .result     (fpu_rsp_result),
        .has_fflags (fpu_rsp_has_fflags),
        .fflags     (fpu_rsp_fflags),
        .tag_out    (fpu_rsp_tag),
        .ready_out  (fpu_rsp_ready)
    );

`elsif VX_CFG_FPU_TYPE_DSP

    VX_fpu_dsp #(
        .NUM_LANES  (NUM_LANES),
        .TAG_WIDTH  (TAG_WIDTH),
        .OUT_BUF    (PARTIAL_BW ? 1 : 3)
    ) fpu_dsp (
        .clk        (clk),
        .reset      (reset),

        .valid_in   (fpu_req_valid),
        .mask_in    (execute_if.data.header.tmask),
        .op_type    (execute_if.data.op_type),
        .fmt        (fpu_fmt),
        .frm        (fpu_req_frm),
        .dataa      (execute_if.data.rs1_data),
        .datab      (execute_if.data.rs2_data),
        .datac      (execute_if.data.rs3_data),
        .tag_in     (fpu_req_tag),
        .ready_in   (fpu_req_ready),

        .valid_out  (fpu_rsp_valid),
        .result     (fpu_rsp_result),
        .has_fflags (fpu_rsp_has_fflags),
        .fflags     (fpu_rsp_fflags),
        .tag_out    (fpu_rsp_tag),
        .ready_out  (fpu_rsp_ready)
    );

`elsif VX_CFG_FPU_TYPE_STD

    VX_fpu_std #(
        .NUM_LANES  (NUM_LANES),
        .TAG_WIDTH  (TAG_WIDTH),
        .OUT_BUF    (PARTIAL_BW ? 1 : 3)
    ) fpu_std (
        .clk        (clk),
        .reset      (reset),

        .valid_in   (fpu_req_valid),
        .mask_in    (execute_if.data.header.tmask),
        .op_type    (execute_if.data.op_type),
        .fmt        (fpu_fmt),
        .frm        (fpu_req_frm),
        .dataa      (execute_if.data.rs1_data),
        .datab      (execute_if.data.rs2_data),
        .datac      (execute_if.data.rs3_data),
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
    `ASSIGN_BLOCKED_WID (fpu_csr_tmp_if.write_wid, fpu_hdr.wid, block_idx, `VX_CFG_NUM_FPU_BLOCKS)
    assign fpu_csr_tmp_if.write_fflags = fpu_rsp_fflags_q;

     VX_pipe_register #(
        .DATAW  (1 + NW_WIDTH + $bits(fflags_t)),
        .RESETW (1)
    ) fpu_csr_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  ({fpu_csr_tmp_if.write_enable,        fpu_csr_tmp_if.write_wid,        fpu_csr_tmp_if.write_fflags}),
        .data_out ({fpu_csr_if.write_enable, fpu_csr_if.write_wid, fpu_csr_if.write_fflags})
    );

    // send response

    VX_elastic_buffer #(
        .DATAW ($bits(fpu_header_t) + (NUM_LANES * `VX_CFG_XLEN)),
        .SIZE  (0)
    ) rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (fpu_rsp_valid),
        .ready_in  (fpu_rsp_ready),
        .data_in   ({fpu_hdr_wb, fpu_rsp_result}),
        .data_out  (result_if.data),
        .valid_out (result_if.valid),
        .ready_out (result_if.ready)
    );

endmodule
