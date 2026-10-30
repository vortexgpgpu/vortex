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
//
// With VX_CFG_NUM_FP64_LANES below NUM_LANES (FLEN 64), the block holds two
// datapaths, as a GPU SM does: an FP32-only one NUM_LANES wide, and a narrow
// FP64 one of NUM_FP64_LANES lanes. An operation on FP64 values goes to the
// narrow datapath in NUM_LANES / NUM_FP64_LANES batches, one per cycle, holding
// the request until its last batch is taken; its tag is taken with the first.
// The batches are reassembled by tag into one result, so the block still
// returns one result per request.
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

    localparam FP64_LANES   = `MIN(`VX_CFG_NUM_FP64_LANES, NUM_LANES);
    localparam FP64_SPLIT   = (`VX_CFG_FLEN >= 64) && (FP64_LANES < NUM_LANES);
    localparam FP64_BATCHES = NUM_LANES / FP64_LANES;
    localparam BATCH_W      = `LOG2UP(FP64_BATCHES);

    `STATIC_ASSERT (`IS_DIVISBLE(NUM_LANES, FP64_LANES), ("invalid parameter: NUM_FPU_LANES=%0d, NUM_FP64_LANES=%0d", NUM_LANES, FP64_LANES))

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

    wire [TAG_WIDTH-1:0] fpu_req_tag, fpu_rsp_tag;
    wire mdata_full;
    wire tag_acquire, tag_release;

    wire [INST_FMT_BITS-1:0] fpu_fmt = execute_if.data.op_args.fpu.fmt;
    wire [INST_FRM_BITS-1:0] fpu_frm = execute_if.data.op_args.fpu.frm;

    wire fpu_rsp_fire = fpu_rsp_valid && fpu_rsp_ready;

    VX_index_buffer #(
        .DATAW  ($bits(fpu_header_t)),
        .SIZE   (`VX_CFG_FPU_QUEUE_SIZE)
    ) tag_store (
        .clk          (clk),
        .reset        (reset),
        .acquire_en   (tag_acquire),
        .write_addr   (fpu_req_tag),
        .write_data   (fpu_hdr_store),
        .read_data    (fpu_hdr),
        .read_addr    (fpu_rsp_tag),
        .release_en   (tag_release),
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

    if (FP64_SPLIT) begin : g_split

        wire [INST_FPU_BITS-1:0] fpu_op = INST_FPU_BITS'(execute_if.data.op_type);

        // An operation reads or writes an FP64 value when its format is D; a
        // conversion between floats also when its other end is D (F2F's source
        // is the other of S/D unless it is H, F2H's is fmt[0]).
        wire req_is_d = fpu_fmt[0] || ((fpu_op == INST_FPU_F2F) && ~fpu_fmt[1]);

        // FP32-only datapath, full width

        wire f32_req_valid = execute_if.valid && ~req_is_d && ~mdata_full;
        wire f32_req_ready;

        wire f32_rsp_valid, f32_rsp_ready;
        wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] f32_rsp_result;
        wire f32_rsp_has_fflags;
        fflags_t f32_rsp_fflags;
        wire [TAG_WIDTH-1:0] f32_rsp_tag;

        VX_fpu_datapath #(
            .NUM_LANES     (NUM_LANES),
            .TAG_WIDTH     (TAG_WIDTH),
            .OUT_BUF       (0),
            .FLEN          (32),
            .FMA_LATENCY   (`VX_CFG_FMA_F32_LATENCY),
            .FDIV_LATENCY  (`VX_CFG_FDIV_F32_LATENCY),
            .FSQRT_LATENCY (`VX_CFG_FSQRT_F32_LATENCY)
        ) fp32_datapath (
            .clk        (clk),
            .reset      (reset),
            .valid_in   (f32_req_valid),
            .mask_in    (execute_if.data.header.tmask),
            .op_type    (fpu_op),
            .fmt        (fpu_fmt),
            .frm        (fpu_req_frm),
            .dataa      (execute_if.data.rs1_data),
            .datab      (execute_if.data.rs2_data),
            .datac      (execute_if.data.rs3_data),
            .tag_in     (fpu_req_tag),
            .ready_in   (f32_req_ready),
            .valid_out  (f32_rsp_valid),
            .result     (f32_rsp_result),
            .has_fflags (f32_rsp_has_fflags),
            .fflags     (f32_rsp_fflags),
            .tag_out    (f32_rsp_tag),
            .ready_out  (f32_rsp_ready)
        );

        // FP64 batch serializer: the request stays on execute_if while its
        // batches go out, so its tag and rounding mode are taken with the
        // first batch (the tag is acquired then) and held for the rest.

        reg [BATCH_W-1:0] batch_idx;
        reg [TAG_WIDTH-1:0] batch_tag_r;
        reg [INST_FRM_BITS-1:0] batch_frm_r;

        wire batch_first = (batch_idx == BATCH_W'(0));
        wire batch_last  = (batch_idx == BATCH_W'(FP64_BATCHES-1));

        wire f64_req_valid = execute_if.valid && req_is_d && (~batch_first || ~mdata_full);
        wire f64_req_ready;
        wire f64_req_fire  = f64_req_valid && f64_req_ready;

        always @(posedge clk) begin
            if (reset) begin
                batch_idx <= '0;
            end else if (f64_req_fire) begin
                batch_idx <= batch_last ? BATCH_W'(0) : (batch_idx + BATCH_W'(1));
            end
            if (f64_req_fire && batch_first) begin
                batch_tag_r <= fpu_req_tag;
                batch_frm_r <= fpu_req_frm;
            end
        end

        wire [TAG_WIDTH-1:0] batch_tag = batch_first ? fpu_req_tag : batch_tag_r;
        wire [INST_FRM_BITS-1:0] batch_frm = batch_first ? fpu_req_frm : batch_frm_r;

        wire [FP64_BATCHES-1:0][FP64_LANES-1:0] req_mask_b = execute_if.data.header.tmask;
        wire [FP64_BATCHES-1:0][FP64_LANES-1:0][`VX_CFG_XLEN-1:0] req_rs1_b = execute_if.data.rs1_data;
        wire [FP64_BATCHES-1:0][FP64_LANES-1:0][`VX_CFG_XLEN-1:0] req_rs2_b = execute_if.data.rs2_data;
        wire [FP64_BATCHES-1:0][FP64_LANES-1:0][`VX_CFG_XLEN-1:0] req_rs3_b = execute_if.data.rs3_data;

        localparam F64_TAG_WIDTH = TAG_WIDTH + BATCH_W;
        localparam F64_REQ_DATAW = FP64_LANES + F64_TAG_WIDTH + INST_FPU_BITS + INST_FMT_BITS + INST_FRM_BITS + 3 * FP64_LANES * `VX_CFG_XLEN;

        wire f64_dp_valid, f64_dp_ready;
        wire [FP64_LANES-1:0] f64_dp_mask;
        wire [F64_TAG_WIDTH-1:0] f64_dp_tag;
        wire [INST_FPU_BITS-1:0] f64_dp_op;
        wire [INST_FMT_BITS-1:0] f64_dp_fmt;
        wire [INST_FRM_BITS-1:0] f64_dp_frm;
        wire [FP64_LANES-1:0][`VX_CFG_XLEN-1:0] f64_dp_rs1, f64_dp_rs2, f64_dp_rs3;

        // The batch select is registered before the narrow datapath.
        VX_elastic_buffer #(
            .DATAW   (F64_REQ_DATAW),
            .SIZE    (2),
            .OUT_REG (1)
        ) f64_req_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (f64_req_valid),
            .ready_in  (f64_req_ready),
            .data_in   ({req_mask_b[batch_idx], {batch_tag, batch_idx}, fpu_op, fpu_fmt, batch_frm,
                         req_rs1_b[batch_idx], req_rs2_b[batch_idx], req_rs3_b[batch_idx]}),
            .data_out  ({f64_dp_mask, f64_dp_tag, f64_dp_op, f64_dp_fmt, f64_dp_frm,
                         f64_dp_rs1, f64_dp_rs2, f64_dp_rs3}),
            .valid_out (f64_dp_valid),
            .ready_out (f64_dp_ready)
        );

        wire f64_rsp_valid, f64_rsp_ready;
        wire [FP64_LANES-1:0][`VX_CFG_XLEN-1:0] f64_rsp_result;
        wire f64_rsp_has_fflags;
        fflags_t f64_rsp_fflags;
        wire [F64_TAG_WIDTH-1:0] f64_rsp_tag;

        VX_fpu_datapath #(
            .NUM_LANES  (FP64_LANES),
            .TAG_WIDTH  (F64_TAG_WIDTH),
            .OUT_BUF    (0)
        ) fp64_datapath (
            .clk        (clk),
            .reset      (reset),
            .valid_in   (f64_dp_valid),
            .mask_in    (f64_dp_mask),
            .op_type    (f64_dp_op),
            .fmt        (f64_dp_fmt),
            .frm        (f64_dp_frm),
            .dataa      (f64_dp_rs1),
            .datab      (f64_dp_rs2),
            .datac      (f64_dp_rs3),
            .tag_in     (f64_dp_tag),
            .ready_in   (f64_dp_ready),
            .valid_out  (f64_rsp_valid),
            .result     (f64_rsp_result),
            .has_fflags (f64_rsp_has_fflags),
            .fflags     (f64_rsp_fflags),
            .tag_out    (f64_rsp_tag),
            .ready_out  (f64_rsp_ready)
        );

        // The batches of an operation come back in order, but interleave with
        // other operations' across the FP cores: each but the last is parked by
        // tag, and the last one completes the result with the flags gathered.
        wire [TAG_WIDTH-1:0] f64_rsp_tid   = f64_rsp_tag[BATCH_W +: TAG_WIDTH];
        wire [BATCH_W-1:0]   f64_rsp_batch = f64_rsp_tag[0 +: BATCH_W];
        wire f64_rsp_last = (f64_rsp_batch == BATCH_W'(FP64_BATCHES-1));
        wire f64_out_ready;
        assign f64_rsp_ready = ~f64_rsp_last || f64_out_ready;
        wire f64_rsp_fire = f64_rsp_valid && f64_rsp_ready;

        fflags_t [`VX_CFG_FPU_QUEUE_SIZE-1:0] f64_fflags_acc;
        fflags_t f64_fflags_sum;
        assign f64_fflags_sum = f64_fflags_acc[f64_rsp_tid] | f64_rsp_fflags;

        always @(posedge clk) begin
            if (reset) begin
                f64_fflags_acc <= '0;
            end else if (f64_rsp_fire) begin
                f64_fflags_acc[f64_rsp_tid] <= f64_rsp_last ? '0 : f64_fflags_sum;
            end
        end

        localparam PARK_LANES = NUM_LANES - FP64_LANES;

        wire [FP64_BATCHES-2:0] park_wren;
        for (genvar b = 0; b < FP64_BATCHES-1; ++b) begin : g_park_wren
            assign park_wren[b] = (f64_rsp_batch == BATCH_W'(b));
        end

        wire [PARK_LANES-1:0][`VX_CFG_XLEN-1:0] park_rdata;

        VX_dp_ram #(
            .DATAW    (PARK_LANES * `VX_CFG_XLEN),
            .SIZE     (`VX_CFG_FPU_QUEUE_SIZE),
            .WRENW    (FP64_BATCHES-1),
            .RDW_MODE ("W"),
            .LUTRAM   (1)
        ) park_ram (
            .clk   (clk),
            .reset (reset),
            .read  (1'b1),
            .write (f64_rsp_fire && ~f64_rsp_last),
            .wren  (park_wren),
            .waddr (f64_rsp_tid),
            .wdata ({(FP64_BATCHES-1){f64_rsp_result}}),
            .raddr (f64_rsp_tid),
            .rdata (park_rdata)
        );

        localparam RSP_DATAW = (NUM_LANES * `VX_CFG_XLEN) + 1 + $bits(fflags_t) + TAG_WIDTH;

        VX_stream_arb #(
            .NUM_INPUTS (2),
            .DATAW      (RSP_DATAW),
            .ARBITER    ("R"),
            .OUT_BUF    (PARTIAL_BW ? 1 : 3)
        ) rsp_arb (
            .clk       (clk),
            .reset     (reset),
            .valid_in  ({(f64_rsp_valid && f64_rsp_last), f32_rsp_valid}),
            .ready_in  ({f64_out_ready, f32_rsp_ready}),
            .data_in   ({{f64_rsp_result, park_rdata, f64_rsp_has_fflags, f64_fflags_sum, f64_rsp_tid},
                         {f32_rsp_result, f32_rsp_has_fflags, f32_rsp_fflags, f32_rsp_tag}}),
            .data_out  ({fpu_rsp_result, fpu_rsp_has_fflags, fpu_rsp_fflags, fpu_rsp_tag}),
            .valid_out (fpu_rsp_valid),
            .ready_out (fpu_rsp_ready),
            `UNUSED_PIN (sel_out)
        );

        assign execute_if.ready = req_is_d ? (f64_req_ready && batch_last)
                                           : (f32_req_ready && ~mdata_full);
        assign tag_acquire = (f32_req_valid && f32_req_ready) || (f64_req_fire && batch_first);

    end else begin : g_shared

        wire fpu_req_valid = execute_if.valid && ~mdata_full;
        wire fpu_req_ready;

        assign execute_if.ready = fpu_req_ready && ~mdata_full;

        VX_fpu_datapath #(
            .NUM_LANES  (NUM_LANES),
            .TAG_WIDTH  (TAG_WIDTH),
            .OUT_BUF    (PARTIAL_BW ? 1 : 3)
        ) fpu_datapath (
            .clk        (clk),
            .reset      (reset),
            .valid_in   (fpu_req_valid),
            .mask_in    (execute_if.data.header.tmask),
            .op_type    (INST_FPU_BITS'(execute_if.data.op_type)),
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

        assign tag_acquire = execute_if.valid && execute_if.ready;

    end

    assign tag_release = fpu_rsp_fire;

    always_comb begin
        fpu_hdr_wb = fpu_hdr;
    end

    // handle CSR update
    fflags_t fpu_rsp_fflags_q;

    if (PID_BITS != 0) begin : g_fflags_pid
        fflags_t fpu_rsp_fflags_r;
        always @(posedge clk) begin
            if (reset) begin
                fpu_rsp_fflags_r <= '0;
            end else if (fpu_rsp_fire) begin
                fpu_rsp_fflags_r <= fpu_hdr_wb.eop ? '0 : (fpu_rsp_fflags_r | fpu_rsp_fflags);
            end
        end
        assign fpu_rsp_fflags_q = fpu_rsp_fflags_r | fpu_rsp_fflags;
    end else begin : g_fflags_no_pid
        assign fpu_rsp_fflags_q = fpu_rsp_fflags;
    end

    VX_fpu_csr_if fpu_csr_tmp_if();
    assign fpu_csr_tmp_if.write_enable = fpu_rsp_fire && fpu_hdr_wb.eop && fpu_rsp_has_fflags;
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
