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

// One FP datapath of the configured FPU type. FLEN is the widest format it
// implements; at FLEN 32 in a FLEN 64 core it still NaN-boxes per the core.
module VX_fpu_datapath import VX_gpu_pkg::*, VX_fpu_pkg::*; #(
    parameter NUM_LANES     = 1,
    parameter TAG_WIDTH     = 1,
    parameter OUT_BUF       = 0,
    parameter FLEN          = `VX_CFG_FLEN,
    parameter FMA_LATENCY   = `VX_CFG_FMA_LATENCY,
    parameter FDIV_LATENCY  = `VX_CFG_FDIV_LATENCY,
    parameter FSQRT_LATENCY = `VX_CFG_FSQRT_LATENCY
) (
    input wire clk,
    input wire reset,

    input wire  valid_in,
    output wire ready_in,

    input wire [NUM_LANES-1:0] mask_in,

    input wire [TAG_WIDTH-1:0] tag_in,

    input wire [INST_FPU_BITS-1:0] op_type,
    input wire [INST_FMT_BITS-1:0] fmt,
    input wire [INST_FRM_BITS-1:0] frm,

    input wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0]  dataa,
    input wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0]  datab,
    input wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0]  datac,
    output wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] result,

    output wire has_fflags,
    output wire [`FP_FLAGS_BITS-1:0] fflags,

    output wire [TAG_WIDTH-1:0] tag_out,

    input wire  ready_out,
    output wire valid_out
);
`ifdef VX_CFG_FPU_TYPE_DPI

    VX_fpu_dpi #(
        .NUM_LANES     (NUM_LANES),
        .TAG_WIDTH     (TAG_WIDTH),
        .OUT_BUF       (OUT_BUF),
        .FLEN          (FLEN),
        .FMA_LATENCY   (FMA_LATENCY),
        .FDIV_LATENCY  (FDIV_LATENCY),
        .FSQRT_LATENCY (FSQRT_LATENCY)
    ) fpu_dpi (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (valid_in),
        .mask_in    (mask_in),
        .op_type    (op_type),
        .fmt        (fmt),
        .frm        (frm),
        .dataa      (dataa),
        .datab      (datab),
        .datac      (datac),
        .tag_in     (tag_in),
        .ready_in   (ready_in),
        .valid_out  (valid_out),
        .result     (result),
        .has_fflags (has_fflags),
        .fflags     (fflags),
        .tag_out    (tag_out),
        .ready_out  (ready_out)
    );

`elsif VX_CFG_FPU_TYPE_FPNEW

    // FPnew is configured from the core's formats and latencies.
    `UNUSED_PARAM (FLEN)
    `STATIC_ASSERT ((FMA_LATENCY == `VX_CFG_FMA_LATENCY)
                 && (FDIV_LATENCY == `VX_CFG_FDIV_LATENCY)
                 && (FSQRT_LATENCY == `VX_CFG_FSQRT_LATENCY), ("invalid parameter"))

    VX_fpu_fpnew #(
        .NUM_LANES  (NUM_LANES),
        .TAG_WIDTH  (TAG_WIDTH),
        .OUT_BUF    (OUT_BUF)
    ) fpu_fpnew (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (valid_in),
        .mask_in    (mask_in),
        .op_type    (op_type),
        .fmt        (fmt),
        .frm        (frm),
        .dataa      (dataa),
        .datab      (datab),
        .datac      (datac),
        .tag_in     (tag_in),
        .ready_in   (ready_in),
        .valid_out  (valid_out),
        .result     (result),
        .has_fflags (has_fflags),
        .fflags     (fflags),
        .tag_out    (tag_out),
        .ready_out  (ready_out)
    );

`elsif VX_CFG_FPU_TYPE_DSP

    VX_fpu_dsp #(
        .NUM_LANES     (NUM_LANES),
        .TAG_WIDTH     (TAG_WIDTH),
        .OUT_BUF       (OUT_BUF),
        .FLEN          (FLEN),
        .FMA_LATENCY   (FMA_LATENCY),
        .FDIV_LATENCY  (FDIV_LATENCY),
        .FSQRT_LATENCY (FSQRT_LATENCY)
    ) fpu_dsp (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (valid_in),
        .mask_in    (mask_in),
        .op_type    (op_type),
        .fmt        (fmt),
        .frm        (frm),
        .dataa      (dataa),
        .datab      (datab),
        .datac      (datac),
        .tag_in     (tag_in),
        .ready_in   (ready_in),
        .valid_out  (valid_out),
        .result     (result),
        .has_fflags (has_fflags),
        .fflags     (fflags),
        .tag_out    (tag_out),
        .ready_out  (ready_out)
    );

`elsif VX_CFG_FPU_TYPE_STD

    VX_fpu_std #(
        .NUM_LANES     (NUM_LANES),
        .TAG_WIDTH     (TAG_WIDTH),
        .OUT_BUF       (OUT_BUF),
        .FLEN          (FLEN),
        .FMA_LATENCY   (FMA_LATENCY),
        .FDIV_LATENCY  (FDIV_LATENCY),
        .FSQRT_LATENCY (FSQRT_LATENCY)
    ) fpu_std (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (valid_in),
        .mask_in    (mask_in),
        .op_type    (op_type),
        .fmt        (fmt),
        .frm        (frm),
        .dataa      (dataa),
        .datab      (datab),
        .datac      (datac),
        .tag_in     (tag_in),
        .ready_in   (ready_in),
        .valid_out  (valid_out),
        .result     (result),
        .has_fflags (has_fflags),
        .fflags     (fflags),
        .tag_out    (tag_out),
        .ready_out  (ready_out)
    );

`endif

endmodule
