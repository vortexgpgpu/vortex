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

module VX_tcu_fpnew_mulfp32 #(
    parameter fpnew_pkg::fp_format_e SRC_FMT = fpnew_pkg::FP16,
    parameter fpnew_pkg::fmt_logic_t FMT_CFG = 5'b10100,
    parameter NUM_PIPE_REGS = 3
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        enable,
    input  wire [15:0] a,
    input  wire [15:0] b,
    output wire [31:0] y
);
    localparam type tag_t = logic[0:0];
    localparam type aux_t = logic[0:0];

    logic [2:0][31:0] operands;
    logic [fpnew_pkg::NUM_FP_FORMATS-1:0][2:0] is_boxed;

    wire [31:0] result;

    assign operands[0] = {16'hffff, a};
    assign operands[1] = {16'hffff, b};
    assign operands[2] = 32'h0;
    assign is_boxed = '1;

    fpnew_fma_multi #(
        .FpFmtConfig (FMT_CFG),
        .NumPipeRegs (NUM_PIPE_REGS),
        .PipeConfig  (fpnew_pkg::DISTRIBUTED),
        .TagType     (tag_t),
        .AuxType     (aux_t)
    ) mul (
        .clk_i           (clk),
        .rst_ni          (~reset),
        .operands_i      (operands),
        .is_boxed_i      (is_boxed),
        .rnd_mode_i      (fpnew_pkg::RNE),
        .op_i            (fpnew_pkg::MUL),
        .op_mod_i        (1'b0),
        .src_fmt_i       (SRC_FMT),
        .dst_fmt_i       (fpnew_pkg::FP32),
        .tag_i           ('0),
        .mask_i          (1'b1),
        .aux_i           ('0),
        .in_valid_i      (enable),
        `UNUSED_PIN      (in_ready_o),
        .flush_i         (1'b0),
        .result_o        (result),
        `UNUSED_PIN      (status_o),
        `UNUSED_PIN      (extension_bit_o),
        `UNUSED_PIN      (tag_o),
        `UNUSED_PIN      (mask_o),
        `UNUSED_PIN      (aux_o),
        `UNUSED_PIN      (out_valid_o),
        .out_ready_i     (enable),
        `UNUSED_PIN      (busy_o),
        .reg_ena_i       ('0)
    );

    assign y = result;
endmodule
