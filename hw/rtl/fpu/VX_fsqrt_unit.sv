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

// FSQRT backend selector. On a Xilinx (VIVADO) / Altera (QUARTUS) FPGA flow with
// USE_DSP and flush-to-zero, the F32 path (and on Xilinx the F64 path) maps onto
// the hardened floating-point operator IP; otherwise it uses the portable pure-RTL
// core VX_fsqrt_unit_rtl. The vendor IP is flush-to-zero and round-to-nearest-even
// only (gated on SUBNORM_ENABLE=0). At FLEN 64 the vendor unit holds an F32 and an
// F64 core and fmt[0] selects the result; each core's output is delayed to LATENCY.

`include "VX_fpu_define.vh"

module VX_fsqrt_unit import VX_gpu_pkg::*, VX_fpu_pkg::*; #(
    parameter LATENCY = 17,
    parameter FLEN    = 32,
    // 1: use the FPGA vendor FP IP on Xilinx/Altera. 0: ASIC soft core.
    parameter USE_DSP = 0,
    // 1: full IEEE subnormals (soft core only). 0: flush-to-zero; required for IP.
    parameter SUBNORM_ENABLE = 1,
    // 1: produce IEEE special results + fflags. 0: assume finite, tie 0.
    parameter EXCEPT_ENABLE = 1
) (
    input  wire clk,
    input  wire reset,
    input  wire enable,
    input  wire mask,

    input  wire [INST_FMT_BITS-1:0] fmt,
    input  wire [INST_FRM_BITS-1:0] frm,

    input  wire [FLEN-1:0] dataa,   // radicand

    output wire [FLEN-1:0] result,
    output wire [`FP_FLAGS_BITS-1:0] fflags
);
    // Vendor FP IP: Xilinx/Altera FPGA flows only, flush-to-zero.
`ifdef VIVADO
    localparam VENDOR_OK = 1;
`elsif QUARTUS
    localparam VENDOR_OK = 1;
`else
    localparam VENDOR_OK = 0;
`endif
    localparam IS_F32 = (FLEN == 32);
    localparam IS_F64 = (FLEN == 64);

    // Each vendor core's latency (Xilinx: its C_Latency in xilinx_ip_gen.tcl, the maximum).
`ifdef QUARTUS
    localparam ACL_FSQRT_LATENCY = 10;
    localparam VENDOR_F64 = 0; // acl_fsqrt is F32-only
    localparam VENDOR_LATENCY = ACL_FSQRT_LATENCY;
`else
    localparam XIL_FSQRT_LATENCY   = 28;
    localparam XIL_FSQRT64_LATENCY = 57;
    localparam VENDOR_F64 = 1;
    localparam VENDOR_LATENCY = IS_F64 ? XIL_FSQRT64_LATENCY : XIL_FSQRT_LATENCY;
`endif
    // Vendor IP is selected only with SUBNORM_ENABLE=0 (flush-to-zero); SUBNORM_ENABLE=1
    // falls back to the IEEE soft core.
    localparam USE_VENDOR_IP = (USE_DSP != 0) && (SUBNORM_ENABLE == 0) && (VENDOR_OK != 0)
                            && (IS_F32 || (IS_F64 && (VENDOR_F64 != 0)));

    `STATIC_ASSERT(!USE_VENDOR_IP || (LATENCY >= VENDOR_LATENCY),
        ("vendor FSQRT core latency is %0d; VX_CFG_FSQRT_LATENCY=%0d is below it", VENDOR_LATENCY, LATENCY))

    if (USE_VENDOR_IP) begin : g_vendor
        // The vendor IP rounds round-to-nearest-even only (frm is ignored).
        `UNUSED_VAR (mask)
        `UNUSED_VAR (fmt)
        `UNUSED_VAR (frm)

        wire [31:0] res_s;
        wire [`FP_FLAGS_BITS-1:0] fflags_s;
    `ifdef QUARTUS
        wire [31:0] q_s;
        acl_fsqrt fsqrt (
            .clk    (clk),
            .areset (1'b0),
            .en     (enable),
            .a      (dataa[31:0]),
            .q      (q_s)
        );
        VX_shift_register #(
            .DATAW (32),
            .DEPTH (LATENCY - ACL_FSQRT_LATENCY)
        ) pad_s (
            .clk      (clk),
            .reset    (reset),
            .enable   (enable),
            .data_in  (q_s),
            .data_out (res_s)
        );
        assign fflags_s = '0; // acl_fsqrt does not expose exception flags
    `else // VIVADO
        wire [31:0] q_s;
        wire tuser_s;
        xil_fsqrt fsqrt (
            .aclk                (clk),
            .aclken              (enable),
            .s_axis_a_tvalid     (1'b1),
            .s_axis_a_tdata      (dataa[31:0]),
            `UNUSED_PIN (m_axis_result_tvalid),
            .m_axis_result_tdata (q_s),
            .m_axis_result_tuser (tuser_s)
        );
        // tuser = invalid -> fflags {NV, DZ, OF, UF, NX}.
        wire [`FP_FLAGS_BITS-1:0] qf_s = EXCEPT_ENABLE ? {tuser_s, 4'b0} : '0;
        VX_shift_register #(
            .DATAW (`FP_FLAGS_BITS + 32),
            .DEPTH (LATENCY - XIL_FSQRT_LATENCY)
        ) pad_s (
            .clk      (clk),
            .reset    (reset),
            .enable   (enable),
            .data_in  ({qf_s, q_s}),
            .data_out ({fflags_s, res_s})
        );
    `endif

        if (IS_F64) begin : g_f64
        `ifndef QUARTUS
            wire [63:0] q_d, res_d;
            wire tuser_d;
            wire [`FP_FLAGS_BITS-1:0] fflags_d;
            xil_fsqrt64 fsqrt64 (
                .aclk                (clk),
                .aclken              (enable),
                .s_axis_a_tvalid     (1'b1),
                .s_axis_a_tdata      (dataa),
                `UNUSED_PIN (m_axis_result_tvalid),
                .m_axis_result_tdata (q_d),
                .m_axis_result_tuser (tuser_d)
            );
            wire [`FP_FLAGS_BITS-1:0] qf_d = EXCEPT_ENABLE ? {tuser_d, 4'b0} : '0;
            VX_shift_register #(
                .DATAW (`FP_FLAGS_BITS + 64),
                .DEPTH (LATENCY - XIL_FSQRT64_LATENCY)
            ) pad_d (
                .clk      (clk),
                .reset    (reset),
                .enable   (enable),
                .data_in  ({qf_d, q_d}),
                .data_out ({fflags_d, res_d})
            );
            wire is_d;
            VX_shift_register #(
                .DATAW (1),
                .DEPTH (LATENCY)
            ) is_d_pipe (
                .clk      (clk),
                .reset    (reset),
                .enable   (enable),
                .data_in  (fmt[0]),
                .data_out (is_d)
            );
            // F32 results are NaN-boxed into the F64 register.
            assign result = is_d ? res_d : {32'hffffffff, res_s};
            assign fflags = is_d ? fflags_d : fflags_s;
        `endif
        end else begin : g_f32
            assign result = FLEN'(res_s);
            assign fflags = fflags_s;
        end
    end else begin : g_rtl
        VX_fsqrt_unit_rtl #(
            .LATENCY       (LATENCY),
            .FLEN          (FLEN),
            .USE_DSP       (USE_DSP),
            .SUBNORM_ENABLE  (SUBNORM_ENABLE),
            .EXCEPT_ENABLE (EXCEPT_ENABLE)
        ) core (
            .clk     (clk),
            .reset   (reset),
            .enable  (enable),
            .mask    (mask),
            .fmt     (fmt),
            .frm     (frm),
            .dataa   (dataa),
            .result  (result),
            .fflags  (fflags)
        );
    end

endmodule
