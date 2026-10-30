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

// FMA backend selector. On a Xilinx (VIVADO) or Altera (QUARTUS) FPGA flow with
// USE_DSP and flush-to-zero, the F32 path (and on Xilinx the F64 path) maps onto
// the hardened floating-point operator IP, which is far smaller and faster than
// soft RTL on FPGA. Every other case (generic FPGA, ASIC, simulation, F64 on
// Altera, or any config needing subnormals) uses the portable pure-RTL core
// VX_fma_unit_rtl. The vendor IP is flush-to-zero and round-to-nearest-even only,
// so it is selected only when SUBNORM_ENABLE=0. A LATENCY above the core's own
// is made up by a delay line on its output.

`include "VX_fpu_define.vh"

module VX_fma_unit import VX_gpu_pkg::*, VX_fpu_pkg::*; #(
    parameter LATENCY  = 6,
    parameter MAN_BITS = 23,  // mantissa bits (excluding hidden bit): 23=F32, 52=F64
    parameter EXP_BITS = 8,   // exponent bits: 8=F32, 11=F64
    // 1: use the FPGA DSP-optimized path — the vendor FP IP on Xilinx/Altera,
    //    else an inferred DSP48 mantissa multiply in the soft core. 0: ASIC soft.
    parameter USE_DSP  = 0,
    // 1: full IEEE subnormal support (soft core only). 0: flush-to-zero; required
    //    to use the vendor IP, which has no subnormal support.
    parameter SUBNORM_ENABLE = 1,
    // 1: produce IEEE special results + fflags. 0: assume finite operands, tie 0.
    parameter EXCEPT_ENABLE  = 1
) (
    input  wire clk,
    input  wire reset,
    input  wire enable,
    input  wire mask,

    input  wire [INST_FPU_BITS-1:0] op_type,
    input  wire [INST_FMT_BITS-1:0] fmt,
    input  wire [INST_FRM_BITS-1:0] frm,

    input  wire [MAN_BITS+EXP_BITS:0] dataa,
    input  wire [MAN_BITS+EXP_BITS:0] datab,
    input  wire [MAN_BITS+EXP_BITS:0] datac,

    output wire [MAN_BITS+EXP_BITS:0] result,
    output wire [`FP_FLAGS_BITS-1:0]  fflags
);
    // The vendor FP IP exists only on Xilinx/Altera FPGA flows and is
    // flush-to-zero, so it is gated on USE_DSP and SUBNORM_ENABLE=0.
`ifdef VIVADO
    localparam VENDOR_OK = 1;
`elsif QUARTUS
    localparam VENDOR_OK = 1;
`else
    localparam VENDOR_OK = 0;
`endif
    localparam IS_F32 = (MAN_BITS == 23) && (EXP_BITS == 8);
    localparam IS_F64 = (MAN_BITS == 52) && (EXP_BITS == 11);
    localparam W = MAN_BITS + EXP_BITS + 1;

    // Each vendor core's latency (Xilinx: its C_Latency in xilinx_ip_gen.tcl, the maximum).
`ifdef QUARTUS
    localparam ACL_FMA_LATENCY = 4;
    localparam VENDOR_F64 = 0; // acl_fmadd is F32-only
    localparam VENDOR_LATENCY = ACL_FMA_LATENCY;
`else
    localparam XIL_FMA_LATENCY   = 16;
    localparam XIL_FMA64_LATENCY = 26;
    localparam VENDOR_F64 = 1;
    localparam VENDOR_LATENCY = IS_F64 ? XIL_FMA64_LATENCY : XIL_FMA_LATENCY;
`endif
    localparam USE_VENDOR_IP = (USE_DSP != 0) && (SUBNORM_ENABLE == 0) && (VENDOR_OK != 0)
                            && (IS_F32 || (IS_F64 && (VENDOR_F64 != 0)));

    `STATIC_ASSERT(!USE_VENDOR_IP || (LATENCY >= VENDOR_LATENCY),
        ("vendor FMA core latency is %0d; VX_CFG_FMA_LATENCY=%0d is below it", VENDOR_LATENCY, LATENCY))

    if (USE_VENDOR_IP) begin : g_vendor
        // xil_fma(64) / acl_fmadd compute a*b+c, so the FMA-core opcodes are remapped:
        //   MUL        : a*b + -0  (-0 is the additive identity: a -0 product stays -0)
        //   ADD/SUB    : a*1.0 (+/-) b
        //   MADD/NMADD : (+/-)a*b (+/-) c
        // The vendor IP rounds round-to-nearest-even only (frm is ignored).
        wire is_madd = op_type[1];
        wire is_neg  = op_type[0];
        wire is_sub  = fmt[1];

        localparam [W-1:0] ONE      = {2'b00, {(EXP_BITS-1){1'b1}}, {MAN_BITS{1'b0}}};
        localparam [W-1:0] NEG_ZERO = {1'b1, {(W-1){1'b0}}};

        reg [W-1:0] a_in, b_in, c_in;
        always @(*) begin
            if (is_madd) begin
                a_in = {is_neg ^ dataa[W-1], dataa[0 +: W-1]};
                b_in = datab;
                c_in = {(is_neg ^ is_sub) ^ datac[W-1], datac[0 +: W-1]};
            end else begin
                if (is_neg) begin // MUL
                    a_in = dataa;
                    b_in = datab;
                    c_in = NEG_ZERO;
                end else begin // ADD/SUB
                    a_in = dataa;
                    b_in = ONE;
                    c_in = {is_sub ^ datab[W-1], datab[0 +: W-1]};
                end
            end
        end

        wire [W-1:0] ip_result;
        wire [`FP_FLAGS_BITS-1:0] ip_fflags;

        `UNUSED_VAR (mask)
        `UNUSED_VAR (op_type)
        `UNUSED_VAR (fmt)
        `UNUSED_VAR (frm)
    `ifdef QUARTUS
        acl_fmadd fmadd (
            .clk    (clk),
            .areset (1'b0),
            .en     (enable),
            .a      (a_in),
            .b      (b_in),
            .c      (c_in),
            .q      (ip_result)
        );
        assign ip_fflags = '0; // acl_fmadd does not expose exception flags
    `else // VIVADO
        wire [2:0] tuser;
        if (IS_F64) begin : g_f64
            xil_fma64 fma (
                .aclk                (clk),
                .aclken              (enable),
                .s_axis_a_tvalid     (1'b1),
                .s_axis_a_tdata      (a_in),
                .s_axis_b_tvalid     (1'b1),
                .s_axis_b_tdata      (b_in),
                .s_axis_c_tvalid     (1'b1),
                .s_axis_c_tdata      (c_in),
                `UNUSED_PIN (m_axis_result_tvalid),
                .m_axis_result_tdata (ip_result),
                .m_axis_result_tuser (tuser)
            );
        end else begin : g_f32
            xil_fma fma (
                .aclk                (clk),
                .aclken              (enable),
                .s_axis_a_tvalid     (1'b1),
                .s_axis_a_tdata      (a_in),
                .s_axis_b_tvalid     (1'b1),
                .s_axis_b_tdata      (b_in),
                .s_axis_c_tvalid     (1'b1),
                .s_axis_c_tdata      (c_in),
                `UNUSED_PIN (m_axis_result_tvalid),
                .m_axis_result_tdata (ip_result),
                .m_axis_result_tuser (tuser)
            );
        end
        // tuser = {invalid, overflow, underflow} -> fflags {NV, DZ, OF, UF, NX}.
        assign ip_fflags = EXCEPT_ENABLE ? {tuser[2], 1'b0, tuser[1], tuser[0], 1'b0} : '0;
    `endif

        VX_shift_register #(
            .DATAW (`FP_FLAGS_BITS + W),
            .DEPTH (LATENCY - VENDOR_LATENCY)
        ) pad (
            .clk      (clk),
            .reset    (reset),
            .enable   (enable),
            .data_in  ({ip_fflags, ip_result}),
            .data_out ({fflags, result})
        );
    end else begin : g_rtl
        VX_fma_unit_rtl #(
            .LATENCY        (LATENCY),
            .MAN_BITS       (MAN_BITS),
            .EXP_BITS       (EXP_BITS),
            .USE_DSP        (USE_DSP),
            .SUBNORM_ENABLE   (SUBNORM_ENABLE),
            .EXCEPT_ENABLE  (EXCEPT_ENABLE)
        ) core (
            .clk     (clk),
            .reset   (reset),
            .enable  (enable),
            .mask    (mask),
            .op_type (op_type),
            .fmt     (fmt),
            .frm     (frm),
            .dataa   (dataa),
            .datab   (datab),
            .datac   (datac),
            .result  (result),
            .fflags  (fflags)
        );
    end

endmodule
