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
// USE_DSP and flush-to-zero, the F32 path maps onto the hardened floating-point
// operator IP, which is far smaller and faster than soft RTL on FPGA. Every other
// case (generic FPGA, ASIC, simulation, F64, or any config needing subnormals)
// uses the portable pure-RTL core VX_fma_unit_rtl. The vendor IP is flush-to-zero
// and round-to-nearest-even only, so it is selected only when SNORM_ENABLE=0.

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
    parameter SNORM_ENABLE = 1,
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
    // The vendor FP IP exists only on Xilinx/Altera FPGA flows, is F32-only here,
    // and is flush-to-zero, so it is gated on USE_DSP and SNORM_ENABLE=0.
`ifdef VIVADO
    localparam VENDOR_OK = 1;
`elsif QUARTUS
    localparam VENDOR_OK = 1;
`else
    localparam VENDOR_OK = 0;
`endif
    // Vendor IP is selected only with SNORM_ENABLE=0 (it is flush-to-zero); any
    // config needing subnormals (SNORM_ENABLE=1) falls back to the IEEE soft core,
    // so USE_DSP also keeps its soft meaning (DSP-inferred multiply) there.
    localparam IS_F32 = (MAN_BITS == 23) && (EXP_BITS == 8);
    localparam USE_VENDOR_IP = (USE_DSP != 0) && (SNORM_ENABLE == 0) && IS_F32 && (VENDOR_OK != 0);

    // The vendor IP latency is fixed by xilinx_ip_gen.tcl (C_Latency=16); the
    // surrounding pipeline assumes LATENCY cycles, so the two must agree.
    `STATIC_ASSERT(!USE_VENDOR_IP || (LATENCY == 16),
        ("vendor xil_fma latency is 16; set VX_CFG_LATENCY_FMA=16"))

    if (USE_VENDOR_IP) begin : g_vendor
        // xil_fma / acl_fmadd compute a*b+c, so the FMA-core opcodes are remapped:
        //   MUL        : a*b + 0
        //   ADD/SUB    : a*1.0 (+/-) b
        //   MADD/NMADD : (+/-)a*b (+/-) c
        // The vendor IP rounds round-to-nearest-even only (frm is ignored).
        wire is_madd = op_type[1];
        wire is_neg  = op_type[0];
        wire is_sub  = fmt[1];

        reg [31:0] a32, b32, c32;
        always @(*) begin
            if (is_madd) begin
                a32 = {is_neg ^ dataa[31], dataa[0 +: 31]};
                b32 = datab[31:0];
                c32 = {(is_neg ^ is_sub) ^ datac[31], datac[0 +: 31]};
            end else begin
                if (is_neg) begin // MUL
                    a32 = dataa[31:0];
                    b32 = datab[31:0];
                    c32 = '0;
                end else begin // ADD/SUB
                    a32 = dataa[31:0];
                    b32 = 32'h3f800000; // 1.0f
                    c32 = {is_sub ^ datab[31], datab[0 +: 31]};
                end
            end
        end

        `UNUSED_VAR (reset)
        `UNUSED_VAR (mask)
        `UNUSED_VAR (frm)
    `ifdef QUARTUS
        acl_fmadd fmadd (
            .clk    (clk),
            .areset (1'b0),
            .en     (enable),
            .a      (a32),
            .b      (b32),
            .c      (c32),
            .q      (result)
        );
        assign fflags = '0; // acl_fmadd does not expose exception flags
    `else // VIVADO
        wire [2:0] tuser;
        xil_fma fma (
            .aclk                (clk),
            .aclken              (enable),
            .s_axis_a_tvalid     (1'b1),
            .s_axis_a_tdata      (a32),
            .s_axis_b_tvalid     (1'b1),
            .s_axis_b_tdata      (b32),
            .s_axis_c_tvalid     (1'b1),
            .s_axis_c_tdata      (c32),
            `UNUSED_PIN (m_axis_result_tvalid),
            .m_axis_result_tdata (result),
            .m_axis_result_tuser (tuser)
        );
        // tuser = {invalid, overflow, underflow} -> fflags {NV, DZ, OF, UF, NX}.
        assign fflags = EXCEPT_ENABLE ? {tuser[2], 1'b0, tuser[1], tuser[0], 1'b0} : '0;
    `endif
    end else begin : g_rtl
        VX_fma_unit_rtl #(
            .LATENCY        (LATENCY),
            .MAN_BITS       (MAN_BITS),
            .EXP_BITS       (EXP_BITS),
            .USE_DSP        (USE_DSP),
            .SNORM_ENABLE   (SNORM_ENABLE),
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
