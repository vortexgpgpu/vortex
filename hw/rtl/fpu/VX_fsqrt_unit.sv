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
// USE_DSP and flush-to-zero, the F32 path maps onto the hardened floating-point
// operator IP; otherwise it uses the portable pure-RTL core VX_fsqrt_unit_rtl. The
// vendor IP is flush-to-zero and round-to-nearest-even only (gated on SNORM_ENABLE=0).

`include "VX_fpu_define.vh"

module VX_fsqrt_unit import VX_gpu_pkg::*, VX_fpu_pkg::*; #(
    parameter LATENCY = 17,
    parameter FLEN    = 32,
    // 1: use the FPGA vendor FP IP on Xilinx/Altera (F32). 0: ASIC soft core.
    parameter USE_DSP = 0,
    // 1: full IEEE subnormals (soft core only). 0: flush-to-zero; required for IP.
    parameter SNORM_ENABLE = 1,
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
    // Vendor FP IP: Xilinx/Altera FPGA flows only, F32-only, flush-to-zero.
`ifdef VIVADO
    localparam VENDOR_OK = 1;
`elsif QUARTUS
    localparam VENDOR_OK = 1;
`else
    localparam VENDOR_OK = 0;
`endif
    // Vendor IP is selected only with SNORM_ENABLE=0 (flush-to-zero); SNORM_ENABLE=1
    // falls back to the IEEE soft core.
    localparam IS_F32 = (FLEN == 32);
    localparam USE_VENDOR_IP = (USE_DSP != 0) && (SNORM_ENABLE == 0) && IS_F32 && (VENDOR_OK != 0);

    `STATIC_ASSERT(!USE_VENDOR_IP || (LATENCY == 28),
        ("vendor xil_fsqrt latency is 28; set VX_CFG_LATENCY_FSQRT=28"))

    if (USE_VENDOR_IP) begin : g_vendor
        // The vendor IP rounds round-to-nearest-even only (frm is ignored).
        `UNUSED_VAR (reset)
        `UNUSED_VAR (mask)
        `UNUSED_VAR (fmt)
        `UNUSED_VAR (frm)
    `ifdef QUARTUS
        acl_fsqrt fsqrt (
            .clk    (clk),
            .areset (1'b0),
            .en     (enable),
            .a      (dataa),
            .q      (result)
        );
        assign fflags = '0; // acl_fsqrt does not expose exception flags
    `else // VIVADO
        wire tuser;
        xil_fsqrt fsqrt (
            .aclk                (clk),
            .aclken              (enable),
            .s_axis_a_tvalid     (1'b1),
            .s_axis_a_tdata      (dataa),
            `UNUSED_PIN (m_axis_result_tvalid),
            .m_axis_result_tdata (result),
            .m_axis_result_tuser (tuser)
        );
        // tuser = invalid -> fflags {NV, DZ, OF, UF, NX}.
        assign fflags = EXCEPT_ENABLE ? {tuser, 1'b0, 1'b0, 1'b0, 1'b0} : '0;
    `endif
    end else begin : g_rtl
        VX_fsqrt_unit_rtl #(
            .LATENCY       (LATENCY),
            .FLEN          (FLEN),
            .USE_DSP       (USE_DSP),
            .SNORM_ENABLE  (SNORM_ENABLE),
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
