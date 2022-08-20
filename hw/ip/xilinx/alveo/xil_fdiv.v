// (c) Copyright 1995-2022 Xilinx, Inc. All rights reserved.
// 
// This file contains confidential and proprietary information
// of Xilinx, Inc. and is protected under U.S. and
// international copyright and other intellectual property
// laws.
// 
// DISCLAIMER
// This disclaimer is not a license and does not grant any
// rights to the materials distributed herewith. Except as
// otherwise provided in a valid license issued to you by
// Xilinx, and to the maximum extent permitted by applicable
// law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
// WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
// AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
// BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
// INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
// (2) Xilinx shall not be liable (whether in contract or tort,
// including negligence, or under any other theory of
// liability) for any loss or damage of any kind or nature
// related to, arising under or in connection with these
// materials, including for any direct, or any indirect,
// special, incidental, or consequential loss or damage
// (including loss of data, profits, goodwill, or any type of
// loss or damage suffered as a result of any action brought
// by a third party) even if such damage or loss was
// reasonably foreseeable or Xilinx had been advised of the
// possibility of the same.
// 
// CRITICAL APPLICATIONS
// Xilinx products are not designed or intended to be fail-
// safe, or for use in any application requiring fail-safe
// performance, such as life-support or safety devices or
// systems, Class III medical devices, nuclear facilities,
// applications related to the deployment of airbags, or any
// other applications that could lead to death, personal
// injury, or severe property or environmental damage
// (individually and collectively, "Critical
// Applications"). Customer assumes the sole risk and
// liability of any use of Xilinx products in Critical
// Applications, subject only to applicable laws and
// regulations governing limitations on product liability.
// 
// THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
// PART OF THIS FILE AT ALL TIMES.
// 
// DO NOT MODIFY THIS FILE.


// IP VLNV: xilinx.com:ip:floating_point:7.1
// IP Revision: 11

(* X_CORE_INFO = "floating_point_v7_1_11,Vivado 2020.2.2" *)
(* CHECK_LICENSE_TYPE = "fdiv,floating_point_v7_1_11,{}" *)
(* CORE_GENERATION_INFO = "fdiv,floating_point_v7_1_11,{x_ipProduct=Vivado 2020.2.2,x_ipVendor=xilinx.com,x_ipLibrary=ip,x_ipName=floating_point,x_ipVersion=7.1,x_ipCoreRevision=11,x_ipLanguage=VERILOG,x_ipSimLanguage=VERILOG,C_XDEVICEFAMILY=virtexuplusHBM,C_PART=xcu280-fsvh2892-2L-e,C_HAS_ADD=0,C_HAS_SUBTRACT=0,C_HAS_MULTIPLY=0,C_HAS_DIVIDE=1,C_HAS_SQRT=0,C_HAS_COMPARE=0,C_HAS_FIX_TO_FLT=0,C_HAS_FLT_TO_FIX=0,C_HAS_FLT_TO_FLT=0,C_HAS_RECIP=0,C_HAS_RECIP_SQRT=0,C_HAS_ABSOLUTE=0,C_HAS_LOGARITHM=0,C_HAS_EXPONENTIAL=0,C_HAS_F\
MA=0,C_HAS_FMS=0,C_HAS_UNFUSED_MULTIPLY_ADD=0,C_HAS_UNFUSED_MULTIPLY_SUB=0,C_HAS_UNFUSED_MULTIPLY_ACCUMULATOR_A=0,C_HAS_UNFUSED_MULTIPLY_ACCUMULATOR_S=0,C_HAS_ACCUMULATOR_A=0,C_HAS_ACCUMULATOR_S=0,C_HAS_ACCUMULATOR_PRIMITIVE_A=0,C_HAS_ACCUMULATOR_PRIMITIVE_S=0,C_A_WIDTH=32,C_A_FRACTION_WIDTH=24,C_B_WIDTH=32,C_B_FRACTION_WIDTH=24,C_C_WIDTH=32,C_C_FRACTION_WIDTH=24,C_RESULT_WIDTH=32,C_RESULT_FRACTION_WIDTH=24,C_COMPARE_OPERATION=8,C_LATENCY=28,C_OPTIMIZATION=1,C_MULT_USAGE=0,C_BRAM_USAGE=0,C_RATE=\
1,C_ACCUM_INPUT_MSB=32,C_ACCUM_MSB=32,C_ACCUM_LSB=-31,C_HAS_UNDERFLOW=1,C_HAS_OVERFLOW=1,C_HAS_INVALID_OP=1,C_HAS_DIVIDE_BY_ZERO=1,C_HAS_ACCUM_OVERFLOW=0,C_HAS_ACCUM_INPUT_OVERFLOW=0,C_HAS_ACLKEN=0,C_HAS_ARESETN=0,C_THROTTLE_SCHEME=3,C_HAS_A_TUSER=0,C_HAS_A_TLAST=0,C_HAS_B=1,C_HAS_B_TUSER=0,C_HAS_B_TLAST=0,C_HAS_C=0,C_HAS_C_TUSER=0,C_HAS_C_TLAST=0,C_HAS_OPERATION=0,C_HAS_OPERATION_TUSER=0,C_HAS_OPERATION_TLAST=0,C_HAS_RESULT_TUSER=1,C_HAS_RESULT_TLAST=0,C_TLAST_RESOLUTION=0,C_A_TDATA_WIDTH=32,C_\
A_TUSER_WIDTH=1,C_B_TDATA_WIDTH=32,C_B_TUSER_WIDTH=1,C_C_TDATA_WIDTH=32,C_C_TUSER_WIDTH=1,C_OPERATION_TDATA_WIDTH=8,C_OPERATION_TUSER_WIDTH=1,C_RESULT_TDATA_WIDTH=32,C_RESULT_TUSER_WIDTH=4,C_FIXED_DATA_UNSIGNED=0}" *)
(* DowngradeIPIdentifiedWarnings = "yes" *)
module xil_fdiv (
  aclk,
  s_axis_a_tvalid,
  s_axis_a_tdata,
  s_axis_b_tvalid,
  s_axis_b_tdata,
  m_axis_result_tvalid,
  m_axis_result_tdata,
  m_axis_result_tuser
);

(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME aclk_intf, ASSOCIATED_BUSIF S_AXIS_OPERATION:M_AXIS_RESULT:S_AXIS_C:S_AXIS_B:S_AXIS_A, ASSOCIATED_RESET aresetn, ASSOCIATED_CLKEN aclken, FREQ_HZ 10000000, FREQ_TOLERANCE_HZ 0, PHASE 0.000, INSERT_VIP 0" *)
(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk_intf CLK" *)
input wire aclk;
(* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_A TVALID" *)
input wire s_axis_a_tvalid;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME S_AXIS_A, TDATA_NUM_BYTES 4, TDEST_WIDTH 0, TID_WIDTH 0, TUSER_WIDTH 0, HAS_TREADY 0, HAS_TSTRB 0, HAS_TKEEP 0, HAS_TLAST 0, FREQ_HZ 100000000, PHASE 0.000, LAYERED_METADATA undef, INSERT_VIP 0" *)
(* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_A TDATA" *)
input wire [31 : 0] s_axis_a_tdata;
(* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_B TVALID" *)
input wire s_axis_b_tvalid;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME S_AXIS_B, TDATA_NUM_BYTES 4, TDEST_WIDTH 0, TID_WIDTH 0, TUSER_WIDTH 0, HAS_TREADY 0, HAS_TSTRB 0, HAS_TKEEP 0, HAS_TLAST 0, FREQ_HZ 100000000, PHASE 0.000, LAYERED_METADATA undef, INSERT_VIP 0" *)
(* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_B TDATA" *)
input wire [31 : 0] s_axis_b_tdata;
(* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS_RESULT TVALID" *)
output wire m_axis_result_tvalid;
(* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS_RESULT TDATA" *)
output wire [31 : 0] m_axis_result_tdata;
(* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME M_AXIS_RESULT, TDATA_NUM_BYTES 4, TDEST_WIDTH 0, TID_WIDTH 0, TUSER_WIDTH 4, HAS_TREADY 0, HAS_TSTRB 0, HAS_TKEEP 0, HAS_TLAST 0, FREQ_HZ 100000000, PHASE 0.000, LAYERED_METADATA undef, INSERT_VIP 0" *)
(* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS_RESULT TUSER" *)
output wire [3 : 0] m_axis_result_tuser;

  floating_point_v7_1_11 #(
    .C_XDEVICEFAMILY("virtexuplusHBM"),
    .C_PART("xcu280-fsvh2892-2L-e"),
    .C_HAS_ADD(0),
    .C_HAS_SUBTRACT(0),
    .C_HAS_MULTIPLY(0),
    .C_HAS_DIVIDE(1),
    .C_HAS_SQRT(0),
    .C_HAS_COMPARE(0),
    .C_HAS_FIX_TO_FLT(0),
    .C_HAS_FLT_TO_FIX(0),
    .C_HAS_FLT_TO_FLT(0),
    .C_HAS_RECIP(0),
    .C_HAS_RECIP_SQRT(0),
    .C_HAS_ABSOLUTE(0),
    .C_HAS_LOGARITHM(0),
    .C_HAS_EXPONENTIAL(0),
    .C_HAS_FMA(0),
    .C_HAS_FMS(0),
    .C_HAS_UNFUSED_MULTIPLY_ADD(0),
    .C_HAS_UNFUSED_MULTIPLY_SUB(0),
    .C_HAS_UNFUSED_MULTIPLY_ACCUMULATOR_A(0),
    .C_HAS_UNFUSED_MULTIPLY_ACCUMULATOR_S(0),
    .C_HAS_ACCUMULATOR_A(0),
    .C_HAS_ACCUMULATOR_S(0),
    .C_HAS_ACCUMULATOR_PRIMITIVE_A(0),
    .C_HAS_ACCUMULATOR_PRIMITIVE_S(0),
    .C_A_WIDTH(32),
    .C_A_FRACTION_WIDTH(24),
    .C_B_WIDTH(32),
    .C_B_FRACTION_WIDTH(24),
    .C_C_WIDTH(32),
    .C_C_FRACTION_WIDTH(24),
    .C_RESULT_WIDTH(32),
    .C_RESULT_FRACTION_WIDTH(24),
    .C_COMPARE_OPERATION(8),
    .C_LATENCY(28),
    .C_OPTIMIZATION(1),
    .C_MULT_USAGE(0),
    .C_BRAM_USAGE(0),
    .C_RATE(1),
    .C_ACCUM_INPUT_MSB(32),
    .C_ACCUM_MSB(32),
    .C_ACCUM_LSB(-31),
    .C_HAS_UNDERFLOW(1),
    .C_HAS_OVERFLOW(1),
    .C_HAS_INVALID_OP(1),
    .C_HAS_DIVIDE_BY_ZERO(1),
    .C_HAS_ACCUM_OVERFLOW(0),
    .C_HAS_ACCUM_INPUT_OVERFLOW(0),
    .C_HAS_ACLKEN(0),
    .C_HAS_ARESETN(0),
    .C_THROTTLE_SCHEME(3),
    .C_HAS_A_TUSER(0),
    .C_HAS_A_TLAST(0),
    .C_HAS_B(1),
    .C_HAS_B_TUSER(0),
    .C_HAS_B_TLAST(0),
    .C_HAS_C(0),
    .C_HAS_C_TUSER(0),
    .C_HAS_C_TLAST(0),
    .C_HAS_OPERATION(0),
    .C_HAS_OPERATION_TUSER(0),
    .C_HAS_OPERATION_TLAST(0),
    .C_HAS_RESULT_TUSER(1),
    .C_HAS_RESULT_TLAST(0),
    .C_TLAST_RESOLUTION(0),
    .C_A_TDATA_WIDTH(32),
    .C_A_TUSER_WIDTH(1),
    .C_B_TDATA_WIDTH(32),
    .C_B_TUSER_WIDTH(1),
    .C_C_TDATA_WIDTH(32),
    .C_C_TUSER_WIDTH(1),
    .C_OPERATION_TDATA_WIDTH(8),
    .C_OPERATION_TUSER_WIDTH(1),
    .C_RESULT_TDATA_WIDTH(32),
    .C_RESULT_TUSER_WIDTH(4),
    .C_FIXED_DATA_UNSIGNED(0)
  ) inst (
    .aclk(aclk),
    .aclken(1'H1),
    .aresetn(1'H1),
    .s_axis_a_tvalid(s_axis_a_tvalid),
    .s_axis_a_tready(),
    .s_axis_a_tdata(s_axis_a_tdata),
    .s_axis_a_tuser(1'B0),
    .s_axis_a_tlast(1'H0),
    .s_axis_b_tvalid(s_axis_b_tvalid),
    .s_axis_b_tready(),
    .s_axis_b_tdata(s_axis_b_tdata),
    .s_axis_b_tuser(1'B0),
    .s_axis_b_tlast(1'H0),
    .s_axis_c_tvalid(1'H0),
    .s_axis_c_tready(),
    .s_axis_c_tdata(32'B0),
    .s_axis_c_tuser(1'B0),
    .s_axis_c_tlast(1'H0),
    .s_axis_operation_tvalid(1'H0),
    .s_axis_operation_tready(),
    .s_axis_operation_tdata(8'B0),
    .s_axis_operation_tuser(1'B0),
    .s_axis_operation_tlast(1'H0),
    .m_axis_result_tvalid(m_axis_result_tvalid),
    .m_axis_result_tready(1'H0),
    .m_axis_result_tdata(m_axis_result_tdata),
    .m_axis_result_tuser(m_axis_result_tuser),
    .m_axis_result_tlast()
  );
endmodule
