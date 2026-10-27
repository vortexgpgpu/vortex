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

// Simulation models of the Xilinx floating-point cores (floating_point 7.1)
// that hw/scripts/xilinx_ip_gen.tcl generates, for Verilator, which cannot run
// their encrypted netlists. Simulation only: never synthesized, never read by
// the Foldsim compiler (which sees the black boxes of xilinx_ip_stubs.sv).
//
// Each has its core's ports, latency and clock enable: its result is computed
// from its inputs by Vortex's DPI floating point (softfloat) and delayed by
// the core's C_Latency cycles of aclk with aclken high, the pipeline its
// contract declares. The numerics follow what PG060 documents for these
// configurations: round to nearest even only; denormal inputs taken as zero
// and denormal results flushed to zero, keeping the sign, with underflow
// raised; the flags of each core on m_axis_result_tuser. What the documents
// leave open is softfloat's (RISC-V) choice here: a NaN result is the
// canonical quiet NaN 0x7fc00000, and invalid is raised for signaling NaN
// inputs as for the invalid operations. Bit-exactness to the cores can only be
// confirmed on silicon.

`include "dpi_float.vh"

package xilinx_fp_model_pkg;
  // A denormal input, as the cores read it: zero of its sign.
  function automatic logic [31:0] ftz_in(input logic [31:0] x);
    return (x[30:23] == 8'h0) ? {x[31], 31'h0} : x;
  endfunction
  // A result as the cores give it: a denormal flushed to zero of its sign.
  function automatic logic [31:0] ftz_out(input logic [31:0] x);
    return (x[30:23] == 8'h0) ? {x[31], 31'h0} : x;
  endfunction
  function automatic logic is_denormal(input logic [31:0] x);
    return (x[30:23] == 8'h0) && (x[22:0] != 23'h0);
  endfunction
  function automatic logic [63:0] box(input logic [31:0] x);
    return {32'hffffffff, x};
  endfunction
endpackage

// The cores' pipeline: `D` bits taken on aclk with aclken high, given LATENCY
// such cycles later.
module xilinx_fp_model_pipe #(
  parameter LATENCY = 1,
  parameter D = 1
) (
  input  wire         aclk,
  input  wire         aclken,
  input  wire [D-1:0] d,
  output wire [D-1:0] q
);
  logic [D-1:0] stage [LATENCY];
  always @(posedge aclk) begin
    if (aclken) begin
      stage[0] <= d;
      for (int i = 1; i < LATENCY; ++i)
        stage[i] <= stage[i - 1];
    end
  end
  assign q = stage[LATENCY - 1];
endmodule

module xil_fma (
  input  wire        aclk,
  input  wire        aclken,
  input  wire        s_axis_a_tvalid,
  input  wire [31:0] s_axis_a_tdata,
  input  wire        s_axis_b_tvalid,
  input  wire [31:0] s_axis_b_tdata,
  input  wire        s_axis_c_tvalid,
  input  wire [31:0] s_axis_c_tdata,
  output wire        m_axis_result_tvalid,
  output wire [31:0] m_axis_result_tdata,
  output wire [2:0]  m_axis_result_tuser   // underflow, overflow, invalid
);
  import xilinx_fp_model_pkg::*;
  logic [63:0] r;
  logic [4:0] f;
  always @(*) begin
    dpi_fmadd(1'b1, 0, box(ftz_in(s_axis_a_tdata)), box(ftz_in(s_axis_b_tdata)),
              box(ftz_in(s_axis_c_tdata)), 3'b000, r, f);
  end
  wire tiny = is_denormal(r[31:0]);
  // tuser = {invalid, overflow, underflow}; fflags = {NV, DZ, OF, UF, NX}.
  xilinx_fp_model_pipe #(.LATENCY(16), .D(1 + 32 + 3)) pipe (
    .aclk(aclk), .aclken(aclken),
    .d({s_axis_a_tvalid & s_axis_b_tvalid & s_axis_c_tvalid, ftz_out(r[31:0]), f[4], f[2], f[1] | tiny}),
    .q({m_axis_result_tvalid, m_axis_result_tdata, m_axis_result_tuser}));
endmodule

module xil_fdiv (
  input  wire        aclk,
  input  wire        aclken,
  input  wire        s_axis_a_tvalid,
  input  wire [31:0] s_axis_a_tdata,
  input  wire        s_axis_b_tvalid,
  input  wire [31:0] s_axis_b_tdata,
  output wire        m_axis_result_tvalid,
  output wire [31:0] m_axis_result_tdata,
  output wire [3:0]  m_axis_result_tuser   // underflow, overflow, invalid, divide by zero
);
  import xilinx_fp_model_pkg::*;
  logic [63:0] r;
  logic [4:0] f;
  always @(*) begin
    dpi_fdiv(1'b1, 0, box(ftz_in(s_axis_a_tdata)), box(ftz_in(s_axis_b_tdata)), 3'b000, r, f);
  end
  wire tiny = is_denormal(r[31:0]);
  // tuser = {divide by zero, invalid, overflow, underflow}.
  xilinx_fp_model_pipe #(.LATENCY(28), .D(1 + 32 + 4)) pipe (
    .aclk(aclk), .aclken(aclken),
    .d({s_axis_a_tvalid & s_axis_b_tvalid, ftz_out(r[31:0]), f[3], f[4], f[2], f[1] | tiny}),
    .q({m_axis_result_tvalid, m_axis_result_tdata, m_axis_result_tuser}));
endmodule

module xil_fsqrt (
  input  wire        aclk,
  input  wire        aclken,
  input  wire        s_axis_a_tvalid,
  input  wire [31:0] s_axis_a_tdata,
  output wire        m_axis_result_tvalid,
  output wire [31:0] m_axis_result_tdata,
  output wire [0:0]  m_axis_result_tuser   // invalid
);
  import xilinx_fp_model_pkg::*;
  logic [63:0] r;
  logic [4:0] f;
  always @(*) begin
    dpi_fsqrt(1'b1, 0, box(ftz_in(s_axis_a_tdata)), 3'b000, r, f);
  end
  xilinx_fp_model_pipe #(.LATENCY(28), .D(1 + 32 + 1)) pipe (
    .aclk(aclk), .aclken(aclken),
    .d({s_axis_a_tvalid, ftz_out(r[31:0]), f[4]}),
    .q({m_axis_result_tvalid, m_axis_result_tdata, m_axis_result_tuser}));
endmodule

module xil_fmul (
  input  wire        aclk,
  input  wire        aclken,
  input  wire        s_axis_a_tvalid,
  input  wire [31:0] s_axis_a_tdata,
  input  wire        s_axis_b_tvalid,
  input  wire [31:0] s_axis_b_tdata,
  output wire        m_axis_result_tvalid,
  output wire [31:0] m_axis_result_tdata
);
  import xilinx_fp_model_pkg::*;
  logic [63:0] r;
  logic [4:0] f;
  always @(*) begin
    dpi_fmul(1'b1, 0, box(ftz_in(s_axis_a_tdata)), box(ftz_in(s_axis_b_tdata)), 3'b000, r, f);
  end
  xilinx_fp_model_pipe #(.LATENCY(8), .D(1 + 32)) pipe (
    .aclk(aclk), .aclken(aclken),
    .d({s_axis_a_tvalid & s_axis_b_tvalid, ftz_out(r[31:0])}),
    .q({m_axis_result_tvalid, m_axis_result_tdata}));
endmodule

module xil_fadd (
  input  wire        aclk,
  input  wire        aclken,
  input  wire        s_axis_a_tvalid,
  input  wire [31:0] s_axis_a_tdata,
  input  wire        s_axis_b_tvalid,
  input  wire [31:0] s_axis_b_tdata,
  input  wire        s_axis_operation_tvalid,
  input  wire [7:0]  s_axis_operation_tdata,
  output wire        m_axis_result_tvalid,
  output wire [31:0] m_axis_result_tdata
);
  import xilinx_fp_model_pkg::*;
  logic [63:0] r;
  logic [4:0] f;
  // Operation bit 0: subtract.
  always @(*) begin
    if (s_axis_operation_tdata[0])
      dpi_fsub(1'b1, 0, box(ftz_in(s_axis_a_tdata)), box(ftz_in(s_axis_b_tdata)), 3'b000, r, f);
    else
      dpi_fadd(1'b1, 0, box(ftz_in(s_axis_a_tdata)), box(ftz_in(s_axis_b_tdata)), 3'b000, r, f);
  end
  xilinx_fp_model_pipe #(.LATENCY(11), .D(1 + 32)) pipe (
    .aclk(aclk), .aclken(aclken),
    .d({s_axis_a_tvalid & s_axis_b_tvalid & s_axis_operation_tvalid, ftz_out(r[31:0])}),
    .q({m_axis_result_tvalid, m_axis_result_tdata}));
endmodule
