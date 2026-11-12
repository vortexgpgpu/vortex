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

// Black-box declarations of the Xilinx floating-point cores that
// xilinx_ip_gen.tcl generates: their ports as configured there, no contents.
// The CIRCT elaboration reads them to know each core's interface and keeps
// every instance as an instance; Vivado binds it to the generated core.
// Synthesis never reads this file. Keep it in step with xilinx_ip_gen.tcl.
//
// Each declares the contract the RTL uses it under: a fixed-latency
// pipeline. Its data_out outputs are a function of its data_in inputs of
// exactly fixed_latency cycles of `clock` with `clock_enable` high before,
// and of nothing else (pure_pipeline: non-blocking flow control, no reset, no other state); the `tie`
// inputs are driven with those constants, its other outputs are unused.
// fixed_latency is the core's C_Latency in xilinx_ip_gen.tcl and the latency
// the RTL asserts (hw/scripts/check_ip_contracts.py checks both).

(* black_box, fixed_latency = 28, clock = "aclk", clock_enable = "aclken",
   data_in = "s_axis_a_tdata s_axis_b_tdata",
   data_out = "m_axis_result_tdata m_axis_result_tuser",
   tie = "s_axis_a_tvalid=1 s_axis_b_tvalid=1", pure_pipeline *)
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
endmodule

(* black_box, fixed_latency = 28, clock = "aclk", clock_enable = "aclken",
   data_in = "s_axis_a_tdata",
   data_out = "m_axis_result_tdata m_axis_result_tuser",
   tie = "s_axis_a_tvalid=1", pure_pipeline *)
module xil_fsqrt (
    input  wire        aclk,
    input  wire        aclken,
    input  wire        s_axis_a_tvalid,
    input  wire [31:0] s_axis_a_tdata,
    output wire        m_axis_result_tvalid,
    output wire [31:0] m_axis_result_tdata,
    output wire [0:0]  m_axis_result_tuser   // invalid
);
endmodule

(* black_box, fixed_latency = 16, clock = "aclk", clock_enable = "aclken",
   data_in = "s_axis_a_tdata s_axis_b_tdata s_axis_c_tdata",
   data_out = "m_axis_result_tdata m_axis_result_tuser",
   tie = "s_axis_a_tvalid=1 s_axis_b_tvalid=1 s_axis_c_tvalid=1", pure_pipeline *)
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
endmodule

(* black_box, fixed_latency = 8, clock = "aclk", clock_enable = "aclken",
   data_in = "s_axis_a_tdata s_axis_b_tdata",
   data_out = "m_axis_result_tdata",
   tie = "s_axis_a_tvalid=1 s_axis_b_tvalid=1", pure_pipeline *)
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
endmodule

(* black_box, fixed_latency = 11, clock = "aclk", clock_enable = "aclken",
   data_in = "s_axis_a_tdata s_axis_b_tdata s_axis_operation_tdata",
   data_out = "m_axis_result_tdata",
   tie = "s_axis_a_tvalid=1 s_axis_b_tvalid=1 s_axis_operation_tvalid=1", pure_pipeline *)
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
endmodule
