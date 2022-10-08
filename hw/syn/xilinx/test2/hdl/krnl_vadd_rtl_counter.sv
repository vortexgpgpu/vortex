/**
* Copyright (C) 2019-2021 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

//-----------------------------------------------------------------------------
// Simple up/down counter with reset.
//-----------------------------------------------------------------------------
`default_nettype none
`timescale 1ps/1ps
module krnl_vadd_rtl_counter  #(
  parameter integer C_WIDTH  = 4,
  parameter [C_WIDTH-1:0] C_INIT = {C_WIDTH{1'b0}}
)
( 
  input  wire               clk,
  input  wire               clken,
  input  wire               rst,
  input  wire               load,
  input  wire               incr,
  input  wire               decr,
  input  wire [C_WIDTH-1:0] load_value,
  output wire [C_WIDTH-1:0] count,
  output wire               is_zero
);

  localparam [C_WIDTH-1:0] LP_ZERO = {C_WIDTH{1'b0}};
  localparam [C_WIDTH-1:0] LP_ONE = {{C_WIDTH-1{1'b0}},1'b1};
  localparam [C_WIDTH-1:0] LP_MAX = {C_WIDTH{1'b1}};

  reg [C_WIDTH-1:0] count_r = C_INIT;
  reg   is_zero_r = (C_INIT == LP_ZERO);

  assign count = count_r;

  always @(posedge clk) begin 
    if (rst) begin 
      count_r <= C_INIT;
    end
    else if (clken) begin 
      if (load) begin
        count_r <= load_value; 
      end
      else if (incr & ~decr) begin
        count_r <= count_r + 1'b1;
      end
      else if (~incr & decr) begin 
        count_r <= count_r - 1'b1;
      end
      else 
        count_r <= count_r;
    end
  end

  assign is_zero = is_zero_r;

  always @(posedge clk) begin 
    if (rst) begin 
      is_zero_r <= (C_INIT == LP_ZERO);
    end
    else if (clken) begin 
      if (load) begin 
        is_zero_r <= (load_value == LP_ZERO);
      end
      else begin
        is_zero_r <= incr ^ decr ? (decr && (count_r == LP_ONE)) || (incr && (count_r == LP_MAX)) : is_zero_r;
      end
    end
    else begin
      is_zero_r <= is_zero_r;
    end
  end


endmodule : krnl_vadd_rtl_counter
`default_nettype wire
