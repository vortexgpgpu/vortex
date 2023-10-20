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

`timescale 1ns/1ns
`include "VX_fifo_queue.v"

`define check(x, y) if ((x == y) !== 1) if ((x == y) === 0) $error("x=%h, expected=%h", x, y); else $warning("x=%h, expected=%h", x, y)

module testbench();

  reg clk;
  reg reset;
  reg[3:0] data_in;
  reg push;
  reg pop;
  wire[3:0] data_out;
  wire full;
  wire empty;

  VX_fifo_queue #(
    .DATAW (4), 
    .DEPTH (4)
  ) dut (
    .clk(clk), 
    .reset(reset), 
    .data_in(data_in), 
    .push(push), 
    .pop(pop), 
    .data_out(data_out), 
    .empty(empty), 
    .full(full),
    `UNUSED_PIN (alm_empty),    
    `UNUSED_PIN (alm_full),
    `UNUSED_VAR (size)
  );

  always begin
    #1 clk = !clk;
  end

  initial begin
    $monitor ("%d: clk=%b rst=%b push=%b, pop=%b, din=%h, empty=%b, full=%b, dout=%h", 
              $time, clk, reset, push, pop, data_in, empty, full, data_out);
    #0 clk=0; reset=1; pop=0; push=0;   
    #2 reset=0; data_in=4'ha; pop=0; push=1;
    #2 `check(full, 0); `check(data_out, 4'ha); `check(empty, 0);
    #0 data_in=4'hb;
    #2 `check(full, 0); `check(data_out, 4'ha); `check(empty, 0);
    #0 data_in=4'hc;
    #2 `check(full, 0); `check(data_out, 4'ha); `check(empty, 0);
    #0 data_in=4'hd;
    #2 `check(full, 1); `check(data_out, 4'ha); `check(empty, 0);
    #0 push=0; pop=1;
    #2 `check(full, 0); `check(data_out, 4'hb); `check(empty, 0);
    #2 `check(full, 0); `check(data_out, 4'hc); `check(empty, 0);
    #2 `check(full, 0); `check(data_out, 4'hd); `check(empty, 0);
    #2 `check(full, 0); `check(data_out, 4'ha); `check(empty, 1);
    #0 data_in=4'he; push=1; pop=0;
    #2 `check(full, 0); `check(data_out, 4'he); `check(empty, 0);
    #0 data_in=4'hf; pop=1;
    #2 `check(full, 0); `check(data_out, 4'hf); `check(empty, 0);
    #0 push=0;
    #2 `check(full, 0); `check(data_out, 4'hc); `check(empty, 1);
    #1 $finish;
  end

endmodule
