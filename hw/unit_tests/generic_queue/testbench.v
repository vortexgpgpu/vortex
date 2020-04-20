`timescale 1ns/1ns
`include "VX_generic_queue.v"

`define check(x, y) if ((x == y) !== 1) if ((x == y) === 0) $error("x=%h, expected=%h", x, y); else $warning("x=%h, expected=%h", x, y)

module testbench();

  reg clk;
  reg reset;
  reg[3:0] in_data;
  reg push;
  reg pop;
  wire[3:0] out_data;
  wire full;
  wire empty;

  VX_generic_queue #(.DATAW(4), .SIZE(4)) dut (
                                .clk(clk), 
                                .reset(reset), 
                                .data_i(in_data), 
                                .push(push), 
                                .pop(pop), 
                                .data_o(out_data), 
                                .empty(empty), 
                                .full(full));

  always begin
    #1 clk = !clk;
  end

  initial begin
    $monitor ("%d: clk=%b rst=%b push=%b, pop=%b, din=%h, empty=%b, full=%b, dout=%h", 
              $time, clk, reset, push, pop, in_data, empty, full, out_data);
    #0 clk=0; reset=1; pop=0; push=0;   
    #2 reset=0; in_data=4'ha; pop=0; push=1;
    #2 `check(full, 0); `check(out_data, 4'ha); `check(empty, 0);
    #0 in_data=4'hb;
    #2 `check(full, 0); `check(out_data, 4'ha); `check(empty, 0);
    #0 in_data=4'hc;
    #2 `check(full, 0); `check(out_data, 4'ha); `check(empty, 0);
    #0 in_data=4'hd;
    #2 `check(full, 1); `check(out_data, 4'ha); `check(empty, 0);
    #0 push=0; pop=1;
    #2 `check(full, 0); `check(out_data, 4'hb); `check(empty, 0);
    #2 `check(full, 0); `check(out_data, 4'hc); `check(empty, 0);
    #2 `check(full, 0); `check(out_data, 4'hd); `check(empty, 0);
    #2 `check(full, 0); `check(out_data, 4'ha); `check(empty, 1);
    #0 in_data=4'he; push=1; pop=0;
    #2 `check(full, 0); `check(out_data, 4'he); `check(empty, 0);
    #0 in_data=4'hf; pop=1;
    #2 `check(full, 0); `check(out_data, 4'hf); `check(empty, 0);
    #0 push=0;
    #2 `check(full, 0); `check(out_data, 4'hc); `check(empty, 1);
    #1 $finish;
  end

endmodule
