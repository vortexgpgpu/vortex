`timescale 1ns/1ns
`include "VX_generic_queue_ll.v"

`define check(x, y) if ((x == y) !== 1) if ((x == y) === 0) $error("x=%h, expected=%h", x, y); else $warning("x=%h, expected=%h", x, y)

module testbench();

  reg clk;
  reg reset;
  reg[3:0] in_data;
  reg push;
  reg pop;
  wire io_enq_ready;
  wire[3:0] out_data;
  wire io_deq_valid;

  wire full, empty;

  assign io_enq_ready = !full;
  assign io_deq_valid = !empty;

  VX_generic_queue_ll #(.DATAW(4), .SIZE(4)) dut (
                                .clk(clk), 
                                .reset(reset), 
                                .in_data(in_data), 
                                .push(push), 
                                .pop(pop), 
                                .out_data(out_data), 
                                .empty(empty), 
                                .full(full));

  always begin
    #1 clk = !clk;
  end

  initial begin
    $monitor ("%d: clk=%b rst=%b push=%b, pop=%b, din=%h, empty=%b, full=%b, dout=%h", $time, clk, reset, push, pop, in_data, empty, full, out_data);
    #0 clk=0; reset=1; in_data=4'hd; push=1; pop=1;
    #1 `check(io_enq_ready, 1); `check(out_data, 4'hd); `check(io_deq_valid, 1);
    #1 `check(io_enq_ready, 1); `check(out_data, 4'hx); `check(io_deq_valid, 0);
    #0 reset=0; in_data=4'ha; pop=0;
    #1 `check(io_enq_ready, 1); `check(out_data, 4'hx); `check(io_deq_valid, 0);
    #1 `check(io_enq_ready, 1); `check(out_data, 4'ha); `check(io_deq_valid, 1);
    #0 in_data=4'hb;
    #1 `check(io_enq_ready, 1); `check(out_data, 4'ha); `check(io_deq_valid, 1);
    #1 `check(io_enq_ready, 1); `check(out_data, 4'ha); `check(io_deq_valid, 1);
    #0 in_data=4'hc;
    #1 `check(io_enq_ready, 1); `check(out_data, 4'ha); `check(io_deq_valid, 1);
    #1 `check(io_enq_ready, 1); `check(out_data, 4'ha); `check(io_deq_valid, 1);
    #0 in_data=4'hd;
    #1 `check(io_enq_ready, 1); `check(out_data, 4'ha); `check(io_deq_valid, 1);
    #1 `check(io_enq_ready, 0); `check(out_data, 4'ha); `check(io_deq_valid, 1);
    #0 push=0; pop=1;
    #1 `check(io_enq_ready, 0); `check(out_data, 4'ha); `check(io_deq_valid, 1);
    #1 `check(io_enq_ready, 1); `check(out_data, 4'hb); `check(io_deq_valid, 1);
    #1 `check(io_enq_ready, 1); `check(out_data, 4'hb); `check(io_deq_valid, 1);
    #1 `check(io_enq_ready, 1); `check(out_data, 4'hc); `check(io_deq_valid, 1);
    #1 `check(io_enq_ready, 1); `check(out_data, 4'hc); `check(io_deq_valid, 1);
    #1 `check(io_enq_ready, 1); `check(out_data, 4'hd); `check(io_deq_valid, 1);
    #1 `check(io_enq_ready, 1); `check(out_data, 4'hd); `check(io_deq_valid, 1);
    #1 `check(io_enq_ready, 1); `check(out_data, 4'ha); `check(io_deq_valid, 0);
    #0 in_data=4'ha; push=1; pop=0;
    #1 `check(io_enq_ready, 1); `check(out_data, 4'ha); `check(io_deq_valid, 0);
    #1 `check(io_enq_ready, 1); `check(out_data, 4'ha); `check(io_deq_valid, 1);
    #0 in_data=4'hb; pop=1;
    #1 `check(io_enq_ready, 1); `check(out_data, 4'ha); `check(io_deq_valid, 1);
    #1 `check(io_enq_ready, 1); `check(out_data, 4'hb); `check(io_deq_valid, 1);
    #0 push=0;
    #1 `check(io_enq_ready, 1); `check(out_data, 4'hb); `check(io_deq_valid, 1);
    #1 `check(io_enq_ready, 1); `check(out_data, 4'hc); `check(io_deq_valid, 0);
    #1 $finish;
  end

endmodule
