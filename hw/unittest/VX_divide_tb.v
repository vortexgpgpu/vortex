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

`timescale 1ns/1ps

module VX_tb_divide();

`ifdef TRACE
    initial
    begin
    $dumpfile("trace.vcd");
    $dumpvars(0,test);
    end
`endif

    reg clk;
    reg rst;

    reg [31:0] numer, denom;

    wire [31:0] o_div[0:7], o_rem[0:7];

    for (genvar i = 0; i < 8; ++i) begin
        VX_divide#(
            .WIDTHN(32),
            .WIDTHD(32),
            .WIDTHQ(32),
            .WIDTHR(32),
            .PIPELINE(i)
        ) div(
            .clock(clk),
            .aclr(rst),
            .clken(1'b1),
            .numer(numer),
            .denom(denom),
            .quotient(o_div[i]),
            .remainder(o_rem[i])
        );
    end

    initial begin
        clk = 0; rst = 0;

        numer = 56;
        denom = 11;

        $display("56 / 11 #0");
        if (o_div[0] != 5 || o_rem[0] != 1) begin
            $display("PIPE0: div=", o_div[0], " rem=", o_rem[0]);
            $display("expected 5,1 EXITING");
            $finish();
        end

        if (o_div[1] != 1'bx || o_rem[1] != 1'bx) begin
            $display("PIPE1: div=", o_div[1], " rem=", o_rem[1]);
            $display("expected x,x EXITING");
            $finish();
        end

        if (o_div[2] != 1'bx || o_rem[2] != 1'bx) begin
            $display("PIPE2: div=", o_div[2], " rem=", o_rem[2]);
            $display("expected x,x EXITING");
            $finish();
        end

        if (o_div[3] != 1'bx || o_rem[3] != 1'bx) begin
            $display("PIPE3: div=", o_div[3], " rem=", o_rem[3]);
            $display("expected x,x EXITING");
            $finish();
        end

        #2;

        $display("56 / 11 #2");
         if (o_div[0] != 5 || o_rem[0] != 1) begin
            $display("PIPE0: div=", o_div[0], " rem=", o_rem[0]);
            $display("expected 5,1, EXITING");
            $finish();
        end

        if (o_div[1] != 5 || o_rem[1] != 1) begin
            $display("PIPE1: div=", o_div[1], " rem=", o_rem[1]);
            $display("expected 5,1 EXITING");
            $finish();
        end

        if (o_div[2] != 1'bx || o_rem[2] != 1'bx) begin
            $display("PIPE2: div=", o_div[2], " rem=", o_rem[2]);
            $display("expected x,x EXITING");
            $finish();
        end

        if (o_div[3] != 1'bx || o_rem[3] != 1'bx) begin
            $display("PIPE3: div=", o_div[3], " rem=", o_rem[3]);
            $display("expected x,x EXITING");
            $finish();
        end

        #2;

        $display("56 / 11 #4");
        if (o_div[0] != 5 || o_rem[0] != 1) begin
            $display("PIPE0: div=", o_div[0], " rem=", o_rem[0]);
            $display("expected 5,1 EXITING");
            $finish();
        end

        if (o_div[1] != 5 || o_rem[1] != 1) begin
            $display("PIPE1: div=", o_div[1], " rem=", o_rem[1]);
            $display("expected 5,1 EXITING");
            $finish();
        end

        if (o_div[2] != 5 || o_rem[2] != 1) begin
            $display("PIPE2: div=", o_div[2], " rem=", o_rem[2]);
            $display("expected 5,1 EXITING");
            $finish();
        end

        if (o_div[3] != 1'bx || o_rem[3] != 1'bx) begin
            $display("PIPE3: div=", o_div[3], " rem=", o_rem[3]);
            $display("expected x,x EXITING");
            $finish();
        end

        #2;

        $display("56 / 11 #6");

        if (o_div[0] != 5 || o_rem[0] != 1) begin
            $display("PIPE0: div=", o_div[0], " rem=", o_rem[0]);
            $display("expected 5,1 EXITING");
            $finish();
        end

        if (o_div[1] != 5 || o_rem[1] != 1) begin
            $display("PIPE1: div=", o_div[1], " rem=", o_rem[1]);
            $display("expected 5,1 EXITING");
            $finish();
        end

        if (o_div[2] != 5 || o_rem[2] != 1) begin
            $display("PIPE2: div=", o_div[2], " rem=", o_rem[2]);
            $display("expected 5,1 EXITING");
            $finish();
        end

        if (o_div[3] != 5 || o_rem[3] != 1) begin
            $display("PIPE3: div=", o_div[3], " rem=", o_rem[3]);
            $display("expected 5,1 EXITING");
            $finish();
        end

        $display("PASS");

        $finish();
    end

    always #1
        clk = !clk;

endmodule