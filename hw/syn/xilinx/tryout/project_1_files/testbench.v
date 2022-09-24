`include "VX_define.vh"

`timescale 1ns / 1ps

module testbench;
    // Inpput signals
    reg clk;
    reg resetn;
    reg vx_reset;
    wire vx_busy;
    reg dcr_wr_valid;
    reg [11:0] dcr_wr_addr;
    reg [31:0] dcr_wr_data;

    design_1_wrapper UUD(
        .clk_100MHz     (clk),
        .resetn         (resetn),
        .vx_reset       (vx_reset),
        .dcr_wr_valid   (dcr_wr_valid),
        .dcr_wr_addr    (dcr_wr_addr),
        .dcr_wr_data    (dcr_wr_data),
        .vx_busy        (vx_busy)
    );

    // clock signal creation
    always begin
        clk = 1'b0;
        #1;
        clk = 1'b1;
        #1;
    end
    
    initial begin
        #2;
        resetn          = 1'b0;
        vx_reset        = 1'b1;
        dcr_wr_valid    = 1'b0;
        #2;
        resetn          = 1'b1;
        #2;
        dcr_wr_valid    = 1'b1;
        dcr_wr_addr     = `DCR_BASE_STARTUP_ADDR;
        dcr_wr_data     = `STARTUP_ADDR;
        #2;
        dcr_wr_valid    = 1'b0;
        #20;
        vx_reset        = 1'b0;
    end
    
    always @(posedge clk) begin
        if (resetn && ~vx_reset && ~vx_busy) begin
            $display("done!");
            $finish;
        end    
    end

endmodule