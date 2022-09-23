`timescale 1ns / 1ps

module testbench;
    // Inpput signals
    reg clk_r;
    reg resetn_r;
    reg vx_reset_r;
    wire vx_busy_w;

    design_1_wrapper UUD(
        .clk_100MHz(clk_r),
        .resetn(resetn_r),
        .vx_reset(vx_reset_r),
        .vx_busy(vx_busy_w)
    );

    // clock signal creation
    always begin
        clk_r = 1'b0;
        #1;
        clk_r = 1'b1;
        #1;
    end
    
    initial begin
        #2;
        resetn_r = 1'b0;
        vx_reset_r = 1'b0;
        #2;
        resetn_r = 1'b1;               
        #2;
        vx_reset_r = 1'b1;
        #20;
        vx_reset_r = 1'b0;
    end
    
    always @(posedge clk_r) begin
        if (resetn_r && !vx_busy_w) begin
            $display("done!");
        end    
    end

endmodule