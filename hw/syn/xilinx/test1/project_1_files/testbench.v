`include "VX_define.vh"

`timescale 10ns / 1ns

`define CYCLE_TIME  2

module testbench;
    reg clk;
    reg resetn;
    reg [63:0] cycles;

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

    always #(`CYCLE_TIME/2) 
        clk = ~clk;
    
    initial begin
        clk = 1'b0;
        resetn = 1'b0;
        #8 resetn = 1'b1;
    end

    
    always @(posedge clk) begin
        if (~resetn) begin
            cycles <= 0;
        end else begin
            cycles <= cycles + 1;
        end
    end

    always @(posedge clk) begin
        if (~resetn) begin
            vx_reset     <= 1;
            dcr_wr_valid <= 0;
            dcr_wr_addr  <= 0;
            dcr_wr_data  <= 0;
        end else begin
            case (cycles)
            0:  begin
                dcr_wr_valid <= 1;
                dcr_wr_addr  <= `DCR_BASE_STARTUP_ADDR;
                dcr_wr_data  <= `STARTUP_ADDR;                    
            end
            2: begin
                dcr_wr_valid <= 0;
                dcr_wr_addr  <= 0;
                dcr_wr_data  <= 0;
            end
            `RESET_DELAY: begin
                vx_reset <= 0;
            end
            default:;
            endcase
        end
    end
    
    always @(posedge clk) begin
        if (resetn && ~vx_reset && ~vx_busy) begin
            $display("done!");
            $finish;
        end    
    end

endmodule