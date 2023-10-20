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

`include "VX_define.vh"

`timescale 10ns / 1ns

`define CYCLE_TIME  4

module testbench;
    reg clk;
    reg resetn;
    reg [43:0] cycles;

    reg vx_running;
    reg vx_reset_wait;
    reg vx_busy_wait;
    wire vx_busy;

    reg dcr_wr_valid;
    reg [11:0] dcr_wr_addr;
    reg [31:0] dcr_wr_data;

    design_1_wrapper UUD(
        .clk_100MHz     (clk),
        .resetn         (resetn),
        .vx_reset       (~resetn || ~vx_running),
        .dcr_wr_valid   (dcr_wr_valid),
        .dcr_wr_addr    (dcr_wr_addr),
        .dcr_wr_data    (dcr_wr_data),
        .vx_busy        (vx_busy)
    );

    always #(`CYCLE_TIME/2) 
        clk = ~clk;
    
    initial begin
        clk    = 1'b0;
        resetn = 1'b0;
     #4 resetn = 1'b1;
    end
    
    always @(posedge clk) begin
        if (~resetn) begin
            cycles <= 0;
        end else begin
            cycles <= cycles + 1;
        end
    end

    reg [7:0] vx_reset_ctr;
    always @(posedge clk) begin
        if (vx_reset_wait) begin
            vx_reset_ctr <= vx_reset_ctr + 1;
        end else begin
            vx_reset_ctr <= 0;
        end
    end

    always @(posedge clk) begin
        if (~resetn) begin
            vx_running    <= 0;
            vx_reset_wait <= 0;
            vx_busy_wait  <= 0;
            dcr_wr_valid  <= 0;
            dcr_wr_addr   <= 0;
            dcr_wr_data   <= 0;
        end else begin            
            case (cycles)
            1:  begin
                dcr_wr_valid <= 1;
                dcr_wr_addr  <= `VX_DCR_BASE_STARTUP_ADDR0;
                dcr_wr_data  <= `STARTUP_ADDR;                    
            end
            2: begin
                dcr_wr_valid <= 0;
                dcr_wr_addr  <= 0;
                dcr_wr_data  <= 0;
            end
            3: begin
                vx_reset_wait <= 1;
            end
            default:;
            endcase
            
            if (vx_running) begin
                if (vx_busy_wait) begin
                    if (vx_busy) begin
                        vx_busy_wait <= 0;
                    end
                end else begin
                    if (~vx_busy) begin
                        vx_running <= 0;   
                        $display("done!");
                        $finish;           
                    end
                end
            end else begin
                if (vx_reset_wait && vx_reset_ctr == (`RESET_DELAY-1)) begin
                    $display("start!");
                    vx_reset_wait <= 0;
                    vx_running    <= 1;                    
                    vx_busy_wait  <= 1;
                end
            end
        end
    end

endmodule