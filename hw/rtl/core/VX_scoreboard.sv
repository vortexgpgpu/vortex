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

module VX_scoreboard import VX_gpu_pkg::*; #(
    parameter CORE_ID = 0
) (
    input wire              clk,
    input wire              reset,

    VX_writeback_if.slave   writeback_if [`ISSUE_WIDTH],
    VX_ibuffer_if.slave     ibuffer_if [`ISSUE_WIDTH],
    VX_ibuffer_if.master    scoreboard_if [`ISSUE_WIDTH]
);
    `UNUSED_PARAM (CORE_ID)
    localparam DATAW = `UUID_WIDTH + ISSUE_WIS_W + `NUM_THREADS + `XLEN + `EX_BITS + `INST_OP_BITS + `INST_MOD_BITS + 1 + 1 + `XLEN + (`NR_BITS * 4) + 1;

    for (genvar i = 0; i < `ISSUE_WIDTH; ++i) begin
        reg [`UP(ISSUE_RATIO)-1:0][`NUM_REGS-1:0] inuse_regs, inuse_regs_n;
        reg [3:0] ready_masks, ready_masks_n;        
        VX_ibuffer_if staging_if();
        
        wire writeback_fire = writeback_if[i].valid && writeback_if[i].data.eop;

        always @(*) begin
            inuse_regs_n = inuse_regs;
            ready_masks_n = ready_masks;
            if (writeback_fire) begin
                inuse_regs_n[writeback_if[i].data.wis][writeback_if[i].data.rd] = 0;
                ready_masks_n |= {4{(ISSUE_RATIO == 0) || writeback_if[i].data.wis == staging_if.data.wis}} 
                               & {(writeback_if[i].data.rd == staging_if.data.rd),
                                  (writeback_if[i].data.rd == staging_if.data.rs1),
                                  (writeback_if[i].data.rd == staging_if.data.rs2),
                                  (writeback_if[i].data.rd == staging_if.data.rs3)};
            end   
            if (staging_if.valid && staging_if.ready && staging_if.data.wb) begin
                inuse_regs_n[staging_if.data.wis][staging_if.data.rd] = 1;
                ready_masks_n = '0;
            end
            if (ibuffer_if[i].valid && ibuffer_if[i].ready) begin
                ready_masks_n = ~{inuse_regs_n[ibuffer_if[i].data.wis][ibuffer_if[i].data.rd],
                                  inuse_regs_n[ibuffer_if[i].data.wis][ibuffer_if[i].data.rs1],
                                  inuse_regs_n[ibuffer_if[i].data.wis][ibuffer_if[i].data.rs2],
                                  inuse_regs_n[ibuffer_if[i].data.wis][ibuffer_if[i].data.rs3]};
            end
        end   

        always @(posedge clk) begin
            if (reset) begin
                inuse_regs  <= '0;
                ready_masks <= '0;
            end else begin            
                inuse_regs  <= inuse_regs_n;
                ready_masks <= ready_masks_n;
            end
        end

        // staging buffer

        `RESET_RELAY (stg_buf_reset, reset);
        
        VX_elastic_buffer #(
            .DATAW (DATAW)
        ) stg_buf (
            .clk       (clk),
            .reset     (stg_buf_reset),
            .valid_in  (ibuffer_if[i].valid),
            .ready_in  (ibuffer_if[i].ready),
            .data_in   (ibuffer_if[i].data),
            .data_out  (staging_if.data),
            .valid_out (staging_if.valid),
            .ready_out (staging_if.ready)
        );

        // output buffer

        wire valid_stg, ready_stg;
        wire regs_ready = (& ready_masks);
        assign valid_stg = staging_if.valid && regs_ready;
        assign staging_if.ready = ready_stg && regs_ready;

        `RESET_RELAY (out_buf_reset, reset);

        VX_elastic_buffer #(
            .DATAW   (DATAW),
            .SIZE    (2),
            .OUT_REG (2)
        ) out_buf (
            .clk       (clk),
            .reset     (out_buf_reset),
            .valid_in  (valid_stg),
            .ready_in  (ready_stg),
            .data_in   (staging_if.data),
            .data_out  (scoreboard_if[i].data),
            .valid_out (scoreboard_if[i].valid),
            .ready_out (scoreboard_if[i].ready)
        );

        reg [31:0] timeout_ctr;
    
        always @(posedge clk) begin
            if (reset) begin
                timeout_ctr <= '0;
            end else begin        
                if (staging_if.valid && ~regs_ready) begin
                `ifdef DBG_TRACE_CORE_PIPELINE
                    `TRACE(3, ("%d: *** core%0d-scoreboard-stall: wid=%0d, PC=0x%0h, tmask=%b, cycles=%0d, inuse=%b (#%0d)\n",
                        $time, CORE_ID, wis_to_wid(staging_if.data.wis, i), staging_if.data.PC, staging_if.data.tmask, timeout_ctr,
                        ~ready_masks, staging_if.data.uuid));
                `endif
                    timeout_ctr <= timeout_ctr + 1;
                end else if (staging_if.valid && staging_if.ready) begin
                    timeout_ctr <= '0;
                end
            end
        end

        `RUNTIME_ASSERT((timeout_ctr < `STALL_TIMEOUT),
                        ("%t: *** core%0d-scoreboard-timeout: wid=%0d, PC=0x%0h, tmask=%b, cycles=%0d, inuse=%b (#%0d)",
                            $time, CORE_ID, wis_to_wid(staging_if.data.wis, i), staging_if.data.PC, staging_if.data.tmask, timeout_ctr,
                            ~ready_masks, staging_if.data.uuid));

        `RUNTIME_ASSERT(~writeback_fire || inuse_regs[writeback_if[i].data.wis][writeback_if[i].data.rd] != 0,
            ("%t: *** core%0d: invalid writeback register: wid=%0d, PC=0x%0h, tmask=%b, rd=%0d (#%0d)",
                $time, CORE_ID, wis_to_wid(writeback_if[i].data.wis, i), writeback_if[i].data.PC, writeback_if[i].data.tmask, writeback_if[i].data.rd, writeback_if[i].data.uuid));
    end    

endmodule
