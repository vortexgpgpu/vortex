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

`ifdef PERF_ENABLE
    output reg [`PERF_CTR_BITS-1:0] perf_scb_stalls,
    output reg [`PERF_CTR_BITS-1:0] perf_scb_uses [`NUM_EX_UNITS],
`endif

    VX_writeback_if.slave   writeback_if [`ISSUE_WIDTH],
    VX_ibuffer_if.slave     ibuffer_if [`ISSUE_WIDTH],
    VX_ibuffer_if.master    scoreboard_if [`ISSUE_WIDTH]
);
    `UNUSED_PARAM (CORE_ID)
    localparam DATAW = `UUID_WIDTH + ISSUE_WIS_W + `NUM_THREADS + `XLEN + `EX_BITS + `INST_OP_BITS + `INST_MOD_BITS + 1 + 1 + `XLEN + (`NR_BITS * 4) + 1;

`ifdef PERF_ENABLE
    wire [`CLOG2(`ISSUE_WIDTH+1)-1:0] scoreboard_alu_per_cycle;
`ifdef EXT_F_ENABLE
    wire [`CLOG2(`ISSUE_WIDTH+1)-1:0] scoreboard_fpu_per_cycle;
`endif
    wire [`CLOG2(`ISSUE_WIDTH+1)-1:0] scoreboard_lsu_per_cycle;
    wire [`CLOG2(`ISSUE_WIDTH+1)-1:0] scoreboard_sfu_per_cycle;
    wire [`CLOG2(`ISSUE_WIDTH+1)-1:0] scoreboard_stalls_per_cycle;        
    reg [`EX_BITS-1:0][`ISSUE_WIDTH-1:0] scoreboard_uses;
    wire [`ISSUE_WIDTH-1:0] scoreboard_stalls;
    `POP_COUNT(scoreboard_stalls_per_cycle, scoreboard_stalls);
    `POP_COUNT(scoreboard_alu_per_cycle, scoreboard_uses[`EX_ALU]);
`ifdef EXT_F_ENABLE
    `POP_COUNT(scoreboard_fpu_per_cycle, scoreboard_uses[`EX_FPU]);
`endif
    `POP_COUNT(scoreboard_lsu_per_cycle, scoreboard_uses[`EX_LSU]);
    `POP_COUNT(scoreboard_sfu_per_cycle, scoreboard_uses[`EX_SFU]);
`endif

    for (genvar i = 0; i < `ISSUE_WIDTH; ++i) begin
        reg [`UP(ISSUE_RATIO)-1:0][`NUM_REGS-1:0] inuse_regs;
        VX_ibuffer_if staging_if();

        wire writeback_fire = writeback_if[i].valid && writeback_if[i].data.eop;

        wire inuse_rd  = inuse_regs[ibuffer_if[i].data.wis][ibuffer_if[i].data.rd];
        wire inuse_rs1 = inuse_regs[ibuffer_if[i].data.wis][ibuffer_if[i].data.rs1];
        wire inuse_rs2 = inuse_regs[ibuffer_if[i].data.wis][ibuffer_if[i].data.rs2];
        wire inuse_rs3 = inuse_regs[ibuffer_if[i].data.wis][ibuffer_if[i].data.rs3];

    `ifdef PERF_ENABLE
        reg [`UP(ISSUE_RATIO)-1:0][`NUM_REGS-1:0][`EX_BITS-1:0] inuse_units;        
        always @(*) begin
            scoreboard_uses = '0;
            if (ibuffer_if[i].valid) begin
                if (inuse_rd) begin
                    scoreboard_uses[inuse_units[ibuffer_if[i].data.wis][ibuffer_if[i].data.rd]][i] = 1;
                end
                if (inuse_rs1) begin
                    scoreboard_uses[inuse_units[ibuffer_if[i].data.wis][ibuffer_if[i].data.rs1]][i] = 1;
                end
                if (inuse_rs2) begin
                    scoreboard_uses[inuse_units[ibuffer_if[i].data.wis][ibuffer_if[i].data.rs2]][i] = 1;
                end
                if (inuse_rs3) begin
                    scoreboard_uses[inuse_units[ibuffer_if[i].data.wis][ibuffer_if[i].data.rs3]][i] = 1;
                end
            end
        end
        assign scoreboard_stalls[i] = ibuffer_if[i].valid && ~ibuffer_if[i].ready;
    `endif

        reg [DATAW-1:0] data_out_r;
        reg valid_out_r;

        wire [3:0] ready_masks = ~{inuse_rd, inuse_rs1, inuse_rs2, inuse_rs3};
        wire deps_ready = (& ready_masks);

        always @(posedge clk) begin
            if (reset) begin
                valid_out_r <= 0;
                inuse_regs <= '0;                
            end else begin                
                if (writeback_fire) begin
                    inuse_regs[writeback_if[i].data.wis][writeback_if[i].data.rd] <= 0;            
                end
                if (~valid_out_r) begin
                    valid_out_r <= ibuffer_if[i].valid && deps_ready;
                end else if (staging_if.ready) begin
                    if (staging_if.data.wb) begin
                        inuse_regs[staging_if.data.wis][staging_if.data.rd] <= 1;
                    `ifdef PERF_ENABLE
                        inuse_units[staging_if.data.wis][staging_if.data.rd] <= staging_if.data.ex_type;
                    `endif
                    end
                    valid_out_r <= 0;
                end
            end
            if (~valid_out_r) begin
                data_out_r <= ibuffer_if[i].data;
            end
        end

        assign ibuffer_if[i].ready = ~valid_out_r && deps_ready;
        assign staging_if.valid = valid_out_r;
        assign staging_if.data  = data_out_r;

        VX_elastic_buffer #(
            .DATAW   (DATAW),
            .SIZE    (0),
            .OUT_REG (2)
        ) out_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (staging_if.valid),
            .ready_in  (staging_if.ready),
            .data_in   (staging_if.data),
            .data_out  (scoreboard_if[i].data),
            .valid_out (scoreboard_if[i].valid),
            .ready_out (scoreboard_if[i].ready)
        );

    `ifdef SIMULATION
        reg [31:0] timeout_ctr;       

        always @(posedge clk) begin
            if (reset) begin
                timeout_ctr <= '0;
            end else begin        
                if (ibuffer_if[i].valid && ~ibuffer_if[i].ready) begin
                `ifdef DBG_TRACE_CORE_PIPELINE
                    `TRACE(3, ("%d: *** core%0d-scoreboard-stall: wid=%0d, PC=0x%0h, tmask=%b, cycles=%0d, inuse=%b (#%0d)\n",
                        $time, CORE_ID, wis_to_wid(ibuffer_if[i].data.wis, i), ibuffer_if[i].data.PC, ibuffer_if[i].data.tmask, timeout_ctr,
                        ~ready_masks, ibuffer_if[i].data.uuid));
                `endif
                    timeout_ctr <= timeout_ctr + 1;
                end else if (ibuffer_if[i].valid && ibuffer_if[i].ready) begin
                    timeout_ctr <= '0;
                end
            end
        end        
        
        `RUNTIME_ASSERT((timeout_ctr < `STALL_TIMEOUT),
                        ("%t: *** core%0d-scoreboard-timeout: wid=%0d, PC=0x%0h, tmask=%b, cycles=%0d, inuse=%b (#%0d)",
                            $time, CORE_ID, wis_to_wid(ibuffer_if[i].data.wis, i), ibuffer_if[i].data.PC, ibuffer_if[i].data.tmask, timeout_ctr,
                            ~ready_masks, ibuffer_if[i].data.uuid));

        `RUNTIME_ASSERT(~writeback_fire || inuse_regs[writeback_if[i].data.wis][writeback_if[i].data.rd] != 0,
            ("%t: *** core%0d: invalid writeback register: wid=%0d, PC=0x%0h, tmask=%b, rd=%0d (#%0d)",
                $time, CORE_ID, wis_to_wid(writeback_if[i].data.wis, i), writeback_if[i].data.PC, writeback_if[i].data.tmask, writeback_if[i].data.rd, writeback_if[i].data.uuid));
    `endif
    
    end    

`ifdef PERF_ENABLE
    always @(posedge clk) begin
        if (reset) begin
            perf_scb_stalls <= '0;
            perf_scb_uses[`EX_ALU] <= '0;
        `ifdef EXT_F_ENABLE
            perf_scb_uses[`EX_FPU] <= '0;
        `endif
            perf_scb_uses[`EX_LSU] <= '0;
            perf_scb_uses[`EX_SFU] <= '0;
        end else begin
            perf_scb_stalls <= perf_scb_stalls + `PERF_CTR_BITS'(scoreboard_stalls_per_cycle);
            perf_scb_uses[`EX_ALU] <= perf_scb_uses[`EX_ALU] + `PERF_CTR_BITS'(scoreboard_alu_per_cycle);
        `ifdef EXT_F_ENABLE
            perf_scb_uses[`EX_FPU] <= perf_scb_uses[`EX_FPU] + `PERF_CTR_BITS'(scoreboard_fpu_per_cycle);
        `endif
            perf_scb_uses[`EX_LSU] <= perf_scb_uses[`EX_LSU] + `PERF_CTR_BITS'(scoreboard_lsu_per_cycle);
            perf_scb_uses[`EX_SFU] <= perf_scb_uses[`EX_SFU] + `PERF_CTR_BITS'(scoreboard_sfu_per_cycle);
        end
    end
`endif

endmodule
