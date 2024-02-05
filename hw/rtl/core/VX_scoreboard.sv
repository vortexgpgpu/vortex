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
    output reg [`PERF_CTR_BITS-1:0] perf_units_uses [`NUM_EX_UNITS],
    output reg [`PERF_CTR_BITS-1:0] perf_sfu_uses [`NUM_SFU_UNITS],
`endif

    VX_writeback_if.slave   writeback_if [`ISSUE_WIDTH],
    VX_ibuffer_if.slave     ibuffer_if [`ISSUE_WIDTH],
    VX_ibuffer_if.master    scoreboard_if [`ISSUE_WIDTH]
);
    `UNUSED_PARAM (CORE_ID)
    localparam DATAW = `UUID_WIDTH + ISSUE_WIS_W + `NUM_THREADS + `XLEN + `EX_BITS + `INST_OP_BITS + `INST_MOD_BITS + 1 + 1 + `XLEN + (`NR_BITS * 4) + 1;

`ifdef PERF_ENABLE
    reg [`ISSUE_WIDTH-1:0][`NUM_EX_UNITS-1:0] perf_issue_units_per_cycle;
    wire [`NUM_EX_UNITS-1:0] perf_units_per_cycle, perf_units_per_cycle_r;

    reg [`ISSUE_WIDTH-1:0][`NUM_SFU_UNITS-1:0] perf_issue_sfu_per_cycle;    
    wire [`NUM_SFU_UNITS-1:0] perf_sfu_per_cycle, perf_sfu_per_cycle_r;

    wire [`ISSUE_WIDTH-1:0] perf_issue_stalls_per_cycle;
    wire [`CLOG2(`ISSUE_WIDTH+1)-1:0] perf_stalls_per_cycle, perf_stalls_per_cycle_r;    

    `POP_COUNT(perf_stalls_per_cycle, perf_issue_stalls_per_cycle);    

    VX_reduce #(
        .DATAW_IN (`NUM_EX_UNITS),
        .N  (`ISSUE_WIDTH),
        .OP ("|")
    ) perf_units_reduce (
        .data_in  (perf_issue_units_per_cycle),
        .data_out (perf_units_per_cycle)
    );    

    VX_reduce #(
        .DATAW_IN (`NUM_SFU_UNITS),
        .N  (`ISSUE_WIDTH),
        .OP ("|")
    ) perf_sfu_reduce (
        .data_in  (perf_issue_sfu_per_cycle),
        .data_out (perf_sfu_per_cycle)
    );

    `BUFFER(perf_stalls_per_cycle_r, perf_stalls_per_cycle);
    `BUFFER(perf_units_per_cycle_r, perf_units_per_cycle);
    `BUFFER(perf_sfu_per_cycle_r, perf_sfu_per_cycle);

    always @(posedge clk) begin
        if (reset) begin
            perf_scb_stalls <= '0;            
        end else begin
            perf_scb_stalls <= perf_scb_stalls + `PERF_CTR_BITS'(perf_stalls_per_cycle_r);
        end
    end

    for (genvar i = 0; i < `NUM_EX_UNITS; ++i) begin
        always @(posedge clk) begin
            if (reset) begin
                perf_units_uses[i] <= '0;
            end else begin
                perf_units_uses[i] <= perf_units_uses[i] + `PERF_CTR_BITS'(perf_units_per_cycle_r[i]);
            end
        end
    end
    
    for (genvar i = 0; i < `NUM_SFU_UNITS; ++i) begin
        always @(posedge clk) begin
            if (reset) begin
                perf_sfu_uses[i] <= '0;
            end else begin
                perf_sfu_uses[i] <= perf_sfu_uses[i] + `PERF_CTR_BITS'(perf_sfu_per_cycle_r[i]);
            end
        end
    end
`endif

    for (genvar i = 0; i < `ISSUE_WIDTH; ++i) begin
        reg [`UP(ISSUE_RATIO)-1:0][`NUM_REGS-1:0] inuse_regs;

        wire writeback_fire = writeback_if[i].valid && writeback_if[i].data.eop;

        wire inuse_rd  = inuse_regs[ibuffer_if[i].data.wis][ibuffer_if[i].data.rd];
        wire inuse_rs1 = inuse_regs[ibuffer_if[i].data.wis][ibuffer_if[i].data.rs1];
        wire inuse_rs2 = inuse_regs[ibuffer_if[i].data.wis][ibuffer_if[i].data.rs2];
        wire inuse_rs3 = inuse_regs[ibuffer_if[i].data.wis][ibuffer_if[i].data.rs3];

    `ifdef PERF_ENABLE
        reg [`UP(ISSUE_RATIO)-1:0][`NUM_REGS-1:0][`EX_WIDTH-1:0] inuse_units;   
        reg [`UP(ISSUE_RATIO)-1:0][`NUM_REGS-1:0][`SFU_WIDTH-1:0] inuse_sfu;        

        reg [`SFU_WIDTH-1:0] sfu_type;
        always @(*) begin
            case (ibuffer_if[i].data.op_type)
            `INST_SFU_CSRRW,
            `INST_SFU_CSRRS,
            `INST_SFU_CSRRC: sfu_type = `SFU_CSRS;
            default: sfu_type = `SFU_WCTL;
            endcase
        end

        always @(*) begin
            perf_issue_units_per_cycle[i] = '0;
            perf_issue_sfu_per_cycle[i] = '0;
            if (ibuffer_if[i].valid) begin
                if (inuse_rd) begin
                    perf_issue_units_per_cycle[i][inuse_units[ibuffer_if[i].data.wis][ibuffer_if[i].data.rd]] = 1;
                    if (inuse_units[ibuffer_if[i].data.wis][ibuffer_if[i].data.rd] == `EX_SFU) begin
                        perf_issue_sfu_per_cycle[i][inuse_sfu[ibuffer_if[i].data.wis][ibuffer_if[i].data.rd]] = 1;
                    end
                end
                if (inuse_rs1) begin
                    perf_issue_units_per_cycle[i][inuse_units[ibuffer_if[i].data.wis][ibuffer_if[i].data.rs1]] = 1;
                    if (inuse_units[ibuffer_if[i].data.wis][ibuffer_if[i].data.rs1] == `EX_SFU) begin
                        perf_issue_sfu_per_cycle[i][inuse_sfu[ibuffer_if[i].data.wis][ibuffer_if[i].data.rs1]] = 1;
                    end
                end
                if (inuse_rs2) begin
                    perf_issue_units_per_cycle[i][inuse_units[ibuffer_if[i].data.wis][ibuffer_if[i].data.rs2]] = 1;
                    if (inuse_units[ibuffer_if[i].data.wis][ibuffer_if[i].data.rs2] == `EX_SFU) begin
                        perf_issue_sfu_per_cycle[i][inuse_sfu[ibuffer_if[i].data.wis][ibuffer_if[i].data.rs2]] = 1;
                    end
                end
                if (inuse_rs3) begin
                    perf_issue_units_per_cycle[i][inuse_units[ibuffer_if[i].data.wis][ibuffer_if[i].data.rs3]] = 1;
                    if (inuse_units[ibuffer_if[i].data.wis][ibuffer_if[i].data.rs3] == `EX_SFU) begin
                        perf_issue_sfu_per_cycle[i][inuse_sfu[ibuffer_if[i].data.wis][ibuffer_if[i].data.rs3]] = 1;
                    end
                end
            end
        end
        assign perf_issue_stalls_per_cycle[i] = ibuffer_if[i].valid && ~ibuffer_if[i].ready;
    `endif

        wire [3:0] operands_busy = {inuse_rd, inuse_rs1, inuse_rs2, inuse_rs3};
        wire operands_ready = ~(| operands_busy);
        
        wire stg_valid_in, stg_ready_in;
        assign stg_valid_in = ibuffer_if[i].valid && operands_ready;
        assign ibuffer_if[i].ready = stg_ready_in && operands_ready;        

        VX_stream_buffer #(
            .DATAW (DATAW)
        ) staging_buffer (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (stg_valid_in),
            .data_in   (ibuffer_if[i].data),
            .ready_in  (stg_ready_in),
            .valid_out (scoreboard_if[i].valid),
            .data_out  (scoreboard_if[i].data),
            .ready_out (scoreboard_if[i].ready)
        );

        always @(posedge clk) begin
            if (reset) begin
                inuse_regs <= '0;
            end else begin
                if (writeback_fire) begin
                    inuse_regs[writeback_if[i].data.wis][writeback_if[i].data.rd] <= 0;            
                end
                if (ibuffer_if[i].valid && ibuffer_if[i].ready && ibuffer_if[i].data.wb) begin
                    inuse_regs[ibuffer_if[i].data.wis][ibuffer_if[i].data.rd] <= 1;
                end
            end
        `ifdef PERF_ENABLE
            if (ibuffer_if[i].valid && ibuffer_if[i].ready && ibuffer_if[i].data.wb) begin
                inuse_units[ibuffer_if[i].data.wis][ibuffer_if[i].data.rd] <= ibuffer_if[i].data.ex_type;
                if (ibuffer_if[i].data.ex_type == `EX_SFU) begin
                    inuse_sfu[ibuffer_if[i].data.wis][ibuffer_if[i].data.rd] <= sfu_type;
                end
            end
        `endif
        end

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
                        operands_busy, ibuffer_if[i].data.uuid));
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
                            operands_busy, ibuffer_if[i].data.uuid));

        `RUNTIME_ASSERT(~writeback_fire || inuse_regs[writeback_if[i].data.wis][writeback_if[i].data.rd] != 0,
            ("%t: *** core%0d: invalid writeback register: wid=%0d, PC=0x%0h, tmask=%b, rd=%0d (#%0d)",
                $time, CORE_ID, wis_to_wid(writeback_if[i].data.wis, i), writeback_if[i].data.PC, writeback_if[i].data.tmask, writeback_if[i].data.rd, writeback_if[i].data.uuid));
    `endif
    
    end

endmodule
