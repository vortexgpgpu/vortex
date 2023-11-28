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

module VX_commit import VX_gpu_pkg::*; #(
    parameter CORE_ID = 0
) (
    input wire              clk,
    input wire              reset,

    // inputs
    VX_commit_if.slave      alu_commit_if [`ISSUE_WIDTH],
    VX_commit_if.slave      lsu_commit_if [`ISSUE_WIDTH],
`ifdef EXT_F_ENABLE
    VX_commit_if.slave      fpu_commit_if [`ISSUE_WIDTH],
`endif
    VX_commit_if.slave      sfu_commit_if [`ISSUE_WIDTH],

    // outputs
    VX_writeback_if.master  writeback_if  [`ISSUE_WIDTH],
    VX_commit_csr_if.master commit_csr_if,
    VX_commit_sched_if.master commit_sched_if,

    // simulation helper signals
    output wire [`NUM_REGS-1:0][`XLEN-1:0] sim_wb_value
);
    `UNUSED_PARAM (CORE_ID)
    localparam DATAW = `UUID_WIDTH + `NW_WIDTH + `NUM_THREADS + `XLEN + 1 + `NR_BITS + `NUM_THREADS * `XLEN + 1 + 1 + 1;
    localparam COMMIT_SIZEW = `CLOG2(`NUM_THREADS + 1);
    localparam COMMIT_ALL_SIZEW = COMMIT_SIZEW + `ISSUE_WIDTH - 1;

    // commit arbitration

    VX_commit_if commit_if[`ISSUE_WIDTH]();

    wire [`ISSUE_WIDTH-1:0] commit_fire;
    wire [`ISSUE_WIDTH-1:0][`NW_WIDTH-1:0] commit_wid;
    wire [`ISSUE_WIDTH-1:0][`NUM_THREADS-1:0] commit_tmask;
    wire [`ISSUE_WIDTH-1:0] commit_eop;

    for (genvar i = 0; i < `ISSUE_WIDTH; ++i) begin

        `RESET_RELAY (arb_reset, reset);

        VX_stream_arb #(
            .NUM_INPUTS (`NUM_EX_UNITS),
            .DATAW      (DATAW),
            .ARBITER    ("R"),
            .OUT_REG    (1)
        ) commit_arb (
            .clk       (clk),
            .reset     (arb_reset),
            .valid_in  ({            
                sfu_commit_if[i].valid,
            `ifdef EXT_F_ENABLE
                fpu_commit_if[i].valid,
            `endif
                alu_commit_if[i].valid,
                lsu_commit_if[i].valid
            }),
            .ready_in  ({           
                sfu_commit_if[i].ready,
            `ifdef EXT_F_ENABLE
                fpu_commit_if[i].ready,
            `endif
                alu_commit_if[i].ready,
                lsu_commit_if[i].ready                
            }),
            .data_in   ({
                sfu_commit_if[i].data,
            `ifdef EXT_F_ENABLE
                fpu_commit_if[i].data,
            `endif
                alu_commit_if[i].data,
                lsu_commit_if[i].data       
            }),
            .data_out  (commit_if[i].data),
            .valid_out (commit_if[i].valid),
            .ready_out (commit_if[i].ready),
            `UNUSED_PIN (sel_out)
        );

        assign commit_fire[i] = commit_if[i].valid && commit_if[i].ready;        
        assign commit_tmask[i]= {`NUM_THREADS{commit_fire[i]}} & commit_if[i].data.tmask;
        assign commit_wid[i]  = commit_if[i].data.wid;
        assign commit_eop[i]  = commit_if[i].data.eop;
    end

    // CSRs update
    
    wire [`ISSUE_WIDTH-1:0][COMMIT_SIZEW-1:0] commit_size, commit_size_r;
    wire [COMMIT_ALL_SIZEW-1:0] commit_size_all_r, commit_size_all_rr;
    wire commit_fire_any, commit_fire_any_r, commit_fire_any_rr;

    assign commit_fire_any = (| commit_fire);

    for (genvar i = 0; i < `ISSUE_WIDTH; ++i) begin
        wire [COMMIT_SIZEW-1:0] count;
        `POP_COUNT(count, commit_tmask[i]);
        assign commit_size[i] = count;
    end

    VX_pipe_register #(
        .DATAW  (1 + `ISSUE_WIDTH * COMMIT_SIZEW),
        .RESETW (1)
    ) commit_size_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  ({commit_fire_any, commit_size}),
        .data_out ({commit_fire_any_r, commit_size_r})
    );

    VX_reduce #(
        .DATAW_IN (COMMIT_SIZEW),
        .DATAW_OUT (COMMIT_ALL_SIZEW),
        .N  (`ISSUE_WIDTH),
        .OP ("+")
    ) commit_size_reduce (
        .data_in  (commit_size_r),
        .data_out (commit_size_all_r)
    );

    VX_pipe_register #(
        .DATAW  (1 + COMMIT_ALL_SIZEW),
        .RESETW (1)
    ) commit_size_reg2 (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  ({commit_fire_any_r, commit_size_all_r}),
        .data_out ({commit_fire_any_rr, commit_size_all_rr})
    );

    reg [`PERF_CTR_BITS-1:0] instret;
    always @(posedge clk) begin
       if (reset) begin
            instret <= '0;
        end else begin
            if (commit_fire_any_rr) begin
                instret <= instret + `PERF_CTR_BITS'(commit_size_all_rr);
            end
        end
    end
    assign commit_csr_if.instret = instret;

    // Committed instructions

    wire [`ISSUE_WIDTH-1:0] committed = commit_fire & commit_eop;

    VX_pipe_register #(
        .DATAW  (`ISSUE_WIDTH * (1 + `NW_WIDTH)),
        .RESETW (`ISSUE_WIDTH)
    ) committed_pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  ({committed, commit_wid}),
        .data_out ({commit_sched_if.committed, commit_sched_if.committed_wid})
    );

    // Writeback

    for (genvar i = 0; i < `ISSUE_WIDTH; ++i) begin
        assign writeback_if[i].valid     = commit_if[i].valid && commit_if[i].data.wb;
        assign writeback_if[i].data.uuid = commit_if[i].data.uuid; 
        assign writeback_if[i].data.wis  = wid_to_wis(commit_if[i].data.wid);
        assign writeback_if[i].data.PC   = commit_if[i].data.PC; 
        assign writeback_if[i].data.tmask= commit_if[i].data.tmask; 
        assign writeback_if[i].data.rd   = commit_if[i].data.rd; 
        assign writeback_if[i].data.data = commit_if[i].data.data; 
        assign writeback_if[i].data.sop  = commit_if[i].data.sop; 
        assign writeback_if[i].data.eop  = commit_if[i].data.eop;
        assign commit_if[i].ready = 1'b1; // writeback has no backpressure
    end
    
    // simulation helper signal to get RISC-V tests Pass/Fail status
    reg [`NUM_REGS-1:0][`XLEN-1:0] sim_wb_value_r;
    always @(posedge clk) begin
        if (writeback_if[0].valid) begin
            sim_wb_value_r[writeback_if[0].data.rd] <= writeback_if[0].data.data[0];
        end
    end
    assign sim_wb_value = sim_wb_value_r;
    
`ifdef DBG_TRACE_CORE_PIPELINE
    for (genvar i = 0; i < `ISSUE_WIDTH; ++i) begin
        always @(posedge clk) begin
            if (alu_commit_if[i].valid && alu_commit_if[i].ready) begin
                `TRACE(1, ("%d: core%0d-commit: wid=%0d, PC=0x%0h, ex=ALU, tmask=%b, wb=%0d, rd=%0d, sop=%b, eop=%b, data=", $time, CORE_ID, alu_commit_if[i].data.wid, alu_commit_if[i].data.PC, alu_commit_if[i].data.tmask, alu_commit_if[i].data.wb, alu_commit_if[i].data.rd, alu_commit_if[i].data.sop, alu_commit_if[i].data.eop));
                `TRACE_ARRAY1D(1, alu_commit_if[i].data.data, `NUM_THREADS);
                `TRACE(1, (" (#%0d)\n", alu_commit_if[i].data.uuid));
            end
            if (lsu_commit_if[i].valid && lsu_commit_if[i].ready) begin
                `TRACE(1, ("%d: core%0d-commit: wid=%0d, PC=0x%0h, ex=LSU, tmask=%b, wb=%0d, rd=%0d, sop=%b, eop=%b, data=", $time, CORE_ID, lsu_commit_if[i].data.wid, lsu_commit_if[i].data.PC, lsu_commit_if[i].data.tmask, lsu_commit_if[i].data.wb, lsu_commit_if[i].data.rd, lsu_commit_if[i].data.sop, lsu_commit_if[i].data.eop));
                `TRACE_ARRAY1D(1, lsu_commit_if[i].data.data, `NUM_THREADS);
                `TRACE(1, (" (#%0d)\n", lsu_commit_if[i].data.uuid));
            end
        `ifdef EXT_F_ENABLE
            if (fpu_commit_if[i].valid && fpu_commit_if[i].ready) begin
                `TRACE(1, ("%d: core%0d-commit: wid=%0d, PC=0x%0h, ex=FPU, tmask=%b, wb=%0d, rd=%0d, sop=%b, eop=%b, data=", $time, CORE_ID, fpu_commit_if[i].data.wid, fpu_commit_if[i].data.PC, fpu_commit_if[i].data.tmask, fpu_commit_if[i].data.wb, fpu_commit_if[i].data.rd, fpu_commit_if[i].data.sop, fpu_commit_if[i].data.eop));
                `TRACE_ARRAY1D(1, fpu_commit_if[i].data.data, `NUM_THREADS);
                `TRACE(1, (" (#%0d)\n", fpu_commit_if[i].data.uuid));
            end
        `endif
            if (sfu_commit_if[i].valid && sfu_commit_if[i].ready) begin
                `TRACE(1, ("%d: core%0d-commit: wid=%0d, PC=0x%0h, ex=SFU, tmask=%b, wb=%0d, rd=%0d, sop=%b, eop=%b, data=", $time, CORE_ID, sfu_commit_if[i].data.wid, sfu_commit_if[i].data.PC, sfu_commit_if[i].data.tmask, sfu_commit_if[i].data.wb, sfu_commit_if[i].data.rd, sfu_commit_if[i].data.sop, sfu_commit_if[i].data.eop));
                `TRACE_ARRAY1D(1, sfu_commit_if[i].data.data, `NUM_THREADS);
                `TRACE(1, (" (#%0d)\n", sfu_commit_if[i].data.uuid));
            end
        end
    end
`endif

endmodule
