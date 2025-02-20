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
    parameter `STRING INSTANCE_ID = ""
) (
    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    output reg [`PERF_CTR_BITS-1:0] perf_stalls,
    output reg [`NUM_EX_UNITS-1:0][`PERF_CTR_BITS-1:0] perf_units_uses,
    output reg [`NUM_SFU_UNITS-1:0][`PERF_CTR_BITS-1:0] perf_sfu_uses,
`endif

    VX_writeback_if.slave   writeback_if,
    VX_ibuffer_if.slave     ibuffer_if [PER_ISSUE_WARPS],
    VX_scoreboard_if.master scoreboard_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    localparam NUM_SRC_OPDS = 3;
    localparam NUM_OPDS = NUM_SRC_OPDS + 1;
    localparam DATAW = `UUID_WIDTH + `NUM_THREADS + `PC_BITS + `EX_BITS + `INST_OP_BITS + `INST_ARGS_BITS + (`NR_BITS * 4) + 1;

    VX_ibuffer_if staging_if [PER_ISSUE_WARPS]();
    reg [PER_ISSUE_WARPS-1:0] operands_ready;

`ifdef PERF_ENABLE
    reg [PER_ISSUE_WARPS-1:0][`NUM_EX_UNITS-1:0] perf_inuse_units_per_cycle;
    wire [`NUM_EX_UNITS-1:0] perf_units_per_cycle, perf_units_per_cycle_r;

    reg [PER_ISSUE_WARPS-1:0][`NUM_SFU_UNITS-1:0] perf_inuse_sfu_per_cycle;
    wire [`NUM_SFU_UNITS-1:0] perf_sfu_per_cycle, perf_sfu_per_cycle_r;

    VX_reduce_tree #(
        .DATAW_IN (`NUM_EX_UNITS),
        .N  (PER_ISSUE_WARPS),
        .OP ("|")
    ) perf_units_reduce (
        .data_in  (perf_inuse_units_per_cycle),
        .data_out (perf_units_per_cycle)
    );

    VX_reduce_tree #(
        .DATAW_IN (`NUM_SFU_UNITS),
        .N  (PER_ISSUE_WARPS),
        .OP ("|")
    ) perf_sfu_reduce (
        .data_in  (perf_inuse_sfu_per_cycle),
        .data_out (perf_sfu_per_cycle)
    );

    `BUFFER_EX(perf_units_per_cycle_r, perf_units_per_cycle, 1'b1, 0, `CDIV(PER_ISSUE_WARPS, `MAX_FANOUT));
    `BUFFER_EX(perf_sfu_per_cycle_r, perf_sfu_per_cycle, 1'b1, 0, `CDIV(PER_ISSUE_WARPS, `MAX_FANOUT));

    wire [PER_ISSUE_WARPS-1:0] stg_valid_in;
    for (genvar w = 0; w < PER_ISSUE_WARPS; ++w) begin : g_stg_valid_in
        assign stg_valid_in[w] = staging_if[w].valid;
    end

    wire perf_stall_per_cycle = (|stg_valid_in) && ~(|(stg_valid_in & operands_ready));

    always @(posedge clk) begin : g_perf_stalls
        if (reset) begin
            perf_stalls <= '0;
        end else begin
            perf_stalls <= perf_stalls + `PERF_CTR_BITS'(perf_stall_per_cycle);
        end
    end

    for (genvar i = 0; i < `NUM_EX_UNITS; ++i) begin : g_perf_units_uses
        always @(posedge clk) begin
            if (reset) begin
                perf_units_uses[i] <= '0;
            end else begin
                perf_units_uses[i] <= perf_units_uses[i] + `PERF_CTR_BITS'(perf_units_per_cycle_r[i]);
            end
        end
    end

    for (genvar i = 0; i < `NUM_SFU_UNITS; ++i) begin : g_perf_sfu_uses
        always @(posedge clk) begin
            if (reset) begin
                perf_sfu_uses[i] <= '0;
            end else begin
                perf_sfu_uses[i] <= perf_sfu_uses[i] + `PERF_CTR_BITS'(perf_sfu_per_cycle_r[i]);
            end
        end
    end
`endif

    for (genvar w = 0; w < PER_ISSUE_WARPS; ++w) begin : g_stanging_bufs
        VX_pipe_buffer #(
            .DATAW (DATAW)
        ) stanging_buf (
            .clk      (clk),
            .reset    (reset),
            .valid_in (ibuffer_if[w].valid),
            .data_in  (ibuffer_if[w].data),
            .ready_in (ibuffer_if[w].ready),
            .valid_out(staging_if[w].valid),
            .data_out (staging_if[w].data),
            .ready_out(staging_if[w].ready)
        );
    end

    for (genvar w = 0; w < PER_ISSUE_WARPS; ++w) begin : g_scoreboard
        reg [`NUM_REGS-1:0] inuse_regs;

        reg [NUM_OPDS-1:0] operands_busy, operands_busy_n;

        wire ibuffer_fire = ibuffer_if[w].valid && ibuffer_if[w].ready;

        wire staging_fire = staging_if[w].valid && staging_if[w].ready;

        wire writeback_fire = writeback_if.valid
                           && (writeback_if.data.wis == ISSUE_WIS_W'(w))
                           && writeback_if.data.eop;

        wire [NUM_OPDS-1:0][`NR_BITS-1:0] ibuf_opds, stg_opds;
        assign ibuf_opds = {ibuffer_if[w].data.rs3, ibuffer_if[w].data.rs2, ibuffer_if[w].data.rs1, ibuffer_if[w].data.rd};
        assign stg_opds = {staging_if[w].data.rs3, staging_if[w].data.rs2, staging_if[w].data.rs1, staging_if[w].data.rd};

    `ifdef PERF_ENABLE
        reg [`NUM_REGS-1:0][`EX_WIDTH-1:0] inuse_units;
        reg [`NUM_REGS-1:0][`SFU_WIDTH-1:0] inuse_sfu;

        always @(*) begin
            perf_inuse_units_per_cycle[w] = '0;
            perf_inuse_sfu_per_cycle[w] = '0;
            for (integer i = 0; i < NUM_OPDS; ++i) begin
                if (staging_if[w].valid && operands_busy[i]) begin
                    perf_inuse_units_per_cycle[w][inuse_units[stg_opds[i]]] = 1;
                    if (inuse_units[stg_opds[i]] == `EX_SFU) begin
                        perf_inuse_sfu_per_cycle[w][inuse_sfu[stg_opds[i]]] = 1;
                    end
                end
            end
        end
    `endif

        for (genvar i = 0; i < NUM_OPDS; ++i) begin : g_operands_busy_n
            always @(*) begin
                operands_busy_n[i] = operands_busy[i];
                if (ibuffer_fire) begin
                    operands_busy_n[i] = inuse_regs[ibuf_opds[i]];
                    if (staging_fire && staging_if[w].data.wb && staging_if[w].data.rd == ibuf_opds[i]) begin
                        operands_busy_n[i] = 1;
                    end
                end
                if (writeback_fire) begin
                    if (ibuffer_fire) begin
                        if (writeback_if.data.rd == ibuf_opds[i]) begin
                            operands_busy_n[i] = 0;
                        end
                    end else begin
                        if (writeback_if.data.rd == stg_opds[i]) begin
                            operands_busy_n[i] = 0;
                       end
                    end
                end
            end
        end

        always @(posedge clk) begin
            if (reset) begin
                inuse_regs <= '0;
            end else begin
                if (writeback_fire) begin
                    inuse_regs[writeback_if.data.rd] <= 0;
                end
                if (staging_fire && staging_if[w].data.wb) begin
                    inuse_regs[staging_if[w].data.rd] <= 1;
                end
            end

            operands_busy <= operands_busy_n;
            operands_ready[w] <= ~(| operands_busy_n);

        `ifdef PERF_ENABLE
            if (staging_fire && staging_if[w].data.wb) begin
                inuse_units[staging_if[w].data.rd] <= staging_if[w].data.ex_type;
                if (staging_if[w].data.ex_type == `EX_SFU) begin
                    inuse_sfu[staging_if[w].data.rd] <= op_to_sfu_type(staging_if[w].data.op_type);
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
                if (staging_if[w].valid && ~staging_if[w].ready) begin
                `ifdef DBG_TRACE_PIPELINE
                    `TRACE(4, ("%t: *** %s-stall: wid=%0d, PC=0x%0h, tmask=%b, cycles=%0d, inuse=%b (#%0d)\n",
                        $time, INSTANCE_ID, w, {staging_if[w].data.PC, 1'b0}, staging_if[w].data.tmask, timeout_ctr,
                        operands_busy, staging_if[w].data.uuid))
                `endif
                    timeout_ctr <= timeout_ctr + 1;
                end else if (ibuffer_fire) begin
                    timeout_ctr <= '0;
                end
            end
        end

        `RUNTIME_ASSERT((timeout_ctr < `STALL_TIMEOUT),
                        ("%t: *** %s timeout: wid=%0d, PC=0x%0h, tmask=%b, cycles=%0d, inuse=%b (#%0d)",
                            $time, INSTANCE_ID, w, {staging_if[w].data.PC, 1'b0}, staging_if[w].data.tmask, timeout_ctr,
                            operands_busy, staging_if[w].data.uuid))

        `RUNTIME_ASSERT(~writeback_fire || inuse_regs[writeback_if.data.rd] != 0,
            ("%t: *** %s invalid writeback register: wid=%0d, PC=0x%0h, tmask=%b, rd=%0d (#%0d)",
                $time, INSTANCE_ID, w, {writeback_if.data.PC, 1'b0}, writeback_if.data.tmask, writeback_if.data.rd, writeback_if.data.uuid))
    `endif

    end

    wire [PER_ISSUE_WARPS-1:0] arb_valid_in;
    wire [PER_ISSUE_WARPS-1:0][DATAW-1:0] arb_data_in;
    wire [PER_ISSUE_WARPS-1:0] arb_ready_in;

    for (genvar w = 0; w < PER_ISSUE_WARPS; ++w) begin : g_arb_data_in
        assign arb_valid_in[w] = staging_if[w].valid && operands_ready[w];
        assign arb_data_in[w] = staging_if[w].data;
        assign staging_if[w].ready = arb_ready_in[w] && operands_ready[w];
    end

    VX_stream_arb #(
        .NUM_INPUTS (PER_ISSUE_WARPS),
        .DATAW      (DATAW),
        .ARBITER    ("C"),
        .OUT_BUF    (3)
    ) out_arb (
        .clk      (clk),
        .reset    (reset),
        .valid_in (arb_valid_in),
        .ready_in (arb_ready_in),
        .data_in  (arb_data_in),
        .data_out ({
            scoreboard_if.data.uuid,
            scoreboard_if.data.tmask,
            scoreboard_if.data.PC,
            scoreboard_if.data.ex_type,
            scoreboard_if.data.op_type,
            scoreboard_if.data.op_args,
            scoreboard_if.data.wb,
            scoreboard_if.data.rd,
            scoreboard_if.data.rs1,
            scoreboard_if.data.rs2,
            scoreboard_if.data.rs3
        }),
        .valid_out (scoreboard_if.valid),
        .ready_out (scoreboard_if.ready),
        .sel_out   (scoreboard_if.data.wis)
    );

endmodule
