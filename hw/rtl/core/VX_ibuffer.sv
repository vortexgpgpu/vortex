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

// Instruction buffer of one issue slice: up to IBUF_SIZE entries per warp, in
// order. The instructions of every warp share one RAM, read for the warp the
// scoreboard grants, so a warp costs only the registers holding the fields the
// scoreboard checks. Macro-ops expand here, through one sequencer per slice.
module VX_ibuffer import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire          clk,
    input wire          reset,

`ifdef PERF_ENABLE
    output wire [PERF_CTR_BITS-1:0] perf_stalls,
`endif

    // inputs
    VX_decode_if.slave  decode_if,

    // per warp: the head entry and the one behind it
    output wire [PER_ISSUE_WARPS-1:0]      head_valid,
    output ibuf_hz_t [PER_ISSUE_WARPS-1:0] head_hz,
    output wire [PER_ISSUE_WARPS-1:0]      next_valid,
    output ibuf_hz_t [PER_ISSUE_WARPS-1:0] next_hz,

    // the entry written this cycle, and every warp's entries and their count,
    // for the scoreboard's check at insertion
    output wire                            ins_valid,
    output wire [ISSUE_WIS_W-1:0]          ins_wis,
    output ibuf_hz_t                       ins_hz,
    output ibuf_hz_t [PER_ISSUE_WARPS-1:0][`VX_CFG_IBUF_SIZE-1:0] slots_hz,
    output wire [PER_ISSUE_WARPS-1:0][`CLOG2(`VX_CFG_IBUF_SIZE + 1)-1:0] slots_count,

    // the granted warp's head instruction, popped when it issues
    input wire [ISSUE_WIS_W-1:0]           read_wis,
    output ibuffer_t                       read_data,
    input wire [PER_ISSUE_WARPS-1:0]       head_pop
);
    localparam DEPTH     = `VX_CFG_IBUF_SIZE;
    localparam CNTW      = `CLOG2(DEPTH + 1);
    localparam PTRW      = `LOG2UP(DEPTH);
    localparam RAM_SIZE  = PER_ISSUE_WARPS * DEPTH;
    localparam RAM_ADDRW = `LOG2UP(RAM_SIZE);

    VX_ibuffer_if seq_in_if();
    VX_ibuffer_if seq_out_if();
    wire seq_out_last;

    assign seq_in_if.valid = decode_if.valid;
    assign seq_in_if.data = {
        decode_if.data.uuid,
        decode_if.data.wid,
        decode_if.data.cta_id,
        decode_if.data.tmask,
        decode_if.data.PC,
        decode_if.data.ex_type,
        decode_if.data.op_type,
        decode_if.data.op_args,
        decode_if.data.wb,
        decode_if.data.rd_xregs,
        decode_if.data.wr_xregs,
        decode_if.data.used_rs,
        decode_if.data.rd,
        decode_if.data.bytesel,
        decode_if.data.rs1,
        decode_if.data.rs2,
        decode_if.data.rs3,
        1'b1, // fu_lock
        1'b1  // fu_unlock
    `ifdef VX_CFG_EXT_LDST_WIDE_ENABLE
        , 1'b0 // issue_yield
    `endif
    };
    assign decode_if.ready = seq_in_if.ready;

    VX_uop_sequencer #(
        .INSTANCE_ID (`SFORMATF(("%s-uop", INSTANCE_ID)))
    ) uop_sequencer (
        .clk         (clk),
        .reset       (reset),
        .input_if    (seq_in_if),
        .output_if   (seq_out_if),
        .output_last (seq_out_last)
    );

    wire [ISSUE_WIS_W-1:0] wr_wis = wid_to_wis(seq_out_if.data.wid);
    wire [PER_ISSUE_WARPS-1:0] buf_full;
    assign seq_out_if.ready = ~buf_full[wr_wis];
    wire wr_fire = seq_out_if.valid && seq_out_if.ready;
    ibuf_hz_t wr_hz;
    assign wr_hz = to_ibuf_hz(seq_out_if.data);

    assign ins_valid = wr_fire;
    assign ins_wis   = wr_wis;
    assign ins_hz    = wr_hz;

    wire [PER_ISSUE_WARPS-1:0][PTRW-1:0] head_ptrs, tail_ptrs;

    for (genvar w = 0; w < PER_ISSUE_WARPS; ++w) begin : g_warps
        reg [CNTW-1:0] count;
        reg [PTRW-1:0] head_ptr, tail_ptr;
        // slot 0 is the head; a pop shifts the slots down
        ibuf_hz_t [DEPTH-1:0] slot_hz;
        reg [DEPTH-1:0] slot_last;

        wire push = wr_fire && (wr_wis == ISSUE_WIS_W'(w));
        wire pop  = head_pop[w];
        wire [CNTW-1:0] wr_slot = count - CNTW'(pop);

        always @(posedge clk) begin
            if (reset) begin
                count    <= '0;
                head_ptr <= '0;
                tail_ptr <= '0;
            end else begin
                count <= count + CNTW'(push) - CNTW'(pop);
                if (push) begin
                    tail_ptr <= (tail_ptr == PTRW'(DEPTH-1)) ? '0 : tail_ptr + PTRW'(1);
                end
                if (pop) begin
                    head_ptr <= (head_ptr == PTRW'(DEPTH-1)) ? '0 : head_ptr + PTRW'(1);
                end
            end
        end

        for (genvar i = 0; i < DEPTH; ++i) begin : g_slots
            if (i + 1 < DEPTH) begin : g_shift
                always @(posedge clk) begin
                    if (push && wr_slot == CNTW'(i)) begin
                        slot_hz[i]   <= wr_hz;
                        slot_last[i] <= seq_out_last;
                    end else if (pop) begin
                        slot_hz[i]   <= slot_hz[i + 1];
                        slot_last[i] <= slot_last[i + 1];
                    end
                end
            end else begin : g_tail
                always @(posedge clk) begin
                    if (push && wr_slot == CNTW'(i)) begin
                        slot_hz[i]   <= wr_hz;
                        slot_last[i] <= seq_out_last;
                    end
                end
            end
        end

        assign slots_hz[w]    = slot_hz;
        assign slots_count[w] = count;

        assign buf_full[w]   = (count == CNTW'(DEPTH));
        assign head_valid[w] = (count != '0);
        assign head_hz[w]    = slot_hz[0];
        if (DEPTH > 1) begin : g_next
            assign next_valid[w] = (count > CNTW'(1));
            assign next_hz[w]    = slot_hz[`MIN(1, DEPTH - 1)];
        end else begin : g_no_next
            assign next_valid[w] = 1'b0;
            assign next_hz[w]    = slot_hz[0];
        end
        assign head_ptrs[w] = head_ptr;
        assign tail_ptrs[w] = tail_ptr;

        // the fetch credit returns with the instruction's last entry
        assign decode_if.ibuf_pop[w] = pop && slot_last[0];

        `RUNTIME_ASSERT(~pop || (count != '0), ("%t: *** %s pop from an empty buffer: wis=%0d", $time, INSTANCE_ID, w))
    end

    wire [RAM_ADDRW-1:0] ram_waddr = RAM_ADDRW'(wr_wis * DEPTH) + RAM_ADDRW'(tail_ptrs[wr_wis]);
    wire [RAM_ADDRW-1:0] ram_raddr = RAM_ADDRW'(read_wis * DEPTH) + RAM_ADDRW'(head_ptrs[read_wis]);

    VX_dp_ram #(
        .DATAW    ($bits(ibuffer_t)),
        .SIZE     (RAM_SIZE),
        .LUTRAM   (1),
        .OUT_REG  (0),
        .RDW_MODE ("R")
    ) instr_ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (wr_fire),
        .wren  (1'b1),
        .waddr (ram_waddr),
        .wdata (seq_out_if.data),
        .raddr (ram_raddr),
        .rdata (read_data)
    );

`ifdef PERF_ENABLE
    reg [PERF_CTR_BITS-1:0] perf_ibf_stalls;

    wire decode_if_stall = decode_if.valid && ~decode_if.ready;

    always @(posedge clk) begin
        if (reset) begin
            perf_ibf_stalls <= '0;
        end else begin
            perf_ibf_stalls <= perf_ibf_stalls + PERF_CTR_BITS'(decode_if_stall);
        end
    end

    assign perf_stalls = perf_ibf_stalls;
`endif

endmodule
