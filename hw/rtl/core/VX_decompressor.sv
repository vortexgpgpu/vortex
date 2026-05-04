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

`ifdef EXT_C_ENABLE

// VX_decompressor sits between the icache response and decode when
// EXT_C_ENABLE is set. Per warp it tracks one of:
//   BUF_EMPTY : nothing buffered
//   BUF_RVC   : a 16-bit RVC half-word, ready to emit decompressed
//   BUF_32HI  : low half of a cross-word 32-bit, awaiting the next word
//
// Storage layout (scales to NUM_WARPS=64):
//   - state[]  + buf_pc[]    : flops (small, 64x33b ≈ 2K flops at NW=64,
//                               needed combinationally for sched_buffered_match
//                               and fast-path gating)
//   - {hw, uuid, tmask}      : VX_dp_ram, OUT_REG=1, async write/read-first
//                               (the bulky payload, ~80b × NW)
//
// Pipelining (for U55C @ 300 MHz):
//   - Fast path (rsp_valid && pc_low && !rsp_low_c && state[rsp_wid]==BUF_EMPTY
//     && !s1_valid): zero added latency, drives fetch_if directly from
//     icache rsp; no decompress16/BRAM dependency.
//   - Slow path: 2 stages.
//       Stage 0 (combinational): decode case, drive BRAM read addr +
//         BRAM write, update state[]/buf_pc[], capture s1_ctx into the
//         stage-1 register. sched_buffered_match runs here on flop reads
//         only (no BRAM dep) so schedule_if.ready is short-path.
//       Stage 1 (combinational from registered s1_ctx + registered BRAM
//         output): mux dec_hw, run decompress16, build fsm_data, drive
//         fetch_if.
//
// Slow-path FSM cases handled in stage 0:
//   C. BUF_RVC emit  : sched_buffered_match && state[sched_wid]==BUF_RVC
//                      reads BRAM at sched_wid (cycle N+1 emit)
//   B. BUF_32HI emit : rsp_valid && state[rsp_wid]==BUF_32HI
//                      reads BRAM at rsp_wid; combines with rsp_low
//   D. BUF_EMPTY     : fresh rsp, RVC or cross-word (NOT fast path)
//                      D-RVC sub-cases (pc_low+low_c, !pc_low+high_c) emit
//                      via s1; D-cross-word stashes + follow_req, no emit.
//
// follow_req_* (cross-word fetch) is asserted in stage 0 alongside the
// BRAM write. VX_fetch prioritises it over scheduler requests.
//
// Backpressure: when fetch_if.ready=0 and s1_valid=1, stage 0 stalls —
// state[]/buf_pc[]/BRAM/follow_req/rsp_ready and sched_buffered_match all
// gated on s1_can_accept.
module VX_decompressor import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input  wire                         clk,
    input  wire                         reset,

    // Scheduler peek (read-only). VX_fetch decides ready/fire.
    input  wire                         sched_valid,
    input  wire [PC_BITS-1:0]           sched_PC,
    input  wire [NW_WIDTH-1:0]          sched_wid,
    output wire                         sched_buffered_match,

    // Icache response.
    input  wire                         rsp_valid,
    input  wire [31:0]                  rsp_word,
    input  wire [PC_BITS-1:0]           rsp_PC,
    input  wire [`NUM_THREADS-1:0]      rsp_tmask,
    input  wire [NW_WIDTH-1:0]          rsp_wid,
    input  wire [UUID_WIDTH-1:0]        rsp_uuid,
    output logic                        rsp_ready,

    // Follow-up icache request (cross-word 32-bit).
    output logic                        follow_req_valid,
    output logic [PC_BITS-1:0]          follow_req_PC,
    output logic [`NUM_THREADS-1:0]     follow_req_tmask,
    output logic [NW_WIDTH-1:0]         follow_req_wid,
    output logic [UUID_WIDTH-1:0]       follow_req_uuid,

    // Decoded fetch interface.
    VX_fetch_if.master                  fetch_if
);
    `UNUSED_SPARAM (INSTANCE_ID)

    // ------------------------------------------------------------------
    // Local types
    // ------------------------------------------------------------------

    typedef enum logic [1:0] {
        BUF_EMPTY = 2'b00,
        BUF_RVC   = 2'b01,
        BUF_32HI  = 2'b10
    } buf_state_e;

    typedef struct packed {
        logic [15:0]              hw;
        logic [UUID_WIDTH-1:0]    uuid;
        logic [`NUM_THREADS-1:0]  tmask;
    } buf_data_t;

    // Stage-1 emit kind (which slow-path case captured this entry).
    typedef enum logic [1:0] {
        S1_NONE = 2'b00,
        S1_C    = 2'b01, // BUF_RVC emit: instr = decompress16(BRAM.hw)
        S1_B    = 2'b10, // BUF_32HI emit: instr = {ctx.inline_hw, BRAM.hw}
        S1_D    = 2'b11  // BUF_EMPTY RVC: instr = decompress16(ctx.inline_hw)
    } s1_kind_e;

    // Stage-1 context. For S1_C/S1_B, BRAM read at sched/rsp wid supplies
    // {hw, uuid, tmask}; ctx.PC/wid latch the metadata flop arrays read.
    // For S1_D, BRAM read is don't-care; ctx.{uuid, tmask} are latched
    // from rsp.
    typedef struct packed {
        logic [15:0]               inline_hw;
        logic [PC_BITS-1:0]        PC;
        logic [NW_WIDTH-1:0]       wid;
        logic [UUID_WIDTH-1:0]     uuid;
        logic [`NUM_THREADS-1:0]   tmask;
    } s1_ctx_t;

    // ------------------------------------------------------------------
    // Local functions
    // ------------------------------------------------------------------

    function automatic logic is_rvc16 (input logic [1:0] op);
        return (op != 2'b11);
    endfunction

    function automatic logic [31:0] decompress16 (
        input logic [15:0] instr_i
    );
        localparam logic [6:0] OPC_L    = 7'b0000011;
        localparam logic [6:0] OPC_S    = 7'b0100011;
        localparam logic [6:0] OPC_B    = 7'b1100011;
        localparam logic [6:0] OPC_JAL  = 7'b1101111;
        localparam logic [6:0] OPC_JALR = 7'b1100111;
        localparam logic [6:0] OPC_LUI  = 7'b0110111;
        localparam logic [6:0] OPC_I    = 7'b0010011;
        localparam logic [6:0] OPC_R    = 7'b0110011;
        localparam logic [6:0] OPC_FL   = 7'b0000111;
        localparam logic [6:0] OPC_FS   = 7'b0100111;
    `ifdef XLEN_64
        localparam logic [6:0] OPC_I_W  = 7'b0011011;
        localparam logic [6:0] OPC_R_W  = 7'b0111011;
    `endif
        logic [2:0]  func3;
        logic [4:0]  rd, rs1, rs2;
        logic [4:0]  rdp, rs1p, rs2p;
        logic [11:0] lsw_imm, lwsp_imm, swsp_imm;
    `ifdef XLEN_64
        logic [11:0] lsd_imm, ldsp_imm, sdsp_imm;
    `endif
        logic [11:0] i_imm, b_imm, x_imm, w_imm;
        logic [19:0] j_imm;
        logic [31:0] instr_o;

        func3 = instr_i[15:13];
        rd    = instr_i[11:7];
        rs1   = rd;
        rs2   = instr_i[6:2];
        rdp   = {2'b01, instr_i[4:2]};
        rs1p  = {2'b01, instr_i[9:7]};
        rs2p  = rdp;
        lsw_imm  = {5'b0, instr_i[5], instr_i[12:10], instr_i[6], 2'b0};
        lwsp_imm = {4'b0, instr_i[3:2], instr_i[12], instr_i[6:4], 2'b0};
        swsp_imm = {4'b0, instr_i[8:7], instr_i[12:9], 2'b0};
    `ifdef XLEN_64
        lsd_imm  = {4'b0, instr_i[6:5], instr_i[12:10], 3'b0};
        ldsp_imm = {3'b0, instr_i[4:2], instr_i[12], instr_i[6:5], 3'b0};
        sdsp_imm = {3'b0, instr_i[9:7], instr_i[12:10], 3'b0};
    `endif
        i_imm  = {{7{instr_i[12]}}, instr_i[6:2]};
        b_imm  = {{5{instr_i[12]}}, instr_i[6:5], instr_i[2], instr_i[11:10], instr_i[4:3]};
        j_imm  = {{10{instr_i[12]}}, instr_i[8], instr_i[10:9], instr_i[6], instr_i[7],
                  instr_i[2], instr_i[11], instr_i[5:3]};
        x_imm  = {{3{instr_i[12]}}, instr_i[4:3], instr_i[5], instr_i[2], instr_i[6], 4'b0};
        w_imm  = {2'd0, instr_i[10:7], instr_i[12:11], instr_i[5], instr_i[6], 2'b0};
        instr_o = '0;

        case (instr_i[1:0])
        2'b00: begin // Quadrant 0
            case (func3)
            3'b000: instr_o = {w_imm, 5'd2, 3'b000, rdp, OPC_I};
        `ifdef FLEN_64
            3'b001: instr_o = {lsd_imm, rs1p, 3'b011, rdp, OPC_FL};
        `endif
            3'b010: instr_o = {lsw_imm, rs1p, 3'b010, rdp, OPC_L};
        `ifdef XLEN_64
            3'b011: instr_o = {lsd_imm, rs1p, 3'b011, rdp, OPC_L};
        `else
            3'b011: instr_o = {lsw_imm, rs1p, 3'b010, rdp, OPC_FL};
        `endif
        `ifdef FLEN_64
            3'b101: instr_o = {lsd_imm[11:5], rs2p, rs1p, 3'b011, lsd_imm[4:0], OPC_FS};
        `endif
            3'b110: instr_o = {lsw_imm[11:5], rs2p, rs1p, 3'b010, lsw_imm[4:0], OPC_S};
        `ifdef XLEN_64
            3'b111: instr_o = {lsd_imm[11:5], rs2p, rs1p, 3'b011, lsd_imm[4:0], OPC_S};
        `else
            3'b111: instr_o = {lsw_imm[11:5], rs2p, rs1p, 3'b010, lsw_imm[4:0], OPC_FS};
        `endif
            default: instr_o = 32'h00000013;
            endcase
        end
        2'b01: begin // Quadrant 1
            case (func3)
            3'b000: begin
                if ((rd == 5'd0) && (i_imm == 12'd0))
                    instr_o = 32'h00000013;
                else
                    instr_o = {i_imm, rd, 3'b000, rd, OPC_I};
            end
        `ifdef XLEN_64
            3'b001: instr_o = {i_imm, rd, 3'b000, rd, OPC_I_W};
        `endif
            3'b010: instr_o = {i_imm, 5'd0, 3'b000, rd, OPC_I};
            3'b011: begin
                if (rd == 5'd2)
                    instr_o = {x_imm, 5'd2, 3'b000, 5'd2, OPC_I};
                else
                    instr_o = {{{8{i_imm[11]}}, i_imm}, rd, OPC_LUI};
            end
            3'b100: begin
                case (instr_i[11:10])
                2'b00: instr_o = {{6'b000000, i_imm[5:0]}, rs1p, 3'b101, rs1p, OPC_I};
                2'b01: instr_o = {{6'b010000, i_imm[5:0]}, rs1p, 3'b101, rs1p, OPC_I};
                2'b10: instr_o = {i_imm, rs1p, 3'b111, rs1p, OPC_I};
                2'b11: begin
                    case ({instr_i[12], instr_i[6:5]})
                    3'b000: instr_o = {7'b0100000, rs2p, rs1p, 3'b000, rs1p, OPC_R};
                    3'b001: instr_o = {7'b0000000, rs2p, rs1p, 3'b100, rs1p, OPC_R};
                    3'b010: instr_o = {7'b0000000, rs2p, rs1p, 3'b110, rs1p, OPC_R};
                    3'b011: instr_o = {7'b0000000, rs2p, rs1p, 3'b111, rs1p, OPC_R};
        `ifdef XLEN_64
                    3'b100: instr_o = {7'b0100000, rs2p, rs1p, 3'b000, rs1p, OPC_R_W};
                    3'b101: instr_o = {7'b0000000, rs2p, rs1p, 3'b000, rs1p, OPC_R_W};
        `endif
                    default: instr_o = 32'h00000013;
                    endcase
                end
                default: instr_o = 32'h00000013;
                endcase
            end
            3'b101: instr_o = {j_imm[19], j_imm[9:0], j_imm[10], j_imm[18:11], 5'd0, OPC_JAL};
            3'b110: instr_o = {b_imm[11], b_imm[9:4], 5'd0, rs1p, 3'b000, b_imm[3:0], b_imm[10], OPC_B};
            3'b111: instr_o = {b_imm[11], b_imm[9:4], 5'd0, rs1p, 3'b001, b_imm[3:0], b_imm[10], OPC_B};
            default: instr_o = 32'h00000013;
            endcase
        end
        2'b10: begin // Quadrant 2
            case (func3)
            3'b000: instr_o = {{6'b000000, i_imm[5:0]}, rd, 3'b001, rd, OPC_I};
        `ifdef FLEN_64
            3'b001: instr_o = {ldsp_imm, 5'd2, 3'b011, rd, OPC_FL};
        `endif
            3'b010: instr_o = {lwsp_imm, 5'd2, 3'b010, rd, OPC_L};
        `ifdef XLEN_64
            3'b011: instr_o = {ldsp_imm, 5'd2, 3'b011, rd, OPC_L};
        `else
            3'b011: instr_o = {lwsp_imm, 5'd2, 3'b010, rd, OPC_FL};
        `endif
            3'b100: begin
                if (instr_i[12] == 1'b0) begin
                    if (rs2 == 5'd0)
                        instr_o = {12'd0, rs1, 3'b000, 5'd0, OPC_JALR};
                    else
                        instr_o = {7'b0000000, rs2, 5'd0, 3'b000, rd, OPC_R};
                end else begin
                    if (rs2 == 5'd0) begin
                        if (rs1 == 5'd0)
                            instr_o = 32'b000000000001_00000_000_00000_1110011;
                        else
                            instr_o = {12'd0, rs1, 3'b000, 5'd1, OPC_JALR};
                    end else
                        instr_o = {7'b0000000, rs2, rd, 3'b000, rd, OPC_R};
                end
            end
        `ifdef FLEN_64
            3'b101: instr_o = {sdsp_imm[11:5], rs2, 5'd2, 3'b011, sdsp_imm[4:0], OPC_FS};
        `endif
            3'b110: instr_o = {swsp_imm[11:5], rs2, 5'd2, 3'b010, swsp_imm[4:0], OPC_S};
        `ifdef XLEN_64
            3'b111: instr_o = {sdsp_imm[11:5], rs2, 5'd2, 3'b011, sdsp_imm[4:0], OPC_S};
        `else
            3'b111: instr_o = {swsp_imm[11:5], rs2, 5'd2, 3'b010, swsp_imm[4:0], OPC_FS};
        `endif
            default: instr_o = 32'h00000013;
            endcase
        end
        default: instr_o = 32'h00000013;
        endcase
        return instr_o;
    endfunction

    // ------------------------------------------------------------------
    // State storage
    // ------------------------------------------------------------------

    buf_state_e         state    [`NUM_WARPS];
    buf_state_e         state_n  [`NUM_WARPS];
    logic [PC_BITS-1:0] buf_pc   [`NUM_WARPS];
    logic [PC_BITS-1:0] buf_pc_n [`NUM_WARPS];

    logic                       buf_we;
    logic [NW_WIDTH-1:0]        buf_waddr;
    buf_data_t                  buf_wdata;
    logic [NW_WIDTH-1:0]        buf_raddr;
    buf_data_t                  buf_rdata;       // registered (OUT_REG=1)
    logic                       buf_read;        // gates rdata_r update during stall

    VX_dp_ram #(
        .DATAW   ($bits(buf_data_t)),
        .SIZE    (`NUM_WARPS),
        .OUT_REG (1),
        .LUTRAM  (1),
        .RDW_MODE("R")
    ) buffer_ram (
        .clk   (clk),
        .reset (reset),
        .read  (buf_read),
        .write (buf_we),
        .wren  (1'b1),
        .waddr (buf_waddr),
        .wdata (buf_wdata),
        .raddr (buf_raddr),
        .rdata (buf_rdata)
    );

    // ------------------------------------------------------------------
    // Pre-computed inputs (combinational)
    // ------------------------------------------------------------------

    wire [15:0] rsp_low    = rsp_word[15:0];
    wire [15:0] rsp_high   = rsp_word[31:16];
    wire        rsp_low_c  = is_rvc16(rsp_low[1:0]);
    wire        rsp_high_c = is_rvc16(rsp_high[1:0]);
    // PC_BITS = XLEN-1 under EXT_C: rsp_PC[0] is byte-PC[1] (halfword sel).
    wire        pc_low     = (rsp_PC[0] == 1'b0);

    // ------------------------------------------------------------------
    // Stage 1 register: pending slow-path emit (drains into fetch_if)
    // ------------------------------------------------------------------

    s1_kind_e   s1_kind;     // S1_NONE = empty
    s1_ctx_t    s1_ctx;
    wire        s1_valid    = (s1_kind != S1_NONE);
    wire        s1_consumed = s1_valid && fetch_if.ready;
    wire        s1_can_accept = ~s1_valid || s1_consumed;

    // Stage 0 advance gate: nothing latches/updates while stage 1 is full.
    wire        st0_advance = s1_can_accept;

    // ------------------------------------------------------------------
    // sched_buffered_match — flop-only PC compare, no BRAM dep.
    // Gated by st0_advance so the scheduler isn't acked while stage 0 is
    // stalled.
    // ------------------------------------------------------------------

    assign sched_buffered_match = sched_valid
                               && (state[sched_wid] != BUF_EMPTY)
                               && (buf_pc[sched_wid] == sched_PC)
                               && st0_advance;

    // ------------------------------------------------------------------
    // Fast path (zero latency, bypasses FSM and BRAM)
    // ------------------------------------------------------------------

    wire fast_path = rsp_valid
                  && pc_low
                  && !rsp_low_c
                  && (state[rsp_wid] == BUF_EMPTY)
                  && ~s1_valid;  // s1 has priority over fast (avoids starvation)

    // ------------------------------------------------------------------
    // Stage 0 case detection + BRAM read addr
    // ------------------------------------------------------------------
    //
    // Priority: C (BUF_RVC emit at sched) > B (BUF_32HI emit at rsp) > D.
    // Case A (fast path) is mutually exclusive with C/B/D when state==EMPTY.

    wire have_C = sched_valid
               && (state[sched_wid] != BUF_EMPTY)
               && (buf_pc[sched_wid] == sched_PC)
               && (state[sched_wid] == BUF_RVC);
    wire have_B = rsp_valid && (state[rsp_wid] == BUF_32HI);
    wire have_D_emit_lo = rsp_valid
                       && (state[rsp_wid] == BUF_EMPTY)
                       && pc_low && rsp_low_c;
    wire have_D_emit_hi = rsp_valid
                       && (state[rsp_wid] == BUF_EMPTY)
                       && !pc_low && rsp_high_c;
    wire have_D_xword   = rsp_valid
                       && (state[rsp_wid] == BUF_EMPTY)
                       && !pc_low && !rsp_high_c;

    // BRAM raddr (cycle-N read latches data for cycle-N+1 stage 1 use).
    assign buf_raddr = have_C ? sched_wid : rsp_wid;
    // Only update raddr_r when stage 0 actually advances; keeps the
    // registered rdata_r aligned with the registered s1_ctx.
    assign buf_read  = st0_advance;

    // ------------------------------------------------------------------
    // Unified halfword-stash PC (for state/buf_pc updates + follow_req)
    // ------------------------------------------------------------------

    logic [PC_BITS-1:0] buf_hw_pc;
    always_comb begin
        if (have_B) begin
            // BUF_32HI emit: stash rsp_high at buf_pc[rsp_wid] + 4 bytes.
            buf_hw_pc = buf_pc[rsp_wid] + PC_BITS'(2);
        end else begin
            // BUF_EMPTY: stash position is rsp_high's PC.
            //   pc_low  -> rsp_PC + 1 (= rsp_PC + 2 bytes)
            //   !pc_low -> rsp_PC     (rsp_high IS at rsp_PC)
            buf_hw_pc = rsp_PC + PC_BITS'(pc_low);
        end
    end
    wire [PC_BITS-1:0] follow_pc = (buf_hw_pc & ~PC_BITS'(1)) + PC_BITS'(2);

    // ------------------------------------------------------------------
    // Stage 0 FSM body (combinational)
    // ------------------------------------------------------------------

    s1_kind_e  s1_kind_in;
    s1_ctx_t   s1_ctx_in;
    logic      slow_rsp_ready;

    always_comb begin : decomp_stage0
        // Defaults
        s1_kind_in = S1_NONE;
        s1_ctx_in  = '0;
        slow_rsp_ready = 1'b0;
        follow_req_valid = 1'b0;
        follow_req_PC    = '0;
        follow_req_tmask = '0;
        follow_req_wid   = '0;
        follow_req_uuid  = '0;
        buf_we    = 1'b0;
        buf_waddr = rsp_wid;
        buf_wdata = '0;
        for (int w = 0; w < `NUM_WARPS; ++w) begin
            state_n[w]  = state[w];
            buf_pc_n[w] = buf_pc[w];
        end

        if (st0_advance) begin
            // Stale flush: scheduler asks for a warp at sched_PC but the
            // warp has stale buffered state at a different PC.
            for (int w = 0; w < `NUM_WARPS; ++w) begin
                if (sched_valid
                 && (sched_wid == w[NW_WIDTH-1:0])
                 && (state[w] != BUF_EMPTY)
                 && (buf_pc[w] != sched_PC)) begin
                    state_n[w] = BUF_EMPTY;
                end
            end

            if (have_C) begin
                // Case C: stage-1 emit a buffered RVC. BRAM read at sched_wid.
                // Don't consume rsp.
                s1_kind_in       = S1_C;
                s1_ctx_in.PC     = buf_pc[sched_wid];
                s1_ctx_in.wid    = sched_wid;
                // uuid/tmask come from BRAM in stage 1.
                state_n[sched_wid] = BUF_EMPTY;

            end else if (have_B) begin
                // Case B: stage-1 emit combined 32b. BRAM read at rsp_wid
                // (returns OLD hw via RDW_MODE="R"). Stash rsp_high.
                s1_kind_in            = S1_B;
                s1_ctx_in.PC          = buf_pc[rsp_wid];
                s1_ctx_in.wid         = rsp_wid;
                s1_ctx_in.inline_hw   = rsp_low; // for combined_32 = {inline_hw, BRAM.hw}
                slow_rsp_ready        = 1'b1;
                buf_we    = 1'b1;
                buf_waddr = rsp_wid;
                buf_wdata = '{hw: rsp_high, uuid: rsp_uuid, tmask: rsp_tmask};
                buf_pc_n[rsp_wid] = buf_hw_pc;
                if (rsp_high_c) begin
                    state_n[rsp_wid] = BUF_RVC;
                end else begin
                    state_n[rsp_wid] = BUF_32HI;
                    follow_req_valid = 1'b1;
                    follow_req_PC    = follow_pc;
                    follow_req_tmask = rsp_tmask;
                    follow_req_wid   = rsp_wid;
                    follow_req_uuid  = rsp_uuid;
                end

            end else if (have_D_emit_lo) begin
                // Case D, pc_low + rsp_low_c: emit RVC (decompress rsp_low),
                // stash rsp_high.
                s1_kind_in          = S1_D;
                s1_ctx_in.PC        = rsp_PC;
                s1_ctx_in.wid       = rsp_wid;
                s1_ctx_in.uuid      = rsp_uuid;
                s1_ctx_in.tmask     = rsp_tmask;
                s1_ctx_in.inline_hw = rsp_low;
                slow_rsp_ready      = 1'b1;
                buf_we    = 1'b1;
                buf_waddr = rsp_wid;
                buf_wdata = '{hw: rsp_high, uuid: rsp_uuid, tmask: rsp_tmask};
                buf_pc_n[rsp_wid] = buf_hw_pc;
                if (rsp_high_c) begin
                    state_n[rsp_wid] = BUF_RVC;
                end else begin
                    state_n[rsp_wid] = BUF_32HI;
                    follow_req_valid = 1'b1;
                    follow_req_PC    = follow_pc;
                    follow_req_tmask = rsp_tmask;
                    follow_req_wid   = rsp_wid;
                    follow_req_uuid  = rsp_uuid;
                end

            end else if (have_D_emit_hi) begin
                // Case D, !pc_low + rsp_high_c: emit RVC at high half,
                // nothing to stash.
                s1_kind_in          = S1_D;
                s1_ctx_in.PC        = rsp_PC;
                s1_ctx_in.wid       = rsp_wid;
                s1_ctx_in.uuid      = rsp_uuid;
                s1_ctx_in.tmask     = rsp_tmask;
                s1_ctx_in.inline_hw = rsp_high;
                slow_rsp_ready      = 1'b1;
                state_n[rsp_wid]    = BUF_EMPTY;

            end else if (have_D_xword) begin
                // Case D, !pc_low + !rsp_high_c: cross-word 32b. Stash
                // rsp_high (low half of new 32b) at PC=rsp_PC, request
                // next aligned word. NO emit this cycle (s1_kind_in
                // stays S1_NONE).
                slow_rsp_ready      = 1'b1;
                buf_we    = 1'b1;
                buf_waddr = rsp_wid;
                buf_wdata = '{hw: rsp_high, uuid: rsp_uuid, tmask: rsp_tmask};
                buf_pc_n[rsp_wid] = rsp_PC;
                state_n[rsp_wid]  = BUF_32HI;
                follow_req_valid  = 1'b1;
                follow_req_PC     = follow_pc;
                follow_req_tmask  = rsp_tmask;
                follow_req_wid    = rsp_wid;
                follow_req_uuid   = rsp_uuid;
            end
        end
    end

    // ------------------------------------------------------------------
    // Stage 1 (combinational): build fsm_data from registered ctx + BRAM
    // ------------------------------------------------------------------

    wire [15:0] dec_hw    = (s1_kind == S1_C) ? buf_rdata.hw : s1_ctx.inline_hw;
    wire [31:0] dec_instr = decompress16(dec_hw);
    wire [31:0] combined_32 = {s1_ctx.inline_hw, buf_rdata.hw};

    fetch_t fsm_data;
    always_comb begin
        fsm_data        = '0;
        fsm_data.PC     = s1_ctx.PC;
        fsm_data.wid    = s1_ctx.wid;
        case (s1_kind)
            S1_C: begin
                fsm_data.instr  = dec_instr;
                fsm_data.is_rvc = 1'b1;
                fsm_data.uuid   = buf_rdata.uuid;
                fsm_data.tmask  = buf_rdata.tmask;
            end
            S1_B: begin
                fsm_data.instr  = combined_32;
                fsm_data.is_rvc = 1'b0;
                fsm_data.uuid   = buf_rdata.uuid;
                fsm_data.tmask  = buf_rdata.tmask;
            end
            S1_D: begin
                fsm_data.instr  = dec_instr;
                fsm_data.is_rvc = 1'b1;
                fsm_data.uuid   = s1_ctx.uuid;
                fsm_data.tmask  = s1_ctx.tmask;
            end
            default: ; // S1_NONE
        endcase
    end

    // ------------------------------------------------------------------
    // Output mux: stage 1 emit takes priority; fast path bypasses
    // ------------------------------------------------------------------

    fetch_t fast_data;
    always_comb begin
        fast_data        = '0;
        fast_data.uuid   = rsp_uuid;
        fast_data.wid    = rsp_wid;
        fast_data.tmask  = rsp_tmask;
        fast_data.PC     = rsp_PC;
        fast_data.instr  = rsp_word;
        fast_data.is_rvc = 1'b0;
    end

    assign fetch_if.valid = s1_valid | fast_path;
    assign fetch_if.data  = s1_valid ? fsm_data : fast_data;

    // rsp_ready: fast path and slow path are mutex (fast_path requires
    // ~s1_valid, slow path captures into s1).
    assign rsp_ready = slow_rsp_ready | (fast_path && fetch_if.ready);

    // ------------------------------------------------------------------
    // Sequential state
    // ------------------------------------------------------------------

    always_ff @(posedge clk) begin
        if (reset) begin
            for (int w = 0; w < `NUM_WARPS; ++w) begin
                state[w]  <= BUF_EMPTY;
                buf_pc[w] <= '0;
            end
            s1_kind <= S1_NONE;
            s1_ctx  <= '0;
        end else begin
            for (int w = 0; w < `NUM_WARPS; ++w) begin
                state[w]  <= state_n[w];
                buf_pc[w] <= buf_pc_n[w];
            end
            // Stage-0 → Stage-1 register
            if (s1_can_accept) begin
                s1_kind <= s1_kind_in;
                s1_ctx  <= s1_ctx_in;
            end
            // (When !s1_can_accept, hold values; stage-0 was stalled so
            // s1_kind_in would be S1_NONE anyway.)
        end
    end

endmodule

`endif
