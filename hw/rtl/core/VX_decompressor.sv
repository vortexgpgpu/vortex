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

`ifdef VX_CFG_EXT_C_ENABLE

// VX_decompressor sits between the icache response and decode when
// EXT_C_ENABLE is set. Per warp it tracks one of:
//   BUF_EMPTY : nothing buffered
//   BUF_RVC   : a 16-bit RVC half-word, ready to emit decompressed
//   BUF_32HI  : low half of a cross-word 32-bit, follow fetch in flight
//   BUF_32RDY : follow arrived (high half latched), awaiting scheduler confirm
//
// Both buffered-emit paths (BUF_RVC and BUF_32RDY) are gated on the scheduler
// re-presenting the warp at the buffered PC — never on the icache response
// alone. This is what keeps a straddler that sits immediately after a taken
// branch/return from being injected onto a wrong path: the follow response may
// arrive before the branch redirect reaches the scheduler, but the emit still
// waits for the scheduler, which reflects the resolved control flow.
//
// Storage layout (scales to NUM_WARPS=64):
//   - state[]  + buf_pc[]    : flops (small, 64x33b ≈ 2K flops at NW=64,
//                               needed combinationally for sched_buffered_match
//                               and fast-path gating)
//   - {payload, uuid, tmask} : VX_dp_ram, OUT_REG=1, async write/read-first
//                               (the bulky payload, ~96b × NW)
//
// Pipelining:
//   - Fast path (rsp_valid && pc_low && !rsp_low_c && state[rsp_wid]==BUF_EMPTY
//     && !s1_valid): zero added latency, drives fetch_if directly from
//     icache rsp; no decompress16/BRAM dependency.
//   - Slow path: 2 stages.
//       Stage 0 (combinational): decode case, drive BRAM read addr +
//         BRAM write, update state[]/buf_pc[], expand any RVC halfword
//         (decompress16) into the registered 32-bit payload, capture s1_ctx
//         into the stage-1 register. sched_buffered_match runs here on flop
//         reads only (no BRAM dep) so schedule_if.ready is short-path.
//       Stage 1 (combinational from registered s1_ctx + registered BRAM
//         output): a 32-bit select over pre-expanded payloads builds
//         fsm_data and drives fetch_if — no decompress16 on this path.
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
    input  wire [`VX_CFG_NUM_THREADS-1:0] sched_tmask,
    input  wire [NCTA_WIDTH-1:0]        sched_cta_id,
    input  wire [NW_WIDTH-1:0]          sched_wid,
    output wire                         sched_buffered_match,
    // Ungated buffered-hit: high whenever the scheduled PC is already in the
    // decompressor buffer, regardless of stage-0 stall. Used by VX_fetch to
    // suppress a redundant icache fetch for a word we already hold.
    output wire                         sched_buffered,

    // Icache response.
    input  wire                         rsp_valid,
    input  wire [31:0]                  rsp_word,
    input  wire [PC_BITS-1:0]           rsp_PC,
    input  wire [`VX_CFG_NUM_THREADS-1:0] rsp_tmask,
    input  wire [NCTA_WIDTH-1:0]        rsp_cta_id,
    input  wire [NW_WIDTH-1:0]          rsp_wid,
    input  wire [UUID_WIDTH-1:0]        rsp_uuid,
    output logic                        rsp_ready,

    // One-outstanding-per-warp tracking (from VX_fetch): high while a warp has
    // an icache request in flight. Used to gate the persistent follow arbiter.
    input  wire [`VX_CFG_NUM_WARPS-1:0] inflight,

    // Follow-up icache request (cross-word 32-bit).
    output logic                        follow_req_valid,
    output logic [PC_BITS-1:0]          follow_req_PC,
    output logic [`VX_CFG_NUM_THREADS-1:0] follow_req_tmask,
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
        BUF_32HI  = 2'b10, // low half of a cross-word 32-bit, follow fetch in flight
        BUF_32RDY = 2'b11  // follow arrived (high half latched), awaiting scheduler
    } buf_state_e;

    // payload holds the already-expanded 32-bit instruction for a buffered RVC
    // (BUF_RVC), or the straddle low half in payload[15:0] with payload[31:16]=0
    // for a cross-word 32-bit (BUF_32HI/BUF_32RDY). The RVC expansion is done at
    // stash time (stage 0) so stage 1 needs no decompress16.
    typedef struct packed {
        logic [31:0]              payload;
        logic [UUID_WIDTH-1:0]    uuid;
        logic [`VX_CFG_NUM_THREADS-1:0]  tmask;
        logic [NCTA_WIDTH-1:0]    cta_id;
    } buf_data_t;

    // Stage-1 emit kind (which slow-path case captured this entry).
    // Scheduler-gated emits (S1_C, S1_B) take tmask from the scheduler's current
    // presentation (post-split), latched into ctx.tmask; the response-gated ones
    // (S1_BR, S1_D) carry their own fetch-time tmask.
    typedef enum logic [2:0] {
        S1_NONE = 3'b000,
        S1_C    = 3'b001, // BUF_RVC emit (scheduler): instr = BRAM.payload
        S1_B    = 3'b010, // BUF_32RDY emit (scheduler): instr = {ctx.payload32[15:0], BRAM.payload[15:0]}
        S1_D    = 3'b011, // BUF_EMPTY RVC (response): instr = ctx.payload32
        S1_BR   = 3'b100  // BUF_32HI emit (response): instr = {ctx.payload32[15:0], BRAM.payload[15:0]}
    } s1_kind_e;

    // Stage-1 context. For S1_C/S1_B, BRAM read at sched/rsp wid supplies
    // {payload, uuid, tmask}; ctx.PC/wid latch the metadata flop arrays read.
    // For S1_D, BRAM read is don't-care; ctx.{uuid, tmask} are latched
    // from rsp. payload32 holds the expanded 32-bit RVC (S1_D), or the straddle
    // high half in payload32[15:0] (S1_B/S1_BR); both are formed in stage 0.
    typedef struct packed {
        logic [31:0]               payload32;
        logic [PC_BITS-1:0]        PC;
        logic [NW_WIDTH-1:0]       wid;
        logic [UUID_WIDTH-1:0]     uuid;
        logic [`VX_CFG_NUM_THREADS-1:0]   tmask;
        logic [NCTA_WIDTH-1:0]     cta_id;
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
    `ifdef VX_CFG_XLEN_64
        localparam logic [6:0] OPC_I_W  = 7'b0011011;
        localparam logic [6:0] OPC_R_W  = 7'b0111011;
    `endif
        logic [2:0]  func3;
        logic [4:0]  rd, rs1, rs2;
        logic [4:0]  rdp, rs1p, rs2p;
        logic [11:0] lsw_imm, lwsp_imm, swsp_imm;
    `ifdef VX_CFG_XLEN_64
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
    `ifdef VX_CFG_XLEN_64
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
        `ifdef VX_CFG_FLEN_64
            3'b001: instr_o = {lsd_imm, rs1p, 3'b011, rdp, OPC_FL};
        `endif
            3'b010: instr_o = {lsw_imm, rs1p, 3'b010, rdp, OPC_L};
        `ifdef VX_CFG_XLEN_64
            3'b011: instr_o = {lsd_imm, rs1p, 3'b011, rdp, OPC_L};
        `else
            3'b011: instr_o = {lsw_imm, rs1p, 3'b010, rdp, OPC_FL};
        `endif
        `ifdef VX_CFG_FLEN_64
            3'b101: instr_o = {lsd_imm[11:5], rs2p, rs1p, 3'b011, lsd_imm[4:0], OPC_FS};
        `endif
            3'b110: instr_o = {lsw_imm[11:5], rs2p, rs1p, 3'b010, lsw_imm[4:0], OPC_S};
        `ifdef VX_CFG_XLEN_64
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
        `ifdef VX_CFG_XLEN_64
            3'b001: instr_o = {i_imm, rd, 3'b000, rd, OPC_I_W};
        `else
            // RV32C: C.JAL -> JAL x1, j_imm (CJ-format).
            3'b001: instr_o = {j_imm[19], j_imm[9:0], j_imm[10], j_imm[18:11], 5'd1, OPC_JAL};
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
        `ifdef VX_CFG_XLEN_64
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
        `ifdef VX_CFG_FLEN_64
            3'b001: instr_o = {ldsp_imm, 5'd2, 3'b011, rd, OPC_FL};
        `endif
            3'b010: instr_o = {lwsp_imm, 5'd2, 3'b010, rd, OPC_L};
        `ifdef VX_CFG_XLEN_64
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
        `ifdef VX_CFG_FLEN_64
            3'b101: instr_o = {sdsp_imm[11:5], rs2, 5'd2, 3'b011, sdsp_imm[4:0], OPC_FS};
        `endif
            3'b110: instr_o = {swsp_imm[11:5], rs2, 5'd2, 3'b010, swsp_imm[4:0], OPC_S};
        `ifdef VX_CFG_XLEN_64
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

    buf_state_e         state    [`VX_CFG_NUM_WARPS];
    buf_state_e         state_n  [`VX_CFG_NUM_WARPS];
    logic [PC_BITS-1:0] buf_pc   [`VX_CFG_NUM_WARPS];
    logic [PC_BITS-1:0] buf_pc_n [`VX_CFG_NUM_WARPS];

    // Follow metadata captured on BUF_32HI entry (the persistent follow arbiter
    // re-issues the request from registered state, so it needs its own copy).
    logic [UUID_WIDTH-1:0]          hi_uuid   [`VX_CFG_NUM_WARPS];
    logic [UUID_WIDTH-1:0]          hi_uuid_n [`VX_CFG_NUM_WARPS];
    logic [`VX_CFG_NUM_THREADS-1:0] hi_tmask   [`VX_CFG_NUM_WARPS];
    logic [`VX_CFG_NUM_THREADS-1:0] hi_tmask_n [`VX_CFG_NUM_WARPS];
    // High half of the cross-word 32-bit, latched from the follow response and
    // held (BUF_32RDY) until the scheduler-gated emit combines it with the
    // BRAM-stashed low half.
    logic [15:0]                    hi_hw   [`VX_CFG_NUM_WARPS];
    logic [15:0]                    hi_hw_n [`VX_CFG_NUM_WARPS];
    // drain[w]: the BUF_32HI straddle on warp w was squashed by a redirect
    // while its follow was in flight; discard the follow response (no emit).
    logic [`VX_CFG_NUM_WARPS-1:0]   drain, drain_n;
    // spec[w]: the BUF_32HI straddle is speculative — it trails an RVC emitted
    // from the same word (have_D_emit_lo), which may be a taken branch, so the
    // warp is NOT yet committed to the straddle PC. Speculative straddles defer
    // their emit to scheduler re-presentation (BUF_32RDY -> have_B_emit); the
    // scheduler presenting a different PC drops them. Non-speculative straddles
    // (have_D_xword: the warp is already at the straddle PC and was acked by its
    // own fetch, so the scheduler will not re-present it) emit on the follow
    // response (have_B_resp).
    logic [`VX_CFG_NUM_WARPS-1:0]   spec, spec_n;

    logic                       buf_we;
    logic [NW_WIDTH-1:0]        buf_waddr;
    buf_data_t                  buf_wdata;
    logic [NW_WIDTH-1:0]        buf_raddr;
    buf_data_t                  buf_rdata;       // registered (OUT_REG=1)
    logic                       buf_read;        // gates rdata_r update during stall

    VX_dp_ram #(
        .DATAW   ($bits(buf_data_t)),
        .SIZE    (`VX_CFG_NUM_WARPS),
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

    // Stash payload for a captured rsp_high halfword: a complete RVC is
    // pre-expanded here (stage 0), a straddle low half is stored raw in [15:0].
    wire [31:0] stash_payload = rsp_high_c ? decompress16(rsp_high)
                                           : {16'b0, rsp_high};

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

    // Ack the scheduler ONLY when the buffered word is actually emitted this
    // cycle (BUF_RVC -> have_C, BUF_32RDY -> have_B_emit). The scheduler stalls
    // an acked warp until decode returns, so acking a BUF_32HI warp (follow
    // still in flight, nothing emitted) would stall it forever. Leaving BUF_32HI
    // un-acked keeps the scheduler presenting buf_pc — fetch suppressed via
    // sched_buffered below — until the follow lands and it turns BUF_32RDY, at
    // which point have_B_emit fires and acks.
    assign sched_buffered_match = (have_C || have_B_emit) && st0_advance;
    // Suppress the scheduler icache fetch for ANY buffered warp (non-EMPTY),
    // regardless of buf_pc match. A warp with buffered state must not have a
    // fresh fetch outstanding: on a redirect (buf_pc != sched_PC) the buffered
    // entry is stale and will be flushed (BUF_RVC/BUF_32RDY -> EMPTY, or BUF_32HI
    // -> drain) — but if a fetch were allowed to issue before the flush, its
    // response would return while the warp is still BUF_RVC/BUF_32RDY, a state no
    // response case consumes (rsp_ready stuck low -> icache deadlock). Holding
    // the fetch until the buffer drains to EMPTY closes that hole. Cross-word
    // follows are issued separately by the persistent arbiter, not here.
    assign sched_buffered = sched_valid && (state[sched_wid] != BUF_EMPTY);

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
               && (buf_pc[sched_wid] == sched_PC)
               && (state[sched_wid] == BUF_RVC);
    // Scheduler-gated straddle emit: the follow response already arrived
    // (BUF_32RDY, high half in hi_hw) and the scheduler now re-presents the warp
    // at the stashed straddle PC, confirming the preceding word did NOT redirect
    // away. Emitting only here (never on the follow response alone) is what stops
    // a straddler sitting right after a taken branch/return from landing on a
    // wrong path.
    wire have_B_emit = sched_valid
                    && (buf_pc[sched_wid] == sched_PC)
                    && (state[sched_wid] == BUF_32RDY);
    // Redirect detector: a BUF_32HI warp re-presented at a PC that DIFFERS from
    // its stashed straddle PC is a genuine branch redirect (buf_pc == sched_PC is
    // normal straddler continuation after an RVC-then-straddler word).
    wire redirect_hit = sched_valid && (state[sched_wid] == BUF_32HI)
                     && (buf_pc[sched_wid] != sched_PC);
    wire rsp_discard = rsp_valid && (state[rsp_wid] == BUF_32HI)
                    && (drain[rsp_wid] || (redirect_hit && (sched_wid == rsp_wid)));
    // Follow response for an in-flight straddle (not squashed):
    //   speculative     -> latch high half, advance to BUF_32RDY (defer emit).
    //   non-speculative -> emit the combined 32b now (have_B_resp).
    wire follow_rsp   = rsp_valid && (state[rsp_wid] == BUF_32HI) && ~rsp_discard;
    wire follow_store = follow_rsp && spec[rsp_wid];
    wire have_B_resp  = follow_rsp && ~spec[rsp_wid];
    wire have_D_emit_lo = rsp_valid
                       && (state[rsp_wid] == BUF_EMPTY)
                       && pc_low && rsp_low_c;
    wire have_D_emit_hi = rsp_valid
                       && (state[rsp_wid] == BUF_EMPTY)
                       && !pc_low && rsp_high_c;
    wire have_D_xword   = rsp_valid
                       && (state[rsp_wid] == BUF_EMPTY)
                       && !pc_low && !rsp_high_c;

    // BRAM raddr (cycle-N read latches data for cycle-N+1 stage 1 use). Both
    // scheduler-gated emits (have_C, have_B_emit) read at sched_wid.
    assign buf_raddr = (have_C || have_B_emit) ? sched_wid : rsp_wid;
    // Only update raddr_r when stage 0 actually advances; keeps the
    // registered rdata_r aligned with the registered s1_ctx.
    assign buf_read  = st0_advance;

    // ------------------------------------------------------------------
    // Unified halfword-stash PC (for state/buf_pc updates + follow_req)
    // ------------------------------------------------------------------

    // BUF_EMPTY stash position is rsp_high's PC:
    //   pc_low  -> rsp_PC + 1 (= rsp_PC + 2 bytes)
    //   !pc_low -> rsp_PC     (rsp_high IS at rsp_PC)
    wire [PC_BITS-1:0] buf_hw_pc = rsp_PC + PC_BITS'(pc_low);

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
        buf_we    = 1'b0;
        buf_waddr = rsp_wid;
        buf_wdata = '0;
        for (int w = 0; w < `VX_CFG_NUM_WARPS; ++w) begin
            state_n[w]    = state[w];
            buf_pc_n[w]   = buf_pc[w];
            hi_uuid_n[w]  = hi_uuid[w];
            hi_tmask_n[w] = hi_tmask[w];
            hi_hw_n[w]    = hi_hw[w];
            drain_n[w]    = drain[w];
            spec_n[w]     = spec[w];
        end

        if (st0_advance) begin
            // Stale flush: scheduler presents a buffered warp at a PC that
            // differs from its stashed PC — a branch/return redirected it, so the
            // buffered straddler/RVC is on the wrong path and must be dropped.
            //   BUF_32HI  : follow fetch still in flight -> mark drain so its
            //               response is discarded (not latched), keeping the warp
            //               BUF_32HI until the response arrives and clears drain.
            //   BUF_32RDY : follow already arrived, no outstanding fetch -> clear
            //               directly.
            //   BUF_RVC   : no outstanding fetch -> clear directly.
            // (buf_pc == sched_PC is normal straddler continuation after an
            // RVC-then-straddler word — must NOT drop.)
            for (int w = 0; w < `VX_CFG_NUM_WARPS; ++w) begin
                if (sched_valid && (sched_wid == w[NW_WIDTH-1:0])
                 && (buf_pc[w] != sched_PC)) begin
                    if (state[w] == BUF_32HI) begin
                        drain_n[w] = 1'b1;
                    end else if ((state[w] == BUF_32RDY) || (state[w] == BUF_RVC)) begin
                        state_n[w] = BUF_EMPTY;
                    end
                end
            end

            if (have_C) begin
                // Case C: stage-1 emit a buffered RVC. BRAM read at sched_wid.
                // Don't consume rsp. tmask is the scheduler's current mask: the
                // trailing RVC may follow a diverging branch in the same word, so
                // the stashed (pre-split) mask is stale — the scheduler reflects
                // the resolved post-split mask. uuid still comes from BRAM.
                s1_kind_in       = S1_C;
                s1_ctx_in.PC     = buf_pc[sched_wid];
                s1_ctx_in.wid    = sched_wid;
                s1_ctx_in.tmask  = sched_tmask;
                s1_ctx_in.cta_id = sched_cta_id;
                state_n[sched_wid] = BUF_EMPTY;

            end else if (have_B_emit) begin
                // Case B (scheduler-confirmed): stage-1 emit the combined 32b.
                // BRAM read at sched_wid returns the stashed low half; the high
                // half is the registered hi_hw. tmask from the scheduler (the
                // straddle may follow a diverging branch — see have_C); uuid from
                // BRAM. Don't consume rsp.
                s1_kind_in          = S1_B;
                s1_ctx_in.PC        = buf_pc[sched_wid];
                s1_ctx_in.wid       = sched_wid;
                s1_ctx_in.tmask     = sched_tmask;
                s1_ctx_in.cta_id    = sched_cta_id;
                s1_ctx_in.payload32 = 32'(hi_hw[sched_wid]); // straddle high half -> [15:0]
                state_n[sched_wid]  = BUF_EMPTY;

            end else if (rsp_discard) begin
                // Squashed straddle: consume the follow response but do NOT
                // emit; return the warp to BUF_EMPTY.
                slow_rsp_ready    = 1'b1;
                state_n[rsp_wid]  = BUF_EMPTY;
                drain_n[rsp_wid]  = 1'b0;

            end else if (have_B_resp) begin
                // Non-speculative straddle: the warp is committed to buf_pc (it
                // arrived there via its own fetch, no intervening branch), so its
                // stashed fetch-time tmask/uuid are still current — emit the
                // combined 32b on the follow response (S1_BR, meta from BRAM).
                // BRAM read at rsp_wid supplies the low half; rsp_low is high.
                s1_kind_in          = S1_BR;
                s1_ctx_in.PC        = buf_pc[rsp_wid];
                s1_ctx_in.wid       = rsp_wid;
                s1_ctx_in.payload32 = 32'(rsp_low); // straddle high half -> [15:0]
                slow_rsp_ready      = 1'b1;
                state_n[rsp_wid]    = BUF_EMPTY;

            end else if (follow_store) begin
                // Speculative straddle: latch the high half and advance to
                // BUF_32RDY. The emit is deferred to have_B_emit once the
                // scheduler re-presents the warp (confirming the trailing RVC
                // did not redirect away).
                slow_rsp_ready   = 1'b1;
                hi_hw_n[rsp_wid] = rsp_low; // high half of the cross-word 32b
                state_n[rsp_wid] = BUF_32RDY;

            end else if (have_D_emit_lo) begin
                // Case D, pc_low + rsp_low_c: emit RVC (decompress rsp_low),
                // stash rsp_high.
                s1_kind_in          = S1_D;
                s1_ctx_in.PC        = rsp_PC;
                s1_ctx_in.wid       = rsp_wid;
                s1_ctx_in.uuid      = rsp_uuid;
                s1_ctx_in.tmask     = rsp_tmask;
                s1_ctx_in.cta_id    = rsp_cta_id;
                s1_ctx_in.payload32 = decompress16(rsp_low);
                slow_rsp_ready      = 1'b1;
                buf_we    = 1'b1;
                buf_waddr = rsp_wid;
                buf_wdata = '{payload: stash_payload, uuid: rsp_uuid, tmask: rsp_tmask, cta_id: rsp_cta_id};
                buf_pc_n[rsp_wid] = buf_hw_pc;
                if (rsp_high_c) begin
                    state_n[rsp_wid] = BUF_RVC;
                end else begin
                    state_n[rsp_wid]   = BUF_32HI;
                    spec_n[rsp_wid]    = 1'b1; // trails an RVC that may branch
                    hi_uuid_n[rsp_wid] = rsp_uuid;
                    hi_tmask_n[rsp_wid]= rsp_tmask;
                end

            end else if (have_D_emit_hi) begin
                // Case D, !pc_low + rsp_high_c: emit RVC at high half,
                // nothing to stash.
                s1_kind_in          = S1_D;
                s1_ctx_in.PC        = rsp_PC;
                s1_ctx_in.wid       = rsp_wid;
                s1_ctx_in.uuid      = rsp_uuid;
                s1_ctx_in.tmask     = rsp_tmask;
                s1_ctx_in.cta_id    = rsp_cta_id;
                s1_ctx_in.payload32 = decompress16(rsp_high);
                slow_rsp_ready      = 1'b1;
                state_n[rsp_wid]    = BUF_EMPTY;

            end else if (have_D_xword) begin
                // Case D, !pc_low + !rsp_high_c: cross-word 32b. Stash
                // rsp_high (low half of new 32b) at PC=rsp_PC. The follow is
                // issued by the persistent arbiter from BUF_32HI state. NO emit
                // this cycle (s1_kind_in stays S1_NONE).
                slow_rsp_ready      = 1'b1;
                buf_we    = 1'b1;
                buf_waddr = rsp_wid;
                buf_wdata = '{payload: stash_payload, uuid: rsp_uuid, tmask: rsp_tmask, cta_id: rsp_cta_id};
                buf_pc_n[rsp_wid]  = rsp_PC;
                state_n[rsp_wid]   = BUF_32HI;
                spec_n[rsp_wid]    = 1'b0; // warp is committed to this straddle PC
                hi_uuid_n[rsp_wid] = rsp_uuid;
                hi_tmask_n[rsp_wid]= rsp_tmask;
            end
        end
    end

    // ------------------------------------------------------------------
    // Persistent follow-request arbiter (Invariant B)
    //
    // Re-derives the cross-word follow from *registered* BUF_32HI state rather
    // than the transient response, so it survives icache backpressure. Skips a
    // warp whose request is already in flight (`inflight`), giving at-most-one
    // outstanding icache request per warp -> the single-slot tag_store can't
    // alias. Lowest-wid priority (reverse scan).
    // ------------------------------------------------------------------
    logic                follow_sel_valid;
    logic [NW_WIDTH-1:0] follow_sel_wid;
    always_comb begin
        follow_sel_valid = 1'b0;
        follow_sel_wid   = '0;
        for (int w = `VX_CFG_NUM_WARPS-1; w >= 0; --w) begin
            if ((state[w] == BUF_32HI) && ~inflight[w[NW_WIDTH-1:0]]) begin
                follow_sel_valid = 1'b1;
                follow_sel_wid   = w[NW_WIDTH-1:0];
            end
        end
    end

    assign follow_req_valid = follow_sel_valid;
    assign follow_req_wid   = follow_sel_wid;
    assign follow_req_PC    = (buf_pc[follow_sel_wid] & ~PC_BITS'(1)) + PC_BITS'(2);
    assign follow_req_tmask = hi_tmask[follow_sel_wid];
    assign follow_req_uuid  = hi_uuid[follow_sel_wid];

    // ------------------------------------------------------------------
    // Stage 1 (combinational): build fsm_data from registered ctx + BRAM
    // ------------------------------------------------------------------

    // Both RVC and straddle sources are registered 32-bit payloads pre-formed in
    // stage 0; this stage only selects, no decompress16.
    wire [31:0] instr_rvc   = (s1_kind == S1_C) ? buf_rdata.payload   // buffered RVC
                                                : s1_ctx.payload32;   // immediate RVC (S1_D)
    wire [31:0] combined_32 = {s1_ctx.payload32[15:0], buf_rdata.payload[15:0]};

    fetch_t fsm_data;
    always_comb begin
        fsm_data        = '0;
        fsm_data.PC     = s1_ctx.PC;
        fsm_data.wid    = s1_ctx.wid;
        // cta_id: response-gated straddle (S1_BR) carries the BRAM-stashed CTA;
        // all other kinds captured the CTA into ctx (sched or rsp) in stage 0.
        fsm_data.cta_id = (s1_kind == S1_BR) ? buf_rdata.cta_id : s1_ctx.cta_id;
        case (s1_kind)
            S1_C: begin
                // Scheduler-gated RVC: tmask from ctx (scheduler), uuid from BRAM.
                fsm_data.instr  = instr_rvc;
                fsm_data.is_rvc = 1'b1;
                fsm_data.uuid   = buf_rdata.uuid;
                fsm_data.tmask  = s1_ctx.tmask;
            end
            S1_B: begin
                // Scheduler-gated straddle: tmask from ctx (scheduler), uuid BRAM.
                fsm_data.instr  = combined_32;
                fsm_data.is_rvc = 1'b0;
                fsm_data.uuid   = buf_rdata.uuid;
                fsm_data.tmask  = s1_ctx.tmask;
            end
            S1_BR: begin
                // Response-gated straddle (non-speculative): uuid/tmask from BRAM
                // (the committed fetch context).
                fsm_data.instr  = combined_32;
                fsm_data.is_rvc = 1'b0;
                fsm_data.uuid   = buf_rdata.uuid;
                fsm_data.tmask  = buf_rdata.tmask;
            end
            S1_D: begin
                fsm_data.instr  = instr_rvc;
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
        fast_data.cta_id = rsp_cta_id;
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
            for (int w = 0; w < `VX_CFG_NUM_WARPS; ++w) begin
                state[w]    <= BUF_EMPTY;
                buf_pc[w]   <= '0;
                hi_uuid[w]  <= '0;
                hi_tmask[w] <= '0;
                hi_hw[w]    <= '0;
            end
            drain   <= '0;
            spec    <= '0;
            s1_kind <= S1_NONE;
            s1_ctx  <= '0;
        end else begin
            for (int w = 0; w < `VX_CFG_NUM_WARPS; ++w) begin
                state[w]    <= state_n[w];
                buf_pc[w]   <= buf_pc_n[w];
                hi_uuid[w]  <= hi_uuid_n[w];
                hi_tmask[w] <= hi_tmask_n[w];
                hi_hw[w]    <= hi_hw_n[w];
            end
            drain <= drain_n;
            spec  <= spec_n;
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
