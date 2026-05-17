// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_cp_engine — per-queue Command Processor Engine (CPE).
//
// Phase 2b: real decode + resource-bid + retire logic. The fetch and
// unpack paths are left wired through to `cmd_in` / `cmd_in_valid` from
// outside (Phase 3 splices VX_cp_fetch + VX_cp_unpack onto these inputs
// once the AXI xbar is real).
//
// FSM:
//   IDLE         : no command in hand; assert cmd_in_ready
//   DECODE       : combinational classification of cmd opcode -> resource
//   BID          : assert bid line for the chosen resource
//   WAIT_DONE    : hold bid until resource signals done
//   RETIRE       : pulse retire_evt + advance seqnum; back to IDLE
//
// For Phase 2b the engine handles:
//   - CMD_NOP (retire immediately)
//   - CMD_LAUNCH (bid KMU)
//   - CMD_DCR_WRITE / CMD_DCR_READ (bid DCR)
//   - CMD_MEM_* (bid DMA)
// Other opcodes (CMD_FENCE, CMD_EVENT_*) are passed through but
// effectively NOP for now (FSM retires them without doing anything).
// Real semantics for those land in Phase 4.
// ============================================================================

module VX_cp_engine
  import VX_cp_pkg::*;
#(
  parameter int QID = 0
)(
  input  wire clk,
  input  wire reset,

  // Per-queue state mirror (driven by AXI-Lite Q_* register writes from
  // the host via VX_cp_core's regfile). Read by this engine.
  input  cpe_state_t              state_in,
  output cpe_state_t              state_out,

  // Decoded command stream input. Phase 3 wires VX_cp_fetch + VX_cp_unpack
  // here; for Phase 2b nothing drives it from outside (the engine just
  // sits in IDLE waiting on cmd_in_valid).
  input  wire                     cmd_in_valid,
  input  cmd_t                    cmd_in,
  output logic                    cmd_in_ready,

  // Bid lines to the three resource arbiters.
  VX_cp_engine_bid_if.bidder      bid_kmu,
  VX_cp_engine_bid_if.bidder      bid_dma,
  VX_cp_engine_bid_if.bidder      bid_dcr,

  // Retirement signaling to VX_cp_completion.
  output logic                    retire_evt,
  output logic [63:0]             retire_seqnum,

  // Profiling sample pulses (Phase 4 hookup).
  output logic                    submit_evt,
  output logic                    start_evt,
  output logic                    end_evt,
  output logic [63:0]             profile_slot
);

  typedef enum logic [2:0] {
    S_IDLE,
    S_DECODE,
    S_BID,
    S_WAIT_DONE,
    S_RETIRE
  } state_e;

  state_e       fsm;
  cmd_t         cur_cmd;
  cp_resource_e cur_res;
  logic         no_resource;        // true for opcodes that bypass arbiters (NOP, FENCE, EVENT_*)
  logic [63:0]  seqnum_r;

  // -------------------------------------------------------------------------
  // Opcode → resource classification (combinational over cur_cmd).
  // -------------------------------------------------------------------------
  function automatic cp_resource_e classify(cp_opcode_e op,
                                            output logic skip);
    skip = 1'b0;
    case (op)
      CMD_LAUNCH:                    return RES_KMU;
      CMD_DCR_WRITE, CMD_DCR_READ:   return RES_DCR;
      CMD_MEM_WRITE,
      CMD_MEM_READ,
      CMD_MEM_COPY:                  return RES_DMA;
      default: begin
        skip = 1'b1;
        return RES_KMU;   // unused when skip=1
      end
    endcase
  endfunction

  // Grant + done signals from the three resource arbiters / consumers.
  // Engine sees which arbiter has granted and waits for the matching done.
  wire kmu_done = bid_kmu.grant;  // VX_cp_launch's done is OR'd into all CPEs; CPE filters on its own grant
  wire dma_done = bid_dma.grant;  // similarly tied for Phase 2b
  wire dcr_done = bid_dcr.grant;
  // NOTE: tying done to grant here is a Phase 2b shortcut — the
  // resource modules' real `done` outputs are aggregated in VX_cp_core
  // and routed back per-CPE in Phase 3. For now we treat "got grant"
  // as "done immediately next cycle" which lets the FSM cycle through
  // states cleanly without external resource feedback.

  // -------------------------------------------------------------------------
  // FSM
  // -------------------------------------------------------------------------

  always_ff @(posedge clk) begin
    automatic cp_resource_e res;
    automatic logic         skip_flag;
    if (reset) begin
      fsm         <= S_IDLE;
      cur_cmd     <= '0;
      cur_res     <= RES_KMU;
      no_resource <= 1'b0;
      seqnum_r    <= '0;
    end else begin
      case (fsm)
        S_IDLE: begin
          if (cmd_in_valid) begin
            cur_cmd <= cmd_in;
            fsm     <= S_DECODE;
          end
        end
        S_DECODE: begin
          res         = classify(cp_opcode_e'(cur_cmd.hdr.opcode), skip_flag);
          cur_res     <= res;
          no_resource <= skip_flag;
          if (skip_flag) begin
            fsm <= S_RETIRE;
          end else begin
            fsm <= S_BID;
          end
        end
        S_BID: begin
          // Wait for our grant.
          case (cur_res)
            RES_KMU: if (bid_kmu.grant) fsm <= S_WAIT_DONE;
            RES_DMA: if (bid_dma.grant) fsm <= S_WAIT_DONE;
            RES_DCR: if (bid_dcr.grant) fsm <= S_WAIT_DONE;
            default: fsm <= S_RETIRE;
          endcase
        end
        S_WAIT_DONE: begin
          // Phase 2b: treat grant as done. Phase 3+ replaces with per-
          // resource done aggregator.
          fsm <= S_RETIRE;
        end
        S_RETIRE: begin
          seqnum_r <= seqnum_r + 64'd1;
          fsm      <= S_IDLE;
        end
        default: fsm <= S_IDLE;
      endcase
    end
  end

  // -------------------------------------------------------------------------
  // Output drivers
  // -------------------------------------------------------------------------

  always_comb begin
    cmd_in_ready = (fsm == S_IDLE);

    // Bid one resource at a time.
    bid_kmu.valid     = (fsm == S_BID) && (cur_res == RES_KMU);
    bid_kmu.priority_ = state_in.prio;
    bid_kmu.cmd       = cur_cmd;

    bid_dma.valid     = (fsm == S_BID) && (cur_res == RES_DMA);
    bid_dma.priority_ = state_in.prio;
    bid_dma.cmd       = cur_cmd;

    bid_dcr.valid     = (fsm == S_BID) && (cur_res == RES_DCR);
    bid_dcr.priority_ = state_in.prio;
    bid_dcr.cmd       = cur_cmd;

    retire_evt    = (fsm == S_RETIRE);
    retire_seqnum = seqnum_r;

    // Profiling hooks (Phase 4 fills these in for real).
    submit_evt   = (fsm == S_DECODE) && cur_cmd.hdr.flags[F_PROFILE];
    start_evt    = (fsm == S_BID) && cur_cmd.hdr.flags[F_PROFILE] &&
                   ((cur_res == RES_KMU && bid_kmu.grant) ||
                    (cur_res == RES_DMA && bid_dma.grant) ||
                    (cur_res == RES_DCR && bid_dcr.grant));
    end_evt      = (fsm == S_RETIRE) && cur_cmd.hdr.flags[F_PROFILE];
    profile_slot = cur_cmd.profile_slot;
  end

  // State mirror passes through with seqnum tracked locally.
  always_comb begin
    state_out         = state_in;
    state_out.seqnum  = seqnum_r;
  end

  `UNUSED_VAR (QID)
  `UNUSED_VAR (kmu_done)
  `UNUSED_VAR (dma_done)
  `UNUSED_VAR (dcr_done)
  `UNUSED_VAR (no_resource)

endmodule : VX_cp_engine
