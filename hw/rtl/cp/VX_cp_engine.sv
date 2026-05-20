// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_cp_engine — per-queue Command Processor Engine (CPE).
//
// Consumes a decoded command stream on `cmd_in`, classifies each command
// onto one of three shared resources (KMU / DMA / DCR), bids for the
// resource through the engine_bid interface, and retires the command
// once the resource signals done.
//
// FSM:
//   IDLE         : no command in hand; assert cmd_in_ready
//   DECODE       : combinational classification of cmd opcode -> resource
//   BID          : assert bid line for the chosen resource
//   WAIT_DONE    : hold bid until resource signals done
//   RETIRE       : pulse retire_evt + advance seqnum; back to IDLE
//
// Opcodes handled:
//   - CMD_NOP / CMD_FENCE                          (retire immediately)
//   - CMD_LAUNCH                                   (bid KMU)
//   - CMD_DCR_WRITE / CMD_DCR_READ                 (bid DCR)
//   - CMD_MEM_*                                    (bid DMA)
//   - CMD_EVENT_SIGNAL / CMD_EVENT_WAIT            (bid EVENT)
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

  // Decoded command stream input (driven by VX_cp_fetch + VX_cp_unpack).
  input  wire                     cmd_in_valid,
  input  cmd_t                    cmd_in,
  output logic                    cmd_in_ready,

  // Bid lines to the four resource arbiters.
  VX_cp_engine_bid_if.bidder      bid_kmu,
  VX_cp_engine_bid_if.bidder      bid_dma,
  VX_cp_engine_bid_if.bidder      bid_dcr,
  VX_cp_engine_bid_if.bidder      bid_event,

  // Per-resource done signals. These come from the resource module
  // (launch/dma/dcr_proxy/event_unit) and pulse high for one cycle when
  // the resource finishes the current command. The engine consumes them
  // in S_WAIT_DONE to know when to retire.
  input  wire                     kmu_done_i,
  input  wire                     dma_done_i,
  input  wire                     dcr_done_i,
  input  wire                     event_done_i,

  // Retirement signaling to VX_cp_completion.
  output logic                    retire_evt,
  output logic [63:0]             retire_seqnum,

  // Profiling sample pulses (consumed by the event unit).
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
      CMD_EVENT_SIGNAL,
      CMD_EVENT_WAIT:                return RES_EVT;
      default: begin
        skip = 1'b1;
        return RES_KMU;   // unused when skip=1
      end
    endcase
  endfunction

  // The done pulses (kmu_done_i / dma_done_i / dcr_done_i) are broadcast
  // from the shared resource modules to every CPE. The bid arbiter grants
  // one CPE per resource at a time and the resource processes one command
  // at a time, so only the granted CPE is in S_WAIT_DONE when the matching
  // pulse arrives; non-granted CPEs ignore it.

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
            RES_KMU:   if (bid_kmu.grant)   fsm <= S_WAIT_DONE;
            RES_DMA:   if (bid_dma.grant)   fsm <= S_WAIT_DONE;
            RES_DCR:   if (bid_dcr.grant)   fsm <= S_WAIT_DONE;
            RES_EVT: if (bid_event.grant) fsm <= S_WAIT_DONE;
            default:                        fsm <= S_RETIRE;
          endcase
        end
        S_WAIT_DONE: begin
          // Wait for the resource's actual done pulse before retiring.
          case (cur_res)
            RES_KMU:   if (kmu_done_i)   fsm <= S_RETIRE;
            RES_DMA:   if (dma_done_i)   fsm <= S_RETIRE;
            RES_DCR:   if (dcr_done_i)   fsm <= S_RETIRE;
            RES_EVT: if (event_done_i) fsm <= S_RETIRE;
            default:                     fsm <= S_RETIRE;
          endcase
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

    bid_event.valid     = (fsm == S_BID) && (cur_res == RES_EVT);
    bid_event.priority_ = state_in.prio;
    bid_event.cmd       = cur_cmd;

    retire_evt    = (fsm == S_RETIRE);
    retire_seqnum = seqnum_r;

    submit_evt   = (fsm == S_DECODE) && cur_cmd.hdr.flags[F_PROFILE];
    start_evt    = (fsm == S_BID) && cur_cmd.hdr.flags[F_PROFILE] &&
                   ((cur_res == RES_KMU   && bid_kmu.grant)   ||
                    (cur_res == RES_DMA   && bid_dma.grant)   ||
                    (cur_res == RES_DCR   && bid_dcr.grant)   ||
                    (cur_res == RES_EVT && bid_event.grant));
    end_evt      = (fsm == S_RETIRE) && cur_cmd.hdr.flags[F_PROFILE];
    profile_slot = cur_cmd.profile_slot;
  end

  // State mirror passes through with seqnum tracked locally.
  always_comb begin
    state_out         = state_in;
    state_out.seqnum  = seqnum_r;
  end

  `UNUSED_VAR (QID)
  `UNUSED_VAR (no_resource)

endmodule : VX_cp_engine
