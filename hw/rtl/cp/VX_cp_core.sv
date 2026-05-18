// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_cp_core — top-level Command Processor wrapper.
//
// Integrates everything in rtl/cp/ into one block the AFU shim can
// instantiate alongside Vortex:
//
//                         ┌──────────────────────────┐
//   AXI4-Lite host ──────►│  VX_cp_axil_regfile      │── per-queue
//   (control plane)       │                          │   cpe_state
//                         └──┬───────────────────────┘
//                            │ q_state[NUM_QUEUES]
//                  ┌─────────┴────────┬──────────────┬──────────┐
//                  │ fetch[NUM_QUEUES] │ engine[N]    │ cmpl     │
//                  │ + embedded unpack │  + 3 bid     │  retire  │
//                  │  → cmd_in stream  │    arbiters  │   slots  │
//                  └─────────┬─────────┴───┬──────────┴────┬─────┘
//                            │              │               │
//                            ▼              ▼               ▼
//                       ┌────────────────────────────────────────┐
//                       │           VX_cp_axi_xbar                │
//                       │   fetch[N] + DMA + completion → 1      │
//                       └────────────────────┬───────────────────┘
//                                            │
//                                            ▼  axi_m (host AXI4)
//
//   The shared KMU launch / DCR proxy connect to gpu_if (Vortex side).
//   Event unit + profiling are reserved for a follow-up commit; the
//   engine retires CMD_EVENT_* / profile-flagged commands as NOPs
//   today so omitting those modules is correctness-safe.
//
// AXI master TID layout (parent §15):
//   bit [ID_W-1 : ID_W-2]  = source index (xbar sets/inspects this 2-bit
//                            field for the 3-source v1 topology)
//   bit [ID_W-3 : 0]       = sub-tag, source-defined
// ============================================================================

module VX_cp_core
  import VX_cp_pkg::*;
#(
  parameter int NUM_QUEUES = VX_CP_NUM_QUEUES_C,
  parameter int ADDR_W     = 64,
  parameter int DATA_W     = 512,
  parameter int ID_W       = VX_CP_AXI_TID_WIDTH_C,
  parameter int AXIL_AW    = 16
)(
  input  wire                       clk,
  input  wire                       reset,

  // Host control plane (AXI4-Lite slave).
  VX_cp_axil_s_if.slave             axil_s,

  // Host data plane (AXI4 master).
  VX_cp_axi_m_if.master             axi_m,

  // GPU-facing handshake (Vortex DCR + start/busy).
  VX_cp_gpu_if.master               gpu_if,

  // Tied to 0 in v1; Phase 6 wires it to a real interrupt source.
  output wire                       interrupt
);

  localparam int N_SOURCES = NUM_QUEUES + 2;   // fetch[N] + DMA + cmpl

  // ----- Regfile-owned per-queue programmable state -----
  cpe_state_t q_state          [NUM_QUEUES];
  logic       q_reset_pulse    [NUM_QUEUES];

  // Telemetry inputs from CPEs to the regfile.
  logic [63:0] q_head_to_reg   [NUM_QUEUES];
  logic [63:0] q_seqnum_to_reg [NUM_QUEUES];
  logic [31:0] q_error_to_reg  [NUM_QUEUES];

  // Aggregated CP status seen by the host through CP_STATUS.
  logic cp_busy;
  logic cp_error;

  wire [`VX_DCR_DATA_BITS-1:0] dcr_last_rsp_data;

  VX_cp_axil_regfile #(
    .NUM_QUEUES (NUM_QUEUES),
    .ADDR_W     (AXIL_AW)
  ) u_regfile (
    .clk            (clk),
    .reset          (reset),
    .axil_s         (axil_s),
    .cp_busy        (cp_busy),
    .cp_error       (cp_error),
    .q_head         (q_head_to_reg),
    .q_seqnum       (q_seqnum_to_reg),
    .q_error        (q_error_to_reg),
    .last_dcr_rsp   (dcr_last_rsp_data),
    .q_state        (q_state),
    .q_reset_pulse  (q_reset_pulse)
  );

  // ----- Per-CPE wires -----
  cpe_state_t state_out  [NUM_QUEUES];

  // Bid lines to the three arbiters.
  VX_cp_engine_bid_if bid_kmu [NUM_QUEUES] ();
  VX_cp_engine_bid_if bid_dma [NUM_QUEUES] ();
  VX_cp_engine_bid_if bid_dcr [NUM_QUEUES] ();

  // Retire + profile pulses from each CPE.
  logic        retire_evt    [NUM_QUEUES];
  logic [63:0] retire_seqnum [NUM_QUEUES];
  logic        submit_evt    [NUM_QUEUES];
  logic        start_evt     [NUM_QUEUES];
  logic        end_evt       [NUM_QUEUES];
  logic [63:0] profile_slot  [NUM_QUEUES];

  // Per-CPE fetch → engine streaming command port.
  logic       cpe_cmd_valid [NUM_QUEUES];
  cmd_t       cpe_cmd       [NUM_QUEUES];
  logic       cpe_cmd_ready [NUM_QUEUES];

  // Per-CPE AXI sub-master ports (fetch is the only AXI user per CPE).
  VX_cp_axi_m_if #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .ID_W(ID_W))
                       fetch_axi [NUM_QUEUES] ();

  // ----- N CPEs (fetch + engine) -----
  generate
    for (genvar q = 0; q < NUM_QUEUES; ++q) begin : g_cpe
      // Per-CPE TID prefix = source index q in the high $clog2(N_SOURCES) bits.
      localparam logic [ID_W-1:0] FETCH_TID_PREFIX =
        ID_W'(q) << ($clog2(N_SOURCES) > 0 ? (ID_W - $clog2(N_SOURCES)) : 0);

      VX_cp_fetch #(.QID(q), .TID_PREFIX(FETCH_TID_PREFIX)) u_fetch (
        .clk           (clk),
        .reset         (reset),
        .state_in      (q_state[q]),
        .head_out      (q_head_to_reg[q]),
        .cmd_out_valid (cpe_cmd_valid[q]),
        .cmd_out       (cpe_cmd[q]),
        .cmd_out_ready (cpe_cmd_ready[q]),
        .axi_m         (fetch_axi[q])
      );

      VX_cp_engine #(.QID(q)) u_engine (
        .clk           (clk),
        .reset         (reset),
        .state_in      (q_state[q]),
        .state_out     (state_out[q]),
        .cmd_in_valid  (cpe_cmd_valid[q]),
        .cmd_in        (cpe_cmd[q]),
        .cmd_in_ready  (cpe_cmd_ready[q]),
        .bid_kmu       (bid_kmu[q]),
        .bid_dma       (bid_dma[q]),
        .bid_dcr       (bid_dcr[q]),
        // Real done pulses from the shared resource modules. Broadcast
        // to every CPE: the bid arbiter only grants one CPE at a time
        // per resource, and the resource processes one command at a
        // time, so only the granted CPE will be in S_WAIT_DONE when the
        // pulse arrives — non-granted CPEs ignore it (they're in
        // S_IDLE / S_DECODE / S_BID).
        .kmu_done_i    (launch_done),
        .dma_done_i    (dma_done),
        .dcr_done_i    (dcr_done),
        .retire_evt    (retire_evt[q]),
        .retire_seqnum (retire_seqnum[q]),
        .submit_evt    (submit_evt[q]),
        .start_evt     (start_evt[q]),
        .end_evt       (end_evt[q]),
        .profile_slot  (profile_slot[q])
      );

      // Telemetry up to the regfile.
      assign q_seqnum_to_reg[q] = state_out[q].seqnum;
      assign q_error_to_reg [q] = 32'd0;   // no per-queue error reporting in v1
    end
  endgenerate

  // ----- Three resource arbiters (round-robin) -----
  wire        kmu_valid [NUM_QUEUES];
  wire [1:0]  kmu_prio  [NUM_QUEUES];
  cmd_t       kmu_cmd   [NUM_QUEUES];
  logic       kmu_grant [NUM_QUEUES];

  wire        dma_valid [NUM_QUEUES];
  wire [1:0]  dma_prio  [NUM_QUEUES];
  cmd_t       dma_cmd   [NUM_QUEUES];
  logic       dma_grant [NUM_QUEUES];

  wire        dcr_valid [NUM_QUEUES];
  wire [1:0]  dcr_prio  [NUM_QUEUES];
  cmd_t       dcr_cmd   [NUM_QUEUES];
  logic       dcr_grant [NUM_QUEUES];

  generate
    for (genvar q = 0; q < NUM_QUEUES; ++q) begin : g_unpack_bids
      assign kmu_valid[q]     = bid_kmu[q].valid;
      assign kmu_prio[q]      = bid_kmu[q].priority_;
      assign kmu_cmd[q]       = bid_kmu[q].cmd;
      assign bid_kmu[q].grant = kmu_grant[q];

      assign dma_valid[q]     = bid_dma[q].valid;
      assign dma_prio[q]      = bid_dma[q].priority_;
      assign dma_cmd[q]       = bid_dma[q].cmd;
      assign bid_dma[q].grant = dma_grant[q];

      assign dcr_valid[q]     = bid_dcr[q].valid;
      assign dcr_prio[q]      = bid_dcr[q].priority_;
      assign dcr_cmd[q]       = bid_dcr[q].cmd;
      assign bid_dcr[q].grant = dcr_grant[q];
    end
  endgenerate

  VX_cp_arbiter #(.N(NUM_QUEUES)) u_arb_kmu (
    .clk(clk), .reset(reset),
    .bid_valid(kmu_valid), .bid_priority(kmu_prio), .bid_grant(kmu_grant)
  );
  VX_cp_arbiter #(.N(NUM_QUEUES)) u_arb_dma (
    .clk(clk), .reset(reset),
    .bid_valid(dma_valid), .bid_priority(dma_prio), .bid_grant(dma_grant)
  );
  VX_cp_arbiter #(.N(NUM_QUEUES)) u_arb_dcr (
    .clk(clk), .reset(reset),
    .bid_valid(dcr_valid), .bid_priority(dcr_prio), .bid_grant(dcr_grant)
  );

  // ----- Pick the granted bid's cmd for each shared resource -----
  logic any_kmu_grant, any_dma_grant, any_dcr_grant;
  cmd_t granted_kmu_cmd, granted_dma_cmd, granted_dcr_cmd;
  always_comb begin
    any_kmu_grant = 1'b0; granted_kmu_cmd = '0;
    any_dma_grant = 1'b0; granted_dma_cmd = '0;
    any_dcr_grant = 1'b0; granted_dcr_cmd = '0;
    for (int i = 0; i < NUM_QUEUES; ++i) begin
      if (kmu_grant[i]) begin any_kmu_grant = 1'b1; granted_kmu_cmd = kmu_cmd[i]; end
      if (dma_grant[i]) begin any_dma_grant = 1'b1; granted_dma_cmd = dma_cmd[i]; end
      if (dcr_grant[i]) begin any_dcr_grant = 1'b1; granted_dcr_cmd = dcr_cmd[i]; end
    end
  end

  `UNUSED_VAR (granted_kmu_cmd)

  // ----- Shared KMU launch (consumes the kmu bid grant) -----
  logic launch_done;
  VX_cp_launch u_launch (
    .clk      (clk),
    .reset    (reset),
    .grant    (any_kmu_grant),
    .start    (gpu_if.start),
    .gpu_busy (gpu_if.busy),
    .done     (launch_done)
  );

  // ----- Shared DCR proxy -----
  logic dcr_done;
  VX_cp_dcr_proxy u_dcr (
    .clk           (clk),
    .reset         (reset),
    .grant         (any_dcr_grant),
    .cmd           (granted_dcr_cmd),
    .done          (dcr_done),
    .last_rsp_data (dcr_last_rsp_data),
    .dcr_req_valid (gpu_if.dcr_req_valid),
    .dcr_req_rw    (gpu_if.dcr_req_rw),
    .dcr_req_addr  (gpu_if.dcr_req_addr),
    .dcr_req_data  (gpu_if.dcr_req_data),
    .dcr_rsp_valid (gpu_if.dcr_rsp_valid),
    .dcr_rsp_data  (gpu_if.dcr_rsp_data)
  );
  `UNUSED_VAR (gpu_if.dcr_req_ready)

  // ----- DMA (AXI source via xbar) -----
  localparam logic [ID_W-1:0] DMA_TID_PREFIX =
    ID_W'(NUM_QUEUES) << ($clog2(N_SOURCES) > 0 ? (ID_W - $clog2(N_SOURCES)) : 0);
  localparam logic [ID_W-1:0] CMPL_TID_PREFIX =
    ID_W'(NUM_QUEUES + 1) << ($clog2(N_SOURCES) > 0 ? (ID_W - $clog2(N_SOURCES)) : 0);

  VX_cp_axi_m_if #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .ID_W(ID_W)) dma_axi  ();
  VX_cp_axi_m_if #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .ID_W(ID_W)) cmpl_axi ();

  logic dma_done;
  VX_cp_dma #(.TID_PREFIX(DMA_TID_PREFIX)) u_dma (
    .clk   (clk),
    .reset (reset),
    .grant (any_dma_grant),
    .cmd   (granted_dma_cmd),
    .done  (dma_done),
    .axi_m (dma_axi)
  );

  // ----- Completion writeback -----
  wire [63:0] cmpl_addr_arr [NUM_QUEUES];
  generate
    for (genvar q = 0; q < NUM_QUEUES; ++q) begin : g_cmpl_addr
      assign cmpl_addr_arr[q] = q_state[q].cmpl_addr;
    end
  endgenerate

  VX_cp_completion #(
    .NUM_QUEUES (NUM_QUEUES),
    .TID_PREFIX (CMPL_TID_PREFIX)
  ) u_completion (
    .clk           (clk),
    .reset         (reset),
    .retire_evt    (retire_evt),
    .retire_seqnum (retire_seqnum),
    .cmpl_addr     (cmpl_addr_arr),
    .axi_m         (cmpl_axi)
  );

  // ----- AXI xbar: fan fetch[N] + DMA + completion → axi_m -----
  VX_cp_axi_m_if #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .ID_W(ID_W))
                       xbar_src [N_SOURCES] ();

  generate
    for (genvar q = 0; q < NUM_QUEUES; ++q) begin : g_xbar_fetch
      // Pass fetch's AXI through to the xbar's source slot q.
      assign xbar_src[q].awvalid = fetch_axi[q].awvalid;
      assign xbar_src[q].awaddr  = fetch_axi[q].awaddr;
      assign xbar_src[q].awid    = fetch_axi[q].awid;
      assign xbar_src[q].awlen   = fetch_axi[q].awlen;
      assign xbar_src[q].awsize  = fetch_axi[q].awsize;
      assign xbar_src[q].awburst = fetch_axi[q].awburst;
      assign fetch_axi[q].awready = xbar_src[q].awready;
      assign xbar_src[q].wvalid  = fetch_axi[q].wvalid;
      assign xbar_src[q].wdata   = fetch_axi[q].wdata;
      assign xbar_src[q].wstrb   = fetch_axi[q].wstrb;
      assign xbar_src[q].wlast   = fetch_axi[q].wlast;
      assign fetch_axi[q].wready = xbar_src[q].wready;
      assign fetch_axi[q].bvalid = xbar_src[q].bvalid;
      assign fetch_axi[q].bid    = xbar_src[q].bid;
      assign fetch_axi[q].bresp  = xbar_src[q].bresp;
      assign xbar_src[q].bready  = fetch_axi[q].bready;
      assign xbar_src[q].arvalid = fetch_axi[q].arvalid;
      assign xbar_src[q].araddr  = fetch_axi[q].araddr;
      assign xbar_src[q].arid    = fetch_axi[q].arid;
      assign xbar_src[q].arlen   = fetch_axi[q].arlen;
      assign xbar_src[q].arsize  = fetch_axi[q].arsize;
      assign xbar_src[q].arburst = fetch_axi[q].arburst;
      assign fetch_axi[q].arready = xbar_src[q].arready;
      assign fetch_axi[q].rvalid = xbar_src[q].rvalid;
      assign fetch_axi[q].rdata  = xbar_src[q].rdata;
      assign fetch_axi[q].rid    = xbar_src[q].rid;
      assign fetch_axi[q].rlast  = xbar_src[q].rlast;
      assign fetch_axi[q].rresp  = xbar_src[q].rresp;
      assign xbar_src[q].rready  = fetch_axi[q].rready;
    end
  endgenerate

  // Wire DMA into source slot NUM_QUEUES.
  assign xbar_src[NUM_QUEUES].awvalid = dma_axi.awvalid;
  assign xbar_src[NUM_QUEUES].awaddr  = dma_axi.awaddr;
  assign xbar_src[NUM_QUEUES].awid    = dma_axi.awid;
  assign xbar_src[NUM_QUEUES].awlen   = dma_axi.awlen;
  assign xbar_src[NUM_QUEUES].awsize  = dma_axi.awsize;
  assign xbar_src[NUM_QUEUES].awburst = dma_axi.awburst;
  assign dma_axi.awready = xbar_src[NUM_QUEUES].awready;
  assign xbar_src[NUM_QUEUES].wvalid  = dma_axi.wvalid;
  assign xbar_src[NUM_QUEUES].wdata   = dma_axi.wdata;
  assign xbar_src[NUM_QUEUES].wstrb   = dma_axi.wstrb;
  assign xbar_src[NUM_QUEUES].wlast   = dma_axi.wlast;
  assign dma_axi.wready = xbar_src[NUM_QUEUES].wready;
  assign dma_axi.bvalid = xbar_src[NUM_QUEUES].bvalid;
  assign dma_axi.bid    = xbar_src[NUM_QUEUES].bid;
  assign dma_axi.bresp  = xbar_src[NUM_QUEUES].bresp;
  assign xbar_src[NUM_QUEUES].bready = dma_axi.bready;
  assign xbar_src[NUM_QUEUES].arvalid = dma_axi.arvalid;
  assign xbar_src[NUM_QUEUES].araddr  = dma_axi.araddr;
  assign xbar_src[NUM_QUEUES].arid    = dma_axi.arid;
  assign xbar_src[NUM_QUEUES].arlen   = dma_axi.arlen;
  assign xbar_src[NUM_QUEUES].arsize  = dma_axi.arsize;
  assign xbar_src[NUM_QUEUES].arburst = dma_axi.arburst;
  assign dma_axi.arready = xbar_src[NUM_QUEUES].arready;
  assign dma_axi.rvalid = xbar_src[NUM_QUEUES].rvalid;
  assign dma_axi.rdata  = xbar_src[NUM_QUEUES].rdata;
  assign dma_axi.rid    = xbar_src[NUM_QUEUES].rid;
  assign dma_axi.rlast  = xbar_src[NUM_QUEUES].rlast;
  assign dma_axi.rresp  = xbar_src[NUM_QUEUES].rresp;
  assign xbar_src[NUM_QUEUES].rready = dma_axi.rready;

  // Wire completion into source slot NUM_QUEUES+1.
  assign xbar_src[NUM_QUEUES+1].awvalid = cmpl_axi.awvalid;
  assign xbar_src[NUM_QUEUES+1].awaddr  = cmpl_axi.awaddr;
  assign xbar_src[NUM_QUEUES+1].awid    = cmpl_axi.awid;
  assign xbar_src[NUM_QUEUES+1].awlen   = cmpl_axi.awlen;
  assign xbar_src[NUM_QUEUES+1].awsize  = cmpl_axi.awsize;
  assign xbar_src[NUM_QUEUES+1].awburst = cmpl_axi.awburst;
  assign cmpl_axi.awready = xbar_src[NUM_QUEUES+1].awready;
  assign xbar_src[NUM_QUEUES+1].wvalid  = cmpl_axi.wvalid;
  assign xbar_src[NUM_QUEUES+1].wdata   = cmpl_axi.wdata;
  assign xbar_src[NUM_QUEUES+1].wstrb   = cmpl_axi.wstrb;
  assign xbar_src[NUM_QUEUES+1].wlast   = cmpl_axi.wlast;
  assign cmpl_axi.wready = xbar_src[NUM_QUEUES+1].wready;
  assign cmpl_axi.bvalid = xbar_src[NUM_QUEUES+1].bvalid;
  assign cmpl_axi.bid    = xbar_src[NUM_QUEUES+1].bid;
  assign cmpl_axi.bresp  = xbar_src[NUM_QUEUES+1].bresp;
  assign xbar_src[NUM_QUEUES+1].bready = cmpl_axi.bready;
  assign xbar_src[NUM_QUEUES+1].arvalid = cmpl_axi.arvalid;
  assign xbar_src[NUM_QUEUES+1].araddr  = cmpl_axi.araddr;
  assign xbar_src[NUM_QUEUES+1].arid    = cmpl_axi.arid;
  assign xbar_src[NUM_QUEUES+1].arlen   = cmpl_axi.arlen;
  assign xbar_src[NUM_QUEUES+1].arsize  = cmpl_axi.arsize;
  assign xbar_src[NUM_QUEUES+1].arburst = cmpl_axi.arburst;
  assign cmpl_axi.arready = xbar_src[NUM_QUEUES+1].arready;
  assign cmpl_axi.rvalid = xbar_src[NUM_QUEUES+1].rvalid;
  assign cmpl_axi.rdata  = xbar_src[NUM_QUEUES+1].rdata;
  assign cmpl_axi.rid    = xbar_src[NUM_QUEUES+1].rid;
  assign cmpl_axi.rlast  = xbar_src[NUM_QUEUES+1].rlast;
  assign cmpl_axi.rresp  = xbar_src[NUM_QUEUES+1].rresp;
  assign xbar_src[NUM_QUEUES+1].rready = cmpl_axi.rready;

  VX_cp_axi_xbar #(
    .N_SOURCES (N_SOURCES),
    .ADDR_W    (ADDR_W),
    .DATA_W    (DATA_W),
    .ID_W      (ID_W)
  ) u_xbar (
    .clk   (clk),
    .reset (reset),
    .src   (xbar_src),
    .axi_m (axi_m)
  );

  // ----- Aggregated status -----
  // Busy if any CPE is not in idle (approximated: any fetch/engine has
  // not yet drained, i.e. arvalid pending or cmd_in_valid asserted) OR
  // any shared resource is active.
  always_comb begin
    cp_busy = 1'b0;
    cp_error = 1'b0;
    for (int i = 0; i < NUM_QUEUES; ++i) begin
      if (cpe_cmd_valid[i]) cp_busy = 1'b1;
    end
    if (any_kmu_grant || any_dma_grant || any_dcr_grant) cp_busy = 1'b1;
  end

  // Reset pulse from regfile (Q_CONTROL.reset / CP_CTRL.reset_all) — v1
  // does NOT propagate this to CPEs as a separate signal. The host can
  // disable the queue (Q_CONTROL.enable=0) and the fetch will park in
  // IDLE; in-flight commands drain naturally. Wiring a hard-stop is a
  // Phase 4 task.
  generate
    for (genvar q = 0; q < NUM_QUEUES; ++q) begin : g_unused_reset
      `UNUSED_VAR (q_reset_pulse[q])
    end
  endgenerate

  // ----- Interrupt: tied low in v1 -----
  assign interrupt = 1'b0;

  // Unused profiling pulses (event_unit + profiling helpers are deferred
  // — engine still fires the pulses, we just don't route them anywhere).
  generate
    for (genvar q = 0; q < NUM_QUEUES; ++q) begin : g_unused_prof
      `UNUSED_VAR (submit_evt[q])
      `UNUSED_VAR (start_evt[q])
      `UNUSED_VAR (end_evt[q])
      `UNUSED_VAR (profile_slot[q])
      `UNUSED_VAR (state_out[q])
    end
  endgenerate

  `UNUSED_PARAM (ADDR_W)
  `UNUSED_PARAM (DATA_W)

endmodule : VX_cp_core
