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
//                  │ + embedded unpack │  + 4 bid     │  retire  │
//                  │  → cmd_in stream  │    arbiters  │   slots  │
//                  └─────────┬─────────┴───┬──────────┴────┬─────┘
//                            │              │               │
//                            ▼              ▼               ▼
//      ┌──────────────────────────────┐   ┌────────────────────────┐
//      │  host xbar: fetch[N] + cmpl  │   │  dev xbar: DMA(dev) +   │
//      │            + DMA(host)       │   │            event       │
//      └───────────────┬──────────────┘   └───────────┬────────────┘
//                      ▼ axi_host (AXI4)              ▼ axi_dev (AXI4)
//
//   Dual data plane: XRT pins each kernel AXI master to exactly one memory
//   resource, so the CP carries two — axi_host reaches host memory (the
//   command ring lives there; it is one end of every upload/download) and
//   axi_dev reaches device memory. The DMA engine straddles both: its
//   opcode picks the read-source and write-destination port.
//
//   The shared KMU launch / DCR proxy connect to gpu_if (Vortex side).
//
// AXI master TID layout (per xbar):
//   bit [ID_W-1 : ID_W-SRC_W]  = source index (xbar sets/inspects this for
//                                response routing)
//   bit [ID_W-SRC_W-1 : 0]     = sub-tag, source-defined
// ============================================================================

// Connect a VX_cp_axi_m_if `src` (master) to a `slot` of an xbar's `src`
// array (the xbar drives that array as an AXI slave).
`define CP_AXI_LINK(slot, src)            \
  assign slot.awvalid = src.awvalid;      \
  assign slot.awaddr  = src.awaddr;       \
  assign slot.awid    = src.awid;         \
  assign slot.awlen   = src.awlen;        \
  assign slot.awsize  = src.awsize;       \
  assign slot.awburst = src.awburst;      \
  assign src.awready  = slot.awready;     \
  assign slot.wvalid  = src.wvalid;       \
  assign slot.wdata   = src.wdata;        \
  assign slot.wstrb   = src.wstrb;        \
  assign slot.wlast   = src.wlast;        \
  assign src.wready   = slot.wready;      \
  assign src.bvalid   = slot.bvalid;      \
  assign src.bid      = slot.bid;         \
  assign src.bresp    = slot.bresp;       \
  assign slot.bready  = src.bready;       \
  assign slot.arvalid = src.arvalid;      \
  assign slot.araddr  = src.araddr;       \
  assign slot.arid    = src.arid;         \
  assign slot.arlen   = src.arlen;        \
  assign slot.arsize  = src.arsize;       \
  assign slot.arburst = src.arburst;      \
  assign src.arready  = slot.arready;     \
  assign src.rvalid   = slot.rvalid;      \
  assign src.rdata    = slot.rdata;       \
  assign src.rid      = slot.rid;         \
  assign src.rlast    = slot.rlast;       \
  assign src.rresp    = slot.rresp;       \
  assign slot.rready  = src.rready

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

  // Host-memory data plane (AXI4 master) — command ring + completion +
  // host side of every upload/download.
  VX_cp_axi_m_if.master             axi_host,

  // Device-memory data plane (AXI4 master) — device side of upload/download
  // and event-counter traffic.
  VX_cp_axi_m_if.master             axi_dev,

  // GPU-facing handshake (Vortex DCR + start/busy).
  VX_cp_gpu_if.master               gpu_if,

  // One-cycle pulse after any queue retires a command (drives the
  // platform irq pin). Named `irq` rather than `interrupt` because the
  // latter is a SystemVerilog reserved keyword (Verilator SYMRSVDWORD).
  output wire                       irq
);

  // Host xbar sources: fetch[NUM_QUEUES] + completion + DMA(host side).
  localparam int N_SRC_HOST = NUM_QUEUES + 2;
  // Device xbar sources: DMA(dev side) + event unit.
  localparam int N_SRC_DEV  = 2;

  // Source-slot indices.
  localparam int SLOT_CMPL     = NUM_QUEUES;       // host xbar
  localparam int SLOT_DMA_HOST = NUM_QUEUES + 1;   // host xbar
  localparam int SLOT_DMA_DEV  = 0;                // dev xbar
  localparam int SLOT_EVENT    = 1;                // dev xbar

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
  logic [63:0] seqnum_out [NUM_QUEUES];

  // Bid lines to the four arbiters.
  VX_cp_engine_bid_if bid_kmu   [NUM_QUEUES] ();
  VX_cp_engine_bid_if bid_dma   [NUM_QUEUES] ();
  VX_cp_engine_bid_if bid_dcr   [NUM_QUEUES] ();
  VX_cp_engine_bid_if bid_event [NUM_QUEUES] ();

  // Retire + profile pulses from each CPE. retire_evt is held by the
  // engine until retire_ready (from VX_cp_completion) handshakes.
  logic        retire_evt    [NUM_QUEUES];
  logic [63:0] retire_seqnum [NUM_QUEUES];
  logic        retire_ready  [NUM_QUEUES];
  logic        submit_evt    [NUM_QUEUES];
  logic        start_evt     [NUM_QUEUES];
  logic        end_evt       [NUM_QUEUES];
  logic [63:0] profile_slot  [NUM_QUEUES];

  // Per-CPE fetch → engine streaming command port.
  logic       cpe_cmd_valid [NUM_QUEUES];
  cmd_t       cpe_cmd       [NUM_QUEUES];
  logic       cpe_cmd_ready [NUM_QUEUES];

  // Shared-resource done pulses.
  logic launch_done, dma_done, dcr_done, event_done;

  // Per-CPE AXI sub-master ports (fetch is the only AXI user per CPE).
  VX_cp_axi_m_if #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .ID_W(ID_W))
                       fetch_axi [NUM_QUEUES] ();

  // ----- N CPEs (fetch + engine) -----
  generate
    for (genvar q = 0; q < NUM_QUEUES; ++q) begin : g_cpe
      VX_cp_fetch #(.QID(q)) u_fetch (
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
        .prio_in       (q_state[q].prio),
        .seqnum_out    (seqnum_out[q]),
        .cmd_in_valid  (cpe_cmd_valid[q]),
        .cmd_in        (cpe_cmd[q]),
        .cmd_in_ready  (cpe_cmd_ready[q]),
        .bid_kmu       (bid_kmu[q]),
        .bid_dma       (bid_dma[q]),
        .bid_dcr       (bid_dcr[q]),
        .bid_event     (bid_event[q]),
        // Done pulses are broadcast from the shared resource modules to
        // every CPE; only the granted CPE is in S_WAIT_DONE when the
        // matching pulse arrives.
        .kmu_done_i    (launch_done),
        .dma_done_i    (dma_done),
        .dcr_done_i    (dcr_done),
        .event_done_i  (event_done),
        .retire_evt    (retire_evt[q]),
        .retire_seqnum (retire_seqnum[q]),
        .retire_ready_i(retire_ready[q]),
        .submit_evt    (submit_evt[q]),
        .start_evt     (start_evt[q]),
        .end_evt       (end_evt[q]),
        .profile_slot  (profile_slot[q])
      );

      // Telemetry up to the regfile.
      assign q_seqnum_to_reg[q] = seqnum_out[q];
      assign q_error_to_reg [q] = 32'd0;   // per-queue error reporting reserved
    end
  endgenerate

  // ----- Four resource arbiters (round-robin) -----
  wire        kmu_valid   [NUM_QUEUES];
  wire [1:0]  kmu_prio    [NUM_QUEUES];
  cmd_t       kmu_cmd     [NUM_QUEUES];
  logic       kmu_grant   [NUM_QUEUES];

  wire        dma_valid   [NUM_QUEUES];
  wire [1:0]  dma_prio    [NUM_QUEUES];
  cmd_t       dma_cmd     [NUM_QUEUES];
  logic       dma_grant   [NUM_QUEUES];

  wire        dcr_valid   [NUM_QUEUES];
  wire [1:0]  dcr_prio    [NUM_QUEUES];
  cmd_t       dcr_cmd     [NUM_QUEUES];
  logic       dcr_grant   [NUM_QUEUES];

  wire        event_valid [NUM_QUEUES];
  wire [1:0]  event_prio  [NUM_QUEUES];
  cmd_t       event_cmd   [NUM_QUEUES];
  logic       event_grant [NUM_QUEUES];

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

      assign event_valid[q]     = bid_event[q].valid;
      assign event_prio[q]      = bid_event[q].priority_;
      assign event_cmd[q]       = bid_event[q].cmd;
      assign bid_event[q].grant = event_grant[q];
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
  VX_cp_arbiter #(.N(NUM_QUEUES)) u_arb_event (
    .clk(clk), .reset(reset),
    .bid_valid(event_valid), .bid_priority(event_prio), .bid_grant(event_grant)
  );

  // ----- Pick the granted bid's cmd for each shared resource -----
  logic any_kmu_grant, any_dma_grant, any_dcr_grant, any_event_grant;
  cmd_t granted_kmu_cmd, granted_dma_cmd, granted_dcr_cmd, granted_event_cmd;
  always_comb begin
    any_kmu_grant = 1'b0; granted_kmu_cmd = '0;
    any_dma_grant = 1'b0; granted_dma_cmd = '0;
    any_dcr_grant = 1'b0; granted_dcr_cmd = '0;
    any_event_grant = 1'b0; granted_event_cmd = '0;
    for (int i = 0; i < NUM_QUEUES; ++i) begin
      if (kmu_grant[i])   begin any_kmu_grant   = 1'b1; granted_kmu_cmd   = kmu_cmd[i];   end
      if (dma_grant[i])   begin any_dma_grant   = 1'b1; granted_dma_cmd   = dma_cmd[i];   end
      if (dcr_grant[i])   begin any_dcr_grant   = 1'b1; granted_dcr_cmd   = dcr_cmd[i];   end
      if (event_grant[i]) begin any_event_grant = 1'b1; granted_event_cmd = event_cmd[i]; end
    end
  end

  `UNUSED_VAR (granted_kmu_cmd)

  // ----- Shared KMU launch (consumes the kmu bid grant) -----
  VX_cp_launch u_launch (
    .clk      (clk),
    .reset    (reset),
    .grant    (any_kmu_grant),
    .start    (gpu_if.start),
    .gpu_busy (gpu_if.busy),
    .done     (launch_done)
  );

  // ----- Shared DCR proxy -----
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

  // ----- DMA (straddles host + dev xbars) -----
  VX_cp_axi_m_if #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .ID_W(ID_W)) dma_host_axi ();
  VX_cp_axi_m_if #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .ID_W(ID_W)) dma_dev_axi  ();
  VX_cp_axi_m_if #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .ID_W(ID_W)) cmpl_axi     ();
  VX_cp_axi_m_if #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .ID_W(ID_W)) event_axi    ();

  VX_cp_dma u_dma (
    .clk      (clk),
    .reset    (reset),
    .grant    (any_dma_grant),
    .cmd      (granted_dma_cmd),
    .done     (dma_done),
    .axi_host (dma_host_axi),
    .axi_dev  (dma_dev_axi)
  );

  // ----- Event unit (CMD_EVENT_SIGNAL / CMD_EVENT_WAIT) -----
  VX_cp_event_unit u_event (
    .clk   (clk),
    .reset (reset),
    .grant (any_event_grant),
    .cmd   (granted_event_cmd),
    .done  (event_done),
    .axi_m (event_axi)
  );

  // ----- Completion writeback (host memory) -----
  wire [63:0] cmpl_addr_arr [NUM_QUEUES];
  generate
    for (genvar q = 0; q < NUM_QUEUES; ++q) begin : g_cmpl_addr
      assign cmpl_addr_arr[q] = q_state[q].cmpl_addr;
    end
  endgenerate

  VX_cp_completion #(
    .NUM_QUEUES (NUM_QUEUES)
  ) u_completion (
    .clk           (clk),
    .reset         (reset),
    .retire_evt    (retire_evt),
    .retire_seqnum (retire_seqnum),
    .cmpl_addr     (cmpl_addr_arr),
    .retire_ready  (retire_ready),
    .axi_m         (cmpl_axi)
  );

  // ============================================================================
  // Host xbar — fetch[N] + completion + DMA(host) → axi_host.
  // ============================================================================
  VX_cp_axi_m_if #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .ID_W(ID_W))
                       xbar_host_src [N_SRC_HOST] ();

  generate
    for (genvar q = 0; q < NUM_QUEUES; ++q) begin : g_host_fetch
      `CP_AXI_LINK(xbar_host_src[q], fetch_axi[q]);
    end
  endgenerate

  `CP_AXI_LINK(xbar_host_src[SLOT_CMPL],     cmpl_axi);
  `CP_AXI_LINK(xbar_host_src[SLOT_DMA_HOST], dma_host_axi);

  // Register slice that breaks the long, routing-dominated path from the CP
  // masters to the far-side host-memory AXI so the kernel clock can close.
  VX_cp_axi_m_if #(
    .ADDR_W (ADDR_W),
    .DATA_W (DATA_W),
    .ID_W   (ID_W)
  ) axi_host_pre ();

  VX_cp_axi_xbar #(
    .N_SOURCES (N_SRC_HOST),
    .ADDR_W    (ADDR_W),
    .DATA_W    (DATA_W),
    .ID_W      (ID_W)
  ) u_xbar_host (
    .clk   (clk),
    .reset (reset),
    .src   (xbar_host_src),
    .axi_m (axi_host_pre)
  );

  VX_cp_axi_slice #(
    .ADDR_W (ADDR_W),
    .DATA_W (DATA_W),
    .ID_W   (ID_W)
  ) u_slice_host (
    .clk   (clk),
    .reset (reset),
    .s     (axi_host_pre),
    .m     (axi_host)
  );

  // ============================================================================
  // Device xbar — DMA(dev) + event → axi_dev.
  // ============================================================================
  VX_cp_axi_m_if #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .ID_W(ID_W))
                       xbar_dev_src [N_SRC_DEV] ();

  `CP_AXI_LINK(xbar_dev_src[SLOT_DMA_DEV], dma_dev_axi);
  `CP_AXI_LINK(xbar_dev_src[SLOT_EVENT],   event_axi);

  // Register slice on the path to device memory (memory subsystem).
  VX_cp_axi_m_if #(
    .ADDR_W (ADDR_W),
    .DATA_W (DATA_W),
    .ID_W   (ID_W)
  ) axi_dev_pre ();

  VX_cp_axi_xbar #(
    .N_SOURCES (N_SRC_DEV),
    .ADDR_W    (ADDR_W),
    .DATA_W    (DATA_W),
    .ID_W      (ID_W)
  ) u_xbar_dev (
    .clk   (clk),
    .reset (reset),
    .src   (xbar_dev_src),
    .axi_m (axi_dev_pre)
  );

  VX_cp_axi_slice #(
    .ADDR_W (ADDR_W),
    .DATA_W (DATA_W),
    .ID_W   (ID_W)
  ) u_slice_dev (
    .clk   (clk),
    .reset (reset),
    .s     (axi_dev_pre),
    .m     (axi_dev)
  );

  // ----- Aggregated status -----
  // Busy if any CPE has a command in flight or any shared resource is active.
  always_comb begin
    cp_busy = 1'b0;
    cp_error = 1'b0;
    for (int i = 0; i < NUM_QUEUES; ++i) begin
      if (cpe_cmd_valid[i]) cp_busy = 1'b1;
    end
    if (any_kmu_grant || any_dma_grant || any_dcr_grant ||
        any_event_grant) cp_busy = 1'b1;
  end

  // Reset pulse from regfile (Q_CONTROL.reset / CP_CTRL.reset_all) is
  // not propagated to CPEs as a separate signal. To stop a queue, the
  // host clears Q_CONTROL.enable and the fetch parks in IDLE while
  // in-flight commands drain naturally.
  generate
    for (genvar q = 0; q < NUM_QUEUES; ++q) begin : g_unused_reset
      `UNUSED_VAR (q_reset_pulse[q])
    end
  endgenerate

  // ----- IRQ: one-cycle pulse after any queue retires a command.
  // No host-visible ack/ISR register yet (runtime polls Q_SEQNUM);
  // this drives the platform irq pin for future interrupt-driven launch-wait. -----
  reg irq_r;
  always_ff @(posedge clk) begin
    if (reset) begin
      irq_r <= 1'b0;
    end else begin
      irq_r <= 1'b0;
      for (int q = 0; q < NUM_QUEUES; ++q) begin
        if (retire_evt[q])
          irq_r <= 1'b1;
      end
    end
  end
  assign irq = irq_r;

  // Profiling pulses fired by each engine are not routed externally yet;
  // suppress unused-signal warnings here.
  generate
    for (genvar q = 0; q < NUM_QUEUES; ++q) begin : g_unused_prof
      `UNUSED_VAR (submit_evt[q])
      `UNUSED_VAR (start_evt[q])
      `UNUSED_VAR (end_evt[q])
      `UNUSED_VAR (profile_slot[q])
    end
  endgenerate

  `UNUSED_PARAM (ADDR_W)
  `UNUSED_PARAM (DATA_W)

endmodule : VX_cp_core

`undef CP_AXI_LINK
