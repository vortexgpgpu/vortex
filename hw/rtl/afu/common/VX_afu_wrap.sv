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
//
// Reference: https://www.xilinx.com/developer/articles/porting-rtl-designs-to-vitis-rtl-kernels.html

`include "vortex_afu.vh"

// ============================================================================
// XRT AFU shim. The Command Processor is the sole command path.
//
// AXI-Lite address space:
//   0x0000..0x0FFF — VX_afu_ctrl: a minimal ap_ctrl stub (0x00) plus the
//                    SCOPE bit-serial register pair (0x28/0x2C).
//   0x1000..0x1FFF — Command Processor regfile, mapped to CP's native
//                    0x000..0xFFF address space (CP sees addr - 0x1000).
//                    The bit-12 split keeps CP_CTRL at CP-offset 0x000
//                    reachable without colliding with the ap_ctrl stub
//                    register at host-offset 0x000.
//
// Data plane:
//   * Vortex memory banks 0..N-1 ride the platform AXI4 master ports.
//   * VX_cp_core has its own axi_m. Bank 0 is shared via VX_mm_axi_arb —
//     the arbiter holds a sticky owner per channel until the response
//     completes, so CP and Vortex can interleave without deadlock.
//
// Launch / DCR: driven solely by the CP through cp_gpu_if (start + DCR).
// ============================================================================

module VX_afu_wrap import VX_gpu_pkg::*; #(
    parameter C_S_AXI_CTRL_ADDR_WIDTH = 16,
    parameter C_S_AXI_CTRL_DATA_WIDTH = 32,
    parameter C_M_AXI_MEM_ID_WIDTH    = `PLATFORM_MEMORY_ID_WIDTH,
    parameter C_M_AXI_MEM_DATA_WIDTH  = `VX_CFG_PLATFORM_MEMORY_DATA_SIZE * 8,
    parameter C_M_AXI_MEM_ADDR_WIDTH  = 64,
`ifdef PLATFORM_MERGED_MEMORY_INTERFACE
    parameter C_M_AXI_MEM_NUM_BANKS   = 1
`else
    parameter C_M_AXI_MEM_NUM_BANKS   = `VX_CFG_PLATFORM_MEMORY_NUM_BANKS
`endif
) (
    // System signals
    input wire clk,
    input wire reset,

    // AXI4 master interface
`ifdef PLATFORM_MERGED_MEMORY_INTERFACE
    `MP_REPEAT (1, GEN_AXI_MEM, MP_COMMA),
`else
    `MP_REPEAT (`VX_CFG_PLATFORM_MEMORY_NUM_BANKS, GEN_AXI_MEM, MP_COMMA),
`endif
    // AXI4 host-memory master interface (CP command ring + host side of DMA)
    `GEN_AXI_HOST,
    // AXI4-Lite slave interface
    input  wire                                 s_axi_ctrl_awvalid,
    output wire                                 s_axi_ctrl_awready,
    input  wire [C_S_AXI_CTRL_ADDR_WIDTH-1:0]   s_axi_ctrl_awaddr,

    input  wire                                 s_axi_ctrl_wvalid,
    output wire                                 s_axi_ctrl_wready,
    input  wire [C_S_AXI_CTRL_DATA_WIDTH-1:0]   s_axi_ctrl_wdata,
    input  wire [C_S_AXI_CTRL_DATA_WIDTH/8-1:0] s_axi_ctrl_wstrb,

    input  wire                                 s_axi_ctrl_arvalid,
    output wire                                 s_axi_ctrl_arready,
    input  wire [C_S_AXI_CTRL_ADDR_WIDTH-1:0]   s_axi_ctrl_araddr,

    output wire                                 s_axi_ctrl_rvalid,
    input  wire                                 s_axi_ctrl_rready,
    output wire [C_S_AXI_CTRL_DATA_WIDTH-1:0]   s_axi_ctrl_rdata,
    output wire [1:0]                           s_axi_ctrl_rresp,

    output wire                                 s_axi_ctrl_bvalid,
    input  wire                                 s_axi_ctrl_bready,
    output wire [1:0]                           s_axi_ctrl_bresp,

    output wire                                 interrupt
);
    localparam M_AXI_MEM_ADDR_WIDTH = `VX_CFG_PLATFORM_MEMORY_ADDR_WIDTH;

    wire                                 m_axi_mem_awvalid_a [C_M_AXI_MEM_NUM_BANKS];
    wire                                 m_axi_mem_awready_a [C_M_AXI_MEM_NUM_BANKS];
    wire [C_M_AXI_MEM_ADDR_WIDTH-1:0]    m_axi_mem_awaddr_a [C_M_AXI_MEM_NUM_BANKS];
    wire [C_M_AXI_MEM_ID_WIDTH-1:0]      m_axi_mem_awid_a [C_M_AXI_MEM_NUM_BANKS];
    wire [7:0]                           m_axi_mem_awlen_a [C_M_AXI_MEM_NUM_BANKS];

    wire                                 m_axi_mem_wvalid_a [C_M_AXI_MEM_NUM_BANKS];
    wire                                 m_axi_mem_wready_a [C_M_AXI_MEM_NUM_BANKS];
    wire [C_M_AXI_MEM_DATA_WIDTH-1:0]    m_axi_mem_wdata_a [C_M_AXI_MEM_NUM_BANKS];
    wire [C_M_AXI_MEM_DATA_WIDTH/8-1:0]  m_axi_mem_wstrb_a [C_M_AXI_MEM_NUM_BANKS];
    wire                                 m_axi_mem_wlast_a [C_M_AXI_MEM_NUM_BANKS];

    wire                                 m_axi_mem_bvalid_a [C_M_AXI_MEM_NUM_BANKS];
    wire                                 m_axi_mem_bready_a [C_M_AXI_MEM_NUM_BANKS];
    wire [C_M_AXI_MEM_ID_WIDTH-1:0]      m_axi_mem_bid_a [C_M_AXI_MEM_NUM_BANKS];
    wire [1:0]                           m_axi_mem_bresp_a [C_M_AXI_MEM_NUM_BANKS];

    wire                                 m_axi_mem_arvalid_a [C_M_AXI_MEM_NUM_BANKS];
    wire                                 m_axi_mem_arready_a [C_M_AXI_MEM_NUM_BANKS];
    wire [C_M_AXI_MEM_ADDR_WIDTH-1:0]    m_axi_mem_araddr_a [C_M_AXI_MEM_NUM_BANKS];
    wire [C_M_AXI_MEM_ID_WIDTH-1:0]      m_axi_mem_arid_a [C_M_AXI_MEM_NUM_BANKS];
    wire [7:0]                           m_axi_mem_arlen_a [C_M_AXI_MEM_NUM_BANKS];

    wire                                 m_axi_mem_rvalid_a [C_M_AXI_MEM_NUM_BANKS];
    wire                                 m_axi_mem_rready_a [C_M_AXI_MEM_NUM_BANKS];
    wire [C_M_AXI_MEM_DATA_WIDTH-1:0]    m_axi_mem_rdata_a [C_M_AXI_MEM_NUM_BANKS];
    wire                                 m_axi_mem_rlast_a [C_M_AXI_MEM_NUM_BANKS];
    wire [C_M_AXI_MEM_ID_WIDTH-1:0]      m_axi_mem_rid_a [C_M_AXI_MEM_NUM_BANKS];
    wire [1:0]                           m_axi_mem_rresp_a [C_M_AXI_MEM_NUM_BANKS];

    // convert memory interface to array
`ifdef PLATFORM_MERGED_MEMORY_INTERFACE
    `MP_REPEAT (1, AXI_MEM_TO_ARRAY, MP_SEMI);
`else
    `MP_REPEAT (`VX_CFG_PLATFORM_MEMORY_NUM_BANKS, AXI_MEM_TO_ARRAY, MP_SEMI);
`endif

    reg [`VX_CFG_RESET_DELAY-1:0] vx_reset_shift_r;
    wire vx_reset;
    wire vx_start;
    wire vx_busy;

    // ---- Final DCR signals delivered to Vortex (legacy ∪ CP) ----
    wire                         dcr_req_valid;
    wire                         dcr_req_rw;
    wire [VX_DCR_ADDR_WIDTH-1:0] dcr_req_addr;
    wire [VX_DCR_DATA_WIDTH-1:0] dcr_req_data;
    wire                         dcr_rsp_valid;
    wire [VX_DCR_DATA_WIDTH-1:0] dcr_rsp_data;

    // ========================================================================
    // AXI-Lite demux: bit 12 picks the slave — addr[12]=0 → legacy AFU_ctrl,
    // addr[12]=1 → CP regfile (which sees its own 0x000-based window).
    // See VX_afu_axil_demux for why W routing must fall through from AW.
    // ========================================================================
    wire                                 lg_awvalid, lg_awready;
    wire [7:0]                           lg_awaddr;
    wire                                 lg_wvalid, lg_wready;
    wire [C_S_AXI_CTRL_DATA_WIDTH-1:0]   lg_wdata;
    wire [C_S_AXI_CTRL_DATA_WIDTH/8-1:0] lg_wstrb;
    wire                                 lg_bvalid, lg_bready;
    wire [1:0]                           lg_bresp;
    wire                                 lg_arvalid, lg_arready;
    wire [7:0]                           lg_araddr;
    wire                                 lg_rvalid, lg_rready;
    wire [C_S_AXI_CTRL_DATA_WIDTH-1:0]   lg_rdata;
    wire [1:0]                           lg_rresp;

    VX_cp_axil_s_if #(.ADDR_W(16)) cp_axil ();

    wire [C_S_AXI_CTRL_ADDR_WIDTH-1:0] lg_awaddr_w, lg_araddr_w;

    VX_afu_axil_demux #(
        .ADDR_WIDTH (C_S_AXI_CTRL_ADDR_WIDTH),
        .DATA_WIDTH (C_S_AXI_CTRL_DATA_WIDTH),
        .SEL_BIT    (12)
    ) axil_demux (
        .clk        (clk),
        .reset      (reset),

        .s_awvalid  (s_axi_ctrl_awvalid),
        .s_awready  (s_axi_ctrl_awready),
        .s_awaddr   (s_axi_ctrl_awaddr),
        .s_wvalid   (s_axi_ctrl_wvalid),
        .s_wready   (s_axi_ctrl_wready),
        .s_wdata    (s_axi_ctrl_wdata),
        .s_wstrb    (s_axi_ctrl_wstrb),
        .s_bvalid   (s_axi_ctrl_bvalid),
        .s_bready   (s_axi_ctrl_bready),
        .s_bresp    (s_axi_ctrl_bresp),
        .s_arvalid  (s_axi_ctrl_arvalid),
        .s_arready  (s_axi_ctrl_arready),
        .s_araddr   (s_axi_ctrl_araddr),
        .s_rvalid   (s_axi_ctrl_rvalid),
        .s_rready   (s_axi_ctrl_rready),
        .s_rdata    (s_axi_ctrl_rdata),
        .s_rresp    (s_axi_ctrl_rresp),

        // port 0 — legacy AFU_ctrl (only the low 8 address bits are decoded)
        .m0_awvalid (lg_awvalid),
        .m0_awready (lg_awready),
        .m0_awaddr  (lg_awaddr_w),
        .m0_wvalid  (lg_wvalid),
        .m0_wready  (lg_wready),
        .m0_wdata   (lg_wdata),
        .m0_wstrb   (lg_wstrb),
        .m0_bvalid  (lg_bvalid),
        .m0_bready  (lg_bready),
        .m0_bresp   (lg_bresp),
        .m0_arvalid (lg_arvalid),
        .m0_arready (lg_arready),
        .m0_araddr  (lg_araddr_w),
        .m0_rvalid  (lg_rvalid),
        .m0_rready  (lg_rready),
        .m0_rdata   (lg_rdata),
        .m0_rresp   (lg_rresp),

        // port 1 — CP regfile
        .m1_awvalid (cp_axil.awvalid),
        .m1_awready (cp_axil.awready),
        .m1_awaddr  (cp_axil.awaddr),
        .m1_wvalid  (cp_axil.wvalid),
        .m1_wready  (cp_axil.wready),
        .m1_wdata   (cp_axil.wdata),
        .m1_wstrb   (cp_axil.wstrb),
        .m1_bvalid  (cp_axil.bvalid),
        .m1_bready  (cp_axil.bready),
        .m1_bresp   (cp_axil.bresp),
        .m1_arvalid (cp_axil.arvalid),
        .m1_arready (cp_axil.arready),
        .m1_araddr  (cp_axil.araddr),
        .m1_rvalid  (cp_axil.rvalid),
        .m1_rready  (cp_axil.rready),
        .m1_rdata   (cp_axil.rdata),
        .m1_rresp   (cp_axil.rresp)
    );

    // AFU_ctrl decodes only the low 8 bits of its window.
    assign lg_awaddr = lg_awaddr_w[7:0];
    assign lg_araddr = lg_araddr_w[7:0];
    `UNUSED_VAR (lg_awaddr_w[C_S_AXI_CTRL_ADDR_WIDTH-1:8])
    `UNUSED_VAR (lg_araddr_w[C_S_AXI_CTRL_ADDR_WIDTH-1:8])

`ifdef SCOPE
    wire scope_bus_in;
    wire scope_bus_out;
    wire scope_reset = reset;
`endif

    initial begin
        vx_reset_shift_r = {`VX_CFG_RESET_DELAY{1'b1}};
    end
    assign vx_reset = vx_reset_shift_r[`VX_CFG_RESET_DELAY-1];

    wire ap_reset;
    wire rst_stop_req;
    wire rst_assert;
    wire rst_busy;
    wire rst_timeout_error;
    wire masters_idle;

    // Vortex reset-delay shift register. The platform reset reloads it
    // directly; a host soft-reset request reaches it only through
    // VX_afu_reset_seq, which first drains the AXI masters. Resetting a
    // master with a transaction in flight is what breaks the interconnect.
    always @(posedge clk) begin
        if (reset || rst_assert) begin
            vx_reset_shift_r <= {`VX_CFG_RESET_DELAY{1'b1}};
        end else begin
            vx_reset_shift_r <= {vx_reset_shift_r[`VX_CFG_RESET_DELAY-2:0], 1'b0};
        end
    end

    // Host-port drain counters, declared before their first use in afu_ctrl
    // below. An implicit-net footgun lives here: connecting these in a port
    // map before the declaration makes Vivado mint 1-bit implicit wires at
    // the use site and keep BOTH nets -- afu_ctrl then reads floating 1-bit
    // stubs while host_drain drives the real vectors, and the debug registers
    // read zero forever. That is not hypothetical; it cost a bitstream.
    wire [9:0] dbg_host_aw_count, dbg_host_w_count, dbg_host_b_count;
    wire [9:0] dbg_host_ar_count, dbg_host_r_count;

    VX_afu_reset_seq reset_seq (
        .clk             (clk),
        .reset           (reset),
        .ap_reset_req    (ap_reset),
        .masters_idle    (masters_idle),
        .vx_reset_active (vx_reset),
        .stop_req        (rst_stop_req),
        .rst_assert      (rst_assert),
        .busy            (rst_busy),
        .timeout_error   (rst_timeout_error)
    );

    // Soft-resettable subsystem domain: every block holding state that must
    // clear on a device soft reset (CP command state, bank-0 arbitration).
    // The AXI-Lite control path stays on `reset` alone so it can complete
    // the very write that triggers the sequence.
    wire subsys_reset = reset || vx_reset;

    VX_afu_ctrl #(
        .S_AXI_ADDR_WIDTH (8),
        .S_AXI_DATA_WIDTH (C_S_AXI_CTRL_DATA_WIDTH)
    ) afu_ctrl (
        .clk       		(clk),
        .reset     		(reset),

        .s_axi_awvalid  (lg_awvalid),
        .s_axi_awready  (lg_awready),
        .s_axi_awaddr   (lg_awaddr),

        .s_axi_wvalid   (lg_wvalid),
        .s_axi_wready   (lg_wready),
        .s_axi_wdata    (lg_wdata),
        .s_axi_wstrb    (lg_wstrb),

        .s_axi_arvalid  (lg_arvalid),
        .s_axi_arready  (lg_arready),
        .s_axi_araddr   (lg_araddr),

        .s_axi_rvalid   (lg_rvalid),
        .s_axi_rready   (lg_rready),
        .s_axi_rdata    (lg_rdata),
        .s_axi_rresp    (lg_rresp),

        .s_axi_bvalid   (lg_bvalid),
        .s_axi_bready   (lg_bready),
        .s_axi_bresp    (lg_bresp),

        .ap_reset        (ap_reset),
        .soft_reset_busy (rst_busy),
        .reset_error     (rst_timeout_error),

        .dbg_host_aw_count (dbg_host_aw_count),
        .dbg_host_w_count  (dbg_host_w_count),
        .dbg_host_b_count  (dbg_host_b_count),
        .dbg_host_ar_count (dbg_host_ar_count),
        .dbg_host_r_count  (dbg_host_r_count)

    `ifdef SCOPE
      , .scope_bus_in   (scope_bus_out),
        .scope_bus_out  (scope_bus_in)
    `endif
    );

    // ========================================================================
    // Command Processor
    // ========================================================================
    VX_cp_gpu_if cp_gpu_if ();
    // CP device-memory master (shares Vortex bank 0 via VX_mm_axi_arb).
    VX_mem_axi_if #(.ADDR_W(64), .DATA_W(C_M_AXI_MEM_DATA_WIDTH), .ID_W(`VX_CP_AXI_TID_WIDTH))
        cp_axi_dev ();
    // CP host-memory master (command ring + host side of DMA → m_axi_host).
    VX_mem_axi_if #(.ADDR_W(64), .DATA_W(C_M_AXI_MEM_DATA_WIDTH), .ID_W(`VX_CP_AXI_TID_WIDTH))
        cp_axi_host ();

    wire cp_interrupt;

    VX_cp_core cp_core (
        .clk        (clk),
        .reset      (subsys_reset),
        .axil_s     (cp_axil),
        .axi_host   (cp_axi_host),
        .axi_dev    (cp_axi_dev),
        .dbg_host_w_counts ({2'b0, dbg_host_b_count, dbg_host_w_count,
                             dbg_host_aw_count}),
        .dbg_host_r_counts ({12'b0, dbg_host_r_count, dbg_host_ar_count}),
        .gpu_if     (cp_gpu_if),
        .irq        (cp_interrupt)
    );

    // ---- CP host-memory master → m_axi_host AFU port ----
    // XRT pins m_axi_host to HOST[0]; host addresses pass straight through
    // (no PLATFORM_MEMORY_OFFSET — that offset is device-memory specific).
    // Gated like the bank ports: no new host-side request leaves the AFU while
    // the reset sequencer is draining. The gate is a VX_afu_req_gate rather
    // than a plain AND so an offer already made to the shell is never
    // withdrawn -- see the header of that module.
    VX_afu_req_gate host_aw_gate (
        .clk       (clk),
        .reset     (reset),
        .stop_req  (rst_stop_req),
        .in_valid  (cp_axi_host.awvalid),
        .in_ready  (cp_axi_host.awready),
        .out_valid (m_axi_host_awvalid),
        .out_ready (m_axi_host_awready)
    );
    assign m_axi_host_awaddr  = cp_axi_host.awaddr;
    assign m_axi_host_awid    = {{(C_M_AXI_MEM_ID_WIDTH-`VX_CP_AXI_TID_WIDTH){1'b0}}, cp_axi_host.awid};
    assign m_axi_host_awlen   = cp_axi_host.awlen;
    assign m_axi_host_wvalid  = cp_axi_host.wvalid;
    assign m_axi_host_wdata   = cp_axi_host.wdata;
    assign m_axi_host_wstrb   = cp_axi_host.wstrb;
    assign m_axi_host_wlast   = cp_axi_host.wlast;
    assign cp_axi_host.wready = m_axi_host_wready;
    assign cp_axi_host.bvalid = m_axi_host_bvalid;
    assign cp_axi_host.bid    = m_axi_host_bid[`VX_CP_AXI_TID_WIDTH-1:0];
    assign cp_axi_host.bresp  = m_axi_host_bresp;
    assign m_axi_host_bready  = cp_axi_host.bready;
    VX_afu_req_gate host_ar_gate (
        .clk       (clk),
        .reset     (reset),
        .stop_req  (rst_stop_req),
        .in_valid  (cp_axi_host.arvalid),
        .in_ready  (cp_axi_host.arready),
        .out_valid (m_axi_host_arvalid),
        .out_ready (m_axi_host_arready)
    );
    assign m_axi_host_araddr  = cp_axi_host.araddr;
    assign m_axi_host_arid    = {{(C_M_AXI_MEM_ID_WIDTH-`VX_CP_AXI_TID_WIDTH){1'b0}}, cp_axi_host.arid};
    assign m_axi_host_arlen   = cp_axi_host.arlen;
    assign cp_axi_host.rvalid = m_axi_host_rvalid;
    assign cp_axi_host.rdata  = m_axi_host_rdata;
    assign cp_axi_host.rid    = m_axi_host_rid[`VX_CP_AXI_TID_WIDTH-1:0];
    assign cp_axi_host.rlast  = m_axi_host_rlast;
    assign cp_axi_host.rresp  = m_axi_host_rresp;
    assign m_axi_host_rready  = cp_axi_host.rready;
    `UNUSED_VAR (m_axi_host_bid)
    `UNUSED_VAR (m_axi_host_rid)
    `UNUSED_VAR (cp_axi_host.awsize)
    `UNUSED_VAR (cp_axi_host.awburst)
    `UNUSED_VAR (cp_axi_host.arsize)
    `UNUSED_VAR (cp_axi_host.arburst)

    // AFU interrupt pin reflects the Command Processor — one-cycle pulse per retired command.
    assign interrupt = cp_interrupt;

    // ---- gpu_if → Vortex DCR (the CP is the sole DCR source) ----
    assign dcr_req_valid = cp_gpu_if.dcr_req_valid;
    assign dcr_req_rw    = cp_gpu_if.dcr_req_rw;
    assign dcr_req_addr  = cp_gpu_if.dcr_req_addr;
    assign dcr_req_data  = cp_gpu_if.dcr_req_data;

    assign cp_gpu_if.dcr_req_ready = 1'b1;          // Vortex DCR always accepts
    assign cp_gpu_if.dcr_rsp_valid = dcr_rsp_valid;
    assign cp_gpu_if.dcr_rsp_data  = dcr_rsp_data;
    assign cp_gpu_if.busy          = vx_busy;

    // The CP is the sole launch source.
    assign vx_start = cp_gpu_if.start;

    wire [M_AXI_MEM_ADDR_WIDTH-1:0] m_axi_mem_awaddr_u [C_M_AXI_MEM_NUM_BANKS];
    wire [M_AXI_MEM_ADDR_WIDTH-1:0] m_axi_mem_araddr_u [C_M_AXI_MEM_NUM_BANKS];

    // Per-bank XRT BO base offset. Each m_axi_mem_<i> port targets a different
    // xrt::bo (one per DDR/HBM channel) that XRT places at a different virtual
    // base address. PLATFORM_MEMORY_OFFSET applies the same synthesis-time
    // offset to every bank.
    wire [C_M_AXI_MEM_ADDR_WIDTH-1:0] platform_memory_offsets [C_M_AXI_MEM_NUM_BANKS];
    for (genvar i = 0; i < C_M_AXI_MEM_NUM_BANKS; ++i) begin : g_pmo
        assign platform_memory_offsets[i] = C_M_AXI_MEM_ADDR_WIDTH'(`PLATFORM_MEMORY_OFFSET);
    end

    for (genvar i = 0; i < C_M_AXI_MEM_NUM_BANKS; ++i) begin : g_addressing
        assign m_axi_mem_awaddr_a[i] = C_M_AXI_MEM_ADDR_WIDTH'(m_axi_mem_awaddr_u[i]) + platform_memory_offsets[i];
        assign m_axi_mem_araddr_a[i] = C_M_AXI_MEM_ADDR_WIDTH'(m_axi_mem_araddr_u[i]) + platform_memory_offsets[i];
    end

    // ---- Intermediate Vortex AXI signals (per-bank) — arbiter sits on bank 0 ----
    wire                              vx_awvalid_a [C_M_AXI_MEM_NUM_BANKS];
    wire                              vx_awready_a [C_M_AXI_MEM_NUM_BANKS];
    wire [M_AXI_MEM_ADDR_WIDTH-1:0]   vx_awaddr_a  [C_M_AXI_MEM_NUM_BANKS];
    wire [C_M_AXI_MEM_ID_WIDTH-1:0]   vx_awid_a    [C_M_AXI_MEM_NUM_BANKS];
    wire [7:0]                        vx_awlen_a   [C_M_AXI_MEM_NUM_BANKS];

    wire                              vx_wvalid_a  [C_M_AXI_MEM_NUM_BANKS];
    wire                              vx_wready_a  [C_M_AXI_MEM_NUM_BANKS];
    wire [C_M_AXI_MEM_DATA_WIDTH-1:0] vx_wdata_a   [C_M_AXI_MEM_NUM_BANKS];
    wire [C_M_AXI_MEM_DATA_WIDTH/8-1:0] vx_wstrb_a [C_M_AXI_MEM_NUM_BANKS];
    wire                              vx_wlast_a   [C_M_AXI_MEM_NUM_BANKS];

    wire                              vx_bvalid_a  [C_M_AXI_MEM_NUM_BANKS];
    wire                              vx_bready_a  [C_M_AXI_MEM_NUM_BANKS];
    wire [C_M_AXI_MEM_ID_WIDTH-1:0]   vx_bid_a     [C_M_AXI_MEM_NUM_BANKS];
    wire [1:0]                        vx_bresp_a   [C_M_AXI_MEM_NUM_BANKS];

    wire                              vx_arvalid_a [C_M_AXI_MEM_NUM_BANKS];
    wire                              vx_arready_a [C_M_AXI_MEM_NUM_BANKS];
    wire [M_AXI_MEM_ADDR_WIDTH-1:0]   vx_araddr_a  [C_M_AXI_MEM_NUM_BANKS];
    wire [C_M_AXI_MEM_ID_WIDTH-1:0]   vx_arid_a    [C_M_AXI_MEM_NUM_BANKS];
    wire [7:0]                        vx_arlen_a   [C_M_AXI_MEM_NUM_BANKS];

    wire                              vx_rvalid_a  [C_M_AXI_MEM_NUM_BANKS];
    wire                              vx_rready_a  [C_M_AXI_MEM_NUM_BANKS];
    wire [C_M_AXI_MEM_DATA_WIDTH-1:0] vx_rdata_a   [C_M_AXI_MEM_NUM_BANKS];
    wire                              vx_rlast_a   [C_M_AXI_MEM_NUM_BANKS];
    wire [C_M_AXI_MEM_ID_WIDTH-1:0]   vx_rid_a     [C_M_AXI_MEM_NUM_BANKS];
    wire [1:0]                        vx_rresp_a   [C_M_AXI_MEM_NUM_BANKS];

    `SCOPE_IO_SWITCH (2);

    Vortex_axi #(
        .AXI_DATA_WIDTH (C_M_AXI_MEM_DATA_WIDTH),
        .AXI_ADDR_WIDTH (M_AXI_MEM_ADDR_WIDTH),
        .AXI_TID_WIDTH  (C_M_AXI_MEM_ID_WIDTH),
        .AXI_NUM_BANKS  (C_M_AXI_MEM_NUM_BANKS)
    ) vortex_axi (
        `SCOPE_IO_BIND  (1)

        .clk			(clk),
        .reset			(vx_reset),

        .m_axi_awvalid	(vx_awvalid_a),
        .m_axi_awready	(vx_awready_a),
        .m_axi_awaddr	(vx_awaddr_a),
        .m_axi_awid		(vx_awid_a),
        .m_axi_awlen    (vx_awlen_a),
        `UNUSED_PIN (m_axi_awsize),
        `UNUSED_PIN (m_axi_awburst),
        `UNUSED_PIN (m_axi_awlock),
        `UNUSED_PIN (m_axi_awcache),
        `UNUSED_PIN (m_axi_awprot),
        `UNUSED_PIN (m_axi_awqos),
        `UNUSED_PIN (m_axi_awregion),

        .m_axi_wvalid	(vx_wvalid_a),
        .m_axi_wready	(vx_wready_a),
        .m_axi_wdata	(vx_wdata_a),
        .m_axi_wstrb	(vx_wstrb_a),
        .m_axi_wlast	(vx_wlast_a),

        .m_axi_bvalid	(vx_bvalid_a),
        .m_axi_bready	(vx_bready_a),
        .m_axi_bid		(vx_bid_a),
        .m_axi_bresp	(vx_bresp_a),

        .m_axi_arvalid	(vx_arvalid_a),
        .m_axi_arready	(vx_arready_a),
        .m_axi_araddr	(vx_araddr_a),
        .m_axi_arid		(vx_arid_a),
        .m_axi_arlen	(vx_arlen_a),
        `UNUSED_PIN (m_axi_arsize),
        `UNUSED_PIN (m_axi_arburst),
        `UNUSED_PIN (m_axi_arlock),
        `UNUSED_PIN (m_axi_arcache),
        `UNUSED_PIN (m_axi_arprot),
        `UNUSED_PIN (m_axi_arqos),
        `UNUSED_PIN (m_axi_arregion),

        .m_axi_rvalid	(vx_rvalid_a),
        .m_axi_rready	(vx_rready_a),
        .m_axi_rdata	(vx_rdata_a),
        .m_axi_rlast	(vx_rlast_a),
        .m_axi_rid    	(vx_rid_a),
        .m_axi_rresp	(vx_rresp_a),

        .dcr_req_valid	(dcr_req_valid),
        .dcr_req_rw		(dcr_req_rw),
        .dcr_req_addr	(dcr_req_addr),
        .dcr_req_data	(dcr_req_data),
        .dcr_rsp_valid	(dcr_rsp_valid),
        .dcr_rsp_data	(dcr_rsp_data),

        .start          (vx_start),
        .busy			(vx_busy)
    );

    // ========================================================================
    // Request gate + drain tracking.
    //
    // Everything upstream drives pre_* instead of the bank port directly. The
    // gate below withholds new AW/AR while the reset sequencer is quiescing,
    // so the outstanding counts can reach zero in bounded time. W, B and R are
    // never gated: a burst whose address the interconnect has already accepted
    // must be allowed to finish.
    // ========================================================================
    wire pre_awvalid_a [C_M_AXI_MEM_NUM_BANKS];
    wire pre_awready_a [C_M_AXI_MEM_NUM_BANKS];
    wire pre_arvalid_a [C_M_AXI_MEM_NUM_BANKS];
    wire pre_arready_a [C_M_AXI_MEM_NUM_BANKS];

    wire mem_idle_a [C_M_AXI_MEM_NUM_BANKS];

    for (genvar i = 0; i < C_M_AXI_MEM_NUM_BANKS; ++i) begin : g_axi_gate
        VX_afu_req_gate mem_aw_gate (
            .clk       (clk),
            .reset     (reset),
            .stop_req  (rst_stop_req),
            .in_valid  (pre_awvalid_a[i]),
            .in_ready  (pre_awready_a[i]),
            .out_valid (m_axi_mem_awvalid_a[i]),
            .out_ready (m_axi_mem_awready_a[i])
        );

        VX_afu_req_gate mem_ar_gate (
            .clk       (clk),
            .reset     (reset),
            .stop_req  (rst_stop_req),
            .in_valid  (pre_arvalid_a[i]),
            .in_ready  (pre_arready_a[i]),
            .out_valid (m_axi_mem_arvalid_a[i]),
            .out_ready (m_axi_mem_arready_a[i])
        );

        VX_afu_axi_drain mem_drain (
            .clk          (clk),
            .reset        (reset),
            .aw_fire      (m_axi_mem_awvalid_a[i] && m_axi_mem_awready_a[i]),
            .w_fire_last  (m_axi_mem_wvalid_a[i] && m_axi_mem_wready_a[i]
                                                 && m_axi_mem_wlast_a[i]),
            .b_fire       (m_axi_mem_bvalid_a[i] && m_axi_mem_bready_a[i]),
            .ar_fire      (m_axi_mem_arvalid_a[i] && m_axi_mem_arready_a[i]),
            .r_fire_last  (m_axi_mem_rvalid_a[i] && m_axi_mem_rready_a[i]
                                                 && m_axi_mem_rlast_a[i]),
            .idle         (mem_idle_a[i]),
            `UNUSED_PIN (dbg_aw_count),
            `UNUSED_PIN (dbg_w_count),
            `UNUSED_PIN (dbg_b_count),
            `UNUSED_PIN (dbg_ar_count),
            `UNUSED_PIN (dbg_r_count)
        );
    end

    wire host_idle;

    VX_afu_axi_drain host_drain (
        .clk          (clk),
        .reset        (reset),
        .aw_fire      (m_axi_host_awvalid && m_axi_host_awready),
        .w_fire_last  (m_axi_host_wvalid && m_axi_host_wready && m_axi_host_wlast),
        .b_fire       (m_axi_host_bvalid && m_axi_host_bready),
        .ar_fire      (m_axi_host_arvalid && m_axi_host_arready),
        .r_fire_last  (m_axi_host_rvalid && m_axi_host_rready && m_axi_host_rlast),
        .idle         (host_idle),
        .dbg_aw_count (dbg_host_aw_count),
        .dbg_w_count  (dbg_host_w_count),
        .dbg_b_count  (dbg_host_b_count),
        .dbg_ar_count (dbg_host_ar_count),
        .dbg_r_count  (dbg_host_r_count)
    );

    // masters_idle gates the reset assertion, so it must cover every master.
    reg mem_idle_all;
    always @(*) begin
        mem_idle_all = 1'b1;
        for (int b = 0; b < C_M_AXI_MEM_NUM_BANKS; ++b) begin
            if (!mem_idle_a[b]) begin
                mem_idle_all = 1'b0;
            end
        end
    end
    assign masters_idle = mem_idle_all && host_idle;

    // ---- Banks 1..N-1: direct passthrough ----
    for (genvar i = 1; i < C_M_AXI_MEM_NUM_BANKS; ++i) begin : g_bank_passthrough
        assign pre_awvalid_a[i] = vx_awvalid_a[i];
        assign m_axi_mem_awaddr_u[i]  = vx_awaddr_a[i];
        assign m_axi_mem_awid_a[i]    = vx_awid_a[i];
        assign m_axi_mem_awlen_a[i]   = vx_awlen_a[i];
        assign vx_awready_a[i]        = pre_awready_a[i];

        assign m_axi_mem_wvalid_a[i]  = vx_wvalid_a[i];
        assign m_axi_mem_wdata_a[i]   = vx_wdata_a[i];
        assign m_axi_mem_wstrb_a[i]   = vx_wstrb_a[i];
        assign m_axi_mem_wlast_a[i]   = vx_wlast_a[i];
        assign vx_wready_a[i]         = m_axi_mem_wready_a[i];

        assign vx_bvalid_a[i]         = m_axi_mem_bvalid_a[i];
        assign vx_bid_a[i]            = m_axi_mem_bid_a[i];
        assign vx_bresp_a[i]          = m_axi_mem_bresp_a[i];
        assign m_axi_mem_bready_a[i]  = vx_bready_a[i];

        assign pre_arvalid_a[i]       = vx_arvalid_a[i];
        assign m_axi_mem_araddr_u[i]  = vx_araddr_a[i];
        assign m_axi_mem_arid_a[i]    = vx_arid_a[i];
        assign m_axi_mem_arlen_a[i]   = vx_arlen_a[i];
        assign vx_arready_a[i]        = pre_arready_a[i];

        assign vx_rvalid_a[i]         = m_axi_mem_rvalid_a[i];
        assign vx_rdata_a[i]          = m_axi_mem_rdata_a[i];
        assign vx_rlast_a[i]          = m_axi_mem_rlast_a[i];
        assign vx_rid_a[i]            = m_axi_mem_rid_a[i];
        assign vx_rresp_a[i]          = m_axi_mem_rresp_a[i];
        assign m_axi_mem_rready_a[i]  = vx_rready_a[i];
    end

    // ---- Bank 0: 2:1 arbiter merges Vortex bank-0 + CP axi_m ----
    // Pad CP's narrower ID into the platform ID width so the arbiter sees
    // identical signal widths from both sources.
    wire [C_M_AXI_MEM_ID_WIDTH-1:0] cp_awid_padded =
        {{(C_M_AXI_MEM_ID_WIDTH - `VX_CP_AXI_TID_WIDTH){1'b0}}, cp_axi_dev.awid};
    wire [C_M_AXI_MEM_ID_WIDTH-1:0] cp_arid_padded =
        {{(C_M_AXI_MEM_ID_WIDTH - `VX_CP_AXI_TID_WIDTH){1'b0}}, cp_axi_dev.arid};

    // The CP's device addresses come from the same host-side allocator as the
    // pointers handed to the cores -- Device::global_mem_, based at
    // VX_MEM_USER_BASE_ADDR -- so they are offset-relative already, exactly
    // like vx_awaddr_a[0]. Feed them to the arbiter unchanged and let
    // PLATFORM_MEMORY_OFFSET be applied once, at the bank port, for both
    // masters. Subtracting it here would cancel that re-offset and leave every
    // CP DMA pointed outside the platform's memory aperture.
    wire [M_AXI_MEM_ADDR_WIDTH-1:0] cp_awaddr_dev =
        M_AXI_MEM_ADDR_WIDTH'(cp_axi_dev.awaddr);
    wire [M_AXI_MEM_ADDR_WIDTH-1:0] cp_araddr_dev =
        M_AXI_MEM_ADDR_WIDTH'(cp_axi_dev.araddr);

    // Packed 2-master AXI arbiter: index 0 = Vortex bank-0 (priority via
    // ARBITER="P"), index 1 = CP device master. Input channels are packed
    // {cp, vx}; the arbiter's slave-side outputs land in local packed wires
    // and are split back to the two masters below.
    localparam BANK0_STRB_W = C_M_AXI_MEM_DATA_WIDTH/8;

    wire [1:0]                            b0_awready;
    wire [1:0]                            b0_wready;
    wire [1:0]                            b0_bvalid;
    wire [1:0][C_M_AXI_MEM_ID_WIDTH-1:0]  b0_bid;
    wire [1:0][1:0]                       b0_bresp;
    wire [1:0]                            b0_arready;
    wire [1:0]                            b0_rvalid;
    wire [1:0][C_M_AXI_MEM_DATA_WIDTH-1:0] b0_rdata;
    wire [1:0]                            b0_rlast;
    wire [1:0][C_M_AXI_MEM_ID_WIDTH-1:0]  b0_rid;
    wire [1:0][1:0]                       b0_rresp;

    VX_mm_axi_arb #(
        .NUM_INPUTS (2),
        .ADDR_WIDTH (M_AXI_MEM_ADDR_WIDTH),
        .DATA_WIDTH (C_M_AXI_MEM_DATA_WIDTH),
        .ID_WIDTH   (C_M_AXI_MEM_ID_WIDTH),
        .ARBITER    ("P"),          // index 0 (Vortex bank-0) > index 1 (CP)
        .STRB_WIDTH (BANK0_STRB_W)
    ) bank0_arb (
        .clk   (clk),
        .reset (subsys_reset),

        .s_awvalid ({cp_axi_dev.awvalid, vx_awvalid_a[0]}),
        .s_awready (b0_awready),
        .s_awaddr  ({cp_awaddr_dev,   vx_awaddr_a[0]}),
        .s_awid    ({cp_awid_padded,     vx_awid_a[0]}),
        .s_awlen   ({cp_axi_dev.awlen,   vx_awlen_a[0]}),

        .s_wvalid  ({cp_axi_dev.wvalid,  vx_wvalid_a[0]}),
        .s_wready  (b0_wready),
        .s_wdata   ({cp_axi_dev.wdata,   vx_wdata_a[0]}),
        .s_wstrb   ({cp_axi_dev.wstrb,   vx_wstrb_a[0]}),
        .s_wlast   ({cp_axi_dev.wlast,   vx_wlast_a[0]}),

        .s_bvalid  (b0_bvalid),
        .s_bready  ({cp_axi_dev.bready,  vx_bready_a[0]}),
        .s_bid     (b0_bid),
        .s_bresp   (b0_bresp),

        .s_arvalid ({cp_axi_dev.arvalid, vx_arvalid_a[0]}),
        .s_arready (b0_arready),
        .s_araddr  ({cp_araddr_dev,   vx_araddr_a[0]}),
        .s_arid    ({cp_arid_padded,     vx_arid_a[0]}),
        .s_arlen   ({cp_axi_dev.arlen,   vx_arlen_a[0]}),

        .s_rvalid  (b0_rvalid),
        .s_rready  ({cp_axi_dev.rready,  vx_rready_a[0]}),
        .s_rdata   (b0_rdata),
        .s_rlast   (b0_rlast),
        .s_rid     (b0_rid),
        .s_rresp   (b0_rresp),

        .m_awvalid  (pre_awvalid_a[0]),       .m_awready (pre_awready_a[0]),
        .m_awaddr   (m_axi_mem_awaddr_u[0]),  .m_awid    (m_axi_mem_awid_a[0]),
        .m_awlen    (m_axi_mem_awlen_a[0]),
        .m_wvalid   (m_axi_mem_wvalid_a[0]),  .m_wready  (m_axi_mem_wready_a[0]),
        .m_wdata    (m_axi_mem_wdata_a[0]),   .m_wstrb   (m_axi_mem_wstrb_a[0]),
        .m_wlast    (m_axi_mem_wlast_a[0]),
        .m_bvalid   (m_axi_mem_bvalid_a[0]),  .m_bready  (m_axi_mem_bready_a[0]),
        .m_bid      (m_axi_mem_bid_a[0]),     .m_bresp   (m_axi_mem_bresp_a[0]),
        .m_arvalid  (pre_arvalid_a[0]),       .m_arready (pre_arready_a[0]),
        .m_araddr   (m_axi_mem_araddr_u[0]),  .m_arid    (m_axi_mem_arid_a[0]),
        .m_arlen    (m_axi_mem_arlen_a[0]),
        .m_rvalid   (m_axi_mem_rvalid_a[0]),  .m_rready  (m_axi_mem_rready_a[0]),
        .m_rdata    (m_axi_mem_rdata_a[0]),   .m_rlast   (m_axi_mem_rlast_a[0]),
        .m_rid      (m_axi_mem_rid_a[0]),     .m_rresp   (m_axi_mem_rresp_a[0])
    );

    // ---- Split the arbiter's packed slave-side outputs to the two masters ----
    // index 0 = Vortex bank-0, index 1 = CP device master.
    // Declared before their first assignment below (the implicit-net footgun:
    // a pre-declaration use makes Vivado mint 1-bit nets and keep both).
    wire [C_M_AXI_MEM_ID_WIDTH-1:0] cp_axi_dev_bid_full;
    wire [C_M_AXI_MEM_ID_WIDTH-1:0] cp_axi_dev_rid_full;
    assign vx_awready_a[0]     = b0_awready[0];
    assign cp_axi_dev.awready  = b0_awready[1];
    assign vx_wready_a[0]      = b0_wready[0];
    assign cp_axi_dev.wready   = b0_wready[1];
    assign vx_bvalid_a[0]      = b0_bvalid[0];
    assign cp_axi_dev.bvalid   = b0_bvalid[1];
    assign vx_bid_a[0]         = b0_bid[0];
    assign cp_axi_dev_bid_full = b0_bid[1];
    assign vx_bresp_a[0]       = b0_bresp[0];
    assign cp_axi_dev.bresp    = b0_bresp[1];
    assign vx_arready_a[0]     = b0_arready[0];
    assign cp_axi_dev.arready  = b0_arready[1];
    assign vx_rvalid_a[0]      = b0_rvalid[0];
    assign cp_axi_dev.rvalid   = b0_rvalid[1];
    assign vx_rdata_a[0]       = b0_rdata[0];
    assign cp_axi_dev.rdata    = b0_rdata[1];
    assign vx_rlast_a[0]       = b0_rlast[0];
    assign cp_axi_dev.rlast    = b0_rlast[1];
    assign vx_rid_a[0]         = b0_rid[0];
    assign cp_axi_dev_rid_full = b0_rid[1];
    assign vx_rresp_a[0]       = b0_rresp[0];
    assign cp_axi_dev.rresp    = b0_rresp[1];

    // Truncate the arbiter's wider ID back to CP's narrower native ID width.
    assign cp_axi_dev.bid = cp_axi_dev_bid_full[`VX_CP_AXI_TID_WIDTH-1:0];
    assign cp_axi_dev.rid = cp_axi_dev_rid_full[`VX_CP_AXI_TID_WIDTH-1:0];
    `UNUSED_VAR (cp_axi_dev_bid_full)
    `UNUSED_VAR (cp_axi_dev_rid_full)

    // The optional AXI4 sideband signals (size/burst) are unused by the
    // reduced VX_mm_axi_arb view — pin them sink-side so lint stays clean.
    `UNUSED_VAR (cp_axi_dev.awsize)
    `UNUSED_VAR (cp_axi_dev.awburst)
    `UNUSED_VAR (cp_axi_dev.arsize)
    `UNUSED_VAR (cp_axi_dev.arburst)

    // We only use addr[12:0] of the AXI-Lite address space; bits 15:13 are
    // always 0 from the kernel.xml-advertised slave size but Verilator
    // still flags them — pin to UNUSED.
    `UNUSED_VAR (s_axi_ctrl_awaddr[15:13])
    `UNUSED_VAR (s_axi_ctrl_araddr[15:13])

    // SCOPE //////////////////////////////////////////////////////////////////////

`ifdef SCOPE
`ifdef DBG_SCOPE_AFU
    wire m_axi_mem_awfire_0 = m_axi_mem_awvalid_a[0] & m_axi_mem_awready_a[0];
    wire m_axi_mem_arfire_0 = m_axi_mem_arvalid_a[0] & m_axi_mem_arready_a[0];
    wire m_axi_mem_wfire_0  = m_axi_mem_wvalid_a[0]  & m_axi_mem_wready_a[0];
    wire m_axi_mem_bfire_0  = m_axi_mem_bvalid_a[0]  & m_axi_mem_bready_a[0];
    wire reset_negedge;
    `NEG_EDGE (reset_negedge, reset);
    `SCOPE_TAP (0, 0, {
            vx_start,
            interrupt,
            vx_reset,
            vx_busy,
            m_axi_mem_awvalid_a[0],
            m_axi_mem_awready_a[0],
            m_axi_mem_wvalid_a[0],
            m_axi_mem_wready_a[0],
            m_axi_mem_bvalid_a[0],
            m_axi_mem_bready_a[0],
            m_axi_mem_arvalid_a[0],
            m_axi_mem_arready_a[0],
            m_axi_mem_rvalid_a[0],
            m_axi_mem_rready_a[0]
        }, {
            dcr_req_valid,
            m_axi_mem_awfire_0,
            m_axi_mem_arfire_0,
            m_axi_mem_wfire_0,
            m_axi_mem_bfire_0
        }, {
            dcr_req_addr,
            dcr_req_data,
            m_axi_mem_awaddr_u[0],
            m_axi_mem_awid_a[0],
            m_axi_mem_bid_a[0],
            m_axi_mem_araddr_u[0],
            m_axi_mem_arid_a[0],
            m_axi_mem_rid_a[0]
        },
        reset_negedge, 1'b0, 4096
    );
`else
    `SCOPE_IO_UNUSED(0)
`endif
`endif

`ifdef CHIPSCOPE
`ifdef DBG_SCOPE_AFU
    ila_afu ila_afu_inst (
        .clk (clk),
        .probe0 ({
            vx_start,
            interrupt
        }),
        .probe1 ({
            vx_busy,
            vx_reset,
            dcr_req_valid,
            dcr_req_addr,
            dcr_req_data
        })
    );
`endif
`endif

`ifdef SIMULATION
`ifndef VERILATOR
    // disable assertions until full reset
    reg [`CLOG2(`VX_CFG_RESET_DELAY+1)-1:0] assert_delay_ctr;
    reg assert_enabled;
    initial begin
        $assertoff(0, vortex_axi);
    end
    always @(posedge clk) begin
        if (reset || vx_reset) begin
            assert_delay_ctr <= '0;
            assert_enabled   <= 0;
            if (assert_enabled) begin
                $assertoff(0, vortex_axi);
            end
        end else begin
            if (~assert_enabled) begin
                if (assert_delay_ctr == (`VX_CFG_RESET_DELAY-1)) begin
                    assert_enabled <= 1;
                    $asserton(0, vortex_axi);
                end else begin
                    assert_delay_ctr <= assert_delay_ctr + 1;
                end
            end
        end
    end
`endif
`endif

`ifdef DBG_TRACE_AFU
    always @(posedge clk) begin
        for (integer i = 0; i < C_M_AXI_MEM_NUM_BANKS; ++i) begin
            if (m_axi_mem_awvalid_a[i] && m_axi_mem_awready_a[i]) begin
                `TRACE(2, ("%t: AXI Wr Req [%0d]: addr=0x%0h, id=0x%0h\n", $time, i, m_axi_mem_awaddr_a[i], m_axi_mem_awid_a[i]))
            end
            if (m_axi_mem_wvalid_a[i] && m_axi_mem_wready_a[i]) begin
                `TRACE(2, ("%t: AXI Wr Req [%0d]: strb=0x%h, data=0x%h\n", $time, i, m_axi_mem_wstrb_a[i], m_axi_mem_wdata_a[i]))
            end
            if (m_axi_mem_bvalid_a[i] && m_axi_mem_bready_a[i]) begin
                `TRACE(2, ("%t: AXI Wr Rsp [%0d]: id=0x%0h\n", $time, i, m_axi_mem_bid_a[i]))
            end
            if (m_axi_mem_arvalid_a[i] && m_axi_mem_arready_a[i]) begin
                `TRACE(2, ("%t: AXI Rd Req [%0d]: addr=0x%0h, id=0x%0h\n", $time, i, m_axi_mem_araddr_a[i], m_axi_mem_arid_a[i]))
            end
            if (m_axi_mem_rvalid_a[i] && m_axi_mem_rready_a[i]) begin
                `TRACE(2, ("%t: AXI Rd Rsp [%0d]: data=0x%h, id=0x%0h\n", $time, i, m_axi_mem_rdata_a[i], m_axi_mem_rid_a[i]))
            end
        end
    end
`endif

endmodule
