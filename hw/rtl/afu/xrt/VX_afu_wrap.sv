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
//                    SCOPE bit-serial register pair (0x28/0x2C). The legacy
//                    launch FSM / DCR / dev_caps registers were removed in
//                    Phase 4.
//   0x1000..0x1FFF — Command Processor regfile, mapped to CP's native
//                    0x000..0xFFF address space (CP sees addr - 0x1000).
//                    The bit-12 split keeps CP_CTRL at CP-offset 0x000
//                    reachable without colliding with the ap_ctrl stub
//                    register at host-offset 0x000.
//
// Data plane:
//   * Vortex memory banks 0..N-1 ride the platform AXI4 master ports.
//   * VX_cp_core has its own axi_m. Bank 0 is shared via VX_axi_arb2 —
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
	// AXI-Lite demux: 0x00..0xFF → legacy AFU_ctrl, 0x100..0xFFFF → CP regfile.
	// Routing is latched at AW/AR fire so mixed-range pipelines stay coherent.
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

	// Bit 12 picks the slave: host addr[12]=1 → CP regfile; addr[12]=0 → legacy.
	wire is_cp_aw = s_axi_ctrl_awaddr[12];
	wire is_cp_ar = s_axi_ctrl_araddr[12];

	reg route_cp_w_r, route_cp_w_valid;
	reg route_cp_r_r, route_cp_r_valid;
	always @(posedge clk) begin
		if (reset) begin
			route_cp_w_r <= 0; route_cp_w_valid <= 0;
			route_cp_r_r <= 0; route_cp_r_valid <= 0;
		end else begin
			if (s_axi_ctrl_awvalid && s_axi_ctrl_awready) begin
				route_cp_w_r     <= is_cp_aw;
				route_cp_w_valid <= 1;
			end else if (s_axi_ctrl_bvalid && s_axi_ctrl_bready) begin
				route_cp_w_valid <= 0;
			end
			if (s_axi_ctrl_arvalid && s_axi_ctrl_arready) begin
				route_cp_r_r     <= is_cp_ar;
				route_cp_r_valid <= 1;
			end else if (s_axi_ctrl_rvalid && s_axi_ctrl_rready) begin
				route_cp_r_valid <= 0;
			end
		end
	end

	wire route_aw = route_cp_w_valid ? route_cp_w_r : is_cp_aw;
	wire route_ar = route_cp_r_valid ? route_cp_r_r : is_cp_ar;

	assign lg_awvalid       = s_axi_ctrl_awvalid && !route_aw;
	assign lg_awaddr        = s_axi_ctrl_awaddr[7:0];
	assign cp_axil.awvalid  = s_axi_ctrl_awvalid &&  route_aw;
	// CP sees its own 0x000-based address — drop the bit-12 select.
	assign cp_axil.awaddr   = {4'd0, s_axi_ctrl_awaddr[11:0]};
	assign s_axi_ctrl_awready = route_aw ? cp_axil.awready : lg_awready;

	assign lg_wvalid        = s_axi_ctrl_wvalid && !route_cp_w_r;
	assign lg_wdata         = s_axi_ctrl_wdata;
	assign lg_wstrb         = s_axi_ctrl_wstrb;
	assign cp_axil.wvalid   = s_axi_ctrl_wvalid &&  route_cp_w_r;
	assign cp_axil.wdata    = s_axi_ctrl_wdata;
	assign cp_axil.wstrb    = s_axi_ctrl_wstrb;
	assign s_axi_ctrl_wready = route_cp_w_r ? cp_axil.wready : lg_wready;

	assign s_axi_ctrl_bvalid = route_cp_w_r ? cp_axil.bvalid : lg_bvalid;
	assign s_axi_ctrl_bresp  = route_cp_w_r ? cp_axil.bresp  : lg_bresp;
	assign cp_axil.bready    = s_axi_ctrl_bready &&  route_cp_w_r;
	assign lg_bready         = s_axi_ctrl_bready && !route_cp_w_r;

	assign lg_arvalid       = s_axi_ctrl_arvalid && !route_ar;
	assign lg_araddr        = s_axi_ctrl_araddr[7:0];
	assign cp_axil.arvalid  = s_axi_ctrl_arvalid &&  route_ar;
	assign cp_axil.araddr   = {4'd0, s_axi_ctrl_araddr[11:0]};
	assign s_axi_ctrl_arready = route_ar ? cp_axil.arready : lg_arready;

	assign s_axi_ctrl_rvalid = route_cp_r_r ? cp_axil.rvalid : lg_rvalid;
	assign s_axi_ctrl_rdata  = route_cp_r_r ? cp_axil.rdata  : lg_rdata;
	assign s_axi_ctrl_rresp  = route_cp_r_r ? cp_axil.rresp  : lg_rresp;
	assign cp_axil.rready    = s_axi_ctrl_rready &&  route_cp_r_r;
	assign lg_rready         = s_axi_ctrl_rready && !route_cp_r_r;

`ifdef SCOPE
	wire scope_bus_in;
	wire scope_bus_out;
  	wire scope_reset = reset;
`endif

    initial begin
        vx_reset_shift_r = {`VX_CFG_RESET_DELAY{1'b1}};
// asserted at initialization
    end
    assign vx_reset = vx_reset_shift_r[`VX_CFG_RESET_DELAY-1];

	// Vortex reset-delay shift register. The CP owns launches; there is no
	// host-driven ap_reset any more, so this keys on `reset` alone.
	always @(posedge clk) begin
		if (reset) begin
			vx_reset_shift_r <= {`VX_CFG_RESET_DELAY{1'b1}};
		end else begin
			vx_reset_shift_r <= {vx_reset_shift_r[`VX_CFG_RESET_DELAY-2:0], 1'b0};
		end
	end

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
		.s_axi_bresp    (lg_bresp)

	`ifdef SCOPE
	  , .scope_bus_in   (scope_bus_out),
		.scope_bus_out  (scope_bus_in)
	`endif
	);

	// ========================================================================
	// Command Processor
	// ========================================================================
	VX_cp_gpu_if cp_gpu_if ();
	// CP device-memory master (shares Vortex bank 0 via VX_axi_arb2).
	VX_cp_axi_m_if #(.ADDR_W(64), .DATA_W(C_M_AXI_MEM_DATA_WIDTH))
	    cp_axi_dev ();
	// CP host-memory master (command ring + host side of DMA → m_axi_host).
	VX_cp_axi_m_if #(.ADDR_W(64), .DATA_W(C_M_AXI_MEM_DATA_WIDTH))
	    cp_axi_host ();

	wire cp_interrupt;

	VX_cp_core u_cp_core (
		.clk        (clk),
		.reset      (reset),
		.axil_s     (cp_axil),
		.axi_host   (cp_axi_host),
		.axi_dev    (cp_axi_dev),
		.gpu_if     (cp_gpu_if),
		.interrupt  (cp_interrupt)
	);

	// ---- CP host-memory master → m_axi_host AFU port ----
	// XRT pins m_axi_host to HOST[0]; host addresses pass straight through
	// (no PLATFORM_MEMORY_OFFSET — that offset is device-memory specific).
	assign m_axi_host_awvalid = cp_axi_host.awvalid;
	assign m_axi_host_awaddr  = cp_axi_host.awaddr;
	assign m_axi_host_awid    = {{(C_M_AXI_MEM_ID_WIDTH-`VX_CP_AXI_TID_WIDTH){1'b0}}, cp_axi_host.awid};
	assign m_axi_host_awlen   = cp_axi_host.awlen;
	assign cp_axi_host.awready = m_axi_host_awready;
	assign m_axi_host_wvalid  = cp_axi_host.wvalid;
	assign m_axi_host_wdata   = cp_axi_host.wdata;
	assign m_axi_host_wstrb   = cp_axi_host.wstrb;
	assign m_axi_host_wlast   = cp_axi_host.wlast;
	assign cp_axi_host.wready = m_axi_host_wready;
	assign cp_axi_host.bvalid = m_axi_host_bvalid;
	assign cp_axi_host.bid    = m_axi_host_bid[`VX_CP_AXI_TID_WIDTH-1:0];
	assign cp_axi_host.bresp  = m_axi_host_bresp;
	assign m_axi_host_bready  = cp_axi_host.bready;
	assign m_axi_host_arvalid = cp_axi_host.arvalid;
	assign m_axi_host_araddr  = cp_axi_host.araddr;
	assign m_axi_host_arid    = {{(C_M_AXI_MEM_ID_WIDTH-`VX_CP_AXI_TID_WIDTH){1'b0}}, cp_axi_host.arid};
	assign m_axi_host_arlen   = cp_axi_host.arlen;
	assign cp_axi_host.arready = m_axi_host_arready;
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

	// P5: the AFU interrupt pin reflects the Command Processor — a one-cycle
	// pulse each time the CP retires a command.
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

	for (genvar i = 0; i < C_M_AXI_MEM_NUM_BANKS; ++i) begin : g_addressing
		assign m_axi_mem_awaddr_a[i] = C_M_AXI_MEM_ADDR_WIDTH'(m_axi_mem_awaddr_u[i]) + C_M_AXI_MEM_ADDR_WIDTH'(`PLATFORM_MEMORY_OFFSET);
		assign m_axi_mem_araddr_a[i] = C_M_AXI_MEM_ADDR_WIDTH'(m_axi_mem_araddr_u[i]) + C_M_AXI_MEM_ADDR_WIDTH'(`PLATFORM_MEMORY_OFFSET);
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

	// ---- Banks 1..N-1: direct passthrough ----
	for (genvar i = 1; i < C_M_AXI_MEM_NUM_BANKS; ++i) begin : g_bank_passthrough
		assign m_axi_mem_awvalid_a[i] = vx_awvalid_a[i];
		assign m_axi_mem_awaddr_u[i]  = vx_awaddr_a[i];
		assign m_axi_mem_awid_a[i]    = vx_awid_a[i];
		assign m_axi_mem_awlen_a[i]   = vx_awlen_a[i];
		assign vx_awready_a[i]        = m_axi_mem_awready_a[i];

		assign m_axi_mem_wvalid_a[i]  = vx_wvalid_a[i];
		assign m_axi_mem_wdata_a[i]   = vx_wdata_a[i];
		assign m_axi_mem_wstrb_a[i]   = vx_wstrb_a[i];
		assign m_axi_mem_wlast_a[i]   = vx_wlast_a[i];
		assign vx_wready_a[i]         = m_axi_mem_wready_a[i];

		assign vx_bvalid_a[i]         = m_axi_mem_bvalid_a[i];
		assign vx_bid_a[i]            = m_axi_mem_bid_a[i];
		assign vx_bresp_a[i]          = m_axi_mem_bresp_a[i];
		assign m_axi_mem_bready_a[i]  = vx_bready_a[i];

		assign m_axi_mem_arvalid_a[i] = vx_arvalid_a[i];
		assign m_axi_mem_araddr_u[i]  = vx_araddr_a[i];
		assign m_axi_mem_arid_a[i]    = vx_arid_a[i];
		assign m_axi_mem_arlen_a[i]   = vx_arlen_a[i];
		assign vx_arready_a[i]        = m_axi_mem_arready_a[i];

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

	// Drop the platform offset from the CP address so the arbiter's slave
	// port sees an offset-relative bank-0 address (matches vx_awaddr_a[0]).
	wire [M_AXI_MEM_ADDR_WIDTH-1:0] cp_awaddr_offset =
	    M_AXI_MEM_ADDR_WIDTH'(cp_axi_dev.awaddr - `PLATFORM_MEMORY_OFFSET);
	wire [M_AXI_MEM_ADDR_WIDTH-1:0] cp_araddr_offset =
	    M_AXI_MEM_ADDR_WIDTH'(cp_axi_dev.araddr - `PLATFORM_MEMORY_OFFSET);

	VX_axi_arb2 #(
		.ADDR_W (M_AXI_MEM_ADDR_WIDTH),
		.DATA_W (C_M_AXI_MEM_DATA_WIDTH),
		.ID_W   (C_M_AXI_MEM_ID_WIDTH)
	) bank0_arb (
		.clk        (clk),
		.reset      (reset),

		.s0_awvalid (vx_awvalid_a[0]),  .s0_awready (vx_awready_a[0]),
		.s0_awaddr  (vx_awaddr_a[0]),   .s0_awid    (vx_awid_a[0]),
		.s0_awlen   (vx_awlen_a[0]),
		.s0_wvalid  (vx_wvalid_a[0]),   .s0_wready  (vx_wready_a[0]),
		.s0_wdata   (vx_wdata_a[0]),    .s0_wstrb   (vx_wstrb_a[0]),
		.s0_wlast   (vx_wlast_a[0]),
		.s0_bvalid  (vx_bvalid_a[0]),   .s0_bready  (vx_bready_a[0]),
		.s0_bid     (vx_bid_a[0]),      .s0_bresp   (vx_bresp_a[0]),
		.s0_arvalid (vx_arvalid_a[0]),  .s0_arready (vx_arready_a[0]),
		.s0_araddr  (vx_araddr_a[0]),   .s0_arid    (vx_arid_a[0]),
		.s0_arlen   (vx_arlen_a[0]),
		.s0_rvalid  (vx_rvalid_a[0]),   .s0_rready  (vx_rready_a[0]),
		.s0_rdata   (vx_rdata_a[0]),    .s0_rlast   (vx_rlast_a[0]),
		.s0_rid     (vx_rid_a[0]),      .s0_rresp   (vx_rresp_a[0]),

		.s1_awvalid (cp_axi_dev.awvalid), .s1_awready (cp_axi_dev.awready),
		.s1_awaddr  (cp_awaddr_offset), .s1_awid    (cp_awid_padded),
		.s1_awlen   (cp_axi_dev.awlen),
		.s1_wvalid  (cp_axi_dev.wvalid),  .s1_wready  (cp_axi_dev.wready),
		.s1_wdata   (cp_axi_dev.wdata),   .s1_wstrb   (cp_axi_dev.wstrb),
		.s1_wlast   (cp_axi_dev.wlast),
		.s1_bvalid  (cp_axi_dev.bvalid),  .s1_bready  (cp_axi_dev.bready),
		.s1_bid     (cp_axi_dev_bid_full),.s1_bresp   (cp_axi_dev.bresp),
		.s1_arvalid (cp_axi_dev.arvalid), .s1_arready (cp_axi_dev.arready),
		.s1_araddr  (cp_araddr_offset), .s1_arid    (cp_arid_padded),
		.s1_arlen   (cp_axi_dev.arlen),
		.s1_rvalid  (cp_axi_dev.rvalid),  .s1_rready  (cp_axi_dev.rready),
		.s1_rdata   (cp_axi_dev.rdata),   .s1_rlast   (cp_axi_dev.rlast),
		.s1_rid     (cp_axi_dev_rid_full),.s1_rresp   (cp_axi_dev.rresp),

		.m_awvalid  (m_axi_mem_awvalid_a[0]), .m_awready (m_axi_mem_awready_a[0]),
		.m_awaddr   (m_axi_mem_awaddr_u[0]),  .m_awid    (m_axi_mem_awid_a[0]),
		.m_awlen    (m_axi_mem_awlen_a[0]),
		.m_wvalid   (m_axi_mem_wvalid_a[0]),  .m_wready  (m_axi_mem_wready_a[0]),
		.m_wdata    (m_axi_mem_wdata_a[0]),   .m_wstrb   (m_axi_mem_wstrb_a[0]),
		.m_wlast    (m_axi_mem_wlast_a[0]),
		.m_bvalid   (m_axi_mem_bvalid_a[0]),  .m_bready  (m_axi_mem_bready_a[0]),
		.m_bid      (m_axi_mem_bid_a[0]),     .m_bresp   (m_axi_mem_bresp_a[0]),
		.m_arvalid  (m_axi_mem_arvalid_a[0]), .m_arready (m_axi_mem_arready_a[0]),
		.m_araddr   (m_axi_mem_araddr_u[0]),  .m_arid    (m_axi_mem_arid_a[0]),
		.m_arlen    (m_axi_mem_arlen_a[0]),
		.m_rvalid   (m_axi_mem_rvalid_a[0]),  .m_rready  (m_axi_mem_rready_a[0]),
		.m_rdata    (m_axi_mem_rdata_a[0]),   .m_rlast   (m_axi_mem_rlast_a[0]),
		.m_rid      (m_axi_mem_rid_a[0]),     .m_rresp   (m_axi_mem_rresp_a[0])
	);

	// Truncate the arbiter's wider ID back to CP's narrower native ID width.
	wire [C_M_AXI_MEM_ID_WIDTH-1:0] cp_axi_dev_bid_full;
	wire [C_M_AXI_MEM_ID_WIDTH-1:0] cp_axi_dev_rid_full;
	assign cp_axi_dev.bid = cp_axi_dev_bid_full[`VX_CP_AXI_TID_WIDTH-1:0];
	assign cp_axi_dev.rid = cp_axi_dev_rid_full[`VX_CP_AXI_TID_WIDTH-1:0];
	`UNUSED_VAR (cp_axi_dev_bid_full)
	`UNUSED_VAR (cp_axi_dev_rid_full)

	// The optional AXI4 sideband signals (size/burst) are unused by the
	// reduced VX_axi_arb2 view — pin them sink-side so lint stays clean.
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
		if (reset) begin
			assert_delay_ctr <= '0;
			assert_enabled   <= 0;
		end else begin
			if (~assert_enabled) begin
				if (assert_delay_ctr == (`VX_CFG_RESET_DELAY-1)) begin
					assert_enabled <= 1;
					$asserton(0, vortex_axi); // enable assertions
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
