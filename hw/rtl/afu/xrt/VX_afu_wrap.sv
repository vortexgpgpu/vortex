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

`include "vortex_afu.vh"

module VX_afu_wrap #(
	  parameter C_S_AXI_CTRL_ADDR_WIDTH = 8,
	  parameter C_S_AXI_CTRL_DATA_WIDTH	= 32,
	  parameter C_M_AXI_MEM_ID_WIDTH    = `M_AXI_MEM_ID_WIDTH,
	  parameter C_M_AXI_MEM_ADDR_WIDTH  = `MEM_ADDR_WIDTH,
	  parameter C_M_AXI_MEM_DATA_WIDTH  = `VX_MEM_DATA_WIDTH
) (
    // System signals
    input wire ap_clk,
    input wire ap_rst_n,

    // AXI4 master interface
	`REPEAT (`M_AXI_MEM_NUM_BANKS, GEN_AXI_MEM, REPEAT_COMMA),

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
	localparam C_M_AXI_MEM_NUM_BANKS = `M_AXI_MEM_NUM_BANKS;

	localparam STATE_IDLE = 0;
    localparam STATE_RUN  = 1;

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
	`REPEAT (`M_AXI_MEM_NUM_BANKS, AXI_MEM_TO_ARRAY, REPEAT_SEMICOLON);

	wire reset = ~ap_rst_n;

	reg [`CLOG2(`RESET_DELAY+1)-1:0] vx_reset_ctr;
	reg [15:0] vx_pending_writes;
	reg vx_busy_wait;
	reg vx_running;

	wire vx_busy;

	wire [63:0] mem_base [C_M_AXI_MEM_NUM_BANKS];

	wire                          dcr_wr_valid;
	wire [`VX_DCR_ADDR_WIDTH-1:0] dcr_wr_addr;
	wire [`VX_DCR_DATA_WIDTH-1:0] dcr_wr_data;

	reg state;

	wire ap_reset;
	wire ap_start;
	wire ap_idle  = ~vx_running;
	wire ap_done  = ~(state == STATE_RUN || vx_pending_writes != 0);
	wire ap_ready = 1'b1;

`ifdef SCOPE
	wire scope_bus_in;
	wire scope_bus_out;
  	wire scope_reset = reset;
`endif

	always @(posedge ap_clk) begin
		if (reset || ap_reset) begin
			state <= STATE_IDLE;
			vx_busy_wait  <= 0;
			vx_running    <= 0;
		end else begin
			case (state)
			STATE_IDLE: begin
				if (ap_start) begin
				`ifdef DBG_TRACE_AFU
					`TRACE(2, ("%d: STATE RUN\n", $time));
				`endif
					state <= STATE_RUN;
					vx_running <= 0;
				end
			end
			STATE_RUN: begin
				if (vx_running) begin
					if (vx_busy_wait) begin
						// wait until processor goes busy
						if (vx_busy) begin
							vx_busy_wait <= 0;
						end
					end else begin
						// wait until the processor is not busy
						if (~vx_busy) begin
							state <= STATE_IDLE;
						`ifdef DBG_TRACE_AFU
							`TRACE(2, ("%d: AFU: End execution\n", $time));
							`TRACE(2, ("%d: STATE IDLE\n", $time));
						`endif
						end
					end
				end else begin
					// wait until the reset sequence is complete
					if (vx_reset_ctr == (`RESET_DELAY-1)) begin
					`ifdef DBG_TRACE_AFU
						`TRACE(2, ("%d: AFU: Begin execution\n", $time));
					`endif
						vx_running    <= 1;
						vx_busy_wait  <= 1;
					end
				end
			end
			endcase
		end
	end

	reg m_axi_mem_wfire;
	reg m_axi_mem_bfire;

	always @(*) begin
		m_axi_mem_wfire = 0;
		m_axi_mem_bfire = 0;
		for (integer i = 0; i < C_M_AXI_MEM_NUM_BANKS; ++i) begin
			m_axi_mem_wfire |= m_axi_mem_wvalid_a[i] && m_axi_mem_wready_a[i];
			m_axi_mem_bfire |= m_axi_mem_bvalid_a[i] && m_axi_mem_bready_a[i];
		end
	end

	always @(posedge ap_clk) begin
		if (reset || ap_reset) begin
			vx_pending_writes <= '0;
		end else begin
			if (m_axi_mem_wfire && ~m_axi_mem_bfire)
				vx_pending_writes <= vx_pending_writes + 1;
			if (~m_axi_mem_wfire && m_axi_mem_bfire)
				vx_pending_writes <= vx_pending_writes - 1;
		end
	end

	always @(posedge ap_clk) begin
		if (state == STATE_RUN) begin
			vx_reset_ctr <= vx_reset_ctr + 1;
		end else begin
			vx_reset_ctr <= '0;
		end
	end

	VX_afu_ctrl #(
		.AXI_ADDR_WIDTH (C_S_AXI_CTRL_ADDR_WIDTH),
		.AXI_DATA_WIDTH (C_S_AXI_CTRL_DATA_WIDTH),
		.AXI_NUM_BANKS  (C_M_AXI_MEM_NUM_BANKS)
	) afu_ctrl (
		.clk       		(ap_clk),
		.reset     		(reset || ap_reset),
		.clk_en         (1'b1),

		.s_axi_awvalid  (s_axi_ctrl_awvalid),
		.s_axi_awready  (s_axi_ctrl_awready),
		.s_axi_awaddr   (s_axi_ctrl_awaddr),
		.s_axi_wvalid   (s_axi_ctrl_wvalid),
		.s_axi_wready   (s_axi_ctrl_wready),
		.s_axi_wdata    (s_axi_ctrl_wdata),
		.s_axi_wstrb    (s_axi_ctrl_wstrb),
		.s_axi_arvalid  (s_axi_ctrl_arvalid),
		.s_axi_arready  (s_axi_ctrl_arready),
		.s_axi_araddr   (s_axi_ctrl_araddr),
		.s_axi_rvalid   (s_axi_ctrl_rvalid),
		.s_axi_rready   (s_axi_ctrl_rready),
		.s_axi_rdata    (s_axi_ctrl_rdata),
		.s_axi_rresp    (s_axi_ctrl_rresp),
		.s_axi_bvalid   (s_axi_ctrl_bvalid),
		.s_axi_bready   (s_axi_ctrl_bready),
		.s_axi_bresp    (s_axi_ctrl_bresp),

		.ap_reset  		(ap_reset),
		.ap_start  		(ap_start),
		.ap_done     	(ap_done),
		.ap_ready       (ap_ready),
		.ap_idle     	(ap_idle),
		.interrupt 		(interrupt),

	`ifdef SCOPE
		.scope_bus_in   (scope_bus_out),
		.scope_bus_out  (scope_bus_in),
	`endif

		.mem_base       (mem_base),

		.dcr_wr_valid	(dcr_wr_valid),
		.dcr_wr_addr	(dcr_wr_addr),
		.dcr_wr_data	(dcr_wr_data)
	);

	wire [`MEM_ADDR_WIDTH-1:0] m_axi_mem_awaddr_w [C_M_AXI_MEM_NUM_BANKS];
	wire [`MEM_ADDR_WIDTH-1:0] m_axi_mem_araddr_w [C_M_AXI_MEM_NUM_BANKS];

	for (genvar i = 0; i < C_M_AXI_MEM_NUM_BANKS; ++i) begin
		assign m_axi_mem_awaddr_a[i] = C_M_AXI_MEM_ADDR_WIDTH'(m_axi_mem_awaddr_w[i]) + C_M_AXI_MEM_ADDR_WIDTH'(mem_base[i]);
		assign m_axi_mem_araddr_a[i] = C_M_AXI_MEM_ADDR_WIDTH'(m_axi_mem_araddr_w[i]) + C_M_AXI_MEM_ADDR_WIDTH'(mem_base[i]);
	end

	`SCOPE_IO_SWITCH (2)

	Vortex_axi #(
		.AXI_DATA_WIDTH (C_M_AXI_MEM_DATA_WIDTH),
		.AXI_ADDR_WIDTH (`MEM_ADDR_WIDTH),
		.AXI_TID_WIDTH  (C_M_AXI_MEM_ID_WIDTH),
		.AXI_NUM_BANKS  (C_M_AXI_MEM_NUM_BANKS)
	) vortex_axi (
		`SCOPE_IO_BIND  (1)

		.clk			(ap_clk),
		.reset			(reset || ap_reset || ~vx_running),

		.m_axi_awvalid	(m_axi_mem_awvalid_a),
		.m_axi_awready	(m_axi_mem_awready_a),
		.m_axi_awaddr	(m_axi_mem_awaddr_w),
		.m_axi_awid		(m_axi_mem_awid_a),
		.m_axi_awlen    (m_axi_mem_awlen_a),
		`UNUSED_PIN (m_axi_awsize),
		`UNUSED_PIN (m_axi_awburst),
		`UNUSED_PIN (m_axi_awlock),
		`UNUSED_PIN (m_axi_awcache),
		`UNUSED_PIN (m_axi_awprot),
		`UNUSED_PIN (m_axi_awqos),
    	`UNUSED_PIN (m_axi_awregion),

		.m_axi_wvalid	(m_axi_mem_wvalid_a),
		.m_axi_wready	(m_axi_mem_wready_a),
		.m_axi_wdata	(m_axi_mem_wdata_a),
		.m_axi_wstrb	(m_axi_mem_wstrb_a),
		.m_axi_wlast	(m_axi_mem_wlast_a),

		.m_axi_bvalid	(m_axi_mem_bvalid_a),
		.m_axi_bready	(m_axi_mem_bready_a),
		.m_axi_bid		(m_axi_mem_bid_a),
		.m_axi_bresp	(m_axi_mem_bresp_a),

		.m_axi_arvalid	(m_axi_mem_arvalid_a),
		.m_axi_arready	(m_axi_mem_arready_a),
		.m_axi_araddr	(m_axi_mem_araddr_w),
		.m_axi_arid		(m_axi_mem_arid_a),
		.m_axi_arlen	(m_axi_mem_arlen_a),
		`UNUSED_PIN (m_axi_arsize),
		`UNUSED_PIN (m_axi_arburst),
		`UNUSED_PIN (m_axi_arlock),
		`UNUSED_PIN (m_axi_arcache),
		`UNUSED_PIN (m_axi_arprot),
		`UNUSED_PIN (m_axi_arqos),
        `UNUSED_PIN (m_axi_arregion),

		.m_axi_rvalid	(m_axi_mem_rvalid_a),
		.m_axi_rready	(m_axi_mem_rready_a),
		.m_axi_rdata	(m_axi_mem_rdata_a),
		.m_axi_rlast	(m_axi_mem_rlast_a),
		.m_axi_rid    	(m_axi_mem_rid_a),
		.m_axi_rresp	(m_axi_mem_rresp_a),

		.dcr_wr_valid	(dcr_wr_valid),
		.dcr_wr_addr	(dcr_wr_addr),
		.dcr_wr_data	(dcr_wr_data),

		.busy			(vx_busy)
	);

    // SCOPE //////////////////////////////////////////////////////////////////////

`ifdef DBG_SCOPE_AFU
	`define TRIGGERS { \
		reset, \
		ap_start, \
		ap_done, \
		ap_idle, \
		interrupt, \
		vx_busy_wait, \
		vx_busy, \
		vx_running \
	}

	`define PROBES { \
		vx_pending_writes \
	}

    VX_scope_tap #(
        .SCOPE_ID (0),
        .TRIGGERW ($bits(`TRIGGERS)),
        .PROBEW ($bits(`PROBES))
    ) scope_tap (
        .clk (clk),
        .reset (scope_reset_w[0]),
        .start (1'b0),
        .stop (1'b0),
        .triggers (`TRIGGERS),
        .probes (`PROBES),
        .bus_in (scope_bus_in_w[0]),
        .bus_out (scope_bus_out_w[0])
    );
`else
    `SCOPE_IO_UNUSED_W(0)
`endif

`ifdef SIMULATION
`ifndef VERILATOR
	// disable assertions until full reset
	reg [`CLOG2(`RESET_DELAY+1)-1:0] assert_delay_ctr;
	reg assert_enabled;
	initial begin
		$assertoff(0, vortex_axi);
	end
	always @(posedge ap_clk) begin
		if (reset) begin
			assert_delay_ctr <= '0;
			assert_enabled   <= 0;
		end else begin
			if (~assert_enabled) begin
				if (assert_delay_ctr == (`RESET_DELAY-1)) begin
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
    always @(posedge ap_clk) begin
		for (integer i = 0; i < C_M_AXI_MEM_NUM_BANKS; ++i) begin
			if (m_axi_mem_awvalid_a[i] && m_axi_mem_awready_a[i]) begin
				`TRACE(2, ("%d: AFU Wr Req [%0d]: addr=0x%0h, tag=0x%0h\n", $time, i, m_axi_mem_awaddr_a[i], m_axi_mem_awid_a[i]));
			end
			if (m_axi_mem_wvalid_a[i] && m_axi_mem_wready_a[i]) begin
				`TRACE(2, ("%d: AFU Wr Req [%0d]: data=0x%h\n", $time, i, m_axi_mem_wdata_a[i]));
			end
			if (m_axi_mem_arvalid_a[i] && m_axi_mem_arready_a[i]) begin
				`TRACE(2, ("%d: AFU Rd Req [%0d]: addr=0x%0h, tag=0x%0h\n", $time, i, m_axi_mem_araddr_a[i], m_axi_mem_arid_a[i]));
			end
			if (m_axi_mem_rvalid_a[i] && m_axi_mem_rready_a[i]) begin
				`TRACE(2, ("%d: AVS Rd Rsp [%0d]: data=0x%h, tag=0x%0h\n", $time, i, m_axi_mem_rdata_a[i], m_axi_mem_rid_a[i]));
			end
		end
  	end
`endif

endmodule
