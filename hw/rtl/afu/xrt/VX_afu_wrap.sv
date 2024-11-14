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
	  parameter C_M_AXI_MEM_ID_WIDTH    = 32,
	  parameter C_M_AXI_MEM_DATA_WIDTH  = 512,
	  parameter C_M_AXI_MEM_ADDR_WIDTH  = 25,
      parameter C_M_AXI_MEM_NUM_BANKS   = 2
) (
    // System signals
    input wire clk,
    input wire reset,

    // AXI4 master interface
`ifdef PLATFORM_MERGED_MEMORY_INTERFACE
	`REPEAT (1, GEN_AXI_MEM, REPEAT_COMMA),
`else
	`REPEAT (`PLATFORM_MEMORY_BANKS, GEN_AXI_MEM, REPEAT_COMMA),
`endif
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
`ifdef PLATFORM_MERGED_MEMORY_INTERFACE
	localparam M_AXI_MEM_ADDR_WIDTH = `PLATFORM_MEMORY_ADDR_WIDTH + $clog2(`PLATFORM_MEMORY_BANKS);
`else
	localparam M_AXI_MEM_ADDR_WIDTH = `PLATFORM_MEMORY_ADDR_WIDTH;
`endif

	localparam STATE_IDLE = 0;
    localparam STATE_RUN  = 1;

	localparam PENDING_SIZEW = 12; // max outstanding requests size
	localparam C_M_AXI_MEM_NUM_BANKS_SW = `CLOG2(C_M_AXI_MEM_NUM_BANKS+1);

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
	`REPEAT (1, AXI_MEM_TO_ARRAY, REPEAT_SEMICOLON);
`else
	`REPEAT (`PLATFORM_MEMORY_BANKS, AXI_MEM_TO_ARRAY, REPEAT_SEMICOLON);
`endif

	reg [`CLOG2(`RESET_DELAY+1)-1:0] vx_reset_ctr;
	reg [PENDING_SIZEW-1:0] vx_pending_writes;
	reg vx_busy_wait;
	reg vx_reset = 1; // asserted at initialization
	wire vx_busy;

	wire                          dcr_wr_valid;
	wire [`VX_DCR_ADDR_WIDTH-1:0] dcr_wr_addr;
	wire [`VX_DCR_DATA_WIDTH-1:0] dcr_wr_data;

	reg state;

	wire ap_reset;
	wire ap_start;
	wire ap_idle  = vx_reset;
	wire ap_done  = (state == STATE_IDLE) && (vx_pending_writes == '0);
	wire ap_ready = 1'b1;

`ifdef SCOPE
	wire scope_bus_in;
	wire scope_bus_out;
  	wire scope_reset = reset;
`endif

	always @(posedge clk) begin
		if (reset || ap_reset) begin
			state    <= STATE_IDLE;
			vx_reset <= 1;
		end else begin
			case (state)
			STATE_IDLE: begin
				if (ap_start) begin
				`ifdef DBG_TRACE_AFU
					`TRACE(2, ("%t: AFU: Goto STATE RUN\n", $time))
				`endif
					state <= STATE_RUN;
					vx_reset_ctr <= (`RESET_DELAY-1);
					vx_reset <= 1;
				end
			end
			STATE_RUN: begin
				if (vx_reset) begin
					// wait until the reset network is ready
					if (vx_reset_ctr == 0) begin
					`ifdef DBG_TRACE_AFU
						`TRACE(2, ("%t: AFU: Begin execution\n", $time))
					`endif
						vx_busy_wait <= 1;
						vx_reset <= 0;
					end
				end else begin
					if (vx_busy_wait) begin
						// wait until processor goes busy
						if (vx_busy) begin
							vx_busy_wait <= 0;
						end
					end else begin
						// wait until the processor is not busy
						if (~vx_busy) begin
						`ifdef DBG_TRACE_AFU
							`TRACE(2, ("%t: AFU: End execution\n", $time))
                            `TRACE(2, ("%t: AFU: Goto STATE IDLE\n", $time))
						`endif
							state <= STATE_IDLE;
						end
					end
				end
			end
			endcase

			// ensure reset network initialization
			if (vx_reset_ctr != '0) begin
				vx_reset_ctr <= vx_reset_ctr - 1;
			end
		end
	end

	wire [C_M_AXI_MEM_NUM_BANKS-1:0] m_axi_wr_req_fire, m_axi_wr_rsp_fire;
	wire [C_M_AXI_MEM_NUM_BANKS_SW-1:0] cur_wr_reqs, cur_wr_rsps;

	for (genvar i = 0; i < C_M_AXI_MEM_NUM_BANKS; ++i) begin : g_awfire
		VX_axi_write_ack axi_write_ack (
            .clk    (clk),
            .reset  (reset),
            .awvalid(m_axi_mem_awvalid_a[i]),
            .awready(m_axi_mem_awready_a[i]),
            .wvalid (m_axi_mem_wvalid_a[i]),
            .wready (m_axi_mem_wready_a[i]),
			.tx_ack (m_axi_wr_req_fire[i]),
			`UNUSED_PIN (aw_ack),
			`UNUSED_PIN (w_ack),
			`UNUSED_PIN (tx_rdy)
        );
		assign m_axi_wr_rsp_fire[i] = m_axi_mem_bvalid_a[i] & m_axi_mem_bready_a[i];
	end

	`POP_COUNT(cur_wr_reqs, m_axi_wr_req_fire);
	`POP_COUNT(cur_wr_rsps, m_axi_wr_rsp_fire);

	wire signed [C_M_AXI_MEM_NUM_BANKS_SW:0] reqs_sub = (C_M_AXI_MEM_NUM_BANKS_SW+1)'(cur_wr_reqs) -
	                                                     (C_M_AXI_MEM_NUM_BANKS_SW+1)'(cur_wr_rsps);

	always @(posedge clk) begin
		if (reset) begin
			vx_pending_writes <= '0;
		end else begin
			vx_pending_writes <= vx_pending_writes + PENDING_SIZEW'(reqs_sub);
		end
	end

	VX_afu_ctrl #(
		.S_AXI_ADDR_WIDTH (C_S_AXI_CTRL_ADDR_WIDTH),
		.S_AXI_DATA_WIDTH (C_S_AXI_CTRL_DATA_WIDTH)
	) afu_ctrl (
		.clk       		(clk),
		.reset     		(reset),

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

		.dcr_wr_valid	(dcr_wr_valid),
		.dcr_wr_addr	(dcr_wr_addr),
		.dcr_wr_data	(dcr_wr_data)
	);

	wire [M_AXI_MEM_ADDR_WIDTH-1:0] m_axi_mem_awaddr_u [C_M_AXI_MEM_NUM_BANKS];
	wire [M_AXI_MEM_ADDR_WIDTH-1:0] m_axi_mem_araddr_u [C_M_AXI_MEM_NUM_BANKS];

	for (genvar i = 0; i < C_M_AXI_MEM_NUM_BANKS; ++i) begin : g_addressing
		localparam [C_M_AXI_MEM_ADDR_WIDTH-1:0] BANK_OFFSET = C_M_AXI_MEM_ADDR_WIDTH'(`PLATFORM_MEMORY_OFFSET) + C_M_AXI_MEM_ADDR_WIDTH'(i) << M_AXI_MEM_ADDR_WIDTH;
		assign m_axi_mem_awaddr_a[i] = C_M_AXI_MEM_ADDR_WIDTH'(m_axi_mem_awaddr_u[i]) + BANK_OFFSET;
		assign m_axi_mem_araddr_a[i] = C_M_AXI_MEM_ADDR_WIDTH'(m_axi_mem_araddr_u[i]) + BANK_OFFSET;
	end

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

		.m_axi_awvalid	(m_axi_mem_awvalid_a),
		.m_axi_awready	(m_axi_mem_awready_a),
		.m_axi_awaddr	(m_axi_mem_awaddr_u),
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
		.m_axi_araddr	(m_axi_mem_araddr_u),
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

`ifdef SCOPE
`ifdef DBG_SCOPE_AFU
	wire m_axi_mem_awfire_0 = m_axi_mem_awvalid_a[0] & m_axi_mem_awready_a[0];
	wire m_axi_mem_arfire_0 = m_axi_mem_arvalid_a[0] & m_axi_mem_arready_a[0];
	wire m_axi_mem_wfire_0 = m_axi_mem_wvalid_a[0]  & m_axi_mem_wready_a[0];
	wire m_axi_mem_bfire_0  = m_axi_mem_bvalid_a[0]  & m_axi_mem_bready_a[0];

	`NEG_EDGE (reset_negedge, reset);
	`SCOPE_TAP (0, 0, {
			ap_reset,
			ap_start,
			ap_done,
			ap_idle,
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
			dcr_wr_valid,
			m_axi_mem_awfire_0,
			m_axi_mem_arfire_0,
			m_axi_mem_wfire_0,
			m_axi_mem_bfire_0
		},{
			dcr_wr_addr,
			dcr_wr_data,
			vx_pending_writes,
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
			ap_reset,
        	ap_start,
        	ap_done,
			ap_idle,
			interrupt
		}),
		.probe1 ({
        	vx_pending_writes,
			vx_busy_wait,
			vx_busy,
			vx_reset,
			dcr_wr_valid,
			dcr_wr_addr,
			dcr_wr_data
		})
    );
`endif
`endif

`ifdef SIMULATION
`ifndef VERILATOR
	// disable assertions until full reset
	reg [`CLOG2(`RESET_DELAY+1)-1:0] assert_delay_ctr;
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
    always @(posedge clk) begin
		for (integer i = 0; i < C_M_AXI_MEM_NUM_BANKS; ++i) begin
			if (m_axi_mem_awvalid_a[i] && m_axi_mem_awready_a[i]) begin
				`TRACE(2, ("%t: AXI Wr Req [%0d]: addr=0x%0h, tag=0x%0h\n", $time, i, m_axi_mem_awaddr_a[i], m_axi_mem_awid_a[i]))
			end
			if (m_axi_mem_wvalid_a[i] && m_axi_mem_wready_a[i]) begin
				`TRACE(2, ("%t: AXI Wr Req [%0d]: data=0x%h\n", $time, i, m_axi_mem_wdata_a[i]))
			end
			if (m_axi_mem_arvalid_a[i] && m_axi_mem_arready_a[i]) begin
				`TRACE(2, ("%t: AXI Rd Req [%0d]: addr=0x%0h, tag=0x%0h\n", $time, i, m_axi_mem_araddr_a[i], m_axi_mem_arid_a[i]))
			end
			if (m_axi_mem_rvalid_a[i] && m_axi_mem_rready_a[i]) begin
				`TRACE(2, ("%t: AXI Rd Rsp [%0d]: data=0x%h, tag=0x%0h\n", $time, i, m_axi_mem_rdata_a[i], m_axi_mem_rid_a[i]))
			end
		end
  	end
`endif

endmodule
