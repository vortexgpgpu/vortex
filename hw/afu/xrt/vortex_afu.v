`include "VX_define.vh"

`define GEN_AXI_MEM(i) \
	output wire                                 m_axi_mem``i``_awvalid, \
	input  wire                                 m_axi_mem``i``_awready, \
	output wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0]   m_axi_mem``i``_awaddr, \
	output wire [C_M_AXI_GMEM_ID_WIDTH - 1:0]   m_axi_mem``i``_awid, \
	output wire [7:0]                           m_axi_mem``i``_awlen, \
	output wire [2:0]                           m_axi_mem``i``_awsize, \
	output wire [1:0]                           m_axi_mem``i``_awburst, \
	output wire [1:0]                           m_axi_mem``i``_awlock, \
	output wire [3:0]                           m_axi_mem``i``_awcache, \
	output wire [2:0]                           m_axi_mem``i``_awprot, \
	output wire [3:0]                           m_axi_mem``i``_awqos, \
	output wire [3:0]                           m_axi_mem``i``_awregion, \
	output wire                                 m_axi_mem``i``_wvalid, \
	input  wire                                 m_axi_mem``i``_wready, \
	output wire [C_M_AXI_GMEM_DATA_WIDTH-1:0]   m_axi_mem``i``_wdata, \
	output wire [C_M_AXI_GMEM_DATA_WIDTH/8-1:0] m_axi_mem``i``_wstrb, \
	output wire                                 m_axi_mem``i``_wlast, \
	output wire                                 m_axi_mem``i``_arvalid, \
	input  wire                                 m_axi_mem``i``_arready, \
	output wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0]   m_axi_mem``i``_araddr, \
	output wire [C_M_AXI_GMEM_ID_WIDTH-1:0]     m_axi_mem``i``_arid, \
	output wire [7:0]                           m_axi_mem``i``_arlen, \
	output wire [2:0]                           m_axi_mem``i``_arsize, \
	output wire [1:0]                           m_axi_mem``i``_arburst, \
	output wire [1:0]                           m_axi_mem``i``_arlock, \
	output wire [3:0]                           m_axi_mem``i``_arcache, \
	output wire [2:0]                           m_axi_mem``i``_arprot, \
	output wire [3:0]                           m_axi_mem``i``_arqos, \
	output wire [3:0]                           m_axi_mem``i``_arregion, \
	input  wire                                 m_axi_mem``i``_rvalid, \
	output wire                                 m_axi_mem``i``_rready, \
	input  wire [C_M_AXI_GMEM_DATA_WIDTH - 1:0] m_axi_mem``i``_rdata, \
	input  wire                                 m_axi_mem``i``_rlast, \
	input  wire [C_M_AXI_GMEM_ID_WIDTH - 1:0]   m_axi_mem``i``_rid, \
	input  wire [1:0]                           m_axi_mem``i``_rresp, \
	input  wire                                 m_axi_mem``i``_bvalid, \
	output wire                                 m_axi_mem``i``_bready, \
	input  wire [1:0]                           m_axi_mem``i``_bresp, \
	input  wire [C_M_AXI_GMEM_ID_WIDTH - 1:0]   m_axi_mem``i``_bid

`define SEP ,

module vortex_afu #(
	parameter C_S_AXI_CONTROL_ADDR_WIDTH = 6,
	parameter C_S_AXI_CONTROL_DATA_WIDTH = 32,
	parameter C_M_AXI_GMEM_ID_WIDTH      = 12,
	parameter C_M_AXI_GMEM_ADDR_WIDTH    = 64,
	parameter C_M_AXI_GMEM_DATA_WIDTH    = `VX_MEM_DATA_WIDTH
) (
	// System signals
	input wire 									ap_clk,
	input wire 									ap_rst_n,

	//`REPEAT (4, GEN_AXI_MEM, SEP),
	
	// AXI4 memory interface 
	output wire                                 m_axi_mem_awvalid,
	input  wire                                 m_axi_mem_awready,
	output wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0]   m_axi_mem_awaddr,
	output wire [C_M_AXI_GMEM_ID_WIDTH - 1:0]   m_axi_mem_awid,
	output wire [7:0]                           m_axi_mem_awlen,
	output wire [2:0]                           m_axi_mem_awsize,
	output wire [1:0]                           m_axi_mem_awburst,
	output wire [1:0]                           m_axi_mem_awlock,
	output wire [3:0]                           m_axi_mem_awcache,
	output wire [2:0]                           m_axi_mem_awprot,
	output wire [3:0]                           m_axi_mem_awqos,
	output wire [3:0]                           m_axi_mem_awregion,
	output wire                                 m_axi_mem_wvalid,
	input  wire                                 m_axi_mem_wready,
	output wire [C_M_AXI_GMEM_DATA_WIDTH-1:0]   m_axi_mem_wdata,
	output wire [C_M_AXI_GMEM_DATA_WIDTH/8-1:0] m_axi_mem_wstrb,
	output wire                                 m_axi_mem_wlast,
	output wire                                 m_axi_mem_arvalid,
	input  wire                                 m_axi_mem_arready,
	output wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0]   m_axi_mem_araddr,
	output wire [C_M_AXI_GMEM_ID_WIDTH-1:0]     m_axi_mem_arid,
	output wire [7:0]                           m_axi_mem_arlen,
	output wire [2:0]                           m_axi_mem_arsize,
	output wire [1:0]                           m_axi_mem_arburst,
	output wire [1:0]                           m_axi_mem_arlock,
	output wire [3:0]                           m_axi_mem_arcache,
	output wire [2:0]                           m_axi_mem_arprot,
	output wire [3:0]                           m_axi_mem_arqos,
	output wire [3:0]                           m_axi_mem_arregion,
	input  wire                                 m_axi_mem_rvalid,
	output wire                                 m_axi_mem_rready,
	input  wire [C_M_AXI_GMEM_DATA_WIDTH - 1:0] m_axi_mem_rdata,
	input  wire                                 m_axi_mem_rlast,
	input  wire [C_M_AXI_GMEM_ID_WIDTH - 1:0]   m_axi_mem_rid,
	input  wire [1:0]                           m_axi_mem_rresp,
	input  wire                                 m_axi_mem_bvalid,
	output wire                                 m_axi_mem_bready,
	input  wire [1:0]                           m_axi_mem_bresp,
	input  wire [C_M_AXI_GMEM_ID_WIDTH - 1:0]	m_axi_mem_bid,

	// AXI4-Lite control interface
	input  wire                                 s_axi_ctrl_awvalid,
	output wire                                 s_axi_ctrl_awready,
	input  wire [C_S_AXI_CONTROL_ADDR_WIDTH-1:0] s_axi_ctrl_awaddr,
	input  wire                                	s_axi_ctrl_wvalid,
	output wire                                 s_axi_ctrl_wready,
	input  wire [C_S_AXI_CONTROL_DATA_WIDTH-1:0] s_axi_ctrl_wdata,
	input  wire [C_S_AXI_CONTROL_DATA_WIDTH/8-1:0] s_axi_ctrl_wstrb,
	input  wire                                	s_axi_ctrl_arvalid,
	output wire                                 s_axi_ctrl_arready,
	input  wire [C_S_AXI_CONTROL_ADDR_WIDTH-1:0] s_axi_ctrl_araddr,
	output wire                               	s_axi_ctrl_rvalid,
	input  wire                                 s_axi_ctrl_rready,
	output wire [C_S_AXI_CONTROL_DATA_WIDTH-1:0] s_axi_ctrl_rdata,
	output wire [1:0]                          	s_axi_ctrl_rresp,
	output wire                                 s_axi_ctrl_bvalid,
	input  wire                                 s_axi_ctrl_bready,
	output wire [1:0]                           s_axi_ctrl_bresp,
	
  	output wire                                 interrupt
);

    // Register and invert reset signal.
	reg reset;
	always @(posedge ap_clk) begin
		reset <= ~ap_rst_n;
	end

	reg  vx_reset;
	wire vx_busy;

	wire                          dcr_wr_valid;
    wire [`VX_DCR_ADDR_WIDTH-1:0] dcr_wr_addr;
    wire [`VX_DCR_DATA_WIDTH-1:0] dcr_wr_data;
	
    wire ap_start;
	wire ap_idle  = ~vx_busy;
	wire ap_done  = ap_idle;
	wire ap_ready = ap_idle;

	reg [$clog2(`RESET_DELAY+1)-1:0] vx_reset_ctr;
	reg vx_running;

	always @(posedge ap_clk) begin
		if (~vx_running && vx_reset == 0 && ap_start) begin
			vx_reset_ctr <= 0;
		end else begin
			vx_reset_ctr <= vx_reset_ctr + 1;
		end
	end

	always @(posedge ap_clk) begin
		if (reset) begin
			vx_reset   <= 0;
			vx_running <= 0;
		end else begin			
			if (vx_running) begin
				if (~vx_busy) begin
					vx_running <= 0;
				end
			end else begin
				if (vx_reset == 0 && ap_start) begin
					vx_reset <= 1;
				end
				if (vx_reset_ctr == (`RESET_DELAY-1)) begin
					vx_running <= 1;
					vx_reset   <= 0;
				end
			end
		end
	end

	VX_afu_control #(
		.AXI_ADDR_WIDTH (C_S_AXI_CONTROL_ADDR_WIDTH),
		.AXI_DATA_WIDTH (C_S_AXI_CONTROL_DATA_WIDTH)
	) afu_control (
		.clk       		(ap_clk),
		.reset     		(reset),	
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

		.ap_start  		(ap_start),		
		.ap_done     	(ap_done),
		.ap_ready     	(ap_ready),
		.ap_idle     	(ap_idle),
		.interrupt 		(interrupt),	

		.dcr_wr_valid	(dcr_wr_valid),
		.dcr_wr_addr	(dcr_wr_addr),
		.dcr_wr_data	(dcr_wr_data)		
	);

	wire m_axi_mem_awvalid_unqual;
	wire m_axi_mem_wvalid_unqual;
	wire m_axi_mem_arvalid_unqual;

	assign m_axi_mem_awvalid = m_axi_mem_awvalid_unqual && vx_running;
	assign m_axi_mem_wvalid  = m_axi_mem_wvalid_unqual  && vx_running;
	assign m_axi_mem_arvalid = m_axi_mem_arvalid_unqual && vx_running;

	wire [`VX_MEM_ADDR_WIDTH-1:0] m_axi_mem_awaddr_unqual;
	wire [`VX_MEM_ADDR_WIDTH-1:0] m_axi_mem_araddr_unqual;

	reg [C_M_AXI_GMEM_ADDR_WIDTH-1:0] m_axi_mem_awaddr_r;
	reg [C_M_AXI_GMEM_ADDR_WIDTH-1:0] m_axi_mem_araddr_r;

	always @(*) begin
		m_axi_mem_awaddr_r = 0;
		m_axi_mem_awaddr_r[`VX_MEM_ADDR_WIDTH-1:0] = m_axi_mem_awaddr_unqual;

		m_axi_mem_araddr_r = 0;
		m_axi_mem_araddr_r[`VX_MEM_ADDR_WIDTH-1:0] = m_axi_mem_araddr_unqual;
	end

	assign m_axi_mem_awaddr = m_axi_mem_awaddr_r;
	assign m_axi_mem_araddr = m_axi_mem_araddr_r;

	Vortex_axi #(
		.AXI_DATA_WIDTH (C_M_AXI_GMEM_DATA_WIDTH),
		.AXI_ADDR_WIDTH (`VX_MEM_ADDR_WIDTH),
		.AXI_TID_WIDTH  (C_M_AXI_GMEM_ID_WIDTH)
	) vortex_axi (
		.clk			(ap_clk),
		.reset			(reset || vx_reset),
		
		.m_axi_awvalid	(m_axi_mem_awvalid_unqual),
		.m_axi_awready	(m_axi_mem_awready),
		.m_axi_awaddr	(m_axi_mem_awaddr_unqual),
		.m_axi_awid		(m_axi_mem_awid),
		.m_axi_awlen	(m_axi_mem_awlen),
		.m_axi_awsize	(m_axi_mem_awsize),
		.m_axi_awburst	(m_axi_mem_awburst),
		.m_axi_awlock	(m_axi_mem_awlock),
		.m_axi_awcache	(m_axi_mem_awcache),
		.m_axi_awprot	(m_axi_mem_awprot),
		.m_axi_awqos	(m_axi_mem_awqos),		  
        .m_axi_awregion (m_axi_mem_awregion),

		.m_axi_wvalid	(m_axi_mem_wvalid_unqual),
		.m_axi_wready	(m_axi_mem_wready),
		.m_axi_wdata	(m_axi_mem_wdata),
		.m_axi_wstrb	(m_axi_mem_wstrb),
		.m_axi_wlast	(m_axi_mem_wlast),

		.m_axi_bvalid	(m_axi_mem_bvalid),
		.m_axi_bready	(m_axi_mem_bready),
		.m_axi_bid		(m_axi_mem_bid),
		.m_axi_bresp	(m_axi_mem_bresp),		

		.m_axi_arvalid	(m_axi_mem_arvalid_unqual),
		.m_axi_arready	(m_axi_mem_arready),
		.m_axi_araddr	(m_axi_mem_araddr_unqual),
		.m_axi_arid		(m_axi_mem_arid),
		.m_axi_arlen	(m_axi_mem_arlen),
		.m_axi_arsize	(m_axi_mem_arsize),
		.m_axi_arburst	(m_axi_mem_arburst),
		.m_axi_arlock	(m_axi_mem_arlock),
		.m_axi_arcache	(m_axi_mem_arcache),
		.m_axi_arprot	(m_axi_mem_arprot),
		.m_axi_arqos	(m_axi_mem_arqos),		
        .m_axi_arregion (m_axi_mem_arregion),

		.m_axi_rvalid	(m_axi_mem_rvalid),
		.m_axi_rready	(m_axi_mem_rready),
		.m_axi_rdata	(m_axi_mem_rdata),	
		.m_axi_rid    	(m_axi_mem_rid),
		.m_axi_rresp	(m_axi_mem_rresp),
		.m_axi_rlast	(m_axi_mem_rlast),

		.dcr_wr_valid	(dcr_wr_valid),
		.dcr_wr_addr	(dcr_wr_addr),
		.dcr_wr_data	(dcr_wr_data),

		.busy			(vx_busy)
	);
	
endmodule