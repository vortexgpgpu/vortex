`include "VX_define.vh"

module vortex_afu #(
	parameter C_S_AXI_CONTROL_DATA_WIDTH = 32,
	parameter C_S_AXI_CONTROL_ADDR_WIDTH = 6,
	parameter C_M_AXI_GMEM_ID_WIDTH      = 10,
	parameter C_M_AXI_GMEM_ADDR_WIDTH    = `VX_MEM_ADDR_WIDTH,
	parameter C_M_AXI_GMEM_DATA_WIDTH    = `VX_MEM_DATA_WIDTH
) (
	// System signals
	input wire ap_clk,
	input wire ap_rst_n,
	
	// AXI4 memory interface 
	output wire                                 m_axi_gmem_awvalid,
	input  wire                                 m_axi_gmem_awready,
	output wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0]   m_axi_gmem_awaddr,
	output wire [C_M_AXI_GMEM_ID_WIDTH - 1:0]   m_axi_gmem_awid,
	output wire [7:0]                           m_axi_gmem_awlen,
	output wire [2:0]                           m_axi_gmem_awsize,
	output wire [1:0]                           m_axi_gmem_awburst,
	output wire [1:0]                           m_axi_gmem_awlock,
	output wire [3:0]                           m_axi_gmem_awcache,
	output wire [2:0]                           m_axi_gmem_awprot,
	output wire [3:0]                           m_axi_gmem_awqos,
	output wire [3:0]                           m_axi_gmem_awregion,
	output wire                                 m_axi_gmem_wvalid,
	input  wire                                 m_axi_gmem_wready,
	output wire [C_M_AXI_GMEM_DATA_WIDTH-1:0]   m_axi_gmem_wdata,
	output wire [C_M_AXI_GMEM_DATA_WIDTH/8-1:0] m_axi_gmem_wstrb,
	output wire                                 m_axi_gmem_wlast,
	output wire                                 m_axi_gmem_arvalid,
	input  wire                                 m_axi_gmem_arready,
	output wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0]   m_axi_gmem_araddr,
	output wire [C_M_AXI_GMEM_ID_WIDTH-1:0]     m_axi_gmem_arid,
	output wire [7:0]                           m_axi_gmem_arlen,
	output wire [2:0]                           m_axi_gmem_arsize,
	output wire [1:0]                           m_axi_gmem_arburst,
	output wire [1:0]                           m_axi_gmem_arlock,
	output wire [3:0]                           m_axi_gmem_arcache,
	output wire [2:0]                           m_axi_gmem_arprot,
	output wire [3:0]                           m_axi_gmem_arqos,
	output wire [3:0]                           m_axi_gmem_arregion,
	input  wire                                 m_axi_gmem_rvalid,
	output wire                                 m_axi_gmem_rready,
	input  wire [C_M_AXI_GMEM_DATA_WIDTH - 1:0] m_axi_gmem_rdata,
	input  wire                                 m_axi_gmem_rlast,
	input  wire [C_M_AXI_GMEM_ID_WIDTH - 1:0]   m_axi_gmem_rid,
	input  wire [1:0]                           m_axi_gmem_rresp,
	input  wire                                 m_axi_gmem_bvalid,
	output wire                                 m_axi_gmem_bready,
	input  wire [1:0]                           m_axi_gmem_bresp,
	input  wire [C_M_AXI_GMEM_ID_WIDTH - 1:0]   m_axi_gmem_bid,

	// AXI4-Lite control interface
	input  wire                                    s_axi_control_awvalid,
	output wire                                    s_axi_control_awready,
	input  wire [C_S_AXI_CONTROL_ADDR_WIDTH-1:0]   s_axi_control_awaddr,
	input  wire                                    s_axi_control_wvalid,
	output wire                                    s_axi_control_wready,
	input  wire [C_S_AXI_CONTROL_DATA_WIDTH-1:0]   s_axi_control_wdata,
	input  wire [C_S_AXI_CONTROL_DATA_WIDTH/8-1:0] s_axi_control_wstrb,
	input  wire                                    s_axi_control_arvalid,
	output wire                                    s_axi_control_arready,
	input  wire [C_S_AXI_CONTROL_ADDR_WIDTH-1:0]   s_axi_control_araddr,
	output wire                                    s_axi_control_rvalid,
	input  wire                                    s_axi_control_rready,
	output wire [C_S_AXI_CONTROL_DATA_WIDTH-1:0]   s_axi_control_rdata,
	output wire [1:0]                              s_axi_control_rresp,
	output wire                                    s_axi_control_bvalid,
	input  wire                                    s_axi_control_bready,
	output wire [1:0]                              s_axi_control_bresp,
	
  	output wire                                    interrupt 
);

    `STATIC_ASSERT((C_M_AXI_GMEM_ID_WIDTH == `VX_MEM_TAG_WIDTH), ("invalid memory tag size: current=%0d, expected=%0d", C_M_AXI_GMEM_ID_WIDTH, `VX_MEM_TAG_WIDTH))

	wire clk   = ap_clk;
	wire reset = ~ap_rst_n;

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

	always @(posedge clk) begin
		if (~vx_running && vx_reset == 0 && ap_start) begin
			vx_reset_ctr <= 0;
		end else begin
			vx_reset_ctr <= vx_reset_ctr + 1;
		end
	end

	always @(posedge clk) begin
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
		.clk       		(clk),
		.reset     		(reset),	
		.clk_en         (1'b1),
		
		.s_axi_awvalid  (s_axi_control_awvalid),
		.s_axi_awready  (s_axi_control_awready),
		.s_axi_awaddr   (s_axi_control_awaddr),
		.s_axi_wvalid   (s_axi_control_wvalid),
		.s_axi_wready   (s_axi_control_wready),
		.s_axi_wdata    (s_axi_control_wdata),
		.s_axi_wstrb    (s_axi_control_wstrb),
		.s_axi_arvalid  (s_axi_control_arvalid),
		.s_axi_arready  (s_axi_control_arready),
		.s_axi_araddr   (s_axi_control_araddr),
		.s_axi_rvalid   (s_axi_control_rvalid),
		.s_axi_rready   (s_axi_control_rready),
		.s_axi_rdata    (s_axi_control_rdata),
		.s_axi_rresp    (s_axi_control_rresp),
		.s_axi_bvalid   (s_axi_control_bvalid),
		.s_axi_bready   (s_axi_control_bready),
		.s_axi_bresp    (s_axi_control_bresp),

		.ap_start  		(ap_start),		
		.ap_done     	(ap_done),
		.ap_ready     	(ap_ready),
		.ap_idle     	(ap_idle),
		.interrupt 		(interrupt),	

		.dcr_wr_valid	(dcr_wr_valid),
		.dcr_wr_addr	(dcr_wr_addr),
		.dcr_wr_data	(dcr_wr_data)		
	);

	wire m_axi_gmem_awvalid_unqual;
	wire m_axi_gmem_wvalid_unqual;
	wire m_axi_gmem_arvalid_unqual;

	assign m_axi_gmem_awvalid = m_axi_gmem_awvalid_unqual && vx_running;
	assign m_axi_gmem_wvalid  = m_axi_gmem_wvalid_unqual && vx_running;
	assign m_axi_gmem_arvalid = m_axi_gmem_arvalid_unqual && vx_running;

	Vortex_axi #(
		.AXI_DATA_WIDTH (C_M_AXI_GMEM_DATA_WIDTH),
		.AXI_ADDR_WIDTH (C_M_AXI_GMEM_ADDR_WIDTH),
		.AXI_TID_WIDTH  (C_M_AXI_GMEM_ID_WIDTH)
	) vortex_axi (
		.clk			(clk),
		.reset			(reset || vx_reset),

		.m_axi_awid		(m_axi_gmem_awid),
		.m_axi_awaddr	(m_axi_gmem_awaddr),
		.m_axi_awlen	(m_axi_gmem_awlen),
		.m_axi_awsize	(m_axi_gmem_awsize),
		.m_axi_awburst	(m_axi_gmem_awburst),
		.m_axi_awlock	(m_axi_gmem_awlock),
		.m_axi_awcache	(m_axi_gmem_awcache),
		.m_axi_awprot	(m_axi_gmem_awprot),
		.m_axi_awqos	(m_axi_gmem_awqos),		  
        .m_axi_awregion (m_axi_gmem_awregion),
		.m_axi_awvalid	(m_axi_gmem_awvalid_unqual),
		.m_axi_awready	(m_axi_gmem_awready),
		.m_axi_wdata	(m_axi_gmem_wdata),
		.m_axi_wstrb	(m_axi_gmem_wstrb),
		.m_axi_wlast	(m_axi_gmem_wlast),
		.m_axi_wvalid	(m_axi_gmem_wvalid_unqual),
		.m_axi_wready	(m_axi_gmem_wready),
		.m_axi_bid		(m_axi_gmem_bid),
		.m_axi_bresp	(m_axi_gmem_bresp),
		.m_axi_bvalid	(m_axi_gmem_bvalid),
		.m_axi_bready	(m_axi_gmem_bready),
		.m_axi_arid		(m_axi_gmem_arid),
		.m_axi_araddr	(m_axi_gmem_araddr),
		.m_axi_arlen	(m_axi_gmem_arlen),
		.m_axi_arsize	(m_axi_gmem_arsize),
		.m_axi_arburst	(m_axi_gmem_arburst),
		.m_axi_arlock	(m_axi_gmem_arlock),
		.m_axi_arcache	(m_axi_gmem_arcache),
		.m_axi_arprot	(m_axi_gmem_arprot),
		.m_axi_arqos	(m_axi_gmem_arqos),		
        .m_axi_arregion (m_axi_gmem_arregion),
		.m_axi_arvalid	(m_axi_gmem_arvalid_unqual),
		.m_axi_arready	(m_axi_gmem_arready),
		.m_axi_rid		(m_axi_gmem_rid),
		.m_axi_rdata	(m_axi_gmem_rdata),
		.m_axi_rresp	(m_axi_gmem_rresp),
		.m_axi_rlast	(m_axi_gmem_rlast),
		.m_axi_rvalid	(m_axi_gmem_rvalid),
		.m_axi_rready	(m_axi_gmem_rready),

		.dcr_wr_valid	(dcr_wr_valid),
		.dcr_wr_addr	(dcr_wr_addr),
		.dcr_wr_data	(dcr_wr_data),

		.busy			(vx_busy)
	);
	
endmodule