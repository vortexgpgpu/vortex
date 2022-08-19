`include "VX_define.vh"

module vortex_afu #(
	parameter AXI_DATA_WIDTH     = `VX_MEM_DATA_WIDTH,
    parameter AXI_ADDR_WIDTH     = `VX_MEM_ADDR_WIDTH,
    parameter AXI_TID_WIDTH      = 12,
    parameter AXI_STROBE_WIDTH   = `VX_MEM_BYTEEN_WIDTH,
	parameter AXI_DCR_ADDR_WIDTH = `VX_DCR_ADDR_WIDTH,
    parameter AXI_DCR_DATA_WIDTH = `VX_DCR_DATA_WIDTH    
) (
	// System signals
	input  wire  ap_clk,
	input  wire  ap_rst_n,
	
	// AXI4 master interface 
	output wire                                 m_axi_gmem_AWVALID,
	input  wire                                 m_axi_gmem_AWREADY,
	output wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0]   m_axi_gmem_AWADDR,
	output wire [C_M_AXI_GMEM_ID_WIDTH - 1:0]   m_axi_gmem_AWID,
	output wire [7:0]                           m_axi_gmem_AWLEN,
	output wire [2:0]                           m_axi_gmem_AWSIZE,
	// Tie-off AXI4 transaction options that are not being used.
	output wire [1:0]                           m_axi_gmem_AWBURST,
	output wire [1:0]                           m_axi_gmem_AWLOCK,
	output wire [3:0]                           m_axi_gmem_AWCACHE,
	output wire [2:0]                           m_axi_gmem_AWPROT,
	output wire [3:0]                           m_axi_gmem_AWQOS,
	output wire [3:0]                           m_axi_gmem_AWREGION,
	output wire                                 m_axi_gmem_WVALID,
	input  wire                                 m_axi_gmem_WREADY,
	output wire [C_M_AXI_GMEM_DATA_WIDTH-1:0]   m_axi_gmem_WDATA,
	output wire [C_M_AXI_GMEM_DATA_WIDTH/8-1:0] m_axi_gmem_WSTRB,
	output wire                                 m_axi_gmem_WLAST,
	output wire                                 m_axi_gmem_ARVALID,
	input  wire                                 m_axi_gmem_ARREADY,
	output wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0]   m_axi_gmem_ARADDR,
	output wire [C_M_AXI_GMEM_ID_WIDTH-1:0]     m_axi_gmem_ARID,
	output wire [7:0]                           m_axi_gmem_ARLEN,
	output wire [2:0]                           m_axi_gmem_ARSIZE,
	output wire [1:0]                           m_axi_gmem_ARBURST,
	output wire [1:0]                           m_axi_gmem_ARLOCK,
	output wire [3:0]                           m_axi_gmem_ARCACHE,
	output wire [2:0]                           m_axi_gmem_ARPROT,
	output wire [3:0]                           m_axi_gmem_ARQOS,
	output wire [3:0]                           m_axi_gmem_ARREGION,
	input  wire                                 m_axi_gmem_RVALID,
	output wire                                 m_axi_gmem_RREADY,
	input  wire [C_M_AXI_GMEM_DATA_WIDTH - 1:0] m_axi_gmem_RDATA,
	input  wire                                 m_axi_gmem_RLAST,
	input  wire [C_M_AXI_GMEM_ID_WIDTH - 1:0]   m_axi_gmem_RID,
	input  wire [1:0]                           m_axi_gmem_RRESP,
	input  wire                                 m_axi_gmem_BVALID,
	output wire                                 m_axi_gmem_BREADY,
	input  wire [1:0]                           m_axi_gmem_BRESP,
	input  wire [C_M_AXI_GMEM_ID_WIDTH - 1:0]   m_axi_gmem_BID,

	// AXI4-Lite slave interface
	input  wire                                    s_axi_control_AWVALID,
	output wire                                    s_axi_control_AWREADY,
	input  wire [C_S_AXI_CONTROL_ADDR_WIDTH-1:0]   s_axi_control_AWADDR,
	input  wire                                    s_axi_control_WVALID,
	output wire                                    s_axi_control_WREADY,
	input  wire [C_S_AXI_CONTROL_DATA_WIDTH-1:0]   s_axi_control_WDATA,
	input  wire [C_S_AXI_CONTROL_DATA_WIDTH/8-1:0] s_axi_control_WSTRB,
	input  wire                                    s_axi_control_ARVALID,
	output wire                                    s_axi_control_ARREADY,
	input  wire [C_S_AXI_CONTROL_ADDR_WIDTH-1:0]   s_axi_control_ARADDR,
	output wire                                    s_axi_control_RVALID,
	input  wire                                    s_axi_control_RREADY,
	output wire [C_S_AXI_CONTROL_DATA_WIDTH-1:0]   s_axi_control_RDATA,
	output wire [1:0]                              s_axi_control_RRESP,
	output wire                                    s_axi_control_BVALID,
	input  wire                                    s_axi_control_BREADY,
	output wire [1:0]                              s_axi_control_BRESP
);

	///////////////////////////////////////////////////////////////////////////////
	// Local Parameters (constants)
	///////////////////////////////////////////////////////////////////////////////
	localparam integer LP_NUM_READ_CHANNELS  = 2;
	localparam integer LP_LENGTH_WIDTH       = 32;
	localparam integer LP_DW_BYTES           = C_M_AXI_GMEM_DATA_WIDTH/8;
	localparam integer LP_AXI_BURST_LEN      = 4096/LP_DW_BYTES < 256 ? 4096/LP_DW_BYTES : 256;
	localparam integer LP_LOG_BURST_LEN      = $clog2(LP_AXI_BURST_LEN);
	localparam integer LP_RD_MAX_OUTSTANDING = 3;
	localparam integer LP_RD_FIFO_DEPTH      = LP_AXI_BURST_LEN*(LP_RD_MAX_OUTSTANDING + 1);
	localparam integer LP_WR_FIFO_DEPTH      = LP_AXI_BURST_LEN;


	///////////////////////////////////////////////////////////////////////////////
	// Variables
	///////////////////////////////////////////////////////////////////////////////
	logic areset = 1'b0;  
	logic ap_start;
	logic ap_start_pulse;
	logic ap_start_r;
	logic ap_ready;
	logic ap_done;
	logic ap_idle = 1'b1;
	logic [C_M_AXI_GMEM_ADDR_WIDTH-1:0] a;
	logic [C_M_AXI_GMEM_ADDR_WIDTH-1:0] b;
	logic [C_M_AXI_GMEM_ADDR_WIDTH-1:0] c;
	logic [LP_LENGTH_WIDTH-1:0]         length_r;

	logic read_done;
	logic [LP_NUM_READ_CHANNELS-1:0] rd_tvalid;
	logic [LP_NUM_READ_CHANNELS-1:0] rd_tready_n; 
	logic [LP_NUM_READ_CHANNELS-1:0] [C_M_AXI_GMEM_DATA_WIDTH-1:0] rd_tdata;
	logic [LP_NUM_READ_CHANNELS-1:0] ctrl_rd_fifo_prog_full;
	logic [LP_NUM_READ_CHANNELS-1:0] rd_fifo_tvalid_n;
	logic [LP_NUM_READ_CHANNELS-1:0] rd_fifo_tready; 
	logic [LP_NUM_READ_CHANNELS-1:0] [C_M_AXI_GMEM_DATA_WIDTH-1:0] rd_fifo_tdata;

	logic                               adder_tvalid;
	logic                               adder_tready_n; 
	logic [C_M_AXI_GMEM_DATA_WIDTH-1:0] adder_tdata;
	logic                               wr_fifo_tvalid_n;
	logic                               wr_fifo_tready; 
	logic [C_M_AXI_GMEM_DATA_WIDTH-1:0] wr_fifo_tdata;

	///////////////////////////////////////////////////////////////////////////////
	// RTL Logic 
	///////////////////////////////////////////////////////////////////////////////
	// Tie-off unused AXI protocol features
	assign m_axi_gmem_AWID     = {C_M_AXI_GMEM_ID_WIDTH{1'b0}};
	assign m_axi_gmem_AWBURST  = 2'b01;
	assign m_axi_gmem_AWLOCK   = 2'b00;
	assign m_axi_gmem_AWCACHE  = 4'b0011;
	assign m_axi_gmem_AWPROT   = 3'b000;
	assign m_axi_gmem_AWQOS    = 4'b0000;
	assign m_axi_gmem_AWREGION = 4'b0000;
	assign m_axi_gmem_ARBURST  = 2'b01;
	assign m_axi_gmem_ARLOCK   = 2'b00;
	assign m_axi_gmem_ARCACHE  = 4'b0011;
	assign m_axi_gmem_ARPROT   = 3'b000;
	assign m_axi_gmem_ARQOS    = 4'b0000;
	assign m_axi_gmem_ARREGION = 4'b0000;

	// Register and invert reset signal for better timing.
	always @(posedge ap_clk) begin 
	areset <= ~ap_rst_n; 
	end

	// create pulse when ap_start transitions to 1
	always @(posedge ap_clk) begin 
	begin 
		ap_start_r <= ap_start;
	end
	end

	assign ap_start_pulse = ap_start & ~ap_start_r;

	// ap_idle is asserted when done is asserted, it is de-asserted when ap_start_pulse 
	// is asserted
	always @(posedge ap_clk) begin 
	if (areset) begin 
		ap_idle <= 1'b1;
	end
	else begin 
		ap_idle <= ap_done        ? 1'b1 : 
				ap_start_pulse ? 1'b0 : 
									ap_idle;
	end
	end

	assign ap_ready = ap_done;

	// AXI4-Lite slave
	krnl_vadd_rtl_control_s_axi #(
	.C_S_AXI_ADDR_WIDTH( C_S_AXI_CONTROL_ADDR_WIDTH ),
	.C_S_AXI_DATA_WIDTH( C_S_AXI_CONTROL_DATA_WIDTH )
	) 
	inst_krnl_vadd_control_s_axi (
	.AWVALID   ( s_axi_control_AWVALID         ) ,
	.AWREADY   ( s_axi_control_AWREADY         ) ,
	.AWADDR    ( s_axi_control_AWADDR          ) ,
	.WVALID    ( s_axi_control_WVALID          ) ,
	.WREADY    ( s_axi_control_WREADY          ) ,
	.WDATA     ( s_axi_control_WDATA           ) ,
	.WSTRB     ( s_axi_control_WSTRB           ) ,
	.ARVALID   ( s_axi_control_ARVALID         ) ,
	.ARREADY   ( s_axi_control_ARREADY         ) ,
	.ARADDR    ( s_axi_control_ARADDR          ) ,
	.RVALID    ( s_axi_control_RVALID          ) ,
	.RREADY    ( s_axi_control_RREADY          ) ,
	.RDATA     ( s_axi_control_RDATA           ) ,
	.RRESP     ( s_axi_control_RRESP           ) ,
	.BVALID    ( s_axi_control_BVALID          ) ,
	.BREADY    ( s_axi_control_BREADY          ) ,
	.BRESP     ( s_axi_control_BRESP           ) ,
	.ACLK      ( ap_clk                        ) ,
	.ARESET    ( areset                        ) ,
	.ACLK_EN   ( 1'b1                          ) ,
	.ap_start  ( ap_start                      ) ,
	.interrupt ( interrupt                     ) ,
	.ap_ready  ( ap_ready                      ) ,
	.ap_done   ( ap_done                       ) ,
	.ap_idle   ( ap_idle                       ) ,
	.a         ( a[0+:C_M_AXI_GMEM_ADDR_WIDTH] ) ,
	.b         ( b[0+:C_M_AXI_GMEM_ADDR_WIDTH] ) ,
	.c         ( c[0+:C_M_AXI_GMEM_ADDR_WIDTH] ) ,
	.length_r  ( length_r[0+:LP_LENGTH_WIDTH]  ) 
	);

	// AXI4 Read Master
	krnl_vadd_rtl_axi_read_master #( 
	.C_ADDR_WIDTH       ( C_M_AXI_GMEM_ADDR_WIDTH ) ,
	.C_DATA_WIDTH       ( C_M_AXI_GMEM_DATA_WIDTH ) ,
	.C_ID_WIDTH         ( C_M_AXI_GMEM_ID_WIDTH   ) ,
	.C_NUM_CHANNELS     ( LP_NUM_READ_CHANNELS    ) ,
	.C_LENGTH_WIDTH     ( LP_LENGTH_WIDTH         ) ,
	.C_BURST_LEN        ( LP_AXI_BURST_LEN        ) ,
	.C_LOG_BURST_LEN    ( LP_LOG_BURST_LEN        ) ,
	.C_MAX_OUTSTANDING  ( LP_RD_MAX_OUTSTANDING   )
	)
	inst_axi_read_master ( 
	.aclk           ( ap_clk                 ) ,
	.areset         ( areset                 ) ,

	.ctrl_start     ( ap_start_pulse         ) ,
	.ctrl_done      ( read_done              ) ,
	.ctrl_offset    ( {b,a}                  ) ,
	.ctrl_length    ( length_r               ) ,
	.ctrl_prog_full ( ctrl_rd_fifo_prog_full ) ,

	.arvalid        ( m_axi_gmem_ARVALID     ) ,
	.arready        ( m_axi_gmem_ARREADY     ) ,
	.araddr         ( m_axi_gmem_ARADDR      ) ,
	.arid           ( m_axi_gmem_ARID        ) ,
	.arlen          ( m_axi_gmem_ARLEN       ) ,
	.arsize         ( m_axi_gmem_ARSIZE      ) ,
	.rvalid         ( m_axi_gmem_RVALID      ) ,
	.rready         ( m_axi_gmem_RREADY      ) ,
	.rdata          ( m_axi_gmem_RDATA       ) ,
	.rlast          ( m_axi_gmem_RLAST       ) ,
	.rid            ( m_axi_gmem_RID         ) ,
	.rresp          ( m_axi_gmem_RRESP       ) ,

	.m_tvalid       ( rd_tvalid              ) ,
	.m_tready       ( ~rd_tready_n           ) ,
	.m_tdata        ( rd_tdata               ) 
	);

	// xpm_fifo_sync: Synchronous FIFO
	// Xilinx Parameterized Macro, Version 2016.4
	xpm_fifo_sync # (
	.FIFO_MEMORY_TYPE          ("auto"),           //string; "auto", "block", "distributed", or "ultra";
	.ECC_MODE                  ("no_ecc"),         //string; "no_ecc" or "en_ecc";
	.FIFO_WRITE_DEPTH          (LP_RD_FIFO_DEPTH),   //positive integer
	.WRITE_DATA_WIDTH          (C_M_AXI_GMEM_DATA_WIDTH),        //positive integer
	.WR_DATA_COUNT_WIDTH       ($clog2(LP_RD_FIFO_DEPTH)+1),       //positive integer, Not used
	.PROG_FULL_THRESH          (LP_AXI_BURST_LEN-2),               //positive integer
	.FULL_RESET_VALUE          (1),                //positive integer; 0 or 1
	.READ_MODE                 ("fwft"),            //string; "std" or "fwft";
	.FIFO_READ_LATENCY         (1),                //positive integer;
	.READ_DATA_WIDTH           (C_M_AXI_GMEM_DATA_WIDTH),               //positive integer
	.RD_DATA_COUNT_WIDTH       ($clog2(LP_RD_FIFO_DEPTH)+1),               //positive integer, not used
	.PROG_EMPTY_THRESH         (10),               //positive integer, not used 
	.DOUT_RESET_VALUE          ("0"),              //string, don't care
	.WAKEUP_TIME               (0)                 //positive integer; 0 or 2;

	) inst_rd_xpm_fifo_sync[LP_NUM_READ_CHANNELS-1:0] (
	.sleep         ( 1'b0             ) ,
	.rst           ( areset           ) ,
	.wr_clk        ( ap_clk           ) ,
	.wr_en         ( rd_tvalid        ) ,
	.din           ( rd_tdata         ) ,
	.full          ( rd_tready_n      ) ,
	.prog_full     ( ctrl_rd_fifo_prog_full) ,
	.wr_data_count (                  ) ,
	.overflow      (                  ) ,
	.wr_rst_busy   (                  ) ,
	.rd_en         ( rd_fifo_tready   ) ,
	.dout          ( rd_fifo_tdata    ) ,
	.empty         ( rd_fifo_tvalid_n ) ,
	.prog_empty    (                  ) ,
	.rd_data_count (                  ) ,
	.underflow     (                  ) ,
	.rd_rst_busy   (                  ) ,
	.injectsbiterr ( 1'b0             ) ,
	.injectdbiterr ( 1'b0             ) ,
	.sbiterr       (                  ) ,
	.dbiterr       (                  ) 

	);

	// Combinatorial Adder
	krnl_vadd_rtl_adder #( 
	.C_DATA_WIDTH   ( C_M_AXI_GMEM_DATA_WIDTH ) ,
	.C_NUM_CHANNELS ( LP_NUM_READ_CHANNELS    ) 
	)
	inst_adder ( 
	.aclk     ( ap_clk            ) ,
	.areset   ( areset            ) ,

	.s_tvalid ( ~rd_fifo_tvalid_n ) ,
	.s_tready ( rd_fifo_tready    ) ,
	.s_tdata  ( rd_fifo_tdata     ) ,

	.m_tvalid ( adder_tvalid      ) ,
	.m_tready ( ~adder_tready_n   ) ,
	.m_tdata  ( adder_tdata       ) 
	);

	// xpm_fifo_sync: Synchronous FIFO
	// Xilinx Parameterized Macro, Version 2016.4
	xpm_fifo_sync # (
	.FIFO_MEMORY_TYPE          ("auto"),           //string; "auto", "block", "distributed", or "ultra";
	.ECC_MODE                  ("no_ecc"),         //string; "no_ecc" or "en_ecc";
	.FIFO_WRITE_DEPTH          (LP_WR_FIFO_DEPTH),   //positive integer
	.WRITE_DATA_WIDTH          (C_M_AXI_GMEM_DATA_WIDTH),               //positive integer
	.WR_DATA_COUNT_WIDTH       ($clog2(LP_WR_FIFO_DEPTH)),               //positive integer, Not used
	.PROG_FULL_THRESH          (10),               //positive integer, Not used 
	.FULL_RESET_VALUE          (1),                //positive integer; 0 or 1
	.READ_MODE                 ("fwft"),            //string; "std" or "fwft";
	.FIFO_READ_LATENCY         (1),                //positive integer;
	.READ_DATA_WIDTH           (C_M_AXI_GMEM_DATA_WIDTH),               //positive integer
	.RD_DATA_COUNT_WIDTH       ($clog2(LP_WR_FIFO_DEPTH)),               //positive integer, not used
	.PROG_EMPTY_THRESH         (10),               //positive integer, not used 
	.DOUT_RESET_VALUE          ("0"),              //string, don't care
	.WAKEUP_TIME               (0)                 //positive integer; 0 or 2;

	) inst_wr_xpm_fifo_sync (
	.sleep         ( 1'b0             ) ,
	.rst           ( areset           ) ,
	.wr_clk        ( ap_clk           ) ,
	.wr_en         ( adder_tvalid     ) ,
	.din           ( adder_tdata      ) ,
	.full          ( adder_tready_n   ) ,
	.prog_full     (                  ) ,
	.wr_data_count (                  ) ,
	.overflow      (                  ) ,
	.wr_rst_busy   (                  ) ,
	.rd_en         ( wr_fifo_tready   ) ,
	.dout          ( wr_fifo_tdata    ) ,
	.empty         ( wr_fifo_tvalid_n ) ,
	.prog_empty    (                  ) ,
	.rd_data_count (                  ) ,
	.underflow     (                  ) ,
	.rd_rst_busy   (                  ) ,
	.injectsbiterr ( 1'b0             ) ,
	.injectdbiterr ( 1'b0             ) ,
	.sbiterr       (                  ) ,
	.dbiterr       (                  ) 

	);


	// AXI4 Write Master
	krnl_vadd_rtl_axi_write_master #( 
	.C_ADDR_WIDTH       ( C_M_AXI_GMEM_ADDR_WIDTH ) ,
	.C_DATA_WIDTH       ( C_M_AXI_GMEM_DATA_WIDTH ) ,
	.C_MAX_LENGTH_WIDTH ( LP_LENGTH_WIDTH     ) ,
	.C_BURST_LEN        ( LP_AXI_BURST_LEN        ) ,
	.C_LOG_BURST_LEN    ( LP_LOG_BURST_LEN        ) 
	)
	inst_axi_write_master ( 
	.aclk        ( ap_clk             ) ,
	.areset      ( areset             ) ,

	.ctrl_start  ( ap_start_pulse     ) ,
	.ctrl_offset ( c                  ) ,
	.ctrl_length ( length_r           ) ,
	.ctrl_done   ( ap_done            ) ,

	.awvalid     ( m_axi_gmem_AWVALID ) ,
	.awready     ( m_axi_gmem_AWREADY ) ,
	.awaddr      ( m_axi_gmem_AWADDR  ) ,
	.awlen       ( m_axi_gmem_AWLEN   ) ,
	.awsize      ( m_axi_gmem_AWSIZE  ) ,

	.s_tvalid    ( ~wr_fifo_tvalid_n   ) ,
	.s_tready    ( wr_fifo_tready     ) ,
	.s_tdata     ( wr_fifo_tdata      ) ,

	.wvalid      ( m_axi_gmem_WVALID  ) ,
	.wready      ( m_axi_gmem_WREADY  ) ,
	.wdata       ( m_axi_gmem_WDATA   ) ,
	.wstrb       ( m_axi_gmem_WSTRB   ) ,
	.wlast       ( m_axi_gmem_WLAST   ) ,

	.bvalid      ( m_axi_gmem_BVALID  ) ,
	.bready      ( m_axi_gmem_BREADY  ) ,
	.bresp       ( m_axi_gmem_BRESP   ) 
	);

	Vortex_axi #(
		.AXI_DATA_WIDTH     (AXI_DATA_WIDTH),
		.AXI_ADDR_WIDTH     (AXI_ADDR_WIDTH),
		.AXI_TID_WIDTH      (AXI_TID_WIDTH),
		.AXI_STROBE_WIDTH   (AXI_STROBE_WIDTH),
		.AXI_DCR_ADDR_WIDTH (AXI_DCR_ADDR_WIDTH),
		.AXI_DCR_DATA_WIDTH (AXI_DCR_DATA_WIDTH)
	) inst (
		.clk(clk),
		.reset(reset),
		.m_axi_awid(m_axi_awid),
		.m_axi_awaddr(m_axi_awaddr),
		.m_axi_awlen(m_axi_awlen),
		.m_axi_awsize(m_axi_awsize),
		.m_axi_awburst(m_axi_awburst),
		.m_axi_awlock(m_axi_awlock),
		.m_axi_awcache(m_axi_awcache),
		.m_axi_awprot(m_axi_awprot),
		.m_axi_awqos(m_axi_awqos),
		.m_axi_awvalid(m_axi_awvalid),
		.m_axi_awready(m_axi_awready),
		.m_axi_wdata(m_axi_wdata),
		.m_axi_wstrb(m_axi_wstrb),
		.m_axi_wlast(m_axi_wlast),
		.m_axi_wvalid(m_axi_wvalid),
		.m_axi_wready(m_axi_wready),
		.m_axi_bid(m_axi_bid),
		.m_axi_bresp(m_axi_bresp),
		.m_axi_bvalid(m_axi_bvalid),
		.m_axi_bready(m_axi_bready),
		.m_axi_arid(m_axi_arid),
		.m_axi_araddr(m_axi_araddr),
		.m_axi_arlen(m_axi_arlen),
		.m_axi_arsize(m_axi_arsize),
		.m_axi_arburst(m_axi_arburst),
		.m_axi_arlock(m_axi_arlock),
		.m_axi_arcache(m_axi_arcache),
		.m_axi_arprot(m_axi_arprot),
		.m_axi_arqos(m_axi_arqos),
		.m_axi_arvalid(m_axi_arvalid),
		.m_axi_arready(m_axi_arready),
		.m_axi_rid(m_axi_rid),
		.m_axi_rdata(m_axi_rdata),
		.m_axi_rresp(m_axi_rresp),
		.m_axi_rlast(m_axi_rlast),
		.m_axi_rvalid(m_axi_rvalid),
		.m_axi_rready(m_axi_rready),
		.busy(busy)
	);
	
endmodule