`ifndef VORTEX_AFU_VH
`define VORTEX_AFU_VH

`ifndef M_AXI_MEM_NUM_BANKS
`define M_AXI_MEM_NUM_BANKS 1
`endif

`ifndef M_AXI_MEM_ID_WIDTH
`ifdef NDEBUG
`define M_AXI_MEM_ID_WIDTH 20
`else
`define M_AXI_MEM_ID_WIDTH 32
`endif
`endif

`define GEN_AXI_MEM(i) \
	output wire                                 m``i``_axi_mem_awvalid, \
	input  wire                                 m``i``_axi_mem_awready, \
	output wire [C_M_AXI_MEM_ADDR_WIDTH-1:0] 	m``i``_axi_mem_awaddr, \
	output wire [C_M_AXI_MEM_ID_WIDTH - 1:0]   	m``i``_axi_mem_awid, \
	output wire [7:0]                           m``i``_axi_mem_awlen, \
	output wire                                 m``i``_axi_mem_wvalid, \
	input  wire                                 m``i``_axi_mem_wready, \
	output wire [C_M_AXI_MEM_DATA_WIDTH-1:0]   	m``i``_axi_mem_wdata, \
	output wire [C_M_AXI_MEM_DATA_WIDTH/8-1:0] 	m``i``_axi_mem_wstrb, \
	output wire                                 m``i``_axi_mem_wlast, \
	output wire                                 m``i``_axi_mem_arvalid, \
	input  wire                                 m``i``_axi_mem_arready, \
	output wire [C_M_AXI_MEM_ADDR_WIDTH-1:0]   	m``i``_axi_mem_araddr, \
	output wire [C_M_AXI_MEM_ID_WIDTH-1:0]     	m``i``_axi_mem_arid, \
	output wire [7:0]                           m``i``_axi_mem_arlen, \
	input  wire                                 m``i``_axi_mem_rvalid, \
	output wire                                 m``i``_axi_mem_rready, \
	input  wire [C_M_AXI_MEM_DATA_WIDTH - 1:0] 	m``i``_axi_mem_rdata, \
	input  wire                                 m``i``_axi_mem_rlast, \
	input  wire [C_M_AXI_MEM_ID_WIDTH - 1:0]   	m``i``_axi_mem_rid, \
	input  wire [1:0]                           m``i``_axi_mem_rresp, \
	input  wire                                 m``i``_axi_mem_bvalid, \
	output wire                                 m``i``_axi_mem_bready, \
	input  wire [1:0]                           m``i``_axi_mem_bresp, \
	input  wire [C_M_AXI_MEM_ID_WIDTH - 1:0]   	m``i``_axi_mem_bid

`define AXI_MEM_ARGS(i) \
    .m``i``_axi_mem_awvalid(m``i``_axi_mem_awvalid), \
    .m``i``_axi_mem_awready(m``i``_axi_mem_awready), \
    .m``i``_axi_mem_awaddr(m``i``_axi_mem_awaddr), \
    .m``i``_axi_mem_awid(m``i``_axi_mem_awid), \
    .m``i``_axi_mem_awlen(m``i``_axi_mem_awlen), \
    .m``i``_axi_mem_wvalid(m``i``_axi_mem_wvalid), \
    .m``i``_axi_mem_wready(m``i``_axi_mem_wready), \
    .m``i``_axi_mem_wdata(m``i``_axi_mem_wdata), \
    .m``i``_axi_mem_wstrb(m``i``_axi_mem_wstrb), \
    .m``i``_axi_mem_wlast(m``i``_axi_mem_wlast), \
    .m``i``_axi_mem_arvalid(m``i``_axi_mem_arvalid), \
    .m``i``_axi_mem_arready(m``i``_axi_mem_arready), \
    .m``i``_axi_mem_araddr(m``i``_axi_mem_araddr), \
    .m``i``_axi_mem_arid(m``i``_axi_mem_arid), \
    .m``i``_axi_mem_arlen(m``i``_axi_mem_arlen), \
    .m``i``_axi_mem_rvalid(m``i``_axi_mem_rvalid), \
    .m``i``_axi_mem_rready(m``i``_axi_mem_rready), \
    .m``i``_axi_mem_rdata(m``i``_axi_mem_rdata), \
    .m``i``_axi_mem_rlast(m``i``_axi_mem_rlast), \
    .m``i``_axi_mem_rid(m``i``_axi_mem_rid), \
    .m``i``_axi_mem_rresp(m``i``_axi_mem_rresp), \
    .m``i``_axi_mem_bvalid(m``i``_axi_mem_bvalid), \
    .m``i``_axi_mem_bready(m``i``_axi_mem_bready), \
    .m``i``_axi_mem_bresp(m``i``_axi_mem_bresp), \
    .m``i``_axi_mem_bid(m``i``_axi_mem_bid)

`define AXI_MEM_TO_ARRAY(i) \
	assign m``i``_axi_mem_awvalid = m_axi_mem_awvalid_a[i]; \
	assign m_axi_mem_awready_a[i] = m``i``_axi_mem_awready; \
	assign m``i``_axi_mem_awaddr = m_axi_mem_awaddr_a[i]; \
	assign m``i``_axi_mem_awid = m_axi_mem_awid_a[i]; \
	assign m``i``_axi_mem_awlen = m_axi_mem_awlen_a[i]; \
	assign m``i``_axi_mem_wvalid = m_axi_mem_wvalid_a[i]; \
	assign m_axi_mem_wready_a[i] = m``i``_axi_mem_wready; \
	assign m``i``_axi_mem_wdata = m_axi_mem_wdata_a[i]; \
	assign m``i``_axi_mem_wstrb = m_axi_mem_wstrb_a[i]; \
	assign m``i``_axi_mem_wlast = m_axi_mem_wlast_a[i]; \
	assign m``i``_axi_mem_arvalid = m_axi_mem_arvalid_a[i]; \
	assign m_axi_mem_arready_a[i] = m``i``_axi_mem_arready; \
	assign m``i``_axi_mem_araddr = m_axi_mem_araddr_a[i]; \
	assign m``i``_axi_mem_arid = m_axi_mem_arid_a[i]; \
	assign m``i``_axi_mem_arlen = m_axi_mem_arlen_a[i]; \
	assign m_axi_mem_rvalid_a[i] = m``i``_axi_mem_rvalid; \
	assign m``i``_axi_mem_rready = m_axi_mem_rready_a[i]; \
	assign m_axi_mem_rdata_a[i] = m``i``_axi_mem_rdata; \
	assign m_axi_mem_rlast_a[i] = m``i``_axi_mem_rlast; \
	assign m_axi_mem_rid_a[i] = m``i``_axi_mem_rid; \
	assign m_axi_mem_rresp_a[i] = m``i``_axi_mem_rresp; \
	assign m_axi_mem_bvalid_a[i] = m``i``_axi_mem_bvalid; \
	assign m``i``_axi_mem_bready = m_axi_mem_bready_a[i]; \
	assign m_axi_mem_bresp_a[i] = m``i``_axi_mem_bresp; \
	assign m_axi_mem_bid_a[i] = m``i``_axi_mem_bid

`endif // VORTEX_AFU_VH
