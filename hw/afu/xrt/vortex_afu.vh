`ifndef VORTEX_AFU_VH
`define VORTEX_AFU_VH

`define PP(x) x

`define GEN_AXI_MEM(i) \
	output wire                                 m`PP(``i)_axi_mem_awvalid, \
	input  wire                                 m`PP(``i)_axi_mem_awready, \
	output wire [C_M_AXI_MEM_ADDR_WIDTH-1:0] 	m`PP(``i)_axi_mem_awaddr, \
	output wire [C_M_AXI_MEM_ID_WIDTH - 1:0]   	m`PP(``i)_axi_mem_awid, \
	output wire [7:0]                           m`PP(``i)_axi_mem_awlen, \
	output wire                                 m`PP(``i)_axi_mem_wvalid, \
	input  wire                                 m`PP(``i)_axi_mem_wready, \
	output wire [C_M_AXI_MEM_DATA_WIDTH-1:0]   	m`PP(``i)_axi_mem_wdata, \
	output wire [C_M_AXI_MEM_DATA_WIDTH/8-1:0] 	m`PP(``i)_axi_mem_wstrb, \
	output wire                                 m`PP(``i)_axi_mem_wlast, \
	output wire                                 m`PP(``i)_axi_mem_arvalid, \
	input  wire                                 m`PP(``i)_axi_mem_arready, \
	output wire [C_M_AXI_MEM_ADDR_WIDTH-1:0]   	m`PP(``i)_axi_mem_araddr, \
	output wire [C_M_AXI_MEM_ID_WIDTH-1:0]     	m`PP(``i)_axi_mem_arid, \
	output wire [7:0]                           m`PP(``i)_axi_mem_arlen, \
	input  wire                                 m`PP(``i)_axi_mem_rvalid, \
	output wire                                 m`PP(``i)_axi_mem_rready, \
	input  wire [C_M_AXI_MEM_DATA_WIDTH - 1:0] 	m`PP(``i)_axi_mem_rdata, \
	input  wire                                 m`PP(``i)_axi_mem_rlast, \
	input  wire [C_M_AXI_MEM_ID_WIDTH - 1:0]   	m`PP(``i)_axi_mem_rid, \
	input  wire [1:0]                           m`PP(``i)_axi_mem_rresp, \
	input  wire                                 m`PP(``i)_axi_mem_bvalid, \
	output wire                                 m`PP(``i)_axi_mem_bready, \
	input  wire [1:0]                           m`PP(``i)_axi_mem_bresp, \
	input  wire [C_M_AXI_MEM_ID_WIDTH - 1:0]   	m`PP(``i)_axi_mem_bid

`define AXI_MEM_ARGS(i) \
    .m`PP(``i)_axi_mem_awvalid(m`PP(``i)_axi_mem_awvalid), \
    .m`PP(``i)_axi_mem_awready(m`PP(``i)_axi_mem_awready), \
    .m`PP(``i)_axi_mem_awaddr(m`PP(``i)_axi_mem_awaddr), \
    .m`PP(``i)_axi_mem_awid(m`PP(``i)_axi_mem_awid), \
    .m`PP(``i)_axi_mem_awlen(m`PP(``i)_axi_mem_awlen), \
    .m`PP(``i)_axi_mem_wvalid(m`PP(``i)_axi_mem_wvalid), \
    .m`PP(``i)_axi_mem_wready(m`PP(``i)_axi_mem_wready), \
    .m`PP(``i)_axi_mem_wdata(m`PP(``i)_axi_mem_wdata), \
    .m`PP(``i)_axi_mem_wstrb(m`PP(``i)_axi_mem_wstrb), \
    .m`PP(``i)_axi_mem_wlast(m`PP(``i)_axi_mem_wlast), \
    .m`PP(``i)_axi_mem_arvalid(m`PP(``i)_axi_mem_arvalid), \
    .m`PP(``i)_axi_mem_arready(m`PP(``i)_axi_mem_arready), \
    .m`PP(``i)_axi_mem_araddr(m`PP(``i)_axi_mem_araddr), \
    .m`PP(``i)_axi_mem_arid(m`PP(``i)_axi_mem_arid), \
    .m`PP(``i)_axi_mem_arlen(m`PP(``i)_axi_mem_arlen), \
    .m`PP(``i)_axi_mem_rvalid(m`PP(``i)_axi_mem_rvalid), \
    .m`PP(``i)_axi_mem_rready(m`PP(``i)_axi_mem_rready), \
    .m`PP(``i)_axi_mem_rdata(m`PP(``i)_axi_mem_rdata), \
    .m`PP(``i)_axi_mem_rlast(m`PP(``i)_axi_mem_rlast), \
    .m`PP(``i)_axi_mem_rid(m`PP(``i)_axi_mem_rid), \
    .m`PP(``i)_axi_mem_rresp(m`PP(``i)_axi_mem_rresp), \
    .m`PP(``i)_axi_mem_bvalid(m`PP(``i)_axi_mem_bvalid), \
    .m`PP(``i)_axi_mem_bready(m`PP(``i)_axi_mem_bready), \
    .m`PP(``i)_axi_mem_bresp(m`PP(``i)_axi_mem_bresp), \
    .m`PP(``i)_axi_mem_bid(m`PP(``i)_axi_mem_bid)

`define AXI_MEM_TO_ARRAY(i) \
	assign m`PP(``i)_axi_mem_awvalid = m_axi_mem_awvalid_w[i]; \
	assign m_axi_mem_awready_w[i] = m`PP(``i)_axi_mem_awready; \
	assign m`PP(``i)_axi_mem_awaddr = m_axi_mem_awaddr_w[i]; \
	assign m`PP(``i)_axi_mem_awid = m_axi_mem_awid_w[i]; \
	assign m`PP(``i)_axi_mem_awlen = m_axi_mem_awlen_w[i]; \
	assign m`PP(``i)_axi_mem_wvalid = m_axi_mem_wvalid_w[i]; \
	assign m_axi_mem_wready_w[i] = m`PP(``i)_axi_mem_wready; \
	assign m`PP(``i)_axi_mem_wdata = m_axi_mem_wdata_w[i]; \
	assign m`PP(``i)_axi_mem_wstrb = m_axi_mem_wstrb_w[i]; \
	assign m`PP(``i)_axi_mem_wlast = m_axi_mem_wlast_w[i]; \
	assign m`PP(``i)_axi_mem_arvalid = m_axi_mem_arvalid_w[i]; \
	assign m_axi_mem_arready_w[i] = m`PP(``i)_axi_mem_arready; \
	assign m`PP(``i)_axi_mem_araddr = m_axi_mem_araddr_w[i]; \
	assign m`PP(``i)_axi_mem_arid = m_axi_mem_arid_w[i]; \
	assign m`PP(``i)_axi_mem_arlen = m_axi_mem_arlen_w[i]; \
	assign m_axi_mem_rvalid_w[i] = m`PP(``i)_axi_mem_rvalid; \
	assign m`PP(``i)_axi_mem_rready = m_axi_mem_rready_w[i]; \
	assign m_axi_mem_rdata_w[i] = m`PP(``i)_axi_mem_rdata; \
	assign m_axi_mem_rlast_w[i] = m`PP(``i)_axi_mem_rlast; \
	assign m_axi_mem_rid_w[i] = m`PP(``i)_axi_mem_rid; \
	assign m_axi_mem_rresp_w[i] = m`PP(``i)_axi_mem_rresp; \
	assign m_axi_mem_bvalid_w[i] = m`PP(``i)_axi_mem_bvalid; \
	assign m`PP(``i)_axi_mem_bready = m_axi_mem_bready_w[i]; \
	assign m_axi_mem_bresp_w[i] = m`PP(``i)_axi_mem_bresp; \
	assign m_axi_mem_bid_w[i] = m`PP(``i)_axi_mem_bid

`endif
