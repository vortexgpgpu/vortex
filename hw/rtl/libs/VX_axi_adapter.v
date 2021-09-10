`include "VX_define.vh"

module VX_axi_adapter #(
    parameter VX_DATA_WIDTH     = 512, 
    parameter VX_ADDR_WIDTH     = (32 - $clog2(VX_DATA_WIDTH/8)),            
    parameter VX_TAG_WIDTH      = 8,
    parameter AXI_DATA_WIDTH    = VX_DATA_WIDTH, 
    parameter AXI_ADDR_WIDTH    = 32,
    parameter AXI_TID_WIDTH     = VX_TAG_WIDTH,
    
    localparam VX_BYTEEN_WIDTH  = (VX_DATA_WIDTH / 8),
    localparam AXI_STROBE_WIDTH = (AXI_DATA_WIDTH / 8)
) (
    // Vortex request
    input wire                          mem_req_valid,
    input wire                          mem_req_rw,
    input wire [VX_BYTEEN_WIDTH-1:0]    mem_req_byteen,
    input wire [VX_ADDR_WIDTH-1:0]      mem_req_addr,
    input wire [VX_DATA_WIDTH-1:0]      mem_req_data,
    input wire [VX_TAG_WIDTH-1:0]       mem_req_tag,

    // Vortex response
    input wire                          mem_rsp_ready,
    output wire                         mem_rsp_valid,        
    output wire [VX_DATA_WIDTH-1:0]     mem_rsp_data,
    output wire [VX_TAG_WIDTH-1:0]      mem_rsp_tag,
    output wire                         mem_req_ready,

    // AXI write request
    output wire                         m_axi_wvalid,
    output wire                         m_axi_awvalid,
    output wire [AXI_TID_WIDTH-1:0]     m_axi_awid,
    output wire [AXI_ADDR_WIDTH-1:0]    m_axi_awaddr,
    output wire [7:0]                   m_axi_awlen,
    output wire [2:0]                   m_axi_awsize,
    output wire [1:0]                   m_axi_awburst,    
    output wire [AXI_DATA_WIDTH-1:0]    m_axi_wdata,
    output wire [AXI_STROBE_WIDTH-1:0]  m_axi_wstrb,    
    input wire                          m_axi_wready,
    input wire                          m_axi_awready,
    
    // AXI read request
    output wire                         m_axi_arvalid,
    output wire [AXI_TID_WIDTH-1:0]     m_axi_arid,
    output wire [AXI_ADDR_WIDTH-1:0]    m_axi_araddr,
    output wire [7:0]                   m_axi_arlen,
    output wire [2:0]                   m_axi_arsize,
    output wire [1:0]                   m_axi_arburst,    
    input wire                          m_axi_arready,
    
    // AXI read response
    input wire                          m_axi_rvalid,
    input wire [AXI_TID_WIDTH-1:0]      m_axi_rid,
    input wire [AXI_DATA_WIDTH-1:0]     m_axi_rdata,  
    output wire                         m_axi_rready
);
    localparam AXSIZE = $clog2(VX_DATA_WIDTH/8);

    `STATIC_ASSERT((AXI_DATA_WIDTH == VX_DATA_WIDTH), ("invalid parameter"))
    `STATIC_ASSERT((AXI_TID_WIDTH == VX_TAG_WIDTH), ("invalid parameter"))

    // AXI write channel    
    assign m_axi_wvalid     = mem_req_valid & mem_req_rw;
    assign m_axi_awvalid    = mem_req_valid & mem_req_rw;
    assign m_axi_awid       = mem_req_tag;
    assign m_axi_awaddr     = AXI_ADDR_WIDTH'(mem_req_addr) << AXSIZE;    
    assign m_axi_awlen      = 8'b00000000;
    assign m_axi_awsize     = 3'(AXSIZE);
    assign m_axi_awburst    = 2'b00;
    assign m_axi_wdata      = mem_req_data;
    assign m_axi_wstrb      = mem_req_byteen;
	
    // AXI read channel
    assign m_axi_arvalid    = mem_req_valid & ~mem_req_rw;
    assign m_axi_arid       = mem_req_tag;
    assign m_axi_araddr     = AXI_ADDR_WIDTH'(mem_req_addr) << AXSIZE;        
    assign m_axi_arlen      = 8'b00000000;
    assign m_axi_arsize     = 3'(AXSIZE);
    assign m_axi_arburst    = 2'b00;
    assign m_axi_rready     = mem_rsp_ready;

    // Vortex inputs
    assign mem_rsp_valid    = m_axi_rvalid;
    assign mem_rsp_tag      = m_axi_rid;
    assign mem_rsp_data     = m_axi_rdata;
	assign mem_req_ready    = mem_req_rw ? (m_axi_awready && m_axi_wready) : m_axi_arready;

endmodule