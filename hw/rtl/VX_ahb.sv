`include "ahb_pkg.sv"
`include "VX_define.vh"

module VX_ahb 
(
    input logic VX_clk, VX_reset,
    ahb_if.manager ahbif
); 
    bus_protocol_if VX_bus_protocol_if(); 
    ahb_manager VX_ahb_m (VX_bus_protocol_if, ahbif); 

    logic                            mem_req_valid; 
    logic                            mem_req_rw;  
    logic [`VX_MEM_BYTEEN_WIDTH-1:0] mem_req_byteen;
    logic [`VX_MEM_ADDR_WIDTH-1:0]   mem_req_addr;
    logic [`VX_MEM_DATA_WIDTH-1:0]   mem_req_data;
    logic [`VX_MEM_TAG_WIDTH-1:0]    mem_req_tag;
    logic                            mem_req_ready;

    logic                            mem_rsp_valid;        
    logic [`VX_MEM_DATA_WIDTH-1:0]   mem_rsp_data;
    logic [`VX_MEM_TAG_WIDTH-1:0]    mem_rsp_tag;
    logic                            mem_rsp_ready;
    logic busy; 

    logic [`VX_MEM_TAG_WIDTH-1:0] dummy_tag = '1; 

    Vortex vortex (
        .clk            (VX_clk),
        .reset          (VX_reset),

        .mem_req_valid  (mem_req_valid),
        .mem_req_rw     (mem_req_rw),
        .mem_req_byteen (mem_req_byteen),
        .mem_req_addr   (mem_req_addr),
        .mem_req_data   (mem_req_data),
        .mem_req_tag    (mem_req_tag),
        .mem_req_ready  (mem_req_ready),

        .mem_rsp_valid  (mem_rsp_valid),
        .mem_rsp_data   (mem_rsp_data),
        .mem_rsp_tag    (mem_rsp_tag),
        .mem_rsp_ready  (mem_rsp_ready),

        .busy           (busy)
    );

    // VX to AHB signals 
    assign VX_bus_protocol_if.wen = mem_req_rw; 
    assign VX_bus_protocol_if.ren = ~mem_req_rw; 
    assign VX_bus_protocol_if.addr = mem_req_addr; 
    assign VX_bus_protocol_if.wdata = mem_req_data; 
    assign VX_bus_protocol_if.strobe = mem_req_byteen; 

    assign mem_rsp_valid = ~VX_bus_protocol_if.error;
    assign mem_rsp_tag = dummy_tag;
    assign mem_rsp_data = VX_bus_protocol_if.rdata; 
    assign mem_req_ready = mem_req_rw ? ~VX_bus_protocol_if.request_stall : ahbif.hready; // Write: can be stalled, Read: HREADY signal

endmodule 