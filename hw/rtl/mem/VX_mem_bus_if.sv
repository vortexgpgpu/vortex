`include "VX_define.vh"

interface VX_mem_bus_if #(
    parameter DATA_WIDTH = 1,    
    parameter DATA_SIZE  = DATA_WIDTH / 8,
    parameter ADDR_WIDTH = `MEM_ADDR_WIDTH - `CLOG2(DATA_SIZE),
    parameter TAG_WIDTH  = 1    
) ();

    wire                    req_valid;    
    wire                    req_rw;    
    wire [DATA_SIZE-1:0]    req_byteen;
    wire [ADDR_WIDTH-1:0]   req_addr;
    wire [DATA_WIDTH-1:0]   req_data;  
    wire [TAG_WIDTH-1:0]    req_tag;  
    wire                    req_ready;

    wire                    rsp_valid;  
    wire [DATA_WIDTH-1:0]   rsp_data;
    wire [TAG_WIDTH-1:0]    rsp_tag;
    wire                    rsp_ready;

    modport master (
        output req_valid,    
        output req_rw,
        output req_byteen,
        output req_addr,
        output req_data,
        output req_tag,
        input  req_ready,

        input  rsp_valid,
        input  rsp_data,
        input  rsp_tag,
        output rsp_ready
    );

    modport slave (
        input  req_valid,   
        input  req_rw,
        input  req_byteen,
        input  req_addr,
        input  req_data,
        input  req_tag,
        output req_ready,

        output rsp_valid,
        output rsp_data,
        output rsp_tag,
        input  rsp_ready
    );

endinterface
