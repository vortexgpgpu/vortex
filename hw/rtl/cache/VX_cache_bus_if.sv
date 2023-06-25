`include "VX_cache_define.vh"

interface VX_cache_bus_if #(
    parameter NUM_REQS   = 1,
    parameter WORD_SIZE  = 1,
    parameter TAG_WIDTH  = 1,    
    parameter ADDR_WIDTH = `MEM_ADDR_WIDTH - `CLOG2(WORD_SIZE),
    parameter DATA_WIDTH = WORD_SIZE * 8
) ();

    wire [NUM_REQS-1:0]                 req_valid;
    wire [NUM_REQS-1:0]                 req_rw;
    wire [NUM_REQS-1:0][WORD_SIZE-1:0]  req_byteen;
    wire [NUM_REQS-1:0][ADDR_WIDTH-1:0] req_addr;
    wire [NUM_REQS-1:0][DATA_WIDTH-1:0] req_data;
    wire [NUM_REQS-1:0][TAG_WIDTH-1:0]  req_tag;
    wire [NUM_REQS-1:0]                 req_ready;

    wire [NUM_REQS-1:0]                 rsp_valid;
    wire [NUM_REQS-1:0][`CS_WORD_WIDTH-1:0] rsp_data;
    wire [NUM_REQS-1:0][TAG_WIDTH-1:0]  rsp_tag;
    wire [NUM_REQS-1:0]                 rsp_ready;

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
