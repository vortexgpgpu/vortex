`ifndef VX_MEM_REQ_IF
`define VX_MEM_REQ_IF

`include "../cache/VX_cache_define.vh"

interface VX_mem_req_if #(
    parameter DATA_WIDTH = 1,
    parameter ADDR_WIDTH = 1,
    parameter TAG_WIDTH  = 1,
    parameter DATA_SIZE  = DATA_WIDTH / 8
) ();

    wire                    valid;    
    wire                    rw;    
    wire [DATA_SIZE-1:0]    byteen;
    wire [ADDR_WIDTH-1:0]   addr;
    wire [DATA_WIDTH-1:0]   data;  
    wire [TAG_WIDTH-1:0]    tag;  
    wire                    ready;

    modport master (
        output valid,    
        output rw,
        output byteen,
        output addr,
        output data,
        output tag,
        input  ready
    );

    modport slave (
        input  valid,   
        input  rw,
        input  byteen,
        input  addr,
        input  data,
        input  tag,
        output ready
    );

endinterface

`define ASSIGN_VX_MEM_REQ_IF(dst, src) \
    assign dst.valid  = src.valid;  \
    assign dst.rw     = src.rw;     \
    assign dst.byteen = src.byteen; \
    assign dst.addr   = src.addr;   \
    assign dst.data   = src.data;   \
    assign dst.tag    = src.tag;    \
    assign src.ready  = dst.ready

`define ASSIGN_VX_MEM_REQ_IF_XTAG(dst, src) \
    assign dst.valid  = src.valid;  \
    assign dst.rw     = src.rw;     \
    assign dst.byteen = src.byteen; \
    assign dst.addr   = src.addr;   \
    assign dst.data   = src.data;   \
    assign src.ready  = dst.ready

`endif
