`ifndef VX_MEM_RSP_IF
`define VX_MEM_RSP_IF

`include "../cache/VX_cache_define.vh"

interface VX_mem_rsp_if #(
    parameter DATA_WIDTH = 1,
    parameter TAG_WIDTH  = 1
) ();

    wire                    valid;    
    wire [DATA_WIDTH-1:0]   data;
    wire [TAG_WIDTH-1:0]    tag;  
    wire                    ready;  

    modport master (
        output valid,
        output data,
        output tag,
        input  ready
    );

    modport slave (
        input  valid,
        input  data,
        input  tag,
        output ready
    );    

endinterface    

`define ASSIGN_VX_MEM_RSP_IF(dst, src) \
    assign dst.valid  = src.valid;  \
    assign dst.data   = src.data;   \
    assign dst.tag    = src.tag;    \
    assign src.ready  = dst.ready

`define ASSIGN_VX_MEM_RSP_IF_XTAG(dst, src) \
    assign dst.valid  = src.valid;  \
    assign dst.data   = src.data;   \
    assign src.ready  = dst.ready

`endif
