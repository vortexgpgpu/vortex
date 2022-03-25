`ifndef VX_LSU_REQ_IF
`define VX_LSU_REQ_IF

`include "VX_define.vh"

interface VX_lsu_req_if ();

    wire                            valid;
    wire [`UUID_BITS-1:0]           uuid; 
    wire [`NW_BITS-1:0]             wid;
    wire [`NUM_THREADS-1:0]         tmask;    
    wire [31:0]                     PC;
    wire [`INST_LSU_BITS-1:0]       op_type;
    wire                            is_fence;
    wire [`NUM_THREADS-1:0][31:0]   store_data;
    wire [`NUM_THREADS-1:0][31:0]   base_addr;    
    wire [31:0]                     offset;
    wire [`NR_BITS-1:0]             rd;
    wire                            wb;
    wire                            ready;
    wire                            is_prefetch;

    modport master (
        output valid,
        output uuid,
        output wid,
        output tmask,
        output PC,
        output op_type,
        output is_fence,
        output store_data,
        output base_addr,   
        output offset,
        output rd,
        output wb,
        output is_prefetch,
        input  ready
    );

    modport slave (
        input  valid,
        input  uuid,
        input  wid,
        input  tmask,
        input  PC,
        input  op_type,
        input  is_fence,
        input  store_data,
        input  base_addr,   
        input  offset,
        input  rd,
        input  wb,
        input  is_prefetch,
        output ready
    );

endinterface

`endif
