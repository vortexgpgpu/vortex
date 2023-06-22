`include "VX_define.vh"

interface VX_lsu_exe_if ();

    wire                            valid;
    wire [`UP(`UUID_BITS)-1:0]      uuid; 
    wire [`UP(`NW_BITS)-1:0]        wid;
    wire [`NUM_THREADS-1:0]         tmask;    
    wire [`XLEN-1:0]                PC;
    wire [`INST_LSU_BITS-1:0]       op_type;
    wire [`NUM_THREADS-1:0][`XLEN-1:0] store_data;
    wire [`NUM_THREADS-1:0][`XLEN-1:0] base_addr;    
    wire [`XLEN-1:0]                offset;
    wire [`NR_BITS-1:0]             rd;
    wire                            wb;
    wire                            ready;

    modport master (
        output valid,
        output uuid,
        output wid,
        output tmask,
        output PC,
        output op_type,
        output store_data,
        output base_addr,   
        output offset,
        output rd,
        output wb,
        input  ready
    );

    modport slave (
        input  valid,
        input  uuid,
        input  wid,
        input  tmask,
        input  PC,
        input  op_type,
        input  store_data,
        input  base_addr,   
        input  offset,
        input  rd,
        input  wb,
        output ready
    );

endinterface
