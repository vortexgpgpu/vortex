`include "VX_define.vh"

interface VX_ifetch_rsp_if ();

    wire                        valid;
    wire [`UP(`UUID_BITS)-1:0]  uuid;
    wire [`NUM_THREADS-1:0]     tmask;    
    wire [`UP(`NW_BITS)-1:0]    wid;
    wire [`XLEN-1:0]            PC;
    wire [31:0]                 data;
    wire [`NUM_WARPS-1:0]       ibuf_pop;
    wire                        ready;

    modport master (
        output valid,
        output uuid, 
        output tmask,
        output wid,
        output PC,
        output data,
        input  ibuf_pop,
        input  ready
    );

    modport slave (
        input  valid,
        input  uuid,
        input  tmask,
        input  wid,
        input  PC,
        input  data,
        output ibuf_pop,
        output ready
    );

endinterface
