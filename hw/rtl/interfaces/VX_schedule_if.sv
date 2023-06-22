`include "VX_define.vh"

interface VX_schedule_if ();

    wire                        valid;
    wire [`UP(`UUID_BITS)-1:0]  uuid;
    wire [`NUM_THREADS-1:0]     tmask;    
    wire [`UP(`NW_BITS)-1:0]    wid;
    wire [`XLEN-1:0]            PC;
    wire                        ready;

    modport master (
        output valid, 
        output uuid,   
        output tmask,
        output wid,
        output PC,
        input  ready
    );

    modport slave (
        input  valid, 
        input  uuid,  
        input  tmask,
        input  wid,
        input  PC,
        output ready
    );

endinterface
