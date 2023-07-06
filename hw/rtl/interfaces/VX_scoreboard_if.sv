`include "VX_define.vh"

interface VX_scoreboard_if ();

    wire                valid;    
    wire [`UP(`UUID_BITS)-1:0] uuid;
    wire [`UP(`NW_BITS)-1:0] wid;
    wire [`NUM_THREADS-1:0] tmask;  
    wire [`XLEN-1:0]    PC;
    wire [`NR_BITS-1:0] rd;
    wire [`NR_BITS-1:0] rs1;
    wire [`NR_BITS-1:0] rs2;
    wire [`NR_BITS-1:0] rs3;
    wire                ready;

    modport master (
        output valid,
        output uuid,
        output wid,
        output tmask,
        output PC,
        output rd,
        output rs1,
        output rs2,
        output rs3,
        input  ready
    );

    modport slave (
        input  valid,
        input  uuid,
        input  wid,
        input  tmask,
        input  PC,
        input  rd,
        input  rs1,
        input  rs2,
        input  rs3,    
        output ready
    );
    
endinterface
