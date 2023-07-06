`include "VX_define.vh"

interface VX_scoreboard_if ();

    wire [`NUM_WARPS-1:0]               valid;    
    wire [`NUM_WARPS-1:0][`UP(`UUID_BITS)-1:0] uuid;
    wire [`NUM_WARPS-1:0][`NUM_THREADS-1:0] tmask;  
    wire [`NUM_WARPS-1:0][`XLEN-1:0]    PC;
    wire [`NUM_WARPS-1:0][`NR_BITS-1:0] rd;
    wire [`NUM_WARPS-1:0][`NR_BITS-1:0] rs1;
    wire [`NUM_WARPS-1:0][`NR_BITS-1:0] rs2;
    wire [`NUM_WARPS-1:0][`NR_BITS-1:0] rs3;
    wire [`NUM_WARPS-1:0]               ready;

    modport master (
        output valid,
        output uuid,
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
        input  tmask,
        input  PC,
        input  rd,
        input  rs1,
        input  rs2,
        input  rs3,    
        output ready
    );
    
endinterface
