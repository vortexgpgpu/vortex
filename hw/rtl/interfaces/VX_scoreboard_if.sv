`include "VX_define.vh"

interface VX_scoreboard_if ();

    wire                        valid;    
    wire [`UP(`UUID_BITS)-1:0]  uuid;
    wire [`UP(`NW_BITS)-1:0]    wid;
    wire [`NUM_THREADS-1:0]     tmask;
    wire [`XLEN-1:0]            PC;   
    wire                        wb;
    wire [`NR_BITS-1:0]         rd;
    
    wire [`NR_BITS-1:0]         rd_n;
    wire [`NR_BITS-1:0]         rs1_n;
    wire [`NR_BITS-1:0]         rs2_n;
    wire [`NR_BITS-1:0]         rs3_n;
    wire [`UP(`NW_BITS)-1:0]    wid_n;

    wire [3:0]                  used_regs;

    wire                        ready;

    modport master (
        output valid,
        output uuid,
        output wid,
        output tmask,
        output PC,
        output wb,
        output rd,
        output rd_n,
        output rs1_n,
        output rs2_n,
        output rs3_n,
        output wid_n,
        input  used_regs, 
        input  ready
    );

    modport slave (
        input  valid,
        input  uuid,
        input  wid,
        input  tmask,
        input  PC,
        input  wb,
        input  rd,
        input  rd_n,
        input  rs1_n,
        input  rs2_n,
        input  rs3_n,
        input  wid_n,
        output used_regs,        
        output ready
    );
    
endinterface
