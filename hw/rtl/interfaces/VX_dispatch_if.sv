`include "VX_define.vh"

interface VX_dispatch_if ();

    wire                        valid;    
    wire [`UP(`UUID_BITS)-1:0]  uuid;
    wire [`UP(`NW_BITS)-1:0]    wid;
    wire [`NUM_THREADS-1:0]     tmask;
    wire [31:0]                 PC;
    wire [`EX_BITS-1:0]         ex_type;    
    wire [`INST_OP_BITS-1:0]    op_type; 
    wire [`INST_MOD_BITS-1:0]   op_mod;    
    wire                        wb;
    wire                        use_PC;
    wire                        use_imm;
    wire [`XLEN-1:0]            imm;
    wire [`NR_BITS-1:0]         rd;

    wire                        ready;

    modport master (
        output valid,
        output uuid,
        output wid,
        output tmask,
        output PC,
        output ex_type,
        output op_type, 
        output op_mod,    
        output wb,
        output use_PC,
        output use_imm,
        output imm,
        output rd,
        input  ready
    );

    modport slave (
        input  valid,
        input  uuid,
        input  wid,
        input  tmask,
        input  PC,
        input  ex_type,
        input  op_type, 
        input  op_mod,    
        input  wb,
        input  use_PC,
        input  use_imm,
        input  imm,
        input  rd,
        output ready
    );
    
endinterface
