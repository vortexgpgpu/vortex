`include "VX_define.vh"

interface VX_dispatch_if ();

    wire                        valid;    
    wire [`UP(`UUID_BITS)-1:0]  uuid;
    wire [`UP(`NW_BITS)-1:0]    wid;
    wire [`NUM_THREADS-1:0]     tmask;
    wire [`XLEN-1:0]            PC;
    wire [`EX_BITS-1:0]         ex_type;    
    wire [`INST_OP_BITS-1:0]    op_type; 
    wire [`INST_MOD_BITS-1:0]   op_mod;    
    wire                        wb;
    wire                        use_PC;
    wire                        use_imm;
    wire [`XLEN-1:0]            imm;
    wire [`NR_BITS-1:0]         rd;

    wire [`NUM_THREADS-1:0][`XLEN-1:0] rs1_data;
    wire [`NUM_THREADS-1:0][`XLEN-1:0] rs2_data;
    wire [`NUM_THREADS-1:0][`XLEN-1:0] rs3_data;

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
        output rs1_data,
        output rs2_data,
        output rs3_data,
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
        input  rs1_data,
        input  rs2_data,
        input  rs3_data,
        output ready
    );
    
endinterface
