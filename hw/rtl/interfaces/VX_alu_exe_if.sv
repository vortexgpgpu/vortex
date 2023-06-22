`include "VX_define.vh"

interface VX_alu_exe_if ();

    wire                        valid;  
    wire [`UP(`UUID_BITS)-1:0]  uuid; 
    wire [`UP(`NW_BITS)-1:0]    wid;
    wire [`NUM_THREADS-1:0]     tmask;
    wire [`XLEN-1:0]            PC;
    wire [`XLEN-1:0]            next_PC;
    wire [`INST_ALU_BITS-1:0]   op_type;
    wire [`INST_MOD_BITS-1:0]   op_mod;
    wire                        use_PC;
    wire                        use_imm;
    wire [`XLEN-1:0]            imm;
    wire [`UP(`NT_BITS)-1:0]    tid;
    wire [`NUM_THREADS-1:0][`XLEN-1:0] rs1_data;
    wire [`NUM_THREADS-1:0][`XLEN-1:0] rs2_data;
    wire [`NR_BITS-1:0]         rd;
    wire                        wb;    
    wire                        ready;

    modport master (
        output valid,
        output uuid,
        output wid,
        output tmask,
        output PC,
        output next_PC,
        output op_type,
        output op_mod,
        output use_PC,
        output use_imm,
        output imm,
        output tid,
        output rs1_data,
        output rs2_data,
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
        input  next_PC,
        input  op_type,
        input  op_mod,
        input  use_PC,
        input  use_imm,
        input  imm,
        input  tid,
        input  rs1_data,
        input  rs2_data,
        input  rd,
        input  wb,    
        output ready
    );

endinterface
