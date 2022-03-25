`ifndef VX_IBUFFER_IF
`define VX_IBUFFER_IF

`include "VX_define.vh"

interface VX_ibuffer_if ();

    wire                    valid;    
    wire [`UUID_BITS-1:0]   uuid;
    wire [`NW_BITS-1:0]     wid;
    wire [`NUM_THREADS-1:0] tmask;
    wire [31:0]             PC;
    wire [`EX_BITS-1:0]     ex_type;    
    wire [`INST_OP_BITS-1:0] op_type; 
    wire [`INST_MOD_BITS-1:0] op_mod;    
    wire                    wb;
    wire                    use_PC;
    wire                    use_imm;
    wire [31:0]             imm;
    wire [`NR_BITS-1:0]     rd;
    wire [`NR_BITS-1:0]     rs1;
    wire [`NR_BITS-1:0]     rs2;
    wire [`NR_BITS-1:0]     rs3;
    
    wire [`NR_BITS-1:0]     rd_n;
    wire [`NR_BITS-1:0]     rs1_n;
    wire [`NR_BITS-1:0]     rs2_n;
    wire [`NR_BITS-1:0]     rs3_n;
    wire [`NW_BITS-1:0]     wid_n;

    wire                    ready;

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
        output rs1,
        output rs2,
        output rs3,
        output rd_n,
        output rs1_n,
        output rs2_n,
        output rs3_n,
        output wid_n,        
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
        input  rs1,
        input  rs2,
        input  rs3,
        input  rd_n,
        input  rs1_n,
        input  rs2_n,
        input  rs3_n,
        input  wid_n,        
        output ready
    );
    
endinterface

`endif
