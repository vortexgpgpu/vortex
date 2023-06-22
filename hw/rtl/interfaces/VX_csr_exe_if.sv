`include "VX_define.vh"

interface VX_csr_exe_if ();

    wire                        valid;
    wire [`UP(`UUID_BITS)-1:0]  uuid;
    wire [`UP(`NW_BITS)-1:0]    wid;
    wire [`NUM_THREADS-1:0]     tmask;
    wire [`XLEN-1:0]            PC;
    wire [`INST_CSR_BITS-1:0]   op_type;
    wire [`CSR_ADDR_BITS-1:0]   addr;
    wire [`UP(`NT_BITS)-1:0]    tid;
    wire [`NUM_THREADS-1:0][31:0] rs1_data;    
    wire                        use_imm;
    wire [`NRI_BITS-1:0]        imm;
    wire [`NR_BITS-1:0]         rd;
    wire                        wb;    
    wire                        ready;

    modport master (
        output valid,
        output uuid,
        output wid,
        output tmask,
        output PC,
        output op_type,
        output addr,
        output tid,
        output rs1_data,        
        output use_imm,
        output imm,
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
        input  addr,
        input  tid,
        input  rs1_data,        
        input  use_imm,
        input  imm,
        input  rd,
        input  wb,
        output ready
    );
    
endinterface
