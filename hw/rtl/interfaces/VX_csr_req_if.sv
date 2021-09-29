`ifndef VX_CSR_REQ_IF
`define VX_CSR_REQ_IF

`include "VX_define.vh"

interface VX_csr_req_if ();

    wire                    valid;
    wire [`NW_BITS-1:0]     wid;
    wire [`NUM_THREADS-1:0] tmask;
    wire [31:0]             PC;
    wire [`INST_CSR_BITS-1:0] op_type;
    wire [`CSR_ADDR_BITS-1:0] addr;
    wire [31:0]             rs1_data;
    wire                    use_imm;
    wire [`NRI_BITS-1:0]    imm;
    wire [`NR_BITS-1:0]     rd;
    wire                    wb;    
    wire                    ready;

    modport master (
        output valid,
        output wid,
        output tmask,
        output PC,
        output op_type,
        output addr,
        output rs1_data,        
        output use_imm,
        output imm,
        output rd,
        output wb,
        input  ready
    );

    modport slave (
        input  valid,
        input  wid,
        input  tmask,
        input  PC,
        input  op_type,
        input  addr,
        input  rs1_data,        
        input  use_imm,
        input  imm,
        input  rd,
        input  wb,
        output ready
    );
    
endinterface

`endif