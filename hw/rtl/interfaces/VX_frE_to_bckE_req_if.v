`ifndef VX_FrE_to_BCKBE_REQ_IF
`define VX_FrE_to_BCKBE_REQ_IF

`include "VX_define.vh"

interface VX_frE_to_bckE_req_if ();

    wire [11:0]              csr_address;
    wire                     is_csr;
    wire                     csr_immed;
    wire [31:0]              csr_mask;
    wire [4:0]               rd;
    wire [4:0]               rs1;
    wire [4:0]               rs2;
    wire [4:0]               alu_op;
    wire [1:0]               wb;
    wire                     rs2_src;
    wire [31:0]              itype_immed;
    wire [`BYTE_EN_BITS-1:0] mem_read;
    wire [`BYTE_EN_BITS-1:0] mem_write;
    wire [2:0]               branch_type;
    wire [19:0]              upper_immed;
    wire [31:0]              curr_PC;
`IGNORE_WARNINGS_BEGIN
    wire                     ebreak;
`IGNORE_WARNINGS_END
    wire                     jalQual;
    wire                     jal;
    wire [31:0]              jal_offset;
    wire [31:0]              PC_next;
    wire [`NUM_THREADS-1:0]  valid;
    wire [`NW_BITS-1:0]      warp_num;

    // GPGPU stuff
    wire                     is_wspawn;
    wire                     is_tmc;   
    wire                     is_split; 
    wire                     is_barrier;

endinterface

`endif