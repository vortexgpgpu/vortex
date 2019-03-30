
`define NT 2
`define NT_M1 1
`define NT_T2_M1 3


`define R_INST 7'd51
`define L_INST 7'd3
`define ALU_INST 7'd19
`define S_INST 7'd35
`define B_INST 7'd99
`define LUI_INST 7'd55
`define AUIPC_INST 7'd23
`define JAL_INST 7'd111
`define JALR_INST 7'd103
`define SYS_INST 7'd115


`define WB_ALU 2'h1
`define WB_MEM 2'h2
`define WB_JAL 2'h3
`define NO_WB  2'h0


`define RS2_IMMED 1
`define RS2_REG 0


`define NO_MEM_READ  3'h7
`define LB_MEM_READ  3'h0
`define LH_MEM_READ  3'h1
`define LW_MEM_READ  3'h2
`define LBU_MEM_READ 3'h4
`define LHU_MEM_READ 3'h5


`define NO_MEM_WRITE 3'h7
`define SB_MEM_WRITE 3'h0
`define SH_MEM_WRITE 3'h1
`define SW_MEM_WRITE 3'h2


`define NO_BRANCH 3'h0
`define BEQ 3'h1
`define BNE 3'h2
`define BLT 3'h3
`define BGT 3'h4
`define BLTU 3'h5
`define BGTU 3'h6


`define NO_ALU 5'd15
`define ADD 5'd0
`define SUB 5'd1
`define SLLA 5'd2
`define SLT 5'd3
`define SLTU 5'd4
`define XOR 5'd5
`define SRL 5'd6
`define SRA 5'd7
`define OR 5'd8
`define AND 5'd9
`define SUBU 5'd10
`define LUI_ALU 5'd11
`define AUIPC_ALU 5'd12
`define CSR_ALU_RW 5'd13
`define CSR_ALU_RS 5'd14
`define CSR_ALU_RC 5'd15
`define MUL 5'd16
`define MULH 5'd17
`define MULHSU 5'd18
`define MULHU 5'd19
`define DIV 5'd20
`define DIVU 5'd21
`define REM 5'd22
`define REMU 5'd23



// WRITEBACK
`define WB_ALU 2'h1
`define WB_MEM 2'h2
`define WB_JAL 2'h3
`define NO_WB  2'h0


// JAL
`define JUMP 1'h1
`define NO_JUMP 1'h0

// STALLS
`define STALL 1'h1
`define NO_STALL 1'h0


`define TAKEN 1'b1
`define NOT_TAKEN 1'b0


`define ZERO_REG 5'h0








