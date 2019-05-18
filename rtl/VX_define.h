

#define NT 4
#define NT_M1 (NT-1)

#define NW 8


#define R_INST 51
#define L_INST 3
#define ALU_INST 19
#define S_INST 35
#define B_INST 99
#define LUI_INST 55
#define AUIPC_INST 23
#define JAL_INST 111
#define JALR_INST 103
#define SYS_INST 115


#define WB_ALU 1
#define WB_MEM 2
#define WB_JAL 3
#define NO_WB  0


#define RS2_IMMED 1
#define RS2_REG 0


#define NO_MEM_READ  7
#define LB_MEM_READ  0
#define LH_MEM_READ  1
#define LW_MEM_READ  2
#define LBU_MEM_READ 4
#define LHU_MEM_READ 5


#define NO_MEM_WRITE 7
#define SB_MEM_WRITE 0
#define SH_MEM_WRITE 1
#define SW_MEM_WRITE 2


#define NO_BRANCH 0
#define BEQ 1
#define BNE 2
#define BLT 3
#define BGT 4
#define BLTU 5
#define BGTU 6


#define NO_ALU 15
#define ADD 0
#define SUB 1
#define SLLA 2
#define SLT 3
#define SLTU 4
#define XOR 5
#define SRL 6
#define SRA 7
#define OR 8
#define AND 9
#define SUBU 10
#define LUI_ALU 11
#define AUIPC_ALU 12
#define CSR_ALU_RW 13
#define CSR_ALU_RS 14
#define CSR_ALU_RC 15



// WRITEBACK
#define WB_ALU 1
#define WB_MEM 2
#define WB_JAL 3
#define NO_WB  0


// JAL
#define JUMP 1
#define NO_JUMP 0

// STALLS
#define STALL 1
#define NO_STALL 0


#define TAKEN 1
#define NOT_TAKEN 0


#define ZERO_REG 0


// COLORS
#define GREEN "\033[32m"
#define RED "\033[31m"
#define DEFAULT "\033[39m"







