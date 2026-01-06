`ifndef DICE_CONFIG_VH
`define DICE_CONFIG_VH

// =========================================================
// Global architectural configuration constants
// =========================================================

// Available registers
`define DICE_GPR_NUM                 16 // General Purpose Registers
`define DICE_PR_NUM                  8  // Predicate Registers
`define DICE_CR_NUM                  8  // Constant Registers

// Available CGRA memory ports
`define DICE_CGRA_MEM_PORTS          4

// Architectural configurations
`define DICE_ADDR_WIDTH              32
`define DICE_MAX_KERNEL_ID           65536
`define DICE_MAX_GRID_SIZE           65536
`define DICE_NUM_CGRA_CLUSTERS        4
`define DICE_NUM_CGRA_CORES           1
`define DICE_NUM_MAX_THREADS_PER_CORE 512
`define DICE_NUM_MAX_CTA_PER_CORE     4
`define DICE_NUM_RETIRE_TABLE_ENTRIES 4
`define DICE_SMEM_SIZE_PER_CORE     16384  // in Bytes
`define DICE_L1_LINE_SIZE            128   // in Bytes
`define DICE_L2_LINE_SIZE            128   // in Bytes
`define DICE_L3_LINE_SIZE            128   // in Bytes

// P-graph configuration
`define DICE_MAX_PGRAPHS              256   // Maximum p-graphs per kernel

`define DICE_GPR_NUM                 16    // General Purpose Registers
`define DICE_PR_NUM                  8     // Predicate Registers
`define DICE_CR_NUM                  8     // Constant Registers
`define DICE_CGRA_MEM_PORTS          4     // Available CGRA memory ports

`endif // DICE_CONFIG_VH
