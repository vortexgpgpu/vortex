`ifndef VX_TYPES
`define VX_TYPES

`include "VX_define.vh"

typedef struct packed {
    logic is_normal;
    logic is_zero;
    logic is_subnormal;
    logic is_inf;
    logic is_nan;
    logic is_quiet;
    logic is_signaling;    
} fp_type_t;

typedef struct packed {
    logic NV; // 4-Invalid
    logic DZ; // 3-Divide by zero
    logic OF; // 2-Overflow
    logic UF; // 1-Underflow
    logic NX; // 0-Inexact
} fflags_t;

`define FFG_BITS  $bits(fflags_t)

typedef struct packed {
    logic                    valid;
    logic [`NUM_THREADS-1:0] tmask;
} gpu_tmc_t;

`define GPU_TMC_BITS (1+`NUM_THREADS)

typedef struct packed {
    logic                   valid;
    logic [`NUM_WARPS-1:0]  wmask;
    logic [31:0]            pc;
} gpu_wspawn_t;

`define GPU_WSPAWN_BITS (1+`NUM_WARPS+32)

typedef struct packed {
    logic                   valid;
    logic                   diverged;
    logic [`NUM_THREADS-1:0] then_mask;
    logic [`NUM_THREADS-1:0] else_mask;
    logic [31:0]            pc;
} gpu_split_t;

`define GPU_SPLIT_BITS (1+1+`NUM_THREADS+`NUM_THREADS+32)

typedef struct packed {
    logic                   valid;
    logic [`NB_BITS-1:0]    id;
    logic [`NW_BITS-1:0]    size_m1;
} gpu_barrier_t;

`define GPU_BARRIER_BITS (1+`NB_BITS+`NW_BITS)

`endif