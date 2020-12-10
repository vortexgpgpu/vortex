`ifndef VX_TYPES
`define VX_TYPES

`include "VX_define.vh"

typedef struct packed {
    logic is_normal;
    logic is_zero;
    logic is_subnormal;
    logic is_inf;
    logic is_nan;
    logic is_signaling;
    logic is_quiet;
} fp_type_t;

typedef struct packed {
    logic NV; // Invalid
    logic DZ; // Divide by zero
    logic OF; // Overflow
    logic UF; // Underflow
    logic NX; // Inexact
} fflags_t;

`define FFG_BITS  $bits(fflags_t)

typedef struct packed {
    logic                    valid;
    logic [`NUM_THREADS-1:0] tmask;
} gpu_tmc_t;

`define GPU_TMC_SIZE (1+`NUM_THREADS)

typedef struct packed {
    logic                   valid;
    logic [`NUM_WARPS-1:0]  wmask;
    logic [31:0]            pc;
} gpu_wspawn_t;

`define GPU_WSPAWN_SIZE (1+`NUM_WARPS+32)

typedef struct packed {
    logic                   valid;
    logic                   diverged;
    logic [`NUM_THREADS-1:0] then_mask;
    logic [`NUM_THREADS-1:0] else_mask;
    logic [31:0]            pc;
} gpu_split_t;

`define GPU_SPLIT_SIZE (1+1+`NUM_THREADS+`NUM_THREADS+32)

typedef struct packed {
    logic                   valid;
    logic [`NB_BITS-1:0]    id;
    logic [`NW_BITS-1:0]    size_m1;
} gpu_barrier_t;

`define GPU_BARRIER_SIZE (1+`NB_BITS+`NW_BITS)

`endif