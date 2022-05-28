`ifndef VX_GPU_TYPES_VH
`define VX_GPU_TYPES_VH

`include "VX_define.vh"

package VX_gpu_types;

typedef struct packed {
    logic                    valid;
    logic [`NUM_THREADS-1:0] tmask;
} gpu_tmc_t;

typedef struct packed {
    logic                   valid;
    logic [`NUM_WARPS-1:0]  wmask;
    logic [31:0]            pc;
} gpu_wspawn_t;

typedef struct packed {
    logic                   valid;
    logic                   diverged;
    logic [`NUM_THREADS-1:0] then_tmask;
    logic [`NUM_THREADS-1:0] else_tmask;
    logic [31:0]            pc;
} gpu_split_t;

typedef struct packed {
    logic                   valid;
    logic [`NB_BITS-1:0]    id;
    logic [`NW_BITS-1:0]    size_m1;
} gpu_barrier_t;

typedef struct packed {
    logic [`XLEN-1:0]       startup_addr;
    logic [7:0]             mpm_class;
} base_dcrs_t;

endpackage

`define GPU_TMC_BITS        $bits(VX_gpu_types::gpu_tmc_t)
`define GPU_WSPAWN_BITS     $bits(VX_gpu_types::gpu_wspawn_t)
`define GPU_SPLIT_BITS      $bits(VX_gpu_types::gpu_split_t)
`define GPU_BARRIER_BITS    $bits(VX_gpu_types::gpu_barrier_t)

`endif
