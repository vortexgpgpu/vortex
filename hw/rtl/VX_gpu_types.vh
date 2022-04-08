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

endpackage