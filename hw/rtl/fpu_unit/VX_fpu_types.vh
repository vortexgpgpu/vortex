`ifndef VX_FPU_TYPES_VH
`define VX_FPU_TYPES_VH

`include "VX_define.vh"

package VX_fpu_types;

typedef struct packed {
    logic is_normal;
    logic is_zero;
    logic is_subnormal;
    logic is_inf;
    logic is_nan;
    logic is_quiet;
    logic is_signaling;    
} fclass_t;

typedef struct packed {
    logic NV; // 4-Invalid
    logic DZ; // 3-Divide by zero
    logic OF; // 2-Overflow
    logic UF; // 1-Underflow
    logic NX; // 0-Inexact
} fflags_t;

endpackage

`define FP_CLASS_BITS   $bits(VX_fpu_types::fclass_t)
`define FP_FLAGS_BITS   $bits(VX_fpu_types::fflags_t)

`endif // VX_FPU_TYPES_VH
