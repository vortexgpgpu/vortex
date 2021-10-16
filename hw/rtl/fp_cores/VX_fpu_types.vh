`ifndef VX_FPU_TYPES
`define VX_FPU_TYPES

`include "VX_define.vh"

package fpu_types;

typedef struct packed {
    logic is_normal;
    logic is_zero;
    logic is_subnormal;
    logic is_inf;
    logic is_nan;
    logic is_quiet;
    logic is_signaling;    
} fp_class_t;

`define FP_CLASS_BITS  $bits(fpu_types::fp_class_t)

typedef struct packed {
    logic NV; // 4-Invalid
    logic DZ; // 3-Divide by zero
    logic OF; // 2-Overflow
    logic UF; // 1-Underflow
    logic NX; // 0-Inexact
} fflags_t;

`define FFLAGS_BITS  $bits(fpu_types::fflags_t)

endpackage

`endif