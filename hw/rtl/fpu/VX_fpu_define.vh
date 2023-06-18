`ifndef VX_FPU_DEFINE_VH
`define VX_FPU_DEFINE_VH

`include "VX_define.vh"

`ifndef SYNTHESIS
`include "float_dpi.vh"
`endif

`include "VX_fpu_types.vh"

`IGNORE_WARNINGS_BEGIN
import VX_fpu_types::*;
`IGNORE_WARNINGS_END

`ifdef XLEN_64
`ifdef FLEN_32
    `define FPU_RV64_F
`endif
`endif

`endif // VX_FPU_DEFINE_VH
