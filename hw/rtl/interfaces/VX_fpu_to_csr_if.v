`ifndef VX_FPU_TO_CSR_IF
`define VX_FPU_TO_CSR_IF

`include "VX_define.vh"

interface VX_fpu_to_csr_if ();

    wire                 write_enable;
    wire [`NW_BITS-1:0]  write_wid;
    fflags_t             write_fflags;

    wire [`NW_BITS-1:0]  read_wid;
    wire [`FRM_BITS-1:0] read_frm;

endinterface

`endif