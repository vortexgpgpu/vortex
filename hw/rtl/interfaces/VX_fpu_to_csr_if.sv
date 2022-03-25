`ifndef VX_FPU_TO_CSR_IF
`define VX_FPU_TO_CSR_IF

`include "VX_define.vh"

interface VX_fpu_to_csr_if ();

    wire                 write_enable;
    wire [`NW_BITS-1:0]  write_wid;
    fpu_types::fflags_t  write_fflags;

    wire [`NW_BITS-1:0]  read_wid;
    wire [`INST_FRM_BITS-1:0] read_frm;

    modport master (
        output write_enable,
        output write_wid,
        output write_fflags,
        output read_wid,
        input  read_frm
    );

    modport slave (
        input  write_enable,
        input  write_wid,
        input  write_fflags,
        input  read_wid,
        output read_frm
    );

endinterface

`endif
