`include "VX_define.vh"
`include "VX_fpu_types.vh"

`IGNORE_WARNINGS_BEGIN
import VX_fpu_types::*;
`IGNORE_WARNINGS_END

interface VX_fpu_to_csr_if ();

    wire                     write_enable;
    wire [`UP(`NW_BITS)-1:0] write_wid;
    fflags_t                 write_fflags;

    wire [`UP(`NW_BITS)-1:0] read_wid;
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
