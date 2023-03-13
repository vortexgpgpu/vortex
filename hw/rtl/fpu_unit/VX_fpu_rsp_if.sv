`include "VX_define.vh"
`include "VX_fpu_types.vh"

`IGNORE_WARNINGS_BEGIN
import VX_fpu_types::*;
`IGNORE_WARNINGS_END

interface VX_fpu_rsp_if #(
    parameter NUM_LANES = 1,
    parameter TAG_WIDTH = 1
) ();

    wire                        valid;
    wire [NUM_LANES-1:0][`XLEN-1:0] result; 
    fflags_t [NUM_LANES-1:0]    fflags;
    wire                        has_fflags;       
    wire [TAG_WIDTH-1:0]        tag;    
    wire                        ready;

    modport master (
        output valid,
        output result,        
        output fflags,
        output has_fflags,
        output tag,
        input  ready
    );

    modport slave (
        input  valid,
        input  result,        
        input  fflags,
        input  has_fflags,
        input  tag,
        output ready
    );

endinterface
