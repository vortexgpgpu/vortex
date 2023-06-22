`include "VX_define.vh"
`include "VX_fpu_types.vh"

`IGNORE_WARNINGS_BEGIN
import VX_fpu_types::*;
`IGNORE_WARNINGS_END

interface VX_fpu_bus_if #(
    parameter NUM_LANES = 1,
    parameter TAG_WIDTH = 1
) ();

    wire                        req_valid;
    wire [`INST_FPU_BITS-1:0]   req_type;
    wire [`INST_FMT_BITS-1:0]   req_fmt;
    wire [`INST_FRM_BITS-1:0]   req_frm;
    wire [NUM_LANES-1:0][`XLEN-1:0] req_dataa;
    wire [NUM_LANES-1:0][`XLEN-1:0] req_datab;
    wire [NUM_LANES-1:0][`XLEN-1:0] req_datac;
    wire [TAG_WIDTH-1:0]        req_tag; 
    wire                        req_ready;

    wire                        rsp_valid;
    wire [NUM_LANES-1:0][`XLEN-1:0] rsp_result; 
    fflags_t [NUM_LANES-1:0]    rsp_fflags;
    wire                        rsp_has_fflags;       
    wire [TAG_WIDTH-1:0]        rsp_tag;    
    wire                        rsp_ready;

    modport master (
        output req_valid,
        output req_type,
        output req_fmt,
        output req_frm,
        output req_dataa,
        output req_datab,
        output req_datac,
        output req_tag,
        input  req_ready,

        input  rsp_valid,
        input  rsp_result,        
        input  rsp_fflags,
        input  rsp_has_fflags,
        input  rsp_tag,
        output rsp_ready
    );

    modport slave (
        input  req_valid,
        input  req_type,
        input  req_fmt,
        input  req_frm,
        input  req_dataa,
        input  req_datab,
        input  req_datac,
        input  req_tag,
        output req_ready,

        output rsp_valid,
        output rsp_result,        
        output rsp_fflags,
        output rsp_has_fflags,
        output rsp_tag,
        input  rsp_ready
    );

endinterface
