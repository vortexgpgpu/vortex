`include "VX_define.vh"
`include "VX_fpu_types.vh"

interface VX_fpu_req_if #(
    parameter NUM_LANES = 1,
    parameter TAG_WIDTH = 1
) ();

    wire                         valid;
    wire [`INST_FPU_BITS-1:0]    op_type;
    wire [`INST_FRM_BITS-1:0]    frm;
    wire [NUM_LANES-1:0][`XLEN-1:0] dataa;
    wire [NUM_LANES-1:0][`XLEN-1:0] datab;
    wire [NUM_LANES-1:0][`XLEN-1:0] datac;
    wire [TAG_WIDTH-1:0]         tag; 
    wire                         ready;

    modport master (
        output valid,
        output op_type,
        output frm,
        output dataa,
        output datab,
        output datac,
        output tag,
        input  ready
    );

    modport slave (
        input  valid,
        input  op_type,
        input  frm,
        input  dataa,
        input  datab,
        input  datac,
        input  tag,
        output ready
    );

endinterface
