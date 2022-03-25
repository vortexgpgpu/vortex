`ifndef VX_TEX_DCR_IF
`define VX_TEX_DCR_IF

`include "VX_tex_define.vh"

interface VX_tex_dcr_if #(
    parameter NUM_STAGES
);
    tex_dcrs_t data [NUM_STAGES-1:0];
    
    modport master (
        output  data
    );

    modport slave (
        input   data
    );

endinterface

`endif
