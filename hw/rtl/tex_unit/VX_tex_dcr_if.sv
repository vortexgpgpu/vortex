`include "VX_tex_define.vh"

import VX_tex_types::*;

interface VX_tex_dcr_if #(
    parameter NUM_STAGES = 1
);
    tex_dcrs_t data [NUM_STAGES-1:0];
    
    modport master (
        output  data
    );

    modport slave (
        input   data
    );

endinterface
