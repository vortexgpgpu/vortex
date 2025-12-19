//
// created by liub92@mcmaster.ca

`include "VX_define.vh"

interface VX_lsu_rsp_if import VX_gpu_pkg::*; ();

    logic        valid;
    logic [NW_WIDTH-1:0] wid;

    modport master (
        output valid,
        output wid
    );

    modport slave (
        input valid,
        input wid
    );

endinterface

