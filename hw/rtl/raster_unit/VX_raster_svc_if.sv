`include "VX_raster_define.vh"

import VX_raster_types::*;

interface VX_raster_svc_if ();

    wire                        valid;

    wire [`UUID_BITS-1:0]       uuid;
    wire [`NW_BITS-1:0]         wid;
    wire [`NUM_THREADS-1:0]     tmask;
    wire [31:0]                 PC;
    wire [`NR_BITS-1:0]         rd;
    wire                        wb;

    wire                        ready;

    modport master (
        output valid,
        output uuid,
        output wid,
        output tmask,
        output PC,
        output rd,
        output wb,
        input  ready
    );

    modport slave (
        input  valid,
        input  uuid,
        input  wid,
        input  tmask,
        input  PC,
        input  rd,
        input  wb,
        output ready
    );

endinterface
