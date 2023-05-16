`include "VX_tex_define.vh"

interface VX_tex_agent_if ();

    wire                                valid;      
    wire [`UP(`UUID_BITS)-1:0]          uuid;
    wire [`UP(`NW_BITS)-1:0]            wid;
    wire [`NUM_THREADS-1:0]             tmask;    
    wire [`XLEN-1:0]                    PC;    
    wire [`NR_BITS-1:0]                 rd;

    wire [1:0][`NUM_THREADS-1:0][31:0]  coords;
    wire [`NUM_THREADS-1:0][`TEX_LOD_BITS-1:0] lod;
    wire [`TEX_STAGE_BITS-1:0]          stage;
    
    wire                                ready;

    modport master (
        output valid,
        output uuid,
        output wid,
        output tmask,
        output PC,
        output rd,
        output coords,
        output lod,
        output stage,
        input  ready
    );

    modport slave (
        input  valid,
        input  uuid,
        input  wid,
        input  tmask,
        input  PC,
        input  rd,
        input  coords,
        input  lod,
        input  stage,
        output ready
    );

endinterface
