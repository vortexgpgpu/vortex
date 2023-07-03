`include "VX_define.vh"

interface VX_gpu_perf_if ();

`ifdef EXT_TEX_ENABLE
    wire [`PERF_CTR_BITS-1:0] tex_stalls;
`endif
`ifdef EXT_RASTER_ENABLE
    wire [`PERF_CTR_BITS-1:0] raster_stalls;
`endif
`ifdef EXT_ROP_ENABLE
    wire [`PERF_CTR_BITS-1:0] rop_stalls;
`endif
    wire [`PERF_CTR_BITS-1:0] wctl_stalls;

    modport master (
    `ifdef EXT_TEX_ENABLE
        output tex_stalls,
    `endif
    `ifdef EXT_RASTER_ENABLE
        output raster_stalls,
    `endif
    `ifdef EXT_ROP_ENABLE
        output rop_stalls,
    `endif
        output wctl_stalls
    );

    modport slave (
    `ifdef EXT_TEX_ENABLE
        input tex_stalls,
    `endif
    `ifdef EXT_RASTER_ENABLE
        input raster_stalls,
    `endif
    `ifdef EXT_ROP_ENABLE
        input rop_stalls,
    `endif
        input wctl_stalls
    );

endinterface
