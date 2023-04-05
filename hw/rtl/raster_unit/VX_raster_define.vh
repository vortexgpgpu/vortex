`ifndef VX_RASTER_DEFINE_VH
`define VX_RASTER_DEFINE_VH

`include "VX_define.vh"
`include "VX_gpu_types.vh"
`include "VX_raster_types.vh"

`IGNORE_WARNINGS_BEGIN
import VX_gpu_types::*;
import VX_raster_types::*;
`IGNORE_WARNINGS_END

`define EDGE_UPDATE(dst, src, eval) \
    assign dst[0][0] = src[0][0];   \
    assign dst[1][0] = src[1][0];   \
    assign dst[2][0] = src[2][0];   \
    assign dst[0][1] = src[0][1];   \
    assign dst[1][1] = src[1][1];   \
    assign dst[2][1] = src[2][1];   \
    assign dst[0][2] = eval[0];     \
    assign dst[1][2] = eval[1];     \
    assign dst[2][2] = eval[2]

`define TRACE_RASTER_DCR(level, state) \
    case (state) \
        `DCR_RASTER_TBUF_ADDR:   `TRACE(level, ("TBUF_ADDR")); \
        `DCR_RASTER_TILE_COUNT:  `TRACE(level, ("TILE_COUNT")); \
        `DCR_RASTER_PBUF_ADDR:   `TRACE(level, ("PBUF_ADDR")); \
        `DCR_RASTER_PBUF_STRIDE: `TRACE(level, ("PBUF_STRIDE")); \
        `DCR_RASTER_SCISSOR_X:   `TRACE(level, ("SCISSOR_X")); \
        `DCR_RASTER_SCISSOR_Y:   `TRACE(level, ("SCISSOR_Y")); \
        default:                 `TRACE(level, ("?")); \
    endcase

`define TRACE_RASTER_CSR(level, addr) \
    case (addr) \
        `CSR_RASTER_POS_MASK:   `TRACE(level, ("POS_MASK")); \
        `CSR_RASTER_BCOORD_X0:  `TRACE(level, ("BCOORD_X0")); \
        `CSR_RASTER_BCOORD_X1:  `TRACE(level, ("BCOORD_X1")); \
        `CSR_RASTER_BCOORD_X2:  `TRACE(level, ("BCOORD_X2")); \
        `CSR_RASTER_BCOORD_X3:  `TRACE(level, ("BCOORD_X3")); \
        `CSR_RASTER_BCOORD_Y0:  `TRACE(level, ("BCOORD_Y0")); \
        `CSR_RASTER_BCOORD_Y1:  `TRACE(level, ("BCOORD_Y1")); \
        `CSR_RASTER_BCOORD_Y2:  `TRACE(level, ("BCOORD_Y2")); \
        `CSR_RASTER_BCOORD_Y3:  `TRACE(level, ("BCOORD_Y3")); \
        `CSR_RASTER_BCOORD_Z0:  `TRACE(level, ("BCOORD_Z0")); \
        `CSR_RASTER_BCOORD_Z1:  `TRACE(level, ("BCOORD_Z1")); \
        `CSR_RASTER_BCOORD_Z2:  `TRACE(level, ("BCOORD_Z2")); \
        `CSR_RASTER_BCOORD_Z3:  `TRACE(level, ("BCOORD_Z3")); \
        default:                `TRACE(level, ("?")); \
    endcase

`define PERF_RASTER_ADD(dst, src, count) \
    `REDUCE_ADD (dst, src, mem_reads, `PERF_CTR_BITS, count); \
    `REDUCE_ADD (dst, src, mem_latency, `PERF_CTR_BITS, count); \
    `REDUCE_ADD (dst, src, stall_cycles, `PERF_CTR_BITS, count)

`endif // VX_RASTER_DEFINE_VH
