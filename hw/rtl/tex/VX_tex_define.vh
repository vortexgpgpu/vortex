`ifndef VX_TEX_DEFINE_VH
`define VX_TEX_DEFINE_VH

`include "VX_define.vh"
`include "VX_gpu_types.vh"
`include "VX_tex_types.vh"

`IGNORE_WARNINGS_BEGIN
import VX_gpu_types::*;
import VX_tex_types::*;
`IGNORE_WARNINGS_END

`define TRACE_TEX_DCR(level, addr) \
    case (addr) \
        `VX_DCR_TEX_ADDR:   `TRACE(level, ("ADDR")); \
        `VX_DCR_TEX_LOGDIM: `TRACE(level, ("LOGDIM")); \
        `VX_DCR_TEX_FORMAT: `TRACE(level, ("FORMAT")); \
        `VX_DCR_TEX_FILTER: `TRACE(level, ("FILTER")); \
        `VX_DCR_TEX_WRAP:   `TRACE(level, ("WRAP")); \
        //`VX_DCR_TEX_MIPOFF \
        default:            `TRACE(level, ("MIPOFF")); \
    endcase

`define TRACE_TEX_CSR(level, addr) \
    case (addr) \
        default: `TRACE(level, ("?")); \
    endcase

`define PERF_TEX_ADD(dst, src, count) \
    `REDUCE_ADD (dst, src, mem_reads,   `PERF_CTR_BITS, count); \
    `REDUCE_ADD (dst, src, mem_latency, `PERF_CTR_BITS, count); \
    `REDUCE_ADD (dst, src, stall_cycles,`PERF_CTR_BITS, count)

`endif // VX_TEX_DEFINE_VH
