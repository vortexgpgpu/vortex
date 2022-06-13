`ifndef VX_TEX_DEFINE_VH
`define VX_TEX_DEFINE_VH

`include "VX_define.vh"
`include "VX_tex_types.vh"

`IGNORE_WARNINGS_BEGIN
import VX_tex_types::*;
`IGNORE_WARNINGS_END

task trace_tex_dcr (
    input int                  level,
    input [`DCR_ADDR_BITS-1:0] addr
);
    case (addr)
        `DCR_TEX_ADDR:      `TRACE(level, ("ADDR"));
        `DCR_TEX_LOGDIM:    `TRACE(level, ("LOGDIM"));
        `DCR_TEX_FORMAT:    `TRACE(level, ("FORMAT"));
        `DCR_TEX_FILTER:    `TRACE(level, ("FILTER"));
        `DCR_TEX_WRAP:      `TRACE(level, ("WRAP"));
        //`DCR_TEX_MIPOFF
        default:            `TRACE(level, ("MIPOFF"));
    endcase  
endtask

task trace_tex_csr (
    input int                  level,
    input [`CSR_ADDR_BITS-1:0] addr
);
    case (addr)
        default:         `TRACE(level, ("?"));
    endcase  
endtask

`define PERF_TEX_ADD(dst, src, count) \
    `REDUCE_ADD (dst, src, mem_reads, `PERF_CTR_BITS, count); \
    `REDUCE_ADD (dst, src, mem_latency, `PERF_CTR_BITS, count)

`endif
