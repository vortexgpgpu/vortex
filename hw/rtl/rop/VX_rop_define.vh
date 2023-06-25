`ifndef VX_ROP_DEFINE_VH
`define VX_ROP_DEFINE_VH

`include "VX_define.vh"
`include "VX_gpu_types.vh"
`include "VX_rop_types.vh"

`IGNORE_WARNINGS_BEGIN
import VX_gpu_types::*;
import VX_rop_types::*;
`IGNORE_WARNINGS_END

`define TRACE_ROP_DCR(level, state) \
    case (state) \
        `VX_DCR_ROP_CBUF_ADDR:          `TRACE(level, ("CBUF_ADDR")); \
        `VX_DCR_ROP_CBUF_PITCH:         `TRACE(level, ("CBUF_PITCH")); \
        `VX_DCR_ROP_CBUF_WRITEMASK:     `TRACE(level, ("CBUF_WRITEMASK")); \
        `VX_DCR_ROP_ZBUF_ADDR:          `TRACE(level, ("ZBUF_ADDR")); \
        `VX_DCR_ROP_ZBUF_PITCH:         `TRACE(level, ("ZBUF_PITCH")); \
        `VX_DCR_ROP_DEPTH_FUNC:         `TRACE(level, ("DEPTH_FUNC")); \
        `VX_DCR_ROP_DEPTH_WRITEMASK:    `TRACE(level, ("DEPTH_WRITEMASK")); \
        `VX_DCR_ROP_STENCIL_FUNC:       `TRACE(level, ("STENCIL_FUNC")); \
        `VX_DCR_ROP_STENCIL_ZPASS:      `TRACE(level, ("STENCIL_ZPASS")); \
        `VX_DCR_ROP_STENCIL_ZFAIL:      `TRACE(level, ("STENCIL_ZFAIL")); \
        `VX_DCR_ROP_STENCIL_FAIL:       `TRACE(level, ("STENCIL_FAIL")); \
        `VX_DCR_ROP_STENCIL_REF:        `TRACE(level, ("STENCIL_REF")); \
        `VX_DCR_ROP_STENCIL_MASK:       `TRACE(level, ("STENCIL_MASK")); \
        `VX_DCR_ROP_STENCIL_WRITEMASK:  `TRACE(level, ("STENCIL_WRITEMASK")); \
        `VX_DCR_ROP_BLEND_MODE:         `TRACE(level, ("BLEND_MODE")); \
        `VX_DCR_ROP_BLEND_FUNC:         `TRACE(level, ("BLEND_FUNC")); \
        `VX_DCR_ROP_BLEND_CONST:        `TRACE(level, ("BLEND_CONST")); \
        `VX_DCR_ROP_LOGIC_OP:           `TRACE(level, ("LOGIC_OP")); \
        default:                        `TRACE(level, ("?")); \
    endcase

`define TRACE_ROP_CSR(level, addr) \
    case (addr) \
        `VX_CSR_ROP_RT_IDX:             `TRACE(level, ("RT_IDX")); \
        `VX_CSR_ROP_SAMPLE_IDX:         `TRACE(level, ("SAMPLE_IDX")); \
        default:                        `TRACE(level, ("?")); \
    endcase

`define PERF_ROP_ADD(dst, src, count) \
    `REDUCE_ADD (dst, src, mem_reads,   `PERF_CTR_BITS, count); \
    `REDUCE_ADD (dst, src, mem_writes,  `PERF_CTR_BITS, count); \
    `REDUCE_ADD (dst, src, mem_latency, `PERF_CTR_BITS, count); \
    `REDUCE_ADD (dst, src, stall_cycles,`PERF_CTR_BITS, count)

`endif // VX_ROP_DEFINE_VH
