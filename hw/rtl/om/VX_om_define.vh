//!/bin/bash

// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`ifndef VX_OM_DEFINE_VH
`define VX_OM_DEFINE_VH

`include "VX_define.vh"

`ifdef XLEN_64
`define OM_ADDR_BITS 32
`else
`define OM_ADDR_BITS 25
`endif

`define TRACE_OM_DCR(level, state) \
    case (state) \
        `VX_DCR_OM_CBUF_ADDR:           `TRACE(level, ("CBUF_ADDR")) \
        `VX_DCR_OM_CBUF_PITCH:          `TRACE(level, ("CBUF_PITCH")) \
        `VX_DCR_OM_CBUF_WRITEMASK:      `TRACE(level, ("CBUF_WRITEMASK")) \
        `VX_DCR_OM_ZBUF_ADDR:           `TRACE(level, ("ZBUF_ADDR")) \
        `VX_DCR_OM_ZBUF_PITCH:          `TRACE(level, ("ZBUF_PITCH")) \
        `VX_DCR_OM_DEPTH_FUNC:          `TRACE(level, ("DEPTH_FUNC")) \
        `VX_DCR_OM_DEPTH_WRITEMASK:     `TRACE(level, ("DEPTH_WRITEMASK")) \
        `VX_DCR_OM_STENCIL_FUNC:        `TRACE(level, ("STENCIL_FUNC")) \
        `VX_DCR_OM_STENCIL_ZPASS:       `TRACE(level, ("STENCIL_ZPASS")) \
        `VX_DCR_OM_STENCIL_ZFAIL:       `TRACE(level, ("STENCIL_ZFAIL")) \
        `VX_DCR_OM_STENCIL_FAIL:        `TRACE(level, ("STENCIL_FAIL")) \
        `VX_DCR_OM_STENCIL_REF:         `TRACE(level, ("STENCIL_REF")) \
        `VX_DCR_OM_STENCIL_MASK:        `TRACE(level, ("STENCIL_MASK")) \
        `VX_DCR_OM_STENCIL_WRITEMASK:   `TRACE(level, ("STENCIL_WRITEMASK")) \
        `VX_DCR_OM_BLEND_MODE:          `TRACE(level, ("BLEND_MODE")) \
        `VX_DCR_OM_BLEND_FUNC:          `TRACE(level, ("BLEND_FUNC")) \
        `VX_DCR_OM_BLEND_CONST:         `TRACE(level, ("BLEND_CONST")) \
        `VX_DCR_OM_LOGIC_OP:            `TRACE(level, ("LOGIC_OP")) \
        default:                        `TRACE(level, ("?")) \
    endcase

`define TRACE_OM_CSR(level, addr) \
    case (addr) \
        `VX_CSR_OM_RT_IDX:              `TRACE(level, ("RT_IDX")) \
        `VX_CSR_OM_SAMPLE_IDX:          `TRACE(level, ("SAMPLE_IDX")) \
        default:                        `TRACE(level, ("?")) \
    endcase

`define PERF_OM_ADD(dst, src, dcount, scount) \
    `PERF_COUNTER_ADD_EX (dst, src, mem_reads,   `PERF_CTR_BITS, dcount, scount, (((scount + dcount - 1) / dcount) > 1)) \
    `PERF_COUNTER_ADD_EX (dst, src, mem_writes,  `PERF_CTR_BITS, dcount, scount, (((scount + dcount - 1) / dcount) > 1)) \
    `PERF_COUNTER_ADD_EX (dst, src, mem_latency, `PERF_CTR_BITS, dcount, scount, (((scount + dcount - 1) / dcount) > 1)) \
    `PERF_COUNTER_ADD_EX (dst, src, stall_cycles,`PERF_CTR_BITS, dcount, scount, (((scount + dcount - 1) / dcount) > 1))

`endif // VX_OM_DEFINE_VH
