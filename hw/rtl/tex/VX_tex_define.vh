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

`ifndef VX_TEX_DEFINE_VH
`define VX_TEX_DEFINE_VH

`include "VX_define.vh"

`define TEX_FXD_INT         (`VX_TEX_FXD_BITS - `VX_TEX_FXD_FRAC)
`define TEX_FXD_ONE         (2 ** `VX_TEX_FXD_FRAC)
`define TEX_FXD_HALF        (`TEX_FXD_ONE >> 1)
`define TEX_FXD_MASK        (`TEX_FXD_ONE - 1)

`ifdef XLEN_64
`define TEX_ADDR_BITS       32
`else
`define TEX_ADDR_BITS       25
`endif
`define TEX_FORMAT_BITS     3
`define TEX_WRAP_BITS       2
`define TEX_FILTER_BITS     1
`define TEX_MIPOFF_BITS     (2*`VX_TEX_DIM_BITS+1)

`define TEX_LGSTRIDE_MAX    2
`define TEX_LGSTRIDE_BITS   2

`define TEX_BLEND_FRAC      `VX_TEX_SUBPIXEL_BITS
`define TEX_BLEND_ONE       (2 ** `TEX_BLEND_FRAC)

`define TRACE_TEX_DCR(level, addr) \
    case (addr) \
        `VX_DCR_TEX_STAGE:  `TRACE(level, ("STAGE")) \
        `VX_DCR_TEX_ADDR:   `TRACE(level, ("ADDR")) \
        `VX_DCR_TEX_LOGDIM: `TRACE(level, ("LOGDIM")) \
        `VX_DCR_TEX_FORMAT: `TRACE(level, ("FORMAT")) \
        `VX_DCR_TEX_FILTER: `TRACE(level, ("FILTER")) \
        `VX_DCR_TEX_WRAP:   `TRACE(level, ("WRAP")) \
        //`VX_DCR_TEX_MIPOFF \
        default:            `TRACE(level, ("MIPOFF")) \
    endcase

`define TRACE_TEX_CSR(level, addr) \
    case (addr) \
        default: `TRACE(level, ("?")) \
    endcase

`define PERF_TEX_ADD(dst, src, dcount, scount) \
    `PERF_COUNTER_ADD_EX (dst, src, mem_reads,   `PERF_CTR_BITS, dcount, scount, (((scount + dcount - 1) / dcount) > 1)) \
    `PERF_COUNTER_ADD_EX (dst, src, mem_latency, `PERF_CTR_BITS, dcount, scount, (((scount + dcount - 1) / dcount) > 1)) \
    `PERF_COUNTER_ADD_EX (dst, src, stall_cycles,`PERF_CTR_BITS, dcount, scount, (((scount + dcount - 1) / dcount) > 1))

`endif // VX_TEX_DEFINE_VH
