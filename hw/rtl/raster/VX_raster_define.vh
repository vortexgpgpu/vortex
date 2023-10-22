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

`ifndef VX_RASTER_DEFINE_VH
`define VX_RASTER_DEFINE_VH

`include "VX_define.vh"

`ifdef XLEN_64
`define RASTER_ADDR_BITS        32
`else
`define RASTER_ADDR_BITS        25
`endif
`define RASTER_DCR_DATA_BITS    32
`define RASTER_DATA_BITS        32
`define RASTER_TILE_BITS        16

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
        `VX_DCR_RASTER_TBUF_ADDR:   `TRACE(level, ("TBUF_ADDR")); \
        `VX_DCR_RASTER_TILE_COUNT:  `TRACE(level, ("TILE_COUNT")); \
        `VX_DCR_RASTER_PBUF_ADDR:   `TRACE(level, ("PBUF_ADDR")); \
        `VX_DCR_RASTER_PBUF_STRIDE: `TRACE(level, ("PBUF_STRIDE")); \
        `VX_DCR_RASTER_SCISSOR_X:   `TRACE(level, ("SCISSOR_X")); \
        `VX_DCR_RASTER_SCISSOR_Y:   `TRACE(level, ("SCISSOR_Y")); \
        default:                    `TRACE(level, ("?")); \
    endcase

`define TRACE_RASTER_CSR(level, addr) \
    case (addr) \
        `VX_CSR_RASTER_POS_MASK:    `TRACE(level, ("POS_MASK")); \
        `VX_CSR_RASTER_BCOORD_X0:   `TRACE(level, ("BCOORD_X0")); \
        `VX_CSR_RASTER_BCOORD_X1:   `TRACE(level, ("BCOORD_X1")); \
        `VX_CSR_RASTER_BCOORD_X2:   `TRACE(level, ("BCOORD_X2")); \
        `VX_CSR_RASTER_BCOORD_X3:   `TRACE(level, ("BCOORD_X3")); \
        `VX_CSR_RASTER_BCOORD_Y0:   `TRACE(level, ("BCOORD_Y0")); \
        `VX_CSR_RASTER_BCOORD_Y1:   `TRACE(level, ("BCOORD_Y1")); \
        `VX_CSR_RASTER_BCOORD_Y2:   `TRACE(level, ("BCOORD_Y2")); \
        `VX_CSR_RASTER_BCOORD_Y3:   `TRACE(level, ("BCOORD_Y3")); \
        `VX_CSR_RASTER_BCOORD_Z0:   `TRACE(level, ("BCOORD_Z0")); \
        `VX_CSR_RASTER_BCOORD_Z1:   `TRACE(level, ("BCOORD_Z1")); \
        `VX_CSR_RASTER_BCOORD_Z2:   `TRACE(level, ("BCOORD_Z2")); \
        `VX_CSR_RASTER_BCOORD_Z3:   `TRACE(level, ("BCOORD_Z3")); \
        default:                    `TRACE(level, ("?")); \
    endcase

`define PERF_RASTER_ADD(dst, src, count) \
    `PERF_REDUCE (dst, src, mem_reads, `PERF_CTR_BITS, count); \
    `PERF_REDUCE (dst, src, mem_latency, `PERF_CTR_BITS, count); \
    `PERF_REDUCE (dst, src, stall_cycles, `PERF_CTR_BITS, count)

`endif // VX_RASTER_DEFINE_VH
