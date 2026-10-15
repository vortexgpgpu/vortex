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

`ifndef VX_TEX_PKG_VH
`define VX_TEX_PKG_VH

`include "VX_tex_define.vh"

package VX_tex_pkg;

// TEX_STAGE_BITS is owned by VX_gpu_pkg (it sizes the core op-args stage field);
// the tex CSR struct below reuses it. TEX_NUM_LEVELS is owned there too, since
// it sizes the unit's memory port count.
import VX_gpu_pkg::TEX_STAGE_BITS;
import VX_gpu_pkg::TEX_NUM_LEVELS;

// TEX field widths, derived locally from the VX_types value leaves rather than
// exported as generated contract macros.
localparam TEX_LOD_BITS    = `CLOG2(`VX_TEX_LOD_MAX + 1);
// The whole lod operand: the level (an integer, or under mip-linear fixed point
// with the blend weight in the low VX_TEX_LOD_FRAC_BITS), then the gather
// fields, the lane's filter overrides and its array layer.
localparam TEX_ARG_BITS    = `VX_TEX_ARG_LAYER_SHIFT + `VX_TEX_ARG_LAYER_BITS;
localparam TEX_FILTER_BITS = `CLOG2(`VX_TEX_FILTER_MIP_LINEAR + 1);   // mag/min + mip bit
// The formats the unit reads: the FF colour formats keep their VX_TEX_FORMAT_*
// codes, and the depth formats a vx_tex_cmp compares take the three above them
// (VX_tex_dcr maps the register's value), so the code stays four bits wide.
localparam TEX_FORMAT_BITS = 4;
localparam logic [TEX_FORMAT_BITS-1:0] TEX_FMT_D16  = TEX_FORMAT_BITS'(`VX_TEX_FORMAT_FF_MAX + 1);
localparam logic [TEX_FORMAT_BITS-1:0] TEX_FMT_D32F = TEX_FORMAT_BITS'(`VX_TEX_FORMAT_FF_MAX + 2);
localparam logic [TEX_FORMAT_BITS-1:0] TEX_FMT_D24  = TEX_FORMAT_BITS'(`VX_TEX_FORMAT_FF_MAX + 3);
localparam TEX_CMP_BITS    = `VX_TEX_FILTER_COMPARE_BITS;
localparam TEX_WRAP_BITS   = `CLOG2(`VX_TEX_WRAP_MIRROR_CLAMP + 1);
localparam TEX_DIM_W       = `VX_TEX_DIM_BITS + 1;                // mip-0 texels, up to 2^DIM_BITS

typedef struct packed {
    logic [(`VX_TEX_LOD_MAX+1)-1:0][`TEX_MIPOFF_BITS-1:0] mipoff;
    logic [1:0][TEX_DIM_W-1:0]     dims;
    logic [1:0][TEX_WRAP_BITS-1:0] wraps;
    logic [`TEX_ADDR_BITS-1:0]     baseaddr;
    logic [TEX_FORMAT_BITS-1:0]    format;
    logic [TEX_FILTER_BITS-1:0]    filter;
    // Weighted average, or the min/max of the nonzero-weight taps.
    logic [1:0]                    reduction;
    // vx_tex_cmp's compare function (VX_OM_DEPTH_FUNC_*).
    logic [TEX_CMP_BITS-1:0]       compare;
    // Colour a tap outside [0,1) returns on an axis that wraps to a border; for
    // a vx_tex_cmp, the border tap's depth as an f32.
    logic [31:0]                   border;
    // Bytes between array layers; a lane's layer field steps its base by it.
    logic [31:0]                   layer_stride;
} tex_dcrs_t;

typedef struct packed {
    logic [TEX_STAGE_BITS-1:0] stage;
} tex_csrs_t;

endpackage

`endif // VX_TEX_PKG_VH
