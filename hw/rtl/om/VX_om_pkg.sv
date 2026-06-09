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

`ifndef VX_OM_PKG_VH
`define VX_OM_PKG_VH

`include "VX_om_define.vh"

package VX_om_pkg;

typedef struct packed {
    logic [31:0] argb;
} om_color_t;

typedef struct packed {
    logic [`OM_ADDR_BITS-1:0]           cbuf_addr;
    logic [`VX_OM_PITCH_BITS-1:0]       cbuf_pitch;
    logic [3:0]                         cbuf_writemask;

    logic [`OM_ADDR_BITS-1:0]           zbuf_addr;
    logic [`VX_OM_PITCH_BITS-1:0]       zbuf_pitch;

    logic                               depth_enable;
    logic [`VX_OM_DEPTH_FUNC_BITS-1:0]  depth_func;
    logic                               depth_writemask;

    logic [1:0]                         stencil_enable;
    logic [1:0][`VX_OM_DEPTH_FUNC_BITS-1:0] stencil_func;
    logic [1:0][`VX_OM_STENCIL_OP_BITS-1:0] stencil_zpass;
    logic [1:0][`VX_OM_STENCIL_OP_BITS-1:0] stencil_zfail;
    logic [1:0][`VX_OM_STENCIL_OP_BITS-1:0] stencil_fail;
    logic [1:0][`VX_OM_STENCIL_BITS-1:0] stencil_ref;
    logic [1:0][`VX_OM_STENCIL_BITS-1:0] stencil_mask;
    logic [1:0][`VX_OM_STENCIL_BITS-1:0] stencil_writemask;

    logic                               blend_enable;
    logic [`VX_OM_BLEND_MODE_BITS-1:0]  blend_mode_rgb;
    logic [`VX_OM_BLEND_MODE_BITS-1:0]  blend_mode_a;
    logic [`VX_OM_BLEND_FUNC_BITS-1:0]  blend_src_rgb;
    logic [`VX_OM_BLEND_FUNC_BITS-1:0]  blend_src_a;
    logic [`VX_OM_BLEND_FUNC_BITS-1:0]  blend_dst_rgb;
    logic [`VX_OM_BLEND_FUNC_BITS-1:0]  blend_dst_a;
    om_color_t                          blend_const;

    logic [`VX_OM_LOGIC_OP_BITS-1:0]    logic_op;
} om_dcrs_t;

typedef struct packed {
    logic [1:0]                 rt_idx;
    logic [`VX_OM_DIM_BITS-1:0] pos_x;
    logic [`VX_OM_DIM_BITS-1:0] pos_y;
    logic [23:0]                depth;
    logic [2:0]                 sample_idx;
    logic [7:0]                 sample_mask;
} om_csrs_t;

endpackage

`endif // VX_OM_PKG_VH
