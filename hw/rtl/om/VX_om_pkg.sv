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

// OM field widths and depth/stencil masks, derived from the VX_types value
// leaves — RTL-owned sizing, not part of the HW<->SW contract header.
localparam OM_DEPTH_FUNC_BITS = `CLOG2(`VX_OM_DEPTH_FUNC_NOTEQUAL + 1);
localparam OM_STENCIL_OP_BITS = `CLOG2(`VX_OM_STENCIL_OP_DECR_WRAP + 1);
localparam OM_BLEND_MODE_BITS = `CLOG2(`VX_OM_BLEND_MODE_LOGICOP + 1);
localparam OM_BLEND_FUNC_BITS = `CLOG2(`VX_OM_BLEND_FUNC_ALPHA_SAT + 1);
localparam OM_LOGIC_OP_BITS   = `CLOG2(`VX_OM_LOGIC_OP_SET + 1);
localparam OM_PITCH_BITS      = `VX_OM_DIM_BITS + `CLOG2(4) + 1;
localparam OM_DEPTH_MASK      = (1 << `VX_OM_DEPTH_BITS) - 1;
// Width of an aperture shift amount (xbits/ybits hold a log2 dimension).
localparam OM_APERTURE_BITS_W = `CLOG2(`VX_OM_DIM_BITS + 1);
localparam OM_STENCIL_MASK    = (1 << `VX_OM_STENCIL_BITS) - 1;

typedef struct packed {
    logic [31:0] argb;
} om_color_t;

typedef struct packed {
    logic [`OM_ADDR_BITS-1:0]           cbuf_addr;
    logic [OM_PITCH_BITS-1:0]           cbuf_pitch;
    logic [3:0]                         cbuf_writemask;

    logic [`OM_ADDR_BITS-1:0]           zbuf_addr;
    logic [OM_PITCH_BITS-1:0]           zbuf_pitch;

    logic                               depth_enable;
    logic [OM_DEPTH_FUNC_BITS-1:0]      depth_func;
    logic                               depth_writemask;

    logic [1:0]                         stencil_enable;
    logic [1:0][OM_DEPTH_FUNC_BITS-1:0] stencil_func;
    logic [1:0][OM_STENCIL_OP_BITS-1:0] stencil_zpass;
    logic [1:0][OM_STENCIL_OP_BITS-1:0] stencil_zfail;
    logic [1:0][OM_STENCIL_OP_BITS-1:0] stencil_fail;
    logic [1:0][`VX_OM_STENCIL_BITS-1:0] stencil_ref;
    logic [1:0][`VX_OM_STENCIL_BITS-1:0] stencil_mask;
    logic [1:0][`VX_OM_STENCIL_BITS-1:0] stencil_writemask;

    logic                               blend_enable;
    logic [OM_BLEND_MODE_BITS-1:0]      blend_mode_rgb;
    logic [OM_BLEND_MODE_BITS-1:0]      blend_mode_a;
    logic [OM_BLEND_FUNC_BITS-1:0]      blend_src_rgb;
    logic [OM_BLEND_FUNC_BITS-1:0]      blend_src_a;
    logic [OM_BLEND_FUNC_BITS-1:0]      blend_dst_rgb;
    logic [OM_BLEND_FUNC_BITS-1:0]      blend_dst_a;
    om_color_t                          blend_const;

    logic [OM_LOGIC_OP_BITS-1:0]        logic_op;

    // Fragment-export aperture (see VX_types.toml [dcr_om]). Shift-only encoding,
    // so the ingress decodes an offset into (face, y, x) by bit-slicing rather
    // than dividing.
    logic [OM_APERTURE_BITS_W-1:0]      aperture_xbits;
    logic [OM_APERTURE_BITS_W-1:0]      aperture_ybits;
    logic [1:0]                         aperture_record_shift;  // 2 = one word, 3 = colour+depth
    logic                               aperture_depth_only;    // disambiguates the one-word modes
} om_dcrs_t;

endpackage

`endif // VX_OM_PKG_VH
