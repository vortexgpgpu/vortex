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

typedef struct packed {
    logic [(`VX_TEX_LOD_MAX+1)-1:0][`TEX_MIPOFF_BITS-1:0] mipoff;
    logic [1:0][`VX_TEX_LOD_BITS-1:0] logdims;
    logic [1:0][`TEX_WRAP_BITS-1:0] wraps;
    logic [`TEX_ADDR_BITS-1:0]      baseaddr;
    logic [`TEX_FORMAT_BITS-1:0]    format;
    logic [`TEX_FILTER_BITS-1:0]    filter;
} tex_dcrs_t;

typedef struct packed {
    logic [`VX_TEX_STAGE_BITS-1:0] stage;
} tex_csrs_t;

endpackage

`endif // VX_TEX_PKG_VH
