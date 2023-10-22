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

`ifndef VX_RASTER_PKG_VH
`define VX_RASTER_PKG_VH

`include "VX_raster_define.vh"

package VX_raster_pkg;

typedef struct packed {
    logic [`RASTER_ADDR_BITS-1:0]   tbuf_addr;     // Tile buffer address
    logic [`RASTER_TILE_BITS-1:0]   tile_count;    // Number of tiles in the tile buffer
    logic [`RASTER_ADDR_BITS-1:0]   pbuf_addr;     // Primitive triangle data buffer start address
    logic [`VX_RASTER_STRIDE_BITS-1:0] pbuf_stride; // Primitive data stride to fetch vertices
    logic [`VX_RASTER_DIM_BITS-1:0] dst_xmin;      // Destination window xmin
    logic [`VX_RASTER_DIM_BITS-1:0] dst_xmax;      // Destination window xmax
    logic [`VX_RASTER_DIM_BITS-1:0] dst_ymin;      // Destination window ymin
    logic [`VX_RASTER_DIM_BITS-1:0] dst_ymax;      // Destination window ymax
} raster_dcrs_t;

typedef struct packed {
    logic [`VX_RASTER_DIM_BITS-2:0] pos_x;     // quad x position
    logic [`VX_RASTER_DIM_BITS-2:0] pos_y;     // quad y position
    logic [3:0]                     mask;      // quad mask
    logic [2:0][3:0][31:0]          bcoords;   // barycentric coordinates
    logic [`VX_RASTER_PID_BITS-1:0] pid;       // primitive index
} raster_stamp_t;

typedef struct packed {
    logic [2:0][3:0][31:0] bcoords;
    logic [31:0]           pos_mask;
} raster_csrs_t;

endpackage

`endif // VX_RASTER_PKG_VH
