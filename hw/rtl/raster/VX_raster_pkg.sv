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
`ifdef VX_CFG_RASTER_EARLYZ_ENABLE
// Early-Z snoops the OM depth-buffer config; the OM package is only present in
// the build when early-Z is enabled (which also enables the OM extension).
import VX_om_pkg::*;
`endif

typedef struct packed {
    logic [`RASTER_ADDR_BITS-1:0]   tbuf_addr;     // Tile buffer address
    logic [`RASTER_TILE_BITS-1:0]   tile_count;    // Number of tiles in the tile buffer
    logic [`RASTER_ADDR_BITS-1:0]   pbuf_addr;     // Primitive triangle data buffer start address
    logic [`VX_RASTER_STRIDE_BITS-1:0] pbuf_stride; // Primitive data stride to fetch vertices
    logic [`VX_RASTER_DIM_BITS-1:0] dst_xmin;      // Destination window xmin
    logic [`VX_RASTER_DIM_BITS-1:0] dst_xmax;      // Destination window xmax
    logic [`VX_RASTER_DIM_BITS-1:0] dst_ymin;      // Destination window ymin
    logic [`VX_RASTER_DIM_BITS-1:0] dst_ymax;      // Destination window ymax
    logic                           msaa;          // 4 samples per pixel
`ifdef VX_CFG_RASTER_EARLYZ_ENABLE
    // Early-Z: shared depth-buffer config snooped from the OM depth DCRs. The
    // raster unit reads committed depth and culls occluded fragments before the
    // packer when earlyz_safe is set (monotonic func, no FS depth-export).
    logic [`RASTER_ADDR_BITS-1:0]   zbuf_addr;     // Depth buffer base address (block)
    logic [OM_PITCH_BITS-1:0]   zbuf_pitch;    // Depth buffer row pitch (bytes)
    logic [OM_SAMPLE_STRIDE_BITS-1:0] zbuf_sample_stride; // words between its sample images
    logic [OM_DEPTH_FUNC_BITS-1:0] depth_func; // Depth compare function
    logic                           earlyz_safe;   // 1 = early-Z cull permitted
    logic                           depth_d24;     // D24S8 target: the only plane early-Z tests
`endif
} raster_dcrs_t;

// A primitive's pixel rectangle, [xmin, xmax) x [ymin, ymax).
typedef struct packed {
    logic [`VX_RASTER_DIM_BITS-1:0] xmin;
    logic [`VX_RASTER_DIM_BITS-1:0] xmax;
    logic [`VX_RASTER_DIM_BITS-1:0] ymin;
    logic [`VX_RASTER_DIM_BITS-1:0] ymax;
} raster_rect_t;

localparam RASTER_RECT_BITS = $bits(raster_rect_t);

// Samples a pixel's coverage is kept for: 4 under multisampling, bit 0 alone at
// one sample. The sample positions below are the 4x pattern's.
localparam RASTER_SAMPLES = 4;

typedef struct packed {
    logic [`VX_RASTER_DIM_BITS-2:0] pos_x;     // quad x position
    logic [`VX_RASTER_DIM_BITS-2:0] pos_y;     // quad y position
    logic [3:0][RASTER_SAMPLES-1:0] mask;      // pixel p's sample coverage
    logic [`VX_RASTER_PID_BITS-1:0] pid;       // primitive index
} raster_stamp_t;
// Per-corner edge values (bcoords) are not carried: the fragment shader
// recomputes them from the primitive edges and its own pixel. A pixel is covered
// when any of its samples is.

// The standard 4x positions as offsets from the pixel centre in 1/16 px:
// (-2,-6), (6,-2), (-6,2), (2,6). Sample k moves a plane with gradients (a, b)
// by (a*DX16[k] + b*DY16[k]) >>> 4; samples 2 and 3 mirror 1 and 0. The sum is
// formed RASTER_SAMPLE_SUM_BITS wide, so it cannot wrap before the shift, and
// the shifted offset fits the data width.
localparam RASTER_SAMPLE_SUM_BITS = `RASTER_DATA_BITS + 4;

function automatic [`RASTER_DATA_BITS-1:0] raster_sample_offset (
    input logic [`RASTER_DATA_BITS-1:0] a,
    input logic [`RASTER_DATA_BITS-1:0] b,
    input logic [1:0]                   k
);
    logic signed [RASTER_SAMPLE_SUM_BITS-1:0] sa, sb, v, d;
    begin
        sa = RASTER_SAMPLE_SUM_BITS'($signed(a));
        sb = RASTER_SAMPLE_SUM_BITS'($signed(b));
        v  = (k[0] ^ k[1]) ? ((sa <<< 2) + (sa <<< 1) - (sb <<< 1))   //  6a - 2b
                           : ((sa <<< 1) + (sb <<< 2) + (sb <<< 1));  //  2a + 6b
        d  = k[0] ? v : -v;
        raster_sample_offset = `RASTER_DATA_BITS'(d >>> 4);
    end
endfunction

// ── stamp delivery on the KMU launch bus ─────────────────────────────────
// A fragment launch is a single kmu_req_t beat carrying the wave's stamps. A quad's
// stamp is striped across the quad's four lanes rather than replicated, which is what
// keeps the whole wave inside the header.
localparam RASTER_STAMP_BITS = $bits(raster_stamp_t);

// Cores a fragment launch may be placed on. Fragment work is bin->core affine,
// and a raster core injects into its OWN cluster's launch stream, so the owner
// index spans that cluster's cores — VX_CFG_NUM_CORES is per-cluster. Those are
// the low bits of kmu `dest` (core-in-socket, then socket-in-cluster), which is
// exactly the slice the socket and cluster fan-outs consume.
localparam RASTER_NUM_DESTS = `VX_CFG_NUM_CORES;
localparam RASTER_DEST_W    = `UP(`CLOG2(RASTER_NUM_DESTS));

// quad -> screen bin: a quad is 2 px, so the bin shift is BIN_LOG_SIZE-1.
localparam RASTER_BIN_Q = `VX_CFG_RASTER_BIN_LOG_SIZE - 1;

// The owner core of a quad: the same deterministic bin->core map the raster bus
// arbiter used, so every pixel still lands on exactly one core and same-pixel
// raster/OM order stays correct by construction.
function automatic [RASTER_DEST_W-1:0] raster_owner (
    input logic [`VX_RASTER_DIM_BITS-2:0] pos_x,
    input logic [`VX_RASTER_DIM_BITS-2:0] pos_y
);
    logic [31:0] bin_lin;
    begin
        bin_lin = (32'(pos_x) >> RASTER_BIN_Q) + (32'(pos_y) >> RASTER_BIN_Q);
        raster_owner = RASTER_DEST_W'(bin_lin % RASTER_NUM_DESTS);
    end
endfunction

endpackage

`endif // VX_RASTER_PKG_VH
