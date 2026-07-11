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
`ifdef VX_CFG_RASTER_EARLYZ_ENABLE
    // Early-Z: shared depth-buffer config snooped from the OM depth DCRs. The
    // raster unit reads committed depth and culls occluded fragments before the
    // packer when earlyz_safe is set (monotonic func, no FS depth-export).
    logic [`RASTER_ADDR_BITS-1:0]   zbuf_addr;     // Depth buffer base address (block)
    logic [OM_PITCH_BITS-1:0]   zbuf_pitch;    // Depth buffer row pitch (bytes)
    logic [OM_DEPTH_FUNC_BITS-1:0] depth_func; // Depth compare function
    logic                           earlyz_safe;   // 1 = early-Z cull permitted
`endif
} raster_dcrs_t;

typedef struct packed {
    logic [`VX_RASTER_DIM_BITS-2:0] pos_x;     // quad x position
    logic [`VX_RASTER_DIM_BITS-2:0] pos_y;     // quad y position
    logic [3:0]                     mask;      // quad coverage mask
    logic [`VX_RASTER_PID_BITS-1:0] pid;       // primitive index
} raster_stamp_t;
// P2: per-corner edge values (bcoords) are no longer carried — the fragment
// shader recomputes them from the primitive edges + the quad origin.

// ── stamp delivery on the KMU launch bus ─────────────────────────────────
// A fragment launch is the kmu_req_t header followed by RASTER_STAMP_BEATS
// payload beats, each packing RASTER_STAMPS_PER_BEAT stamps at bus width. The
// beats reuse the header's wires, so carrying the stamp inside the launch costs
// no extra bus.
localparam RASTER_STAMP_BITS      = $bits(raster_stamp_t);
localparam RASTER_STAMPS_PER_BEAT = VX_gpu_pkg::KMU_DATAW / RASTER_STAMP_BITS;
localparam RASTER_STAMP_BEATS     =
    (`VX_CFG_NUM_SFU_LANES + RASTER_STAMPS_PER_BEAT - 1) / RASTER_STAMPS_PER_BEAT;

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
