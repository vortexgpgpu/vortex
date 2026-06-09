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

// Vortex public host-side graphics API. Triangle setup + tile binning
// for the RASTER hardware unit, plus DCR address helpers. Self-contained:
// the only external dependency is <vx_graphics.h> (canonical on-wire
// ABI) and <VX_types.h> (generated constants).
//
// Consumers: mesa-vortex (vortexpipe Gallium driver), regression tests,
// and any downstream tool that programs the RASTER unit. Implementation
// ships in libvortex.so.

#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <vx_graphics.h>
#include <VX_types.h>

namespace vortex {
namespace graphics {

///////////////////////////////////////////////////////////////////////////////
// DCR address → state-index helpers
//
// VX_types.toml only emits scalar `#define`s, so function-like helpers
// must be declared here rather than in VX_types.h.
///////////////////////////////////////////////////////////////////////////////

#ifndef VX_DCR_TEX_STATE
#define VX_DCR_TEX_STATE(addr)    ((addr) - VX_DCR_TEX_STATE_BEGIN)
#endif
#ifndef VX_DCR_RASTER_STATE
#define VX_DCR_RASTER_STATE(addr) ((addr) - VX_DCR_RASTER_STATE_BEGIN)
#endif
#ifndef VX_DCR_OM_STATE
#define VX_DCR_OM_STATE(addr)     ((addr) - VX_DCR_OM_STATE_BEGIN)
#endif
#ifndef VX_DCR_TEX_MIPOFF
#define VX_DCR_TEX_MIPOFF(lod)    (VX_DCR_TEX_MIPOFF_BASE + (lod))
#endif

///////////////////////////////////////////////////////////////////////////////
// Triangle-setup + tile-binning input types
///////////////////////////////////////////////////////////////////////////////

// One vertex in clip space, plus RASTER-interpolable attributes.
struct vertex_t {
  float pos[4];        // clip-space x, y, z, w
  float color[4];      // r, g, b, a in [0, 1]
  float texcoord[2];   // u, v
};

// One triangle, by vertex index into the caller's vertex container.
struct primitive_t {
  uint32_t i0, i1, i2;
};

///////////////////////////////////////////////////////////////////////////////
// Binning: triangle setup + tile assignment.
//
// Produces two device-ready buffers:
//   - primbuf: contiguous array of rast_prim_t (edge equations + attribute
//     deltas in Q15.16 / Q23.8 fixed-point form the RASTER unit reads).
//   - tilebuf: array of rast_tile_header_t records, each followed by the
//     primitive-ID list for that tile.
//
// Returns the number of tiles produced (>= 0). Width / height are the
// render-target dimensions in pixels. near / far are the depth-range
// extents in [0, 1]. tileLogSize is log2 of the RASTER tile size
// (typically 5 → 32×32 pixel tiles); must match VX_DCR_RASTER_TILE_LOGSIZE.
///////////////////////////////////////////////////////////////////////////////

uint32_t Binning(std::vector<uint8_t>& tilebuf,
                 std::vector<uint8_t>& primbuf,
                 const std::unordered_map<uint32_t, vertex_t>& vertices,
                 const std::vector<primitive_t>& primitives,
                 uint32_t width,
                 uint32_t height,
                 float near,
                 float far,
                 uint32_t tileLogSize);

} // namespace graphics
} // namespace vortex
