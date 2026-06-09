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

// Implementation of vortex::graphics::Binning. Self-contained: uses
// only stdlib + the public graphics.h header. No cocogfx dependency
// (intentional — graphics.h ships in the install tree and downstream
// tools build only against $VORTEX_PATH/runtime/include).

#include <graphics.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <utility>

namespace vortex {
namespace graphics {

namespace {

///////////////////////////////////////////////////////////////////////////////
// Local float-vector and rect helpers (no cocogfx). Used internally by
// Binning for triangle setup; not exposed in the public header.
///////////////////////////////////////////////////////////////////////////////

struct vec2f { float x, y; };
struct vec3f { float x, y, z; };
struct vec4f { float x, y, z, w; };
struct rectf { float left, right, top, bottom; };

// Clip-space (x, y, z, w) → homogeneous-device-coordinate (HDC).
// The W component is preserved so downstream callers can apply the
// perspective divide where appropriate (edge equations stay in HDC).
inline vec4f ClipToHDC(const vec4f& in,
                       int32_t left, int32_t right,
                       int32_t top,  int32_t bottom,
                       float near, float far) {
  float minX   = float(left + right) * 0.5f;
  float scaleX = float(right - left) * 0.5f;
  float minY   = float(top + bottom) * 0.5f;
  float scaleY = float(bottom - top) * 0.5f;
  float minZ   = (near + far) * 0.5f;
  float scaleZ = (far - near) * 0.5f;
  return {
    in.x * scaleX + in.w * minX,
    in.y * scaleY + in.w * minY,
    in.z * scaleZ + in.w * minZ,
    in.w,
  };
}

// Clip-space → normalized-device-coordinate (NDC) via perspective divide.
inline vec4f ClipToNDC(const vec4f& in) {
  float rhw = (in.w != 0.0f) ? (1.0f / in.w) : 0.0f;
  return { in.x * rhw, in.y * rhw, in.z * rhw, rhw };
}

// Clip-space → screen space (NDC then viewport).
inline vec4f ClipToScreen(const vec4f& in,
                          int32_t left, int32_t right,
                          int32_t top,  int32_t bottom,
                          float near, float far) {
  vec4f ndc = ClipToNDC(in);
  float minX   = float(left + right) * 0.5f;
  float scaleX = float(right - left) * 0.5f;
  float minY   = float(top + bottom) * 0.5f;
  float scaleY = float(bottom - top) * 0.5f;
  float minZ   = (near + far) * 0.5f;
  float scaleZ = (far - near) * 0.5f;
  return {
    ndc.x * scaleX + minX,
    ndc.y * scaleY + minY,
    ndc.z * scaleZ + minZ,
    ndc.w,
  };
}

inline rectf CalcBoundingBox(const vec2f& v0, const vec2f& v1, const vec2f& v2) {
  rectf r;
  r.left   = std::min(v0.x, std::min(v1.x, v2.x));
  r.right  = std::max(v0.x, std::max(v1.x, v2.x));
  r.top    = std::min(v0.y, std::min(v1.y, v2.y));
  r.bottom = std::max(v0.y, std::max(v1.y, v2.y));
  return r;
}

///////////////////////////////////////////////////////////////////////////////
// EdgeEquation: produce the three (a, b, c) edge-function coefficients for
// the triangle (v0, v1, v2) in HDC space. Returns false for degenerate
// (zero-area) triangles. det negative → flip winding so the interior
// evaluates positive.
///////////////////////////////////////////////////////////////////////////////

bool EdgeEquation(vec3f edges[3],
                  const vec4f& v0, const vec4f& v1, const vec4f& v2) {
  float a0 = (v1.y * v2.w) - (v2.y * v1.w);
  float a1 = (v2.y * v0.w) - (v0.y * v2.w);
  float a2 = (v0.y * v1.w) - (v1.y * v0.w);

  float b0 = (v2.x * v1.w) - (v1.x * v2.w);
  float b1 = (v0.x * v2.w) - (v2.x * v0.w);
  float b2 = (v1.x * v0.w) - (v0.x * v1.w);

  float c0 = (v1.x * v2.y) - (v2.x * v1.y);
  float c1 = (v2.x * v0.y) - (v0.x * v2.y);
  float c2 = (v0.x * v1.y) - (v1.x * v0.y);

  edges[0] = {a0, b0, c0};
  edges[1] = {a1, b1, c1};
  edges[2] = {a2, b2, c2};

  float det = c0 * v0.w + c1 * v1.w + c2 * v2.w;
  if (det < 0) {
    for (int i = 0; i < 3; i++) {
      edges[i].x *= -1.0f;
      edges[i].y *= -1.0f;
      edges[i].z *= -1.0f;
    }
  }
  return (det != 0.0f);
}

///////////////////////////////////////////////////////////////////////////////
// EdgeToFixed: normalize the edge matrix and convert to Q15.16 fixed.
///////////////////////////////////////////////////////////////////////////////

void EdgeToFixed(vec3e_t out[3], const vec3f in[3]) {
  float maxVal = std::max({std::fabs(in[0].x), std::fabs(in[1].x), std::fabs(in[2].x),
                           std::fabs(in[0].y), std::fabs(in[1].y), std::fabs(in[2].y)});
  float scale = (maxVal != 0.0f) ? (1.0f / maxVal) : 1.0f;
  for (int i = 0; i < 3; i++) {
    out[i] = {
      FloatE(in[i].x * scale),
      FloatE(in[i].y * scale),
      FloatE(in[i].z * scale),
    };
  }
}

} // anonymous namespace

///////////////////////////////////////////////////////////////////////////////
// Binning: scan primitives, build per-tile primitive-ID lists, emit the
// device-ready tile-header + primitive-record buffers.
///////////////////////////////////////////////////////////////////////////////

uint32_t Binning(std::vector<uint8_t>& tilebuf,
                 std::vector<uint8_t>& primbuf,
                 const std::unordered_map<uint32_t, vertex_t>& vertices,
                 const std::vector<primitive_t>& primitives,
                 uint32_t width,
                 uint32_t height,
                 float near,
                 float far,
                 uint32_t tileLogSize) {
  std::map<std::pair<uint16_t, uint16_t>, std::vector<uint32_t>> tiles;

  std::vector<rast_prim_t> rast_prims;
  rast_prims.reserve(primitives.size());

  uint32_t total_prims = 0;

  for (auto& primitive : primitives) {
    auto it0 = vertices.find(primitive.i0);
    auto it1 = vertices.find(primitive.i1);
    auto it2 = vertices.find(primitive.i2);
    if (it0 == vertices.end() || it1 == vertices.end() || it2 == vertices.end()) {
      printf("warning: primitive references missing vertex...\n");
      continue;
    }
    const vertex_t& v0 = it0->second;
    const vertex_t& v1 = it1->second;
    const vertex_t& v2 = it2->second;

    vec4f p0 = { v0.pos[0], v0.pos[1], v0.pos[2], v0.pos[3] };
    vec4f p1 = { v1.pos[0], v1.pos[1], v1.pos[2], v1.pos[3] };
    vec4f p2 = { v2.pos[0], v2.pos[1], v2.pos[2], v2.pos[3] };

    vec3f edges[3];
    rast_bbox_t bbox;
    vec4f ps0, ps1, ps2;

    // Triangle edge equations in HDC
    {
      vec4f ph0 = ClipToHDC(p0, 0, width, 0, height, near, far);
      vec4f ph1 = ClipToHDC(p1, 0, width, 0, height, near, far);
      vec4f ph2 = ClipToHDC(p2, 0, width, 0, height, near, far);

      if (!EdgeEquation(edges, ph0, ph1, ph2)) {
        printf("warning: degenerate primitive...\n");
        continue;
      }
    }

    // Screen-space bounding box (clamped to render target)
    {
      ps0 = ClipToScreen(p0, 0, width, 0, height, near, far);
      ps1 = ClipToScreen(p1, 0, width, 0, height, near, far);
      ps2 = ClipToScreen(p2, 0, width, 0, height, near, far);

      vec2f q0 = { ps0.x, ps0.y };
      vec2f q1 = { ps1.x, ps1.y };
      vec2f q2 = { ps2.x, ps2.y };
      rectf tmp = CalcBoundingBox(q0, q1, q2);

      int32_t tbb_left   = static_cast<int32_t>(std::floor(tmp.left));
      int32_t tbb_right  = static_cast<int32_t>(std::ceil (tmp.right));
      int32_t tbb_top    = static_cast<int32_t>(std::floor(tmp.top));
      int32_t tbb_bottom = static_cast<int32_t>(std::ceil (tmp.bottom));

      bbox.left   = std::max<int32_t>(tbb_left,   0);
      bbox.right  = std::min<int32_t>(tbb_right,  static_cast<int32_t>(width));
      bbox.top    = std::max<int32_t>(tbb_top,    0);
      bbox.bottom = std::min<int32_t>(tbb_bottom, static_cast<int32_t>(height));

      if (bbox.right <= bbox.left || bbox.bottom <= bbox.top)
        continue;
    }

    uint32_t p;
    {
      rast_prim_t rast_prim{};

      // Half-pixel sampling offset
      edges[0].z += edges[0].x * 0.5f + edges[0].y * 0.5f;
      edges[1].z += edges[1].x * 0.5f + edges[1].y * 0.5f;
      edges[2].z += edges[2].x * 0.5f + edges[2].y * 0.5f;

      EdgeToFixed(rast_prim.edges, edges);

      auto attrib_delta = [](rast_attrib_t& d, float a0, float a1, float a2) {
        d.x = FloatA(a0 - a2);
        d.y = FloatA(a1 - a2);
        d.z = FloatA(a2);
      };
      attrib_delta(rast_prim.attribs.z, ps0.z,         ps1.z,         ps2.z);
      attrib_delta(rast_prim.attribs.r, v0.color[0],   v1.color[0],   v2.color[0]);
      attrib_delta(rast_prim.attribs.g, v0.color[1],   v1.color[1],   v2.color[1]);
      attrib_delta(rast_prim.attribs.b, v0.color[2],   v1.color[2],   v2.color[2]);
      attrib_delta(rast_prim.attribs.a, v0.color[3],   v1.color[3],   v2.color[3]);
      attrib_delta(rast_prim.attribs.u, v0.texcoord[0], v1.texcoord[0], v2.texcoord[0]);
      attrib_delta(rast_prim.attribs.v, v0.texcoord[1], v1.texcoord[1], v2.texcoord[1]);

      p = static_cast<uint32_t>(rast_prims.size());
      rast_prims.push_back(rast_prim);
    }

    // Tile coverage
    {
      uint32_t tileSize  = 1u << tileLogSize;
      uint32_t minTileX  = bbox.left   >> tileLogSize;
      uint32_t maxTileX  = (bbox.right  + tileSize - 1) >> tileLogSize;
      uint32_t minTileY  = bbox.top    >> tileLogSize;
      uint32_t maxTileY  = (bbox.bottom + tileSize - 1) >> tileLogSize;

      for (uint32_t ty = minTileY; ty < maxTileY; ty++) {
        for (uint32_t tx = minTileX; tx < maxTileX; tx++) {
          tiles[{static_cast<uint16_t>(tx), static_cast<uint16_t>(ty)}].push_back(p);
          ++total_prims;
        }
      }
    }
  }

  // Emit primbuf (contiguous rast_prim_t array)
  primbuf.resize(rast_prims.size() * sizeof(rast_prim_t));
  if (!rast_prims.empty())
    std::memcpy(primbuf.data(), rast_prims.data(), primbuf.size());

  // Emit tilebuf (tile headers followed by per-tile PID lists)
  tilebuf.resize(tiles.size() * sizeof(rast_tile_header_t)
                 + total_prims * sizeof(uint32_t));
  auto tile_header = reinterpret_cast<rast_tile_header_t*>(tilebuf.data());
  auto pids_buffer = reinterpret_cast<uint8_t*>(
      tilebuf.data() + tiles.size() * sizeof(rast_tile_header_t));
  for (auto& it : tiles) {
    tile_header->tile_x       = it.first.first;
    tile_header->tile_y       = it.first.second;
    tile_header->pids_offset  = static_cast<uint16_t>(
        (pids_buffer - reinterpret_cast<uint8_t*>(tile_header + 1)) / sizeof(uint32_t));
    tile_header->pids_count   = static_cast<uint16_t>(it.second.size());
    ++tile_header;
    std::memcpy(pids_buffer, it.second.data(),
                it.second.size() * sizeof(uint32_t));
    pids_buffer += it.second.size() * sizeof(uint32_t);
  }

  return static_cast<uint32_t>(tiles.size());
}

} // namespace graphics
} // namespace vortex
