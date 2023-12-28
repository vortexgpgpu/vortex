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

#include "gfxutil.h"
#include "graphics.h"
#include <assert.h>
#include <cstring>
#include <math.h>
#include <iostream>
#include <map>
#include <algorithm>

using namespace cocogfx;
using namespace graphics;

using vec2f_t = TVector2<float>;
using vec3f_t = TVector3<float>;
using vec4f_t = TVector4<float>;
using rectf_t = TRect<float>;

///////////////////////////////////////////////////////////////////////////////

static bool EdgeEquation(vec3f_t edges[3], 
                         const vec4f_t& v0, 
                         const vec4f_t& v1, 
                         const vec4f_t& v2) {
  // Calculate edge equation matrix
  auto a0 = (v1.y * v2.w) - (v2.y * v1.w);
  auto a1 = (v2.y * v0.w) - (v0.y * v2.w);
  auto a2 = (v0.y * v1.w) - (v1.y * v0.w);

  auto b0 = (v2.x * v1.w) - (v1.x * v2.w);
  auto b1 = (v0.x * v2.w) - (v2.x * v0.w);
  auto b2 = (v1.x * v0.w) - (v0.x * v1.w);

  auto c0 = (v1.x * v2.y) - (v2.x * v1.y);
  auto c1 = (v2.x * v0.y) - (v0.x * v2.y);
  auto c2 = (v0.x * v1.y) - (v1.x * v0.y);

  edges[0] = {a0, b0, c0};
  edges[1] = {a1, b1, c1};
  edges[2] = {a2, b2, c2};

  /*printf("E0.x=%f, E0.y=%f, E0.z=%f, E1.x=%f, E1.y=%f, E1.z=%f, E2.x=%f, E2.y=%f, E2.z=%f\n", 
      edges[0].x, edges[0].y, edges[0].z,
      edges[1].x, edges[1].y, edges[1].z,
      edges[2].x, edges[2].y, edges[2].z);*/

  auto det = c0 * v0.w + c1 * v1.w + c2 * v2.w;
  if (det < 0) {
    edges[0].x *= -1.0f;
    edges[0].y *= -1.0f;
    edges[0].z *= -1.0f;
    edges[1].x *= -1.0f;
    edges[1].y *= -1.0f;
    edges[1].z *= -1.0f;
    edges[2].x *= -1.0f;
    edges[2].y *= -1.0f;
    edges[2].z *= -1.0f;
  }

  return (det != 0);
}

#ifdef FIXEDPOINT_RASTERIZER

static void EdgeToFixed(vec3e_t out[3], vec3f_t in[3]) {
  // Normalize the matrix
  auto maxVal = std::max({std::abs(in[0].x), std::abs(in[1].x), std::abs(in[2].x),
                          std::abs(in[0].y), std::abs(in[1].y), std::abs(in[2].y)});
  auto scale = 1.0f / maxVal;
  auto t0 = vec3f_t{in[0].x * scale, in[0].y * scale, in[0].z * scale};
  auto t1 = vec3f_t{in[1].x * scale, in[1].y * scale, in[1].z * scale};
  auto t2 = vec3f_t{in[2].x * scale, in[2].y * scale, in[2].z * scale};

  // Convert the edge equation to fixedpoint
  out[0] = {FloatE(t0.x), FloatE(t0.y), FloatE(t0.z)};
  out[1] = {FloatE(t1.x), FloatE(t1.y), FloatE(t1.z)};
  out[2] = {FloatE(t2.x), FloatE(t2.y), FloatE(t2.z)};

  //printf("*** out0=(%d, %d, %d)\n", outs[0].x.data(), outs[0].y.data(), outs[0].z.data());
  //printf("*** out1=(%d, %d, %d)\n", outs[1].x.data(), outs[1].y.data(), outs[1].z.data());
  //printf("*** out2=(%d, %d, %d)\n", outs[2].x.data(), outs[2].y.data(), outs[2].z.data());
}

#endif

namespace graphics {

// scan primitives and perform tile assignment
uint32_t Binning(std::vector<uint8_t>& tilebuf, 
                 std::vector<uint8_t>& primbuf,
                 const std::unordered_map<uint32_t, CGLTrace::vertex_t>& vertices,
                 const std::vector<CGLTrace::primitive_t>& primitives,                 
                 uint32_t width,
                 uint32_t height,
                 float near,
                 float far,
                 uint32_t tileLogSize) {
  std::map<std::pair<uint16_t, uint16_t>, std::vector<uint32_t>> tiles;

  std::vector<rast_prim_t> rast_prims;
  rast_prims.reserve(primitives.size());

  rast_bbox_t global_bbox{-1u, 0, -1u, 0};

  uint32_t total_prims = 0;

  #define POS_TO_V2D(d, s) \
      d.x = s.x; \
      d.y = s.y

  #define POS_TO_V4D(d, s) \
    d.x = s.x; \
    d.y = s.y; \
    d.z = s.z; \
    d.w = s.w
  
  for (auto& primitive : primitives) {
    // get primitive vertices
    auto& v0 = vertices.at(primitive.i0);
    auto& v1 = vertices.at(primitive.i1);
    auto& v2 = vertices.at(primitive.i2);

    vec4f_t p0, p1, p2;
    POS_TO_V4D (p0, v0.pos);
    POS_TO_V4D (p1, v1.pos);
    POS_TO_V4D (p2, v2.pos);

    vec3f_t edges[3];
    rast_bbox_t bbox;
    vec4f_t ps0, ps1, ps2;

    {
      vec4f_t ph0, ph1, ph2;
      
      // Convert position from clip to 2D homogenous device space      
      ClipToHDC(&ph0, p0, 0, width, 0, height, near, far);
      ClipToHDC(&ph1, p1, 0, width, 0, height, near, far);
      ClipToHDC(&ph2, p2, 0, width, 0, height, near, far);

      // Calculate edge equation
      if (!EdgeEquation(edges, ph0, ph1, ph2)) {
        // reject degenerate triangles
        printf("warning: degenerate primitive...\n");
        continue;
      }
    }       

    {
      // Convert position from clip to screen space      
      ClipToScreen(&ps0, p0, 0, width, 0, height, near, far);
      ClipToScreen(&ps1, p1, 0, width, 0, height, near, far);
      ClipToScreen(&ps2, p2, 0, width, 0, height, near, far);

      // Calculate bounding box
      vec2f_t q0, q1, q2;
      POS_TO_V2D (q0, ps0);
      POS_TO_V2D (q1, ps1);
      POS_TO_V2D (q2, ps2);

      //printf("*** screen position: v0=(%f, %f), v1=(%f, %f), v2=(%f, %f)\n", q0.x, q0.y, q1.x, q1.y, q2.x, q2.y);

      rectf_t tmp;
      CalcBoundingBox(&tmp, q0, q1, q2);
      auto tbb_left   = static_cast<int>(std::floor(tmp.left));
      auto tbb_right  = static_cast<int>(std::ceil(tmp.right));
      auto tbb_top    = static_cast<int>(std::floor(tmp.top));
      auto tbb_bottom = static_cast<int>(std::ceil(tmp.bottom));

      // clamp to scissor
      bbox.left   = std::max<int32_t>(tbb_left,   0);
      bbox.right  = std::min<int32_t>(tbb_right,  width);
      bbox.top    = std::max<int32_t>(tbb_top,    0);
      bbox.bottom = std::min<int32_t>(tbb_bottom, height);
      
      // reject excluded primitives
		  if (bbox.right <= bbox.left || 
		      bbox.bottom <= bbox.top)
			  continue;

      //printf("*** bbpx=(%f, %f, %f, %f)\n", tmp.left, tmp.right, tmp.top, tmp.bottom);
      global_bbox.left   = std::min(bbox.left, global_bbox.left);
      global_bbox.right  = std::max(bbox.right, global_bbox.right);
      global_bbox.top    = std::min(bbox.top, global_bbox.top);
      global_bbox.bottom = std::max(bbox.bottom, global_bbox.bottom);
    }

    uint32_t p;

    {
      #define ATTRIBUTE_DELTA(d, x0, x1, x2) \
        d.x = FloatA(x0 - x2); \
        d.y = FloatA(x1 - x2); \
        d.z = FloatA(x2)

      rast_prim_t rast_prim;      		   

			// add half-pixel offset
			edges[0].z += edges[0].x * 0.5f + edges[0].y * 0.5f;
			edges[1].z += edges[1].x * 0.5f + edges[1].y * 0.5f;
			edges[2].z += edges[2].x * 0.5f + edges[2].y * 0.5f;

    #ifdef FIXEDPOINT_RASTERIZER 
      EdgeToFixed(rast_prim.edges, edges);
    #else
      rast_prim.edges[0] = edges[0];
      rast_prim.edges[1] = edges[1];
      rast_prim.edges[2] = edges[2];
    #endif
         
      ATTRIBUTE_DELTA (rast_prim.attribs.z, ps0.z, ps1.z, ps2.z);
      ATTRIBUTE_DELTA (rast_prim.attribs.r, v0.color.r, v1.color.r, v2.color.r);
      ATTRIBUTE_DELTA (rast_prim.attribs.g, v0.color.g, v1.color.g, v2.color.g);
      ATTRIBUTE_DELTA (rast_prim.attribs.b, v0.color.b, v1.color.b, v2.color.b);
      ATTRIBUTE_DELTA (rast_prim.attribs.a, v0.color.a, v1.color.a, v2.color.a);
      ATTRIBUTE_DELTA (rast_prim.attribs.u, v0.texcoord.u, v1.texcoord.u, v2.texcoord.u);
      ATTRIBUTE_DELTA (rast_prim.attribs.v, v0.texcoord.v, v1.texcoord.v, v2.texcoord.v);

      p = rast_prims.size();
      rast_prims.push_back(rast_prim);      
    }

    // calculate tiles coverage
    {    
      auto tileSize = 1 << tileLogSize;
      auto minTileX = bbox.left >> tileLogSize;
      auto maxTileX = (bbox.right + tileSize - 1) >> tileLogSize;
      auto minTileY = bbox.top >> tileLogSize;      
      auto maxTileY = (bbox.bottom + tileSize - 1) >> tileLogSize;

      for (uint32_t ty = minTileY; ty < maxTileY; ++ty) {
				for (uint32_t tx = minTileX; tx < maxTileX; ++tx) {
					tiles[{tx, ty}].push_back(p);
          ++total_prims;
				}
			}
    }
  }

  {
    primbuf.resize(rast_prims.size() * sizeof(rast_prim_t));
    memcpy(primbuf.data(), rast_prims.data(), primbuf.size());
  }
  
  {
    tilebuf.resize(tiles.size() * sizeof(rast_tile_header_t) + total_prims * sizeof(uint32_t));
    auto tile_header = reinterpret_cast<rast_tile_header_t*>(tilebuf.data());
    auto pids_buffer = reinterpret_cast<uint8_t*>(tilebuf.data() + tiles.size() * sizeof(rast_tile_header_t));
    for (auto& it : tiles) {
      tile_header->tile_x = it.first.first;
      tile_header->tile_y = it.first.second;
      tile_header->pids_offset = (pids_buffer - reinterpret_cast<uint8_t*>(tile_header + 1)) / sizeof(uint32_t);
      tile_header->pids_count = it.second.size();
      ++tile_header;
      memcpy(pids_buffer, it.second.data(), it.second.size() * sizeof(uint32_t));
      pids_buffer += it.second.size() * sizeof(uint32_t);      
    }
  }

  //printf("Binning bounding box={l=%d, r=%d, t=%d, b=%d}\n", global_bbox.left, global_bbox.right, global_bbox.top, global_bbox.bottom);

  return tiles.size();
}

///////////////////////////////////////////////////////////////////////////////

uint32_t toVXFormat(ePixelFormat format) {
  switch (format) {
  case FORMAT_A8R8G8B8: return VX_TEX_FORMAT_A8R8G8B8; break;
  case FORMAT_R5G6B5: return VX_TEX_FORMAT_R5G6B5; break;
  case FORMAT_A1R5G5B5: return VX_TEX_FORMAT_A1R5G5B5; break;
  case FORMAT_A4R4G4B4: return VX_TEX_FORMAT_A4R4G4B4; break;
  case FORMAT_A8L8: return VX_TEX_FORMAT_A8L8; break;
  case FORMAT_L8: return VX_TEX_FORMAT_L8; break;
  case FORMAT_A8: return VX_TEX_FORMAT_A8; break;
  default:
    std::cout << "Error: invalid format: " << format << std::endl;
    exit(1);
  }
  return 0;
}

uint32_t toVXCompare(CGLTrace::ecompare compare) {
  switch (compare) {
  case CGLTrace::COMPARE_NEVER: return VX_OM_DEPTH_FUNC_NEVER; break;
  case CGLTrace::COMPARE_LESS: return VX_OM_DEPTH_FUNC_LESS; break;
  case CGLTrace::COMPARE_EQUAL: return VX_OM_DEPTH_FUNC_EQUAL; break;
  case CGLTrace::COMPARE_LEQUAL: return VX_OM_DEPTH_FUNC_LEQUAL; break;
  case CGLTrace::COMPARE_GREATER: return VX_OM_DEPTH_FUNC_GREATER; break;
  case CGLTrace::COMPARE_NOTEQUAL: return VX_OM_DEPTH_FUNC_NOTEQUAL; break;
  case CGLTrace::COMPARE_GEQUAL: return VX_OM_DEPTH_FUNC_GEQUAL; break;
  case CGLTrace::COMPARE_ALWAYS: return VX_OM_DEPTH_FUNC_ALWAYS; break;
  default:
    std::cout << "Error: invalid compare function: " << compare << std::endl;
    exit(1);
  }
  return 0;
}

uint32_t toVXStencilOp(CGLTrace::eStencilOp op) {
  switch (op) {
  case CGLTrace::STENCIL_KEEP: return VX_OM_STENCIL_OP_KEEP; break;
  case CGLTrace::STENCIL_REPLACE: return VX_OM_STENCIL_OP_REPLACE; break;
  case CGLTrace::STENCIL_INCR: return VX_OM_STENCIL_OP_INCR; break;
  case CGLTrace::STENCIL_DECR: return VX_OM_STENCIL_OP_DECR; break;
  case CGLTrace::STENCIL_ZERO: return VX_OM_STENCIL_OP_ZERO; break;
  case CGLTrace::STENCIL_INVERT: return VX_OM_STENCIL_OP_INVERT; break;
  default:
    std::cout << "Error: invalid stencil operation: " << op << std::endl;
    exit(1);
  }
  return 0;
}

uint32_t toVXBlendFunc(CGLTrace::eBlendOp op) {
  switch (op) {
  case CGLTrace::BLEND_ZERO: return VX_OM_BLEND_FUNC_ZERO;
  case CGLTrace::BLEND_ONE: return VX_OM_BLEND_FUNC_ONE;
  case CGLTrace::BLEND_SRC_COLOR: return VX_OM_BLEND_FUNC_SRC_RGB;
  case CGLTrace::BLEND_ONE_MINUS_SRC_COLOR: return VX_OM_BLEND_FUNC_ONE_MINUS_SRC_RGB;
  case CGLTrace::BLEND_SRC_ALPHA: return VX_OM_BLEND_FUNC_SRC_A;
  case CGLTrace::BLEND_ONE_MINUS_SRC_ALPHA: return VX_OM_BLEND_FUNC_ONE_MINUS_SRC_A;
  case CGLTrace::BLEND_DST_ALPHA: return VX_OM_BLEND_FUNC_DST_A;
  case CGLTrace::BLEND_ONE_MINUS_DST_ALPHA: return VX_OM_BLEND_FUNC_ONE_MINUS_DST_A;
  case CGLTrace::BLEND_DST_COLOR: return VX_OM_BLEND_FUNC_DST_RGB;
  case CGLTrace::BLEND_ONE_MINUS_DST_COLOR: return VX_OM_BLEND_FUNC_ONE_MINUS_DST_RGB;
  case CGLTrace::BLEND_SRC_ALPHA_SATURATE: return VX_OM_BLEND_FUNC_ALPHA_SAT;
  default:
    std::cout << "Error: invalid blend function: " << op << std::endl;
    exit(1);
  }
  return 0;
}

} // namespace graphics