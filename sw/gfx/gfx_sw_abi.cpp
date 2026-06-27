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

// gfx_v2 §5 software-fallback C ABI — thin wrappers over the C++ single-source-
// of-truth headers. Built to LLVM bitcode with the libgfx_sw divergence flags so
// the vortexpipe FS can llvm-link + inline these (om_fragment's divergent merge
// needs the Vortex divergence pass over the whole FS kernel; see libgfx_sw.mk).

#include "gfx_sw_abi.h"
#include "gfx_sw.h"      // gfx_sw::TexState / om_state_t / tex_sample_sw / om_fragment
#include "rast_sw.h"     // gfx_rast::rast_walk_primitive — SW fine-rasterizer
#include "vx_gfx_abi.h"  // vortex::graphics::rast_prim_t
#include <cstring>
#include <type_traits>

// The C ABI descriptors must be layout-identical to the C++ state structs so the
// reinterpret_cast below is sound (and the host can fill either form).
static_assert(sizeof(gfx_sw_texstate_t) == sizeof(gfx_sw::TexState),
              "gfx_sw_texstate_t must mirror gfx_sw::TexState");
static_assert(sizeof(gfx_sw_omstate_t) == sizeof(gfx_sw::om_state_t),
              "gfx_sw_omstate_t must mirror gfx_sw::om_state_t");
static_assert(std::is_trivially_copyable<gfx_sw::TexState>::value &&
              std::is_trivially_copyable<gfx_sw::om_state_t>::value,
              "SW state structs must be POD for the C ABI");

extern "C" uint32_t gfx_tex_sample_sw(const gfx_sw_texstate_t* st,
                                      int32_t u, int32_t v, uint32_t lod) {
  return gfx_sw::tex_sample_sw(*reinterpret_cast<const gfx_sw::TexState*>(st), u, v, lod);
}

extern "C" void gfx_om_fragment_sw(const gfx_sw_omstate_t* st,
                                   uint32_t x, uint32_t y, uint32_t face,
                                   uint32_t color, uint32_t depth) {
  gfx_sw::om_fragment(*reinterpret_cast<const gfx_sw::om_state_t*>(st), x, y, face, color, depth);
}

extern "C" uint32_t gfx_rast_walk_tile_sw(const void* prim, uint32_t pid,
                                          uint32_t tx, uint32_t ty, uint32_t tile_logsize,
                                          uint32_t scissor_w, uint32_t scissor_h,
                                          gfx_rast_quad_t* out, uint32_t max) {
  const auto* p = reinterpret_cast<const vortex::graphics::rast_prim_t*>(prim);
  gfx_rast::RastConfig cfg{ tile_logsize, 0, 0, scissor_w, scissor_h };
  uint32_t count = 0;
  gfx_rast::rast_walk_primitive(cfg, tx, ty, pid, p->edges,
    [&](uint32_t pos_mask, const gfx_rast::vec3e_t* bc, uint32_t) {
      if (count >= max) return;
      gfx_rast_quad_t& q = out[count++];
      q.pos_mask = pos_mask;
      // Pack as the FF frag payload: bcoords[axis*4 + corner].
      for (uint32_t c = 0; c < 4; ++c) {
        q.bcoords[0 * 4 + c] = bc[c].x.data();
        q.bcoords[1 * 4 + c] = bc[c].y.data();
        q.bcoords[2 * 4 + c] = bc[c].z.data();
      }
    });
  return count;
}
