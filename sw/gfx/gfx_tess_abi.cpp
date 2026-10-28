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

// The tessellator's C ABI for the shader kernels vortexpipe translates: the
// control stage's epilogue processes its patch's levels, and each evaluation
// thread places its own domain point. Co-compiled into those kernels, so the
// calls inline under the Vortex divergence pass with the rest of the shader.

#include "gfx_tess.h"

// Process a patch's levels (outer[4] then inner[2], as the control stage wrote
// them) for GFX_TESS_MODE `mode` into *desc, with the patch's gl_PrimitiveID,
// and its point, primitive and stitch-unit counts into counts[0..2].
extern "C" void gfx_tess_patch_setup(uint32_t mode, const float* levels, uint32_t prim_id,
                                     gfx_tess_patch_t* desc, uint32_t* counts) {
  gfx_tess::process(mode, levels, levels + 4, desc);
  desc->prim_id = prim_id;
  counts[0] = desc->num_points;
  counts[1] = desc->num_prims;
  counts[2] = desc->num_units;
}

// Domain point g of a batch: out[0] its patch in the batch, out[1] the patch's
// gl_PrimitiveID, out[2] and out[3] the bits of gl_TessCoord.xy.
extern "C" void gfx_tess_eval_point(const gfx_tess_draw_t* d, uint32_t g, uint32_t* out) {
  uint32_t p, i, id;
  const auto scan = reinterpret_cast<const uint32_t*>(d->scan_addr);
  if (scan) {
    const uint32_t k = g + d->first_point;
    p = gfx_tess::locate(scan, d->patches, 0, k);
    i = k - scan[3 * p];
    id = gfx_tess::descriptor(*d, p)->prim_id;
  } else {
    p = g / d->points;
    i = g - p * d->points;
    id = gfx_tess::primitive_id(*d, p);
  }
  gfx_tess::fxp_t u, v;
  gfx_tess::point(*gfx_tess::descriptor(*d, p), i, &u, &v);
  out[0] = p;
  out[1] = id;
  out[2] = gfx_tess::f2u(gfx_tess::to_float(u));
  out[3] = gfx_tess::f2u(gfx_tess::to_float(v));
}
