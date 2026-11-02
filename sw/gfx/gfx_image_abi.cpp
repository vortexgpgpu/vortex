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

// Typeless image access, the part of the software-fallback C ABI a kernel calls
// per texel. vortexpipe links this unit into the kernel's IR before codegen
// rather than beside it as an object, so both entry points inline: as calls,
// each access pays a full frame (the callee-saved spills of a helper that
// decodes any format), which dominated typeless storage-image loops.

#include "gfx_sw_abi.h"
#include "gfx_sw.h"
#include <cstddef>
#include <cstring>

extern "C" __attribute__((always_inline))
void gfx_image_load_sw(const void* base, uint32_t row_stride, uint32_t img_stride,
                       int32_t x, int32_t y, uint32_t layer, uint32_t format,
                       uint32_t* out) {
  const uint32_t bpp = gfx_tex::FormatStride(format);
  const uint8_t* p = static_cast<const uint8_t*>(base) + (size_t)layer * img_stride
                   + (size_t)(uint32_t)y * row_stride + (size_t)(uint32_t)x * bpp;
  float lanes[4];
  gfx_tex::TexDecodeFloat4(format, p, bpp, lanes);
  memcpy(out, lanes, sizeof(lanes));
}

extern "C" __attribute__((always_inline))
void gfx_image_store_sw(void* base, uint32_t row_stride, uint32_t img_stride,
                        int32_t x, int32_t y, uint32_t layer, uint32_t format,
                        const uint32_t* lanes) {
  const uint32_t bpp = gfx_tex::FormatStride(format);
  uint8_t* p = static_cast<uint8_t*>(base) + (size_t)layer * img_stride
             + (size_t)(uint32_t)y * row_stride + (size_t)(uint32_t)x * bpp;
  if (gfx_tex::TexIsFieldFormat(format)) {
    gfx_tex::TexEncodeField4(format, p, lanes);
    return;
  }
  // The float formats, which hold no fields to describe.
  switch (format) {
  case VX_TEX_FORMAT_R16F:
  case VX_TEX_FORMAT_RG16F:
  case VX_TEX_FORMAT_RGBA16F:
    for (uint32_t c = 0; c < bpp / 2; ++c) {
      const uint16_t h = (uint16_t)gfx_sw::om_float_to_half(gfx_sw::om_bits_f(lanes[c]));
      memcpy(p + 2 * c, &h, 2);
    }
    break;
  case VX_TEX_FORMAT_R11G11B10F: {
    const uint32_t t = gfx_sw::om_float_to_ufloat(gfx_sw::om_bits_f(lanes[0]), 6)
                     | (gfx_sw::om_float_to_ufloat(gfx_sw::om_bits_f(lanes[1]), 6) << 11)
                     | (gfx_sw::om_float_to_ufloat(gfx_sw::om_bits_f(lanes[2]), 5) << 22);
    memcpy(p, &t, 4);
    break;
  }
  case VX_TEX_FORMAT_R32F:
  case VX_TEX_FORMAT_RG32F:
  case VX_TEX_FORMAT_RGBA32F:
  case VX_TEX_FORMAT_D32F:
    memcpy(p, lanes, bpp);
    break;
  default:
    // Every storage format is one of the above or a field format; anything
    // else names no layout a store can produce, so the texel is left as is.
    break;
  }
}
