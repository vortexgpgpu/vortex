// gfx_v2 SW texture-sampler parity unit test (§4.2).
//
// Proves the on-device SIMT software fallback for vx_tex4 — gfx_sw.h
// tex_sample_sw() — produces byte-identical results to the fixed-function TEX
// model (gfx_ff_model.cpp TextureSampler::read) for every format / filter / wrap /
// LOD, including the §6.8 trilinear mip blend. Both paths now share the
// gfx_frag_tex.h math (single source of truth, §7), so equality is by construction
// — this test locks that contract and catches any future plumbing drift in the
// TexState mirror, the per-LOD tap selection, or the trilinear wrapper.
//
// The "device memory" is mmap'd at a fixed low, 64-byte-aligned address so the
// real host pointer survives the TexDCRS TEX_ADDR (base >> 6, 32-bit) round trip
// the FF model performs, letting the FF MemoryCB and the SW path's raw-pointer
// loads read the exact same bytes.
//
// Build/run: make -C tests/unittest/gfx_tex_sw run

#include "gfx_ff_model.h"   // FF model (TextureSampler) — the golden
#include "gfx_sw.h"       // SW fallback (tex_sample_sw) — under test
#include <sys/mman.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

using namespace vortex;

namespace {

// Fixed low base for the texture pool: 1GB (2^30) is 64-aligned and < 2^38, so
// base >> 6 fits the 32-bit TEX_ADDR DCR and round-trips exactly.
constexpr uintptr_t TEX_BASE = 0x40000000ull;
constexpr size_t    TEX_POOL = 16u << 20;   // 16 MB

uint32_t g_fail = 0;

uint32_t format_stride(uint32_t fmt) { return gfx_tex::FormatStride(fmt); }

// Full mip chain for a (1<<logw0) x (1<<logh0) base, packed contiguously from
// `base`; returns per-LOD byte offsets and fills each texel with a deterministic
// pattern so format-decode/filter differences would surface.
void build_texture(uint8_t* base, uint32_t logw0, uint32_t logh0, uint32_t fmt,
                   uint32_t mip_off[VX_TEX_LOD_MAX + 1], uint32_t* num_lods) {
  uint32_t stride = format_stride(fmt);
  uint32_t off = 0;
  uint32_t l = 0;
  for (; l <= (uint32_t)VX_TEX_LOD_MAX; ++l) {
    uint32_t lw = (logw0 > l) ? (logw0 - l) : 0;
    uint32_t lh = (logh0 > l) ? (logh0 - l) : 0;
    uint32_t w = 1u << lw, h = 1u << lh;
    mip_off[l] = off;
    for (uint32_t y = 0; y < h; ++y) {
      for (uint32_t x = 0; x < w; ++x) {
        // Deterministic per-texel value; bytes vary so every channel is exercised.
        uint32_t v = (l * 0x9E3779B1u) ^ (y * 0x85EBCA77u) ^ (x * 0xC2B2AE3Du);
        uint8_t* p = base + off + (y * w + x) * stride;
        for (uint32_t b = 0; b < stride; ++b)
          p[b] = (uint8_t)(v >> (b * 8));
      }
    }
    off += w * h * stride;
    if (lw == 0 && lh == 0) break;
  }
  uint32_t last = l < (uint32_t)VX_TEX_LOD_MAX ? l : (uint32_t)VX_TEX_LOD_MAX;
  *num_lods = last + 1;
  // The driver writes an offset for every LOD slot; LODs past the real chain
  // clamp to the last mip (so a trilinear blend that brackets the top reads a
  // valid texel). Populate all slots so the FF DCRs and the SW mirror agree.
  for (uint32_t k = last + 1; k <= (uint32_t)VX_TEX_LOD_MAX; ++k)
    mip_off[k] = mip_off[last];
}

// FF MemoryCB: fetch `size` texels from resident host memory (raw pointers).
void mem_cb(uint32_t* out, const uint64_t* addr, uint32_t stride, uint32_t size, void*) {
  for (uint32_t i = 0; i < size; ++i)
    out[i] = gfx_sw::tex_load_texel(addr[i], stride);
}

const char* fmt_name(uint32_t f) {
  switch (f) {
  case VX_TEX_FORMAT_A8R8G8B8: return "A8R8G8B8";
  case VX_TEX_FORMAT_R5G6B5:   return "R5G6B5";
  case VX_TEX_FORMAT_A1R5G5B5: return "A1R5G5B5";
  case VX_TEX_FORMAT_A4R4G4B4: return "A4R4G4B4";
  case VX_TEX_FORMAT_A8L8:     return "A8L8";
  case VX_TEX_FORMAT_L8:       return "L8";
  case VX_TEX_FORMAT_A8:       return "A8";
  }
  return "?";
}

// One configuration: drive FF read() and SW tex_sample_sw() over a grid of
// (u, v, lod) and assert bit-equality.
void check_config(uint8_t* pool, uint32_t fmt, uint32_t mag_filter, uint32_t mip_linear,
                  uint32_t wrapu, uint32_t wrapv) {
  const uint32_t logw0 = 6, logh0 = 5;       // 64 x 32 base
  uint8_t* base = pool;
  uint32_t mip_off[VX_TEX_LOD_MAX + 1] = {0};
  uint32_t num_lods = 0;
  build_texture(base, logw0, logh0, fmt, mip_off, &num_lods);

  uint32_t logdim = (logh0 << 16) | logw0;
  uint32_t wrap   = (wrapv << 16) | wrapu;
  uint32_t filter = mag_filter | (mip_linear ? VX_TEX_FILTER_MIP_LINEAR : 0u);

  // FF model state via TexDCRS.
  TexDCRS dcrs;
  dcrs.write(VX_DCR_TEX_STAGE,  0);
  dcrs.write(VX_DCR_TEX_ADDR,   (uint32_t)((uintptr_t)base >> 6));
  dcrs.write(VX_DCR_TEX_LOGDIM, logdim);
  dcrs.write(VX_DCR_TEX_FORMAT, fmt);
  dcrs.write(VX_DCR_TEX_FILTER, filter);
  dcrs.write(VX_DCR_TEX_WRAP,   wrap);
  for (uint32_t l = 0; l <= (uint32_t)VX_TEX_LOD_MAX; ++l)
    dcrs.write(VX_DCR_TEX_MIPOFF_BASE + l, mip_off[l]);
  TextureSampler ff(mem_cb, nullptr);
  ff.configure(dcrs);

  // SW state mirror.
  gfx_sw::TexState st{};
  st.base   = (uintptr_t)base;
  st.logdim = logdim;
  st.format = fmt;
  st.filter = filter;
  st.wrap   = wrap;
  for (uint32_t l = 0; l <= (uint32_t)VX_TEX_LOD_MAX; ++l)
    st.mip_off[l] = mip_off[l];

  const int32_t ONE = 1 << VX_TEX_FXD_FRAC;
  uint32_t mismatches = 0, checked = 0;
  for (int32_t ui = -2; ui <= 10; ++ui) {
    for (int32_t vi = -2; vi <= 10; ++vi) {
      int32_t u = (int32_t)((int64_t)ui * ONE / 8);   // covers [-0.25 .. 1.25]
      int32_t v = (int32_t)((int64_t)vi * ONE / 8);
      // Integer LODs plus a few fractional ones for the trilinear path.
      for (uint32_t li = 0; li < num_lods; ++li) {
        for (uint32_t fr = 0; fr < 256; fr += (mip_linear ? 64 : 256)) {
          uint32_t lod = mip_linear ? ((li << VX_TEX_LOD_FRAC_BITS) | fr) : li;
          uint32_t a = ff.read(0, u, v, lod);
          uint32_t b = gfx_sw::tex_sample_sw(st, u, v, lod);
          ++checked;
          if (a != b) {
            if (++mismatches <= 4)
              printf("  MISMATCH fmt=%s magf=%u mipL=%u wrap=(%u,%u) u=%d v=%d lod=0x%x: ff=%08x sw=%08x\n",
                     fmt_name(fmt), mag_filter, mip_linear, wrapu, wrapv, u, v, lod, a, b);
          }
        }
      }
    }
  }
  if (mismatches) {
    g_fail += mismatches;
    printf("FAIL %-8s magf=%u mipL=%u wrap=(%u,%u): %u/%u mismatches\n",
           fmt_name(fmt), mag_filter, mip_linear, wrapu, wrapv, mismatches, checked);
  }
}

} // namespace

int main() {
  void* p = mmap((void*)TEX_BASE, TEX_POOL, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
  if (p == MAP_FAILED || (uintptr_t)p != TEX_BASE) {
    fprintf(stderr, "mmap at fixed base 0x%lx failed (got %p)\n", TEX_BASE, p);
    return 2;
  }
  uint8_t* pool = (uint8_t*)p;

  const uint32_t formats[] = {
    VX_TEX_FORMAT_A8R8G8B8, VX_TEX_FORMAT_R5G6B5, VX_TEX_FORMAT_A1R5G5B5,
    VX_TEX_FORMAT_A4R4G4B4, VX_TEX_FORMAT_A8L8, VX_TEX_FORMAT_L8, VX_TEX_FORMAT_A8,
  };
  const uint32_t wraps[] = { VX_TEX_WRAP_CLAMP, VX_TEX_WRAP_REPEAT, VX_TEX_WRAP_MIRROR };

  uint32_t configs = 0;
  for (uint32_t fmt : formats)
    for (uint32_t magf : {(uint32_t)VX_TEX_FILTER_POINT, (uint32_t)VX_TEX_FILTER_BILINEAR})
      for (uint32_t mipL : {0u, 1u})
        for (uint32_t wu : wraps)
          for (uint32_t wv : wraps) {
            check_config(pool, fmt, magf, mipL, wu, wv);
            ++configs;
          }

  if (g_fail) {
    printf("\nTEX-SW PARITY: FAILED (%u mismatches across %u configs)\n", g_fail, configs);
    return 1;
  }
  printf("TEX-SW PARITY: PASSED (%u configs, SW == FF bit-exact)\n", configs);
  return 0;
}
