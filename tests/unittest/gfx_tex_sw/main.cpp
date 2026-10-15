// SW texture-sampler parity unit test.
//
// Proves the on-device SIMT software fallback for vx_tex4 — gfx_sw.h
// tex_sample_sw() — produces byte-identical results to the fixed-function TEX
// model (gfx_ff_model.cpp TextureSampler::read) for every format / filter / wrap /
// LOD, including the trilinear mip blend. Both paths share the gfx_frag_tex.h
// math (single source of truth), so equality is by construction — this test locks
// that contract and catches any future plumbing drift in the TexState mirror, the
// per-LOD tap selection, or the trilinear wrapper.
//
// The "device memory" is mmap'd at a fixed low, 64-byte-aligned address so the
// real host pointer survives the TexDCRS TEX_ADDR (base >> 6, 32-bit) round trip
// the FF model performs, letting the FF MemoryCB and the SW path's raw-pointer
// loads read the exact same bytes.
//
// Build/run: make -C tests/unittest/gfx_tex_sw run

#include "gfx_ff_model.h"   // FF model (TextureSampler) — the golden
#include "gfx_sw.h"       // SW fallback (tex_sample_sw) — under test
#include "gfx_frontend_abi.h"  // setup_vertex_t / setup_bbox_t (perspective-wrap check)
#include "gfx_setup.h"         // gfx_setup::setup_triangle (perspective-wrap check)
#include <sys/mman.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace vortex;

namespace {

// Fixed low base for the texture pool: 1GB (2^30) is 64-aligned and < 2^38, so
// base >> 6 fits the 32-bit TEX_ADDR DCR and round-trips exactly.
constexpr uintptr_t TEX_BASE = 0x40000000ull;
constexpr size_t    TEX_POOL = 16u << 20;   // 16 MB

uint32_t g_fail = 0;

uint32_t format_stride(uint32_t fmt) { return gfx_tex::FormatStride(fmt); }

// Full mip chain for a w0 x h0 base, packed contiguously from `base`; returns
// per-LOD byte offsets and fills each texel with a deterministic pattern so
// format-decode/filter differences would surface.
void build_texture(uint8_t* base, uint32_t w0, uint32_t h0, uint32_t fmt,
                   uint32_t mip_off[VX_TEX_LOD_MAX + 1], uint32_t* num_lods) {
  uint32_t stride = format_stride(fmt);
  uint32_t off = 0;
  uint32_t l = 0;
  for (; l <= (uint32_t)VX_TEX_LOD_MAX; ++l) {
    uint32_t w = (w0 >> l) ? (w0 >> l) : 1u;
    uint32_t h = (h0 >> l) ? (h0 >> l) : 1u;
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
    if (w == 1 && h == 1) break;
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
                  uint32_t wrapu, uint32_t wrapv, uint32_t w0, uint32_t h0,
                  uint32_t reduce = VX_TEX_REDUCTION_AVG) {
  uint8_t* base = pool;
  uint32_t mip_off[VX_TEX_LOD_MAX + 1] = {0};
  uint32_t num_lods = 0;
  build_texture(base, w0, h0, fmt, mip_off, &num_lods);

  uint32_t wrap   = (wrapv << 16) | wrapu;
  uint32_t filter = mag_filter | (mip_linear ? VX_TEX_FILTER_MIP_LINEAR : 0u)
                  | (reduce << VX_TEX_FILTER_REDUCTION_SHIFT);

  // FF model state via TexDCRS.
  TexDCRS dcrs;
  dcrs.write(VX_DCR_TEX_STAGE,  0);
  dcrs.write(VX_DCR_TEX_ADDR,   (uint32_t)((uintptr_t)base >> 6));
  dcrs.write(VX_DCR_TEX_DIM,    (h0 << 16) | w0);
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
  st.width  = w0;
  st.height = h0;
  st.format = fmt;
  st.filter = filter;
  st.wrap   = wrap;
  for (uint32_t l = 0; l <= (uint32_t)VX_TEX_LOD_MAX; ++l)
    st.mip_off[l] = mip_off[l];

  const int32_t ONE = 1 << TEX_FXD_FRAC;
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
              printf("  MISMATCH %ux%u fmt=%s magf=%u mipL=%u wrap=(%u,%u) u=%d v=%d lod=0x%x: ff=%08x sw=%08x\n",
                     w0, h0, fmt_name(fmt), mag_filter, mip_linear, wrapu, wrapv, u, v, lod, a, b);
          }
        }
      }
    }
  }
  if (mismatches) {
    g_fail += mismatches;
    printf("FAIL %ux%u %-8s magf=%u mipL=%u wrap=(%u,%u) reduce=%u: %u/%u mismatches\n",
           w0, h0, fmt_name(fmt), mag_filter, mip_linear, wrapu, wrapv, reduce, mismatches,
           checked);
  }
}

// One axis's taps against the Vulkan texel selection, evaluated in double: the
// unnormalized coordinate (minus half a texel under linear), floor for the taps
// and their weight, then the integer wrap -- repeat as mod, mirrored repeat as
// (n-1) - mirror((i mod 2n) - n), mirror-clamp as clamp(mirror(i)), clamp to
// [0,n), border when outside. Samples within a hair of a tap or weight boundary
// are skipped: the fixed-point weight truncates, and the mirror fold is one
// coordinate ULP short of exact. Clamp folds the coordinate into [-1,2), so its
// weight is checked only where the two taps differ.
void check_axis_reference() {
  const uint32_t dims[] = { 1, 2, 5, 8, 37, 64, 1000 };
  const uint32_t wraps[] = { VX_TEX_WRAP_CLAMP, VX_TEX_WRAP_REPEAT, VX_TEX_WRAP_MIRROR,
                             VX_TEX_WRAP_BORDER, VX_TEX_WRAP_MIRROR_CLAMP };
  const double one = (double)(1 << TEX_FXD_FRAC);
  uint32_t fails = 0, checked = 0;
  uint32_t seed = 12345;
  auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return seed; };
  for (uint32_t n : dims) {
    for (uint32_t wrap : wraps) {
      for (int linear = 0; linear < 2; ++linear) {
        for (int k = 0; k < 4000; ++k) {
          const int32_t c = (int32_t)(rnd() % (uint32_t)(5 << TEX_FXD_FRAC)) - (2 << TEX_FXD_FRAC);
          const double t = (double)c / one * n - (linear ? 0.5 : 0.0);
          const double fl = std::floor(t), fr = t - fl;
          if (fr < 1e-3 || fr > 1.0 - 1e-3) {
            continue;
          }
          auto ref = [&](int64_t i, bool* b) -> uint32_t {
            const int64_t nn = n;
            *b = false;
            switch (wrap) {
            case VX_TEX_WRAP_REPEAT:
              return (uint32_t)(((i % nn) + nn) % nn);
            case VX_TEX_WRAP_MIRROR: {
              const int64_t m = (((i % (2 * nn)) + 2 * nn) % (2 * nn)) - nn;
              return (uint32_t)((nn - 1) - ((m >= 0) ? m : -(1 + m)));
            }
            case VX_TEX_WRAP_MIRROR_CLAMP: {
              const int64_t m = (i >= 0) ? i : -(1 + i);
              return (uint32_t)((m >= nn) ? nn - 1 : m);
            }
            default:
              *b = (wrap == VX_TEX_WRAP_BORDER) && (i < 0 || i >= nn);
              return (uint32_t)((i < 0) ? 0 : ((i >= nn) ? nn - 1 : i));
            }
          };
          bool rb0, rb1;
          const uint32_t r0 = ref((int64_t)fl, &rb0);
          const uint32_t r1 = ref((int64_t)fl + 1, &rb1);
          const gfx_tex::TexAxisTaps a = gfx_tex::tex_axis_taps(c, n, wrap, linear != 0);
          bool ok = (a.i0 == r0) && (a.b0 == rb0);
          if (linear) {
            // Compared as a blend, not as an ordered pair: in a mirrored period
            // the fold puts the taps in the other order with the complementary
            // weight. Two taps naming the same texel blend to it whatever the weight.
            const double af = a.frac / 256.0;
            // One weight step, plus the fold's coordinate ULP in texels.
            const double tol = 1.0 / 256.0 + n / one + 1e-9;
            const bool distinct = (r0 != r1) || (rb0 != rb1);
            const bool same = (a.i0 == r0) && (a.b0 == rb0) && (a.i1 == r1) && (a.b1 == rb1)
                           && (!distinct || std::fabs(af - fr) <= tol);
            const bool swapped = (a.i0 == r1) && (a.b0 == rb1) && (a.i1 == r0) && (a.b1 == rb0)
                              && std::fabs(af - (1.0 - fr)) <= tol;
            ok = same || swapped;
          }
          ++checked;
          if (!ok && ++fails <= 6) {
            printf("  AXIS MISMATCH n=%u wrap=%u linear=%d c=%d: got (%u%s,%u%s,%u) ref (%u%s,%u%s,%.3f)\n",
                   n, wrap, linear, c, a.i0, a.b0 ? "b" : "", a.i1, a.b1 ? "b" : "", a.frac,
                   r0, rb0 ? "b" : "", r1, rb1 ? "b" : "", fr * 256.0);
          }
        }
      }
    }
  }
  if (fails) {
    g_fail += fails;
    printf("AXIS TAPS: FAILED (%u/%u mismatches against the Vulkan selection)\n", fails, checked);
  } else {
    printf("AXIS TAPS: PASSED (%u samples match the Vulkan texel selection)\n", checked);
  }
}

// Min/max reduction against its definition: for a bilinear sample of an
// A8R8G8B8 texture, the per-channel min or max over the texels whose weight is
// not zero, the taps found in double precision. Samples whose weight would round
// to zero in the fixed-point blend are skipped.
void check_reduce_reference(uint8_t* pool) {
  const uint32_t w = 13, h = 7;
  uint32_t* tex = (uint32_t*)pool;
  uint32_t seed = 777;
  auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return seed; };
  for (uint32_t i = 0; i < w * h; ++i) {
    tex[i] = rnd();
  }
  gfx_sw::TexState st{};
  st.base   = (uintptr_t)pool;
  st.width  = w;
  st.height = h;
  st.format = VX_TEX_FORMAT_A8R8G8B8;
  st.wrap   = (VX_TEX_WRAP_CLAMP << 16) | VX_TEX_WRAP_CLAMP;
  const double one = (double)(1 << TEX_FXD_FRAC);
  uint32_t fails = 0, checked = 0;
  for (uint32_t mode : { (uint32_t)VX_TEX_REDUCTION_MIN, (uint32_t)VX_TEX_REDUCTION_MAX }) {
    st.filter = VX_TEX_FILTER_BILINEAR | (mode << VX_TEX_FILTER_REDUCTION_SHIFT);
    for (int k = 0; k < 3000; ++k) {
      const int32_t u = (int32_t)(rnd() % (uint32_t)(1 << TEX_FXD_FRAC));
      const int32_t v = (int32_t)(rnd() % (uint32_t)(1 << TEX_FXD_FRAC));
      const double x = u / one * w - 0.5, y = v / one * h - 0.5;
      const double fx = x - std::floor(x), fy = y - std::floor(y);
      if (fx < 1.0 / 256 || fy < 1.0 / 256) {
        continue;
      }
      auto cl = [](int64_t i, uint32_t n) { return (uint32_t)(i < 0 ? 0 : (i >= n ? n - 1 : i)); };
      const uint32_t x0 = cl((int64_t)std::floor(x), w), x1 = cl((int64_t)std::floor(x) + 1, w);
      const uint32_t y0 = cl((int64_t)std::floor(y), h), y1 = cl((int64_t)std::floor(y) + 1, h);
      const uint32_t t[4] = { tex[x0 + y0 * w], tex[x1 + y0 * w], tex[x0 + y1 * w], tex[x1 + y1 * w] };
      uint32_t want = 0;
      for (uint32_t c = 0; c < 32; c += 8) {
        uint32_t r = (t[0] >> c) & 0xff;
        for (uint32_t i = 1; i < 4; ++i) {
          const uint32_t b = (t[i] >> c) & 0xff;
          r = (mode == VX_TEX_REDUCTION_MIN) ? (b < r ? b : r) : (b > r ? b : r);
        }
        want |= r << c;
      }
      const uint32_t got = gfx_sw::tex_sample_sw(st, u, v, 0);
      ++checked;
      if (got != want && ++fails <= 4) {
        printf("  REDUCE MISMATCH mode=%u u=%d v=%d: got=%08x want=%08x\n", mode, u, v, got, want);
      }
    }
  }
  if (fails) {
    g_fail += fails;
    printf("MINMAX REDUCTION: FAILED (%u/%u mismatches)\n", fails, checked);
  } else {
    printf("MINMAX REDUCTION: PASSED (%u samples match the per-channel min/max)\n", checked);
  }
}

// Seamless cube filtering: six 4x4 faces, each one solid colour. A bilinear
// sample on the edge between two faces blends them half and half, and one at a
// cube corner gives the average of the three faces meeting there; a sample in a
// face's middle is that face's colour.
void check_cube_seamless(uint8_t* pool) {
  const uint32_t n = 4;
  const uint32_t face_red[6] = { 20, 60, 100, 140, 180, 220 };
  uint32_t* tex = (uint32_t*)pool;
  for (uint32_t f = 0; f < 6; ++f) {
    for (uint32_t i = 0; i < n * n; ++i) {
      tex[f * n * n + i] = 0xff000000u | (face_red[f] << 16);
    }
  }
  gfx_sw::TexState st{};
  st.base = (uintptr_t)pool;
  st.width = n;
  st.height = n;
  st.format = VX_TEX_FORMAT_A8R8G8B8;
  st.filter = VX_TEX_FILTER_BILINEAR | GFX_SW_TEX_FILTER_SEAMLESS;
  st.layer_stride = n * n * 4;
  st.wrap = (VX_TEX_WRAP_CLAMP << 16) | VX_TEX_WRAP_CLAMP;
  struct { float d[3]; double want; const char* what; } probes[] = {
    { { 1.0f, 0.1f, 0.2f }, 20.0, "+X middle" },
    { { 1.0f, 0.0f, 1.0f }, (20.0 + 180.0) / 2, "+X/+Z edge" },
    { { -1.0f, 1.0f, 0.0f }, (60.0 + 100.0) / 2, "-X/+Y edge" },
    { { 1.0f, 1.0f, 1.0f }, (20.0 + 100.0 + 180.0) / 3, "+X+Y+Z corner" },
    { { -1.0f, -1.0f, -1.0f }, (60.0 + 140.0 + 220.0) / 3, "-X-Y-Z corner" },
  };
  uint32_t fails = 0;
  for (const auto& p : probes) {
    const uint32_t got = (gfx_sw::tex_sample_sw_cube(st, p.d[0], p.d[1], p.d[2], 0,
                                                     VX_TEX_FILTER_BILINEAR) >> 16) & 0xff;
    if (std::fabs((double)got - p.want) > 1.5) {
      ++fails;
      printf("  CUBE MISMATCH %s: got %u want %.1f\n", p.what, got, p.want);
    }
  }
  if (fails) {
    g_fail += fails;
    printf("CUBE SEAMLESS: FAILED (%u probes)\n", fails);
  } else {
    printf("CUBE SEAMLESS: PASSED (edges blend across faces, corners average three)\n");
  }
}

// NPOT addressing check against a self-describing texture. Each A8R8G8B8 texel is
// filled with its own linear index (x + y*width); a POINT sample at the texel
// centre must return that index, proving x = floor(u_norm*width),
// addr = x + y*width for a genuinely NPOT surface.
void check_npot(uint8_t* pool) {
  const struct { uint32_t w, h; } dims[] = {
    {5, 3}, {7, 1}, {1, 6}, {13, 11}, {17, 9}, {100, 60},
  };
  const int32_t ONE = 1 << TEX_FXD_FRAC;
  uint32_t fails = 0, checked = 0;

  for (auto d : dims) {
    uint32_t w = d.w, h = d.h;
    uint32_t* tex = (uint32_t*)pool;
    for (uint32_t y = 0; y < h; ++y)
      for (uint32_t x = 0; x < w; ++x)
        tex[x + y * w] = 0xff000000u | (x + y * w);   // A=0xff, index in RGB

    gfx_sw::TexState st{};
    st.base   = (uintptr_t)pool;
    // logdim ceil(log2(dim)); unused by the NPOT multiply path but kept sane.
    uint32_t lw = 0; while ((1u << lw) < w) ++lw;
    uint32_t lh = 0; while ((1u << lh) < h) ++lh;
    st.logdim = (lh << 16) | lw;
    st.format = VX_TEX_FORMAT_A8R8G8B8;
    st.filter = VX_TEX_FILTER_POINT;
    st.wrap   = (VX_TEX_WRAP_REPEAT << 16) | VX_TEX_WRAP_REPEAT;
    st.width  = w;
    st.height = h;
    st.mip_off[0] = 0;

    for (uint32_t y = 0; y < h; ++y) {
      for (uint32_t x = 0; x < w; ++x) {
        // texel-centre normalized coord: (x + 0.5)/w
        int32_t u = (int32_t)(((int64_t)(2 * x + 1) * ONE) / (int64_t)(2 * w));
        int32_t v = (int32_t)(((int64_t)(2 * y + 1) * ONE) / (int64_t)(2 * h));
        uint32_t got = gfx_sw::tex_sample_sw(st, u, v, 0);
        uint32_t exp = 0xff000000u | (x + y * w);
        ++checked;
        if (got != exp) {
          if (++fails <= 4)
            printf("  NPOT MISMATCH %ux%u (x=%u y=%u): got=%08x exp=%08x\n",
                   w, h, x, y, got, exp);
        }
      }
    }
  }
  if (fails) {
    g_fail += fails;
    printf("NPOT ADDR: FAILED (%u/%u mismatches)\n", fails, checked);
  } else {
    printf("NPOT ADDR: PASSED (%u samples, floor(u*w)+y*w exact)\n", checked);
  }
}

// Perspective-correct varyings premultiply UV by 1/w and store the a*(1/w) planes
// in Q7.24 FloatA (range ~+-128). Tiling/wrap UV well beyond 1.0 can exceed that
// fixed-point range, so the setup folds an extra common power-of-2 downscale into
// the stored 1/w when a premultiplied texcoord would leave range — it cancels in
// the FS divide, so the recovered UV stays exact. This drives the real setup +
// FS-divide math for a perspective triangle with heavily tiled UV and asserts the
// recovered UV samples the correct (tiled) REPEAT texel.
void check_persp_wrap(uint8_t* pool) {
  using namespace gfx_setup;
  using vortex::graphics::FloatA;
  using vortex::graphics::rast_attrib_t;
  using vortex::graphics::rast_prim_t;

  const int W = 256, H = 256;
  const uint32_t tw = 100, th = 60;               // NPOT surface, REPEAT wrap
  const float UVBIG = 256.0f;                     // tiles ~256x — overflows pre-fix

  // Self-describing NPOT texture: each texel carries its own linear index.
  uint32_t* tex = (uint32_t*)pool;
  for (uint32_t y = 0; y < th; ++y)
    for (uint32_t x = 0; x < tw; ++x)
      tex[x + y * tw] = 0xff000000u | (x + y * tw);

  gfx_sw::TexState st{};
  st.base = (uintptr_t)pool;
  uint32_t lw = 0; while ((1u << lw) < tw) ++lw;
  uint32_t lh = 0; while ((1u << lh) < th) ++lh;
  st.logdim = (lh << 16) | lw;
  st.format = VX_TEX_FORMAT_A8R8G8B8;
  st.filter = VX_TEX_FILTER_POINT;
  st.wrap   = (VX_TEX_WRAP_REPEAT << 16) | VX_TEX_WRAP_REPEAT;
  st.width  = tw;
  st.height = th;
  st.mip_off[0] = 0;

  // Perspective triangle: distinct w per vertex; large UV on the near vertex.
  const float ws[3] = { 1.0f, 2.0f, 4.0f };
  const float sx[3] = { 40.0f, 200.0f, 120.0f }, sy[3] = { 40.0f, 60.0f, 210.0f };
  const float uv[3][2] = { { UVBIG, UVBIG }, { 0.0f, UVBIG }, { UVBIG, 0.0f } };
  setup_vertex_t vtx[3];
  for (int i = 0; i < 3; ++i) {
    float ndcx = (sx[i] / W) * 2.0f - 1.0f, ndcy = (sy[i] / H) * 2.0f - 1.0f;
    vtx[i].pos[0] = ndcx * ws[i]; vtx[i].pos[1] = ndcy * ws[i];
    vtx[i].pos[2] = 0.5f * ws[i]; vtx[i].pos[3] = ws[i];
    vtx[i].color[0] = vtx[i].color[1] = vtx[i].color[2] = vtx[i].color[3] = 1.0f;
    vtx[i].texcoord[0] = uv[i][0]; vtx[i].texcoord[1] = uv[i][1];
  }

  rast_prim_t prim; setup_bbox_t bbox;
  if (!setup_triangle(vtx[0], vtx[1], vtx[2], W, H, 0.0f, 1.0f, prim, bbox)) {
    printf("PERSP-WRAP: setup culled the triangle (unexpected)\n");
    g_fail += 1; return;
  }

  const double rhw[3] = { 1.0 / ws[0], 1.0 / ws[1], 1.0 / ws[2] };
  const int FXD = TEX_FXD_FRAC;
  uint32_t texel_mismatch = 0, checked = 0;
  double max_texel_err = 0.0;

  for (uint32_t Y = bbox.bbT; Y < bbox.bbB; ++Y) {
    for (uint32_t X = bbox.bbL; X < bbox.bbR; ++X) {
      // FS edge eval (Q15.16 -> float), identical to the kernel BCOORD recompute.
      auto ef = [&](int a) {
        return (float)(prim.edges[a].x.data() * (int)X + prim.edges[a].y.data() * (int)Y
                       + prim.edges[a].z.data()) / (float)(1 << 16);
      };
      float F0 = ef(0), F1 = ef(1), F2 = ef(2);
      if (F0 < 0 || F1 < 0 || F2 < 0) continue;   // outside coverage
      float recip = 1.0f / (F0 + F1 + F2);
      FloatA dx = FloatA(recip * F0), dy = FloatA(recip * F1);
      auto interp = [&](const rast_attrib_t& s) -> float {
        FloatA tmp = s.x * dx + s.z; FloatA d = s.y * dy + tmp; return (float)d;
      };
      float iw = interp(prim.attribs.rhw);
      float inv_w = (FloatA(iw).data() != 0) ? (1.0f / iw) : 0.0f;
      float fu = interp(prim.attribs.u) * inv_w;
      float fv = interp(prim.attribs.v) * inv_w;

      // Ground-truth perspective-correct UV (double; screen bary from same edges).
      double L0 = F0, L1 = F1, L2 = F2, s = L0 + L1 + L2; L0 /= s; L1 /= s; L2 /= s;
      double den = L0 * rhw[0] + L1 * rhw[1] + L2 * rhw[2];
      double gu = (L0 * uv[0][0] * rhw[0] + L1 * uv[1][0] * rhw[1] + L2 * uv[2][0] * rhw[2]) / den;
      double gv = (L0 * uv[0][1] * rhw[0] + L1 * uv[1][1] * rhw[1] + L2 * uv[2][1] * rhw[2]) / den;
      max_texel_err = fmax(max_texel_err, fmax(fabs(fu - gu) * tw, fabs(fv - gv) * th));

      // The recovered UV must sample the same REPEAT-tiled texel as the true UV.
      // Skip pixels whose true UV lands within 0.15 texel of an integer boundary:
      // POINT sampling is 1-off-flaky there for any finite-precision interpolator
      // (not a correctness signal); the max_texel_err gate below covers those.
      double tu = (gu - floor(gu)) * tw, tv = (gv - floor(gv)) * th;
      double du = fmin(tu - floor(tu), ceil(tu) - tu);
      double dv = fmin(tv - floor(tv), ceil(tv) - tv);
      if (du < 0.15 || dv < 0.15) continue;
      using fixeduv_t = vortex::graphics::fixed_t<TEX_FXD_FRAC>;
      auto q = [&](float f) { return (int32_t)llround((double)f * (double)(1 << FXD)); };
      uint32_t got = gfx_sw::tex_sample_sw(st, fixeduv_t::make(q(fu)).data(),
                                               fixeduv_t::make(q(fv)).data(), 0);
      uint32_t exp = gfx_sw::tex_sample_sw(st, fixeduv_t::make(q((float)gu)).data(),
                                               fixeduv_t::make(q((float)gv)).data(), 0);
      ++checked;
      if (got != exp) {
        if (++texel_mismatch <= 4)
          printf("  PERSP-WRAP MISMATCH (X=%u Y=%u): fu=%.3f fv=%.3f gu=%.3f gv=%.3f got=%08x exp=%08x\n",
                 X, Y, fu, fv, gu, gv, got, exp);
      }
    }
  }
  if (texel_mismatch || max_texel_err > 0.5) {
    g_fail += (texel_mismatch ? texel_mismatch : 1);
    printf("PERSP-WRAP: FAILED (UVmax=%.0f: %u/%u texel mismatches, max_texel_err=%.3f)\n",
           UVBIG, texel_mismatch, checked, max_texel_err);
  } else {
    printf("PERSP-WRAP: PASSED (%u px, UVmax=%.0f tiled REPEAT exact, max_texel_err=%.4f texel)\n",
           checked, UVBIG, max_texel_err);
  }
}

// ── Extended-format / border / cube / array / sampler-view-array checks ────
// SW-path only (the FF unit handles formats 0..FF_MAX with no border), so these
// self-check against the known decode rather than an FF golden.

static void tex_point_state(gfx_sw::TexState& st, uint8_t* base, uint32_t logw,
                            uint32_t logh, uint32_t fmt) {
  st = gfx_sw::TexState{};
  st.base   = (uintptr_t)base;
  st.logdim = (logh << 16) | logw;
  st.format = fmt;
  st.filter = VX_TEX_FILTER_POINT;
  st.wrap   = (VX_TEX_WRAP_CLAMP << 16) | VX_TEX_WRAP_CLAMP;
  st.mip_off[0] = 0;
}

// (a) sRGB texture: the sampler must gamma-decode RGB (alpha stays linear).
void check_srgb(uint8_t* pool) {
  uint32_t raw = 0xC0A08040u;              // A=C0, R=A0, G=80, B=40 (ARGB packing)
  *(uint32_t*)pool = raw;                  // 1x1 SRGB8A8 texel
  gfx_sw::TexState st; tex_point_state(st, pool, 0, 0, VX_TEX_FORMAT_SRGB8A8);
  const int32_t ONE = 1 << TEX_FXD_FRAC;
  uint32_t got = gfx_sw::tex_sample_sw(st, ONE / 2, ONE / 2, 0);
  uint32_t exp = (0xC0u << 24)
               | (gfx_tex::SrgbToLinear8(0xA0) << 16)
               | (gfx_tex::SrgbToLinear8(0x80) << 8)
               | (gfx_tex::SrgbToLinear8(0x40));
  bool linearized = (got != raw);          // decode must NOT be the raw bytes
  if (got != exp || !linearized) {
    printf("  SRGB MISMATCH: got=%08x exp=%08x (raw=%08x)\n", got, exp, raw);
    ++g_fail;
  } else {
    printf("SRGB DECODE: PASSED (got=%08x, gamma-decoded, alpha linear)\n", got);
  }
}

// (b1) 2D-array view: integer layer index selects the slice at layer*layer_stride.
void check_array(uint8_t* pool) {
  const uint32_t layers = 4, texw = 1u << 2, texh = 1u << 2; // 4x4 POT per layer
  const uint32_t stride = texw * texh * 4;                    // A8R8G8B8
  for (uint32_t L = 0; L < layers; ++L)
    for (uint32_t i = 0; i < texw * texh; ++i)
      ((uint32_t*)(pool + L * stride))[i] = 0xff000000u | (0x100u * L) | i;

  gfx_sw::TexState st; tex_point_state(st, pool, 2, 2, VX_TEX_FORMAT_A8R8G8B8);
  st.layer_stride = stride;
  const int32_t ONE = 1 << TEX_FXD_FRAC;
  uint32_t fails = 0;
  for (uint32_t L = 0; L < layers; ++L) {
    // sample texel (1,1): u=(1.5)/4, v=(1.5)/4
    int32_t u = (int32_t)((3ll * ONE) / 8), v = (int32_t)((3ll * ONE) / 8);
    uint32_t got = gfx_sw::tex_sample_sw_array(st, u, v, L, 0, st.filter);
    uint32_t exp = 0xff000000u | (0x100u * L) | (1 + 1 * texw);
    if (got != exp) { printf("  ARRAY MISMATCH L=%u: got=%08x exp=%08x\n", L, got, exp); ++fails; }
  }
  if (fails) g_fail += fails;
  else printf("ARRAY VIEW: PASSED (%u layers, per-layer slice select exact)\n", layers);
}

// (b2) cube view: the major axis of the direction vector selects the face.
void check_cube(uint8_t* pool) {
  const uint32_t texw = 1u << 2, texh = 1u << 2, stride = texw * texh * 4;
  for (uint32_t f = 0; f < 6; ++f)
    for (uint32_t i = 0; i < texw * texh; ++i)
      ((uint32_t*)(pool + f * stride))[i] = 0xff000000u | (0xA0u + f);   // face tag in low byte

  gfx_sw::TexState st; tex_point_state(st, pool, 2, 2, VX_TEX_FORMAT_A8R8G8B8);
  st.layer_stride = stride;
  // direction vectors whose major axis is +X,-X,+Y,-Y,+Z,-Z → faces 0..5
  const float dirs[6][3] = {
    { 1, 0.1f, 0.1f}, {-1, 0.1f, 0.1f}, {0.1f, 1, 0.1f},
    {0.1f, -1, 0.1f}, {0.1f, 0.1f, 1}, {0.1f, 0.1f, -1}};
  uint32_t fails = 0;
  for (uint32_t f = 0; f < 6; ++f) {
    uint32_t got = gfx_sw::tex_sample_sw_cube(st, dirs[f][0], dirs[f][1], dirs[f][2], 0, st.filter);
    if ((got & 0xff) != (0xA0u + f)) { printf("  CUBE MISMATCH face=%u: got=%08x\n", f, got); ++fails; }
  }
  if (fails) g_fail += fails;
  else printf("CUBE VIEW: PASSED (6 faces, major-axis face select exact)\n");
}

// (c) sampler-view array: two distinct bound textures sampled independently.
void check_two_textures(uint8_t* pool) {
  uint32_t* t0 = (uint32_t*)pool;              // texture 0 at pool
  uint32_t* t1 = (uint32_t*)(pool + 0x1000);   // texture 1 offset
  *t0 = 0xDEADBEEFu; *t1 = 0xFEEDFACEu;
  gfx_sw::TexState st0, st1;
  tex_point_state(st0, (uint8_t*)t0, 0, 0, VX_TEX_FORMAT_A8R8G8B8);
  tex_point_state(st1, (uint8_t*)t1, 0, 0, VX_TEX_FORMAT_A8R8G8B8);
  const int32_t ONE = 1 << TEX_FXD_FRAC;
  uint32_t g0 = gfx_sw::tex_sample_sw(st0, ONE / 2, ONE / 2, 0);
  uint32_t g1 = gfx_sw::tex_sample_sw(st1, ONE / 2, ONE / 2, 0);
  if (g0 != 0xDEADBEEFu || g1 != 0xFEEDFACEu) {
    printf("  SAMPLER-ARRAY MISMATCH: t0=%08x t1=%08x\n", g0, g1); g_fail += 1;
  } else {
    printf("SAMPLER-VIEW ARRAY: PASSED (two distinct bound textures)\n");
  }
}

// (d) CLAMP_TO_BORDER: out-of-[0,1) taps return the border colour.
void check_border(uint8_t* pool) {
  const uint32_t texw = 1u << 2, texh = 1u << 2;
  for (uint32_t i = 0; i < texw * texh; ++i) ((uint32_t*)pool)[i] = 0xff112233u;
  gfx_sw::TexState st; tex_point_state(st, pool, 2, 2, VX_TEX_FORMAT_A8R8G8B8);
  st.wrap   = (VX_TEX_WRAP_BORDER << 16) | VX_TEX_WRAP_BORDER;
  st.border = 0x00000000u;                        // transparent black
  const int32_t ONE = 1 << TEX_FXD_FRAC;
  uint32_t inside  = gfx_sw::tex_sample_sw(st, ONE / 2, ONE / 2, 0);      // (u,v)=0.5 in range
  uint32_t outside = gfx_sw::tex_sample_sw(st, -ONE / 4, ONE / 2, 0);     // u<0 → border
  uint32_t out2    = gfx_sw::tex_sample_sw(st, ONE / 2, (5 * ONE) / 4, 0);// v>1 → border
  if (inside != 0xff112233u || outside != 0x00000000u || out2 != 0x00000000u) {
    printf("  BORDER MISMATCH: inside=%08x out_u=%08x out_v=%08x\n", inside, outside, out2);
    g_fail += 1;
  } else {
    printf("CLAMP_TO_BORDER: PASSED (in-range texel + out-of-range border)\n");
  }
}

// (f) Block-compressed decode and addressing. Each format's texels over 64
// generated blocks hash to the CTS reference decoder's (tcuCompressedTexture):
// the BC1-BC3 and BC7 8-bit words, BC4/BC5 unsigned floats and BC6H's half
// floats, bit for bit. BC7 cycles through its eight modes and BC6H its fourteen.
// The signed BC4 endpoints are snorm8 (127 and -128 both reach +-1), which the
// reference maps by /128 instead, so that ramp is checked here by hand. Last, a
// 6x5 BC1 texture -- two block columns, one partial -- point-sampled at every
// texel centre must read each texel from its own block.
static uint32_t bc_lcg(uint32_t& s) {
  s = s * 1664525u + 1013904223u;
  return s >> 24;
}

static void bc_fnv(uint64_t& h, const void* p, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    h ^= ((const uint8_t*)p)[i];
    h *= 1099511628211ull;
  }
}

static void bc_gen_block(uint32_t fmt, uint32_t k, uint8_t blk[16]) {
  uint32_t s = 0x9e3779b9u * (k + 1) + fmt;
  for (int i = 0; i < 16; ++i) {
    blk[i] = (uint8_t)bc_lcg(s);
  }
  if (fmt == VX_TEX_FORMAT_BC7) {
    blk[0] = (uint8_t)((blk[0] & ~((2u << (k % 8)) - 1)) | (1u << (k % 8)));
  }
  if (fmt == VX_TEX_FORMAT_BC6UF || fmt == VX_TEX_FORMAT_BC6SF) {
    static const uint8_t modes[14] = { 0x00, 0x01, 0x02, 0x06, 0x0a, 0x0e, 0x12, 0x16, 0x1a,
                                       0x1e, 0x03, 0x07, 0x0b, 0x0f };
    const uint8_t m = modes[k % 14];
    blk[0] = (uint8_t)((blk[0] & ((m & 3) < 2 ? 0xfc : 0xe0)) | m);
  }
}

void check_bc_decode(uint8_t* pool) {
  static const struct { uint32_t fmt; uint64_t hash; } kRef[] = {
    { VX_TEX_FORMAT_BC1, 0xe8d22255cf2bc6cdull },
    { VX_TEX_FORMAT_BC1A, 0x6f31b4a30511ff1aull },
    { VX_TEX_FORMAT_BC2, 0x846a6c1954f626bcull },
    { VX_TEX_FORMAT_BC3, 0xa556c7cde350d1e1ull },
    { VX_TEX_FORMAT_BC4, 0x189d2a6cc5ef6e95ull },
    { VX_TEX_FORMAT_BC5, 0x27b89537b9bd22b6ull },
    { VX_TEX_FORMAT_BC6UF, 0xaf34b08c617881fbull },
    { VX_TEX_FORMAT_BC6SF, 0xada14e5562599129ull },
    { VX_TEX_FORMAT_BC7, 0xee780f0be0fa255aull },
  };
  uint32_t fails = 0;
  for (const auto& r : kRef) {
    uint64_t h = 1469598103934665603ull;
    for (uint32_t k = 0; k < 64; ++k) {
      uint8_t blk[16];
      bc_gen_block(r.fmt, k, blk);
      for (uint32_t t = 0; t < 16; ++t) {
        switch (r.fmt) {
        case VX_TEX_FORMAT_BC4: {
          const float f = gfx_tex::BcRedBlock(blk, t, false);
          bc_fnv(h, &f, 4);
          break;
        }
        case VX_TEX_FORMAT_BC5: {
          const float f[2] = { gfx_tex::BcRedBlock(blk, t, false),
                               gfx_tex::BcRedBlock(blk + 8, t, false) };
          bc_fnv(h, f, 8);
          break;
        }
        case VX_TEX_FORMAT_BC6UF:
        case VX_TEX_FORMAT_BC6SF: {
          uint16_t hf[3];
          gfx_tex::BcDecodeBc6(blk, t, r.fmt == VX_TEX_FORMAT_BC6SF, hf);
          bc_fnv(h, hf, 6);
          break;
        }
        default: {
          const uint32_t w = r.fmt == VX_TEX_FORMAT_BC1  ? gfx_tex::BcDecodeBc1(blk, t, false)
                           : r.fmt == VX_TEX_FORMAT_BC1A ? gfx_tex::BcDecodeBc1(blk, t, true)
                           : r.fmt == VX_TEX_FORMAT_BC2  ? gfx_tex::BcDecodeBc2(blk, t)
                           : r.fmt == VX_TEX_FORMAT_BC3  ? gfx_tex::BcDecodeBc3(blk, t)
                                                         : gfx_tex::BcDecodeBc7(blk, t);
          bc_fnv(h, &w, 4);
          break;
        }
        }
      }
    }
    if (h != r.hash) {
      printf("  BC DECODE MISMATCH: format %u hash %016llx, reference %016llx\n", r.fmt,
             (unsigned long long)h, (unsigned long long)r.hash);
      ++fails;
    }
  }

  // Signed BC4: endpoints 127 and -128 are +1 and -1; texel 2 takes index 2 of
  // the eight-step ramp, texels 3 and 4 the six-step ramp's fixed -1 and +1.
  const uint8_t up[8]   = { 0x7f, 0x80, 0x88, 0x00, 0x00, 0x00, 0x00, 0x00 };   // idx 0,1,2
  const uint8_t down[8] = { 0x80, 0x7f, 0x00, 0x7c, 0x00, 0x00, 0x00, 0x00 };   // t3 = 6, t4 = 7
  const float f2 = (1.0f * 6.0f + -1.0f * 1.0f) / 7.0f;
  if (gfx_tex::BcRedBlock(up, 0, true) != 1.0f || gfx_tex::BcRedBlock(up, 1, true) != -1.0f
      || gfx_tex::BcRedBlock(up, 2, true) != f2 || gfx_tex::BcRedBlock(down, 3, true) != -1.0f
      || gfx_tex::BcRedBlock(down, 4, true) != 1.0f) {
    printf("  BC4 SNORM MISMATCH: %g %g %g %g %g\n", gfx_tex::BcRedBlock(up, 0, true),
           gfx_tex::BcRedBlock(up, 1, true), gfx_tex::BcRedBlock(up, 2, true),
           gfx_tex::BcRedBlock(down, 3, true), gfx_tex::BcRedBlock(down, 4, true));
    ++fails;
  }

  // Addressing: 6x5 BC1 = 2x2 blocks of 8 bytes, block (bx,by) at (by*2+bx)*8.
  const uint32_t W = 6, H = 5;
  for (uint32_t b = 0; b < 4; ++b) {
    bc_gen_block(VX_TEX_FORMAT_BC1, 100 + b, pool + b * 8);
  }
  gfx_sw::TexState st;
  tex_point_state(st, pool, 3, 3, VX_TEX_FORMAT_BC1);
  st.width = W;
  st.height = H;
  const int32_t ONE = 1 << TEX_FXD_FRAC;
  for (uint32_t y = 0; y < H; ++y) {
    for (uint32_t x = 0; x < W; ++x) {
      float got[4], exp[4];
      gfx_sw::tex_sample_f32_layer(st, (int32_t)(((2 * x + 1) * (uint32_t)ONE) / (2 * W)),
                                   (int32_t)(((2 * y + 1) * (uint32_t)ONE) / (2 * H)), 0, 0,
                                   VX_TEX_FILTER_POINT, got);
      gfx_tex::TexDecodeBlockF4(VX_TEX_FORMAT_BC1, pool + ((y >> 2) * 2 + (x >> 2)) * 8,
                                ((y & 3) << 2) | (x & 3), exp);
      if (__builtin_memcmp(got, exp, sizeof got)) {
        printf("  BC ADDRESSING MISMATCH at (%u,%u)\n", x, y);
        ++fails;
      }
    }
  }
  if (fails) {
    g_fail += fails;
  } else {
    printf("BC DECODE: PASSED (BC1-BC7 match the reference decoder; block addressing OK)\n");
  }
}

// (g) The field formats named after their Vulkan layouts: one texel of each,
// its fields written by hand, decodes to the channels the layout names.
void check_field_formats() {
  struct Case {
    uint32_t fmt;
    uint8_t  bytes[12];
    float    expect[4];   // an integer format's channels as their integer bits
  };
  const Case cases[] = {
    { VX_TEX_FORMAT_B5G6R5_PACK16,   { 0x1f, 0x00 }, { 1.0f, 0.0f, 0.0f, 1.0f } },
    { VX_TEX_FORMAT_R5G5B5A1_PACK16, { 0x01, 0xf8 }, { 1.0f, 0.0f, 0.0f, 1.0f } },
    { VX_TEX_FORMAT_B5G5R5A1_PACK16, { 0x01, 0xf8 }, { 0.0f, 0.0f, 1.0f, 1.0f } },
    { VX_TEX_FORMAT_A1B5G5R5_PACK16, { 0x1f, 0x80 }, { 1.0f, 0.0f, 0.0f, 1.0f } },
    { VX_TEX_FORMAT_A4R4G4B4_PACK16, { 0x00, 0xf8 }, { 8.0f / 15.0f, 0.0f, 0.0f, 1.0f } },
    { VX_TEX_FORMAT_A4B4G4R4_PACK16, { 0x01, 0x0f }, { 1.0f / 15.0f, 0.0f, 1.0f, 0.0f } },
    { VX_TEX_FORMAT_A8_UNORM,        { 0x80 },       { 0.0f, 0.0f, 0.0f, 128.0f / 255.0f } },
    { VX_TEX_FORMAT_BGRA8SN,         { 0x81, 0x7f, 0x00, 0x40 }, { 0.0f, 1.0f, -1.0f, 64.0f / 127.0f } },
    { VX_TEX_FORMAT_A2R10G10B10,     { 0xff, 0x03, 0x00, 0xc0 }, { 0.0f, 0.0f, 1.0f, 1.0f } },
    { VX_TEX_FORMAT_A2B10G10R10SN,   { 0xff, 0x01, 0x00, 0x40 }, { 1.0f, 0.0f, 0.0f, 1.0f } },
  };
  uint32_t fails = 0;
  for (const Case& c : cases) {
    float got[4];
    gfx_tex::TexDecodeFloat4(c.fmt, c.bytes, gfx_tex::FormatStride(c.fmt), got);
    for (uint32_t ch = 0; ch < 4; ++ch) {
      if (std::fabs(got[ch] - c.expect[ch]) > 1e-6f) {
        printf("  FIELD MISMATCH: format %u channel %u = %g, expected %g\n", c.fmt, ch, got[ch],
               c.expect[ch]);
        ++fails;
      }
    }
  }
  // The integer and 12-byte layouts, checked on their bits.
  const uint8_t ui[4] = { 0x00, 0x00, 0xf0, 0x3f };   // R = 1023 at bits 29:20
  uint32_t lanes[4];
  gfx_tex::TexDecodeFloat4(VX_TEX_FORMAT_A2R10G10B10UI, ui, 4, (float*)lanes);
  if (lanes[0] != 1023u || lanes[1] || lanes[2] || lanes[3]) {
    printf("  FIELD MISMATCH: A2R10G10B10UI = %u,%u,%u,%u\n", lanes[0], lanes[1], lanes[2], lanes[3]);
    ++fails;
  }
  const float rgb[3] = { 1.5f, -2.0f, 3.0e6f };
  float f[4];
  gfx_tex::TexDecodeFloat4(VX_TEX_FORMAT_RGB32F, rgb, 12, f);
  if (f[0] != rgb[0] || f[1] != rgb[1] || f[2] != rgb[2] || f[3] != 1.0f) {
    printf("  FIELD MISMATCH: RGB32F = %g,%g,%g,%g\n", f[0], f[1], f[2], f[3]);
    ++fails;
  }
  if (gfx_tex::FormatStride(VX_TEX_FORMAT_RGB32F) != 12) {
    printf("  FIELD MISMATCH: RGB32F stride %u\n", gfx_tex::FormatStride(VX_TEX_FORMAT_RGB32F));
    ++fails;
  }
  if (fails) {
    g_fail += fails;
  } else {
    printf("FIELD FORMATS: PASSED (Vulkan-named packed, BGRA8, 2-10-10-10, RGB32)\n");
  }
}

} // namespace

// TexDepthCompare against the exact rational compare, ref * (2^n - 1) against
// d in long double (48 significant bits, exact), for D16 and D24 texels: every
// operator, references on and around each code's value and around the clamp
// points, and NaN and -0.
void check_depth_compare() {
  const uint32_t funcs[8] = { VX_OM_DEPTH_FUNC_ALWAYS, VX_OM_DEPTH_FUNC_NEVER,
                              VX_OM_DEPTH_FUNC_LESS, VX_OM_DEPTH_FUNC_LEQUAL,
                              VX_OM_DEPTH_FUNC_EQUAL, VX_OM_DEPTH_FUNC_GEQUAL,
                              VX_OM_DEPTH_FUNC_GREATER, VX_OM_DEPTH_FUNC_NOTEQUAL };
  auto bits_of = [](float f) { uint32_t b; memcpy(&b, &f, 4); return b; };
  uint32_t fails = 0, checked = 0;
  for (uint32_t n : { 16u, 24u }) {
    const uint32_t k = (1u << n) - 1u;
    const uint32_t fmt = n == 16 ? (uint32_t)VX_TEX_FORMAT_D16 : (uint32_t)VX_TEX_FORMAT_D24;
    uint32_t seed = 12345u + n;
    for (uint32_t it = 0; it < 4000; ++it) {
      seed = seed * 1664525u + 1013904223u;
      const uint32_t d = (it < 4) ? (it < 2 ? it : k - (it - 2)) : seed % (k + 1u);
      const float v = (float)d / (float)k;
      float refs[7] = { v, nextafterf(v, 0.0f), nextafterf(v, 2.0f), (float)(d + 1) / (float)k,
                        -0.0f, 1.5f, nanf("") };
      for (float r : refs) {
        long double rr = std::isnan(r) || r <= 0.0f ? 0.0L : (r >= 1.0f ? 1.0L : (long double)r);
        const long double lhs = rr * (long double)k, rhs = (long double)d;
        const gfx_tex::TexDepthTexel t = gfx_tex::TexDepthFromWord(fmt, d);
        for (uint32_t f : funcs) {
          bool want;
          switch (f) {
          case VX_OM_DEPTH_FUNC_NEVER:    want = false;        break;
          case VX_OM_DEPTH_FUNC_LESS:     want = lhs <  rhs;   break;
          case VX_OM_DEPTH_FUNC_LEQUAL:   want = lhs <= rhs;   break;
          case VX_OM_DEPTH_FUNC_EQUAL:    want = lhs == rhs;   break;
          case VX_OM_DEPTH_FUNC_GEQUAL:   want = lhs >= rhs;   break;
          case VX_OM_DEPTH_FUNC_GREATER:  want = lhs >  rhs;   break;
          case VX_OM_DEPTH_FUNC_NOTEQUAL: want = lhs != rhs;   break;
          default:                        want = true;         break;
          }
          ++checked;
          if (gfx_tex::TexDepthCompare(f, bits_of(r), t) != want && ++fails <= 6) {
            printf("  D%u depth compare: func %u ref %.9g d %u -> %d, want %d\n", n, f,
                   (double)r, d, !want, want);
          }
        }
      }
    }
  }
  // D32F: a float compare, -0 equal to +0, NaN equal to nothing.
  const gfx_tex::TexDepthTexel z = gfx_tex::TexDepthFromWord(VX_TEX_FORMAT_D32F, bits_of(0.0f));
  ++checked;
  if (!gfx_tex::TexDepthCompare(VX_OM_DEPTH_FUNC_EQUAL, bits_of(-0.0f), z) ||
      gfx_tex::TexDepthCompare(VX_OM_DEPTH_FUNC_EQUAL, bits_of(nanf("")), z) ||
      !gfx_tex::TexDepthCompare(VX_OM_DEPTH_FUNC_NOTEQUAL, bits_of(nanf("")), z)) {
    ++fails;
    printf("  D32F depth compare: signed zero or NaN wrong\n");
  }
  if (fails) {
    g_fail += fails;
    printf("DEPTH COMPARE: FAILED (%u/%u mismatches against the exact compare)\n", fails, checked);
  } else {
    printf("  depth compare exact (%u cases, D16/D24/D32F)\n", checked);
  }
}

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
            check_config(pool, fmt, magf, mipL, wu, wv, 64, 32);
            check_config(pool, fmt, magf, mipL, wu, wv, 37, 23);
            check_config(pool, fmt, magf, mipL, wu, wv, 37, 23, VX_TEX_REDUCTION_MIN);
            check_config(pool, fmt, magf, mipL, wu, wv, 37, 23, VX_TEX_REDUCTION_MAX);
            configs += 4;
          }

  check_axis_reference();
  check_reduce_reference(pool);
  check_cube_seamless(pool);

  check_npot(pool);
  check_persp_wrap(pool);

  // SW-path breadth (extended formats, border, cube/array, sampler-view array).
  check_srgb(pool);
  check_array(pool);
  check_cube(pool);
  check_two_textures(pool);
  check_border(pool);
  check_bc_decode(pool);
  check_field_formats();
  check_depth_compare();

  if (g_fail) {
    printf("\nTEX-SW PARITY: FAILED (%u mismatches across %u configs)\n", g_fail, configs);
    return 1;
  }
  printf("TEX-SW PARITY: PASSED (%u configs, SW == FF bit-exact; NPOT addressing OK)\n", configs);
  return 0;
}
