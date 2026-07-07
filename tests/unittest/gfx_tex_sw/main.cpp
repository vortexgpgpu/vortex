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

// NPOT addressing check: the FF TEX unit is power-of-two only, so there is
// no FF golden for non-power-of-two textures — the multiply-addressing SW path
// is validated against a self-describing texture instead. Each A8R8G8B8 texel is
// filled with its own linear index (x + y*width); a POINT sample at the texel
// centre must return that index, proving x = floor(u_norm*width),
// addr = x + y*width for a genuinely NPOT surface.
void check_npot(uint8_t* pool) {
  const struct { uint32_t w, h; } dims[] = {
    {5, 3}, {7, 1}, {1, 6}, {13, 11}, {17, 9}, {100, 60},
  };
  const int32_t ONE = 1 << VX_TEX_FXD_FRAC;
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
  const int FXD = VX_TEX_FXD_FRAC;
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
      using fixeduv_t = vortex::graphics::fixed_t<VX_TEX_FXD_FRAC>;
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
  const int32_t ONE = 1 << VX_TEX_FXD_FRAC;
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
  const int32_t ONE = 1 << VX_TEX_FXD_FRAC;
  uint32_t fails = 0;
  for (uint32_t L = 0; L < layers; ++L) {
    // sample texel (1,1): u=(1.5)/4, v=(1.5)/4
    int32_t u = (int32_t)((3ll * ONE) / 8), v = (int32_t)((3ll * ONE) / 8);
    uint32_t got = gfx_sw::tex_sample_sw_array(st, u, v, L, 0);
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
    uint32_t got = gfx_sw::tex_sample_sw_cube(st, dirs[f][0], dirs[f][1], dirs[f][2], 0);
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
  const int32_t ONE = 1 << VX_TEX_FXD_FRAC;
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
  const int32_t ONE = 1 << VX_TEX_FXD_FRAC;
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

  check_npot(pool);
  check_persp_wrap(pool);

  // SW-path breadth (extended formats, border, cube/array, sampler-view array).
  check_srgb(pool);
  check_array(pool);
  check_cube(pool);
  check_two_textures(pool);
  check_border(pool);

  if (g_fail) {
    printf("\nTEX-SW PARITY: FAILED (%u mismatches across %u configs)\n", g_fail, configs);
    return 1;
  }
  printf("TEX-SW PARITY: PASSED (%u configs, SW == FF bit-exact; NPOT addressing OK)\n", configs);
  return 0;
}
