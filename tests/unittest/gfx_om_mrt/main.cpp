// gfx_v2 SW output-merger MRT unit test.
//
// Proves the on-device SIMT software fallback for multiple render targets —
// gfx_sw.h om_fragment_mrt() — writes N colour attachments from one fragment
// with a SHARED depth/stencil test and INDEPENDENT per-attachment colour state
// (base, pitch, write-mask, blend). This is the device primitive the driver
// calls (gfx_om_fragment_mrt_sw) when a draw targets >1 colour attachment.
//
// Build/run: make -C tests/unittest/gfx_om_mrt run

#include "gfx_sw.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace gfx_sw;

namespace {

uint32_t g_fail = 0;

void expect_eq(const char* what, uint32_t got, uint32_t exp) {
  if (got != exp) {
    printf("  FAIL %s: got=%08x exp=%08x\n", what, got, exp);
    ++g_fail;
  }
}

} // namespace

int main() {
  const uint32_t W = 4, H = 4, PITCH = W * 4;
  std::vector<uint32_t> buf0(W * H, 0x11111111u);   // attachment 0
  std::vector<uint32_t> buf1(W * H, 0x22222222u);   // attachment 1
  std::vector<uint32_t> zbuf(W * H, 0xffffffffu);   // depth = far

  // Shared depth/stencil state: depth LESS + write, stencil off.
  om_state_t s{};
  s.depth_func      = VX_OM_DEPTH_FUNC_LESS;
  s.depth_writemask = 1;
  for (int f = 0; f < 2; ++f) {
    s.stencil_func[f]  = VX_OM_DEPTH_FUNC_ALWAYS;
    s.stencil_zpass[f] = VX_OM_STENCIL_OP_KEEP;
    s.stencil_zfail[f] = VX_OM_STENCIL_OP_KEEP;
  }
  s.zbuf_base  = (uint64_t)(uintptr_t)zbuf.data();
  s.zbuf_pitch = PITCH;
  resolve_om_state(s);

  // Two colour attachments. RT0: full write-mask, no blend (replace). RT1:
  // red-only write-mask, no blend — so its green/blue/alpha keep their init,
  // proving per-attachment write-mask independence.
  om_color_t rt[2] = {};
  rt[0].cbuf_base      = (uint64_t)(uintptr_t)buf0.data();
  rt[0].cbuf_pitch     = PITCH;
  rt[0].blend_mode_rgb = VX_OM_BLEND_MODE_ADD;
  rt[0].blend_mode_a   = VX_OM_BLEND_MODE_ADD;
  rt[0].blend_src_rgb  = VX_OM_BLEND_FUNC_ONE;
  rt[0].blend_src_a    = VX_OM_BLEND_FUNC_ONE;
  rt[0].blend_dst_rgb  = VX_OM_BLEND_FUNC_ZERO;
  rt[0].blend_dst_a    = VX_OM_BLEND_FUNC_ZERO;
  rt[0].cbuf_writemask4 = 0xf;
  resolve_om_color(rt[0]);

  rt[1] = rt[0];
  rt[1].cbuf_base       = (uint64_t)(uintptr_t)buf1.data();
  rt[1].cbuf_writemask4 = 0x4;   // bit2 -> red byte (0x00ff0000) only
  resolve_om_color(rt[1]);

  // --- fragment 1: near depth at (1,1), passes; writes both attachments. ---
  const uint32_t src_a[2] = { 0xaabbccddu, 0x55667788u };
  om_fragment_mrt(s, rt, 2, /*x*/1, /*y*/1, /*face*/0, src_a, /*depth*/0x00001000u);

  expect_eq("rt0 (1,1) full replace", buf0[1 * W + 1], 0xaabbccddu);
  // rt1: only red byte (0x00ff0000) taken from src, rest keeps init 0x22222222.
  expect_eq("rt1 (1,1) red-only mask",
            buf1[1 * W + 1], (0x22222222u & ~0x00ff0000u) | (0x55667788u & 0x00ff0000u));
  // depth written once, shared (only the depth bits; stencil byte untouched):
  expect_eq("depth (1,1) written", zbuf[1 * W + 1] & OM_DEPTH_MASK, 0x00001000u);

  // --- fragment 2: FARTHER depth at same pixel — shared depth test rejects it
  //     for BOTH attachments (neither colour buffer changes). ---
  uint32_t b0_before = buf0[1 * W + 1], b1_before = buf1[1 * W + 1];
  const uint32_t src_b[2] = { 0x01020304u, 0x0a0b0c0du };
  om_fragment_mrt(s, rt, 2, 1, 1, 0, src_b, /*depth*/0x00002000u);   // > 0x1000, fails LESS
  expect_eq("rt0 (1,1) depth-rejected", buf0[1 * W + 1], b0_before);
  expect_eq("rt1 (1,1) depth-rejected", buf1[1 * W + 1], b1_before);
  expect_eq("depth (1,1) unchanged",    zbuf[1 * W + 1] & OM_DEPTH_MASK, 0x00001000u);

  // --- a different pixel stays at init (no cross-pixel writes). ---
  expect_eq("rt0 (2,2) untouched", buf0[2 * W + 2], 0x11111111u);
  expect_eq("rt1 (2,2) untouched", buf1[2 * W + 2], 0x22222222u);

  // --- blend on RT1: additive ONE/ONE over a known dst, small values (exact). ---
  om_color_t bc = rt[0];
  std::vector<uint32_t> bbuf(1, 0x00141414u);   // dst = (20,20,20)
  bc.cbuf_base      = (uint64_t)(uintptr_t)bbuf.data();
  bc.cbuf_pitch     = 4;
  bc.blend_dst_rgb  = VX_OM_BLEND_FUNC_ONE;      // src*ONE + dst*ONE
  bc.blend_dst_a    = VX_OM_BLEND_FUNC_ONE;
  resolve_om_color(bc);
  om_state_t sd = s;
  std::vector<uint32_t> z1(1, 0xffffffffu);
  sd.zbuf_base = (uint64_t)(uintptr_t)z1.data();
  sd.zbuf_pitch = 4;
  const uint32_t src_c[1] = { 0x000a0a0au };     // (10,10,10)
  om_fragment_mrt(sd, &bc, 1, 0, 0, 0, src_c, 0x1000u);
  expect_eq("blend additive 10+20=30", bbuf[0] & 0xff, 30u);

  // ── W6 attachment-format breadth ────────────────────────────────────────
  // Helper: a depth/stencil-disabled, blend-passthrough single-RT state.
  auto make_color_state = [](uint64_t cbase, uint32_t fmt, uint32_t wm4) {
    om_state_t o{};
    o.depth_func = VX_OM_DEPTH_FUNC_ALWAYS; o.depth_writemask = 0;
    for (int f = 0; f < 2; ++f) {
      o.stencil_func[f]  = VX_OM_DEPTH_FUNC_ALWAYS;
      o.stencil_zpass[f] = VX_OM_STENCIL_OP_KEEP;
      o.stencil_zfail[f] = VX_OM_STENCIL_OP_KEEP;
    }
    o.blend_mode_rgb = VX_OM_BLEND_MODE_ADD; o.blend_mode_a = VX_OM_BLEND_MODE_ADD;
    o.blend_src_rgb  = VX_OM_BLEND_FUNC_ONE; o.blend_src_a  = VX_OM_BLEND_FUNC_ONE;
    o.blend_dst_rgb  = VX_OM_BLEND_FUNC_ZERO; o.blend_dst_a = VX_OM_BLEND_FUNC_ZERO;
    o.cbuf_base = cbase; o.cbuf_pitch = 64; o.color_format = fmt; o.cbuf_writemask4 = wm4;
    resolve_om_state(o);
    return o;
  };

  // (1) sRGB8 colour target: linear → sRGB encode on write.
  {
    uint32_t cb = 0;
    om_state_t o = make_color_state((uint64_t)(uintptr_t)&cb, VX_OM_COLOR_FORMAT_SRGB8A8, 0xf);
    uint32_t lin = 0xFF804020u;                       // a=FF r=80 g=40 b=20 (linear)
    om_fragment(o, 0, 0, 0, lin, 0);
    uint32_t exp = (0xFFu << 24)
                 | (Linear8ToSrgb(0x80) << 16) | (Linear8ToSrgb(0x40) << 8) | Linear8ToSrgb(0x20);
    expect_eq("sRGB8 encode-on-write", cb, exp);
    if (cb == lin) { printf("  FAIL sRGB8: value not gamma-encoded\n"); ++g_fail; }
  }

  // (2) D32F depth attachment: correct depth test on float bits + non-default clear.
  {
    float clearf = 0.9f; uint32_t zf; std::memcpy(&zf, &clearf, 4);   // non-default clear
    uint32_t cb = 0;
    om_state_t o{};
    o.depth_func = VX_OM_DEPTH_FUNC_LESS; o.depth_writemask = 1;
    for (int f = 0; f < 2; ++f) { o.stencil_func[f]=VX_OM_DEPTH_FUNC_ALWAYS;
      o.stencil_zpass[f]=VX_OM_STENCIL_OP_KEEP; o.stencil_zfail[f]=VX_OM_STENCIL_OP_KEEP; }
    o.blend_mode_rgb=VX_OM_BLEND_MODE_ADD; o.blend_mode_a=VX_OM_BLEND_MODE_ADD;
    o.blend_src_rgb=VX_OM_BLEND_FUNC_ONE; o.blend_src_a=VX_OM_BLEND_FUNC_ONE;
    o.blend_dst_rgb=VX_OM_BLEND_FUNC_ZERO; o.blend_dst_a=VX_OM_BLEND_FUNC_ZERO;
    o.depth_format = VX_OM_DEPTH_FORMAT_D32F;
    o.zbuf_base=(uint64_t)(uintptr_t)&zf; o.zbuf_pitch=64;
    o.cbuf_base=(uint64_t)(uintptr_t)&cb; o.cbuf_pitch=64; o.cbuf_writemask4=0xf;
    resolve_om_state(o);
    auto fbits = [](float f){ uint32_t u; std::memcpy(&u,&f,4); return u; };
    om_fragment(o, 0, 0, 0, 0xAAAAAAAAu, fbits(0.5f));   // 0.5 < 0.9 → passes
    expect_eq("D32F pass color",  cb, 0xAAAAAAAAu);
    expect_eq("D32F pass depth",  zf, fbits(0.5f));
    om_fragment(o, 0, 0, 0, 0xBBBBBBBBu, fbits(0.8f));   // 0.8 !< 0.5 → fails
    expect_eq("D32F fail color",  cb, 0xAAAAAAAAu);
    expect_eq("D32F fail depth",  zf, fbits(0.5f));
    om_fragment(o, 0, 0, 0, 0xCCCCCCCCu, fbits(0.1f));   // 0.1 < 0.5 → passes
    expect_eq("D32F near color",  cb, 0xCCCCCCCCu);
    expect_eq("D32F near depth",  zf, fbits(0.1f));
  }

  // (3) R8 channel-masked write (single-channel attachment).
  {
    uint8_t cb = 0x00;
    om_state_t o = make_color_state((uint64_t)(uintptr_t)&cb, VX_OM_COLOR_FORMAT_R8, 0xf);
    om_fragment(o, 0, 0, 0, 0x00AB0000u /*r=AB*/, 0);
    expect_eq("R8 write red", cb, 0xABu);
  }

  // (4) blend constant reaches the SW OM (src * CONST_RGB, const=0x40, src=0xff).
  {
    uint32_t cb = 0;
    om_state_t o = make_color_state((uint64_t)(uintptr_t)&cb, VX_OM_COLOR_FORMAT_A8R8G8B8, 0xf);
    o.blend_src_rgb = VX_OM_BLEND_FUNC_CONST_RGB;   // src * const
    o.blend_src_a   = VX_OM_BLEND_FUNC_ONE;
    o.blend_dst_rgb = VX_OM_BLEND_FUNC_ZERO;
    o.blend_dst_a   = VX_OM_BLEND_FUNC_ZERO;
    o.blend_const   = 0xFF404040u;                  // const rgb = 0x40
    resolve_om_state(o);
    om_fragment(o, 0, 0, 0, 0x00FFFFFFu, 0);        // src rgb = 0xff
    uint32_t rr = (cb >> 16) & 0xff;
    if (rr < 0x38 || rr > 0x48) { printf("  FAIL blend-const: r=%02x (expected ~0x40)\n", rr); ++g_fail; }
    else printf("  blend-const OK (r=%02x from const 0x40 * src 0xff)\n", rr);
  }

  // (5) Field-described attachments: snorm rounding, the BGRA and 2-10-10-10
  // bit layouts, 12-byte texels, an alpha-only texel that leaves its neighbour
  // alone, a packed 16-bit order, and an snorm blend reading the destination.
  {
    auto lanes = [](float r, float g, float b, float a, uint32_t out[4]) {
      const float f[4] = { r, g, b, a };
      memcpy(out, f, sizeof f);
    };
    auto store = [&](uint32_t fmt, void* dst, const uint32_t src[4], bool add) {
      om_color_t c{};
      c.cbuf_base = (uint64_t)(uintptr_t)dst;
      c.color_format = fmt;
      c.cbuf_writemask4 = 0xf;
      c.blend_mode_rgb = c.blend_mode_a = VX_OM_BLEND_MODE_ADD;
      c.blend_src_rgb = c.blend_src_a = VX_OM_BLEND_FUNC_ONE;
      c.blend_dst_rgb = c.blend_dst_a = add ? VX_OM_BLEND_FUNC_ONE : VX_OM_BLEND_FUNC_ZERO;
      resolve_om_color(c);
      om_wide_color_rmw(c, (uintptr_t)dst, src);
    };
    uint32_t l[4];
    uint8_t b4[4] = { 0 };
    lanes(1.0f, -1.0f, 0.5f, -0.25f, l);
    store(VX_OM_COLOR_FORMAT_RGBA8SN, b4, l, false);
    uint32_t w; memcpy(&w, b4, 4);
    expect_eq("RGBA8SN 1,-1,.5,-.25", w, 0xe040817fu);
    store(VX_OM_COLOR_FORMAT_BGRA8SN, b4, l, false);
    memcpy(&w, b4, 4);
    expect_eq("BGRA8SN 1,-1,.5,-.25", w, 0xe07f8140u);
    lanes(1.0f, 0.0f, 0.0f, 1.0f, l);
    uint32_t w10 = 0;
    store(VX_OM_COLOR_FORMAT_A2R10G10B10, &w10, l, false);
    expect_eq("A2R10G10B10 red", w10, 0xfff00000u);
    lanes(1.5f, -2.0f, 3.0f, 0.0f, l);
    float f3[4] = { 0, 0, 0, 7.0f };
    store(VX_OM_COLOR_FORMAT_RGB32F, f3, l, false);
    uint32_t w3[4];
    memcpy(w3, f3, sizeof w3);
    expect_eq("RGB32F r", w3[0], l[0]);
    expect_eq("RGB32F b", w3[2], l[2]);
    expect_eq("RGB32F past texel", w3[3], 0x40e00000u);
    uint8_t a8[2] = { 0, 0x5a };
    lanes(0.2f, 0.4f, 0.6f, 1.0f, l);
    store(VX_OM_COLOR_FORMAT_A8, a8, l, false);
    expect_eq("A8 alpha", a8[0], 0xffu);
    expect_eq("A8 neighbour", a8[1], 0x5au);
    uint16_t p16 = 0;
    lanes(1.0f, 0.0f, 0.0f, 0.0f, l);
    store(VX_OM_COLOR_FORMAT_B4G4R4A4_PACK16, &p16, l, false);
    expect_eq("B4G4R4A4 red", p16, 0x00f0u);
    // dst 0x40 = 64/127, plus 0.25 = 0.7539 -> round(95.75) = 96
    uint8_t sn = 0x40;
    lanes(0.25f, 0.0f, 0.0f, 0.0f, l);
    store(VX_OM_COLOR_FORMAT_R8SN, &sn, l, true);
    expect_eq("R8SN additive blend", sn, 0x60u);
    if (!g_fail) {
      printf("  field attachments OK (snorm, BGRA8, 2-10-10-10, RGB32F, A8, 4444)\n");
    }
  }

  if (g_fail) {
    printf("\nOM-MRT: FAILED (%u checks)\n", g_fail);
    return 1;
  }
  printf("OM-MRT: PASSED (2 attachments + W6 formats: sRGB8/R8, D32F depth, blend-const)\n");
  return 0;
}
