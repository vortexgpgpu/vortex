// gfx_v2 SW output-merger MRT unit test (§ W6).
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

  if (g_fail) {
    printf("\nOM-MRT: FAILED (%u checks)\n", g_fail);
    return 1;
  }
  printf("OM-MRT: PASSED (2 attachments + W6 formats: sRGB8/R8, D32F depth, blend-const)\n");
  return 0;
}
