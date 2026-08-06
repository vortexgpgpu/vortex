// gfx_v2 MSAA software path unit test (§6).
//
// Exercises the full on-device MSAA SW pipeline end to end on the host, using
// the same code the device runs (no host-only reimplementation):
//   coverage   gfx_frag_rast.h   gfx_rast::rast_sample_mask  (4x per-sample coverage)
//   ROP        gfx_sw.h    gfx_sw::om_fragment_msaa     (per-sample depth+blend)
//   storage    gfx_sw.h    gfx_sw::msaa_*_addr          (sample-interleaved)
//   resolve    gfx_sw.h    gfx_sw::msaa_resolve_color   (box average)
//
// A triangle is rasterized into resident per-sample color/depth surfaces, then
// resolved. The resolved image is checked against an INDEPENDENT oracle: each
// pixel's expected color is the box average of foreground/background weighted by
// that pixel's covered-sample count (recomputed straight from rast_sample_mask).
// Interior pixels must equal fg, exterior bg, edge pixels a proportional blend —
// i.e. anti-aliased edges. A second pass proves per-sample depth testing: an
// occluded redraw must not change any covered sample.
//
// Self-contained: depends only on gfx_frag_rast.h + gfx_sw.h. No GFX_SW_DIVERGENCE_OK
// needed (that guards the *device* divergence-pass build; the host has no such
// pass and compiles the merge normally).
//
// Build/run: make -C tests/unittest/gfx_msaa run

#include "gfx_frag_rast.h"
#include "gfx_sw.h"
#include <cstdio>
#include <cstdint>
#include <vector>
#include <utility>

using gfx_rast::vec3e_t;
using gfx_rast::FloatE;

namespace {

constexpr uint32_t W = 16, H = 16, S = 4;

uint32_t g_fail = 0;

struct Tri { int x0,y0, x1,y1, x2,y2; };

bool make_edges(const Tri& t, vec3e_t edges[3]) {
  long area2 = (long)(t.x1 - t.x0) * (t.y2 - t.y0) - (long)(t.x2 - t.x0) * (t.y1 - t.y0);
  if (area2 == 0) return false;
  int X[3] = {t.x0, t.x1, t.x2}, Y[3] = {t.y0, t.y1, t.y2};
  if (area2 < 0) { std::swap(X[1], X[2]); std::swap(Y[1], Y[2]); }
  for (int i = 0; i < 3; ++i) {
    int j = (i + 1) % 3;
    edges[i].x = FloatE(Y[i] - Y[j]);
    edges[i].y = FloatE(X[j] - X[i]);
    edges[i].z = FloatE(X[i] * Y[j] - X[j] * Y[i]);
  }
  return true;
}

// Independent per-pixel covered-sample count from the same sample pattern.
uint32_t coverage_count(const vec3e_t edges[3], uint32_t px, uint32_t py) {
  vec3e_t base{ gfx_rast::EvalEdgeFunction(edges[0], (int)px, (int)py),
                gfx_rast::EvalEdgeFunction(edges[1], (int)px, (int)py),
                gfx_rast::EvalEdgeFunction(edges[2], (int)px, (int)py) };
  return __builtin_popcount(gfx_rast::rast_sample_mask(edges, base));
}

uint32_t box_avg(uint32_t fg, uint32_t bg, uint32_t cnt) {
  uint32_t out = 0;
  for (uint32_t ch = 0; ch < 4; ++ch) {
    uint32_t f = (fg >> (ch*8)) & 0xff, b = (bg >> (ch*8)) & 0xff;
    uint32_t v = (cnt * f + (S - cnt) * b + (S >> 1)) / S;
    out |= (v & 0xff) << (ch * 8);
  }
  return out;
}

// Opaque-overwrite OM state (no depth, no blend) for the coverage/resolve test.
void make_state_opaque(gfx_sw::om_state_t& s, uint32_t* color, uint32_t* depth) {
  s = {};
  s.depth_func = VX_OM_DEPTH_FUNC_ALWAYS; s.depth_writemask = 0;
  for (int f = 0; f < 2; ++f) {
    s.stencil_func[f] = VX_OM_DEPTH_FUNC_ALWAYS;
    s.stencil_zpass[f] = s.stencil_zfail[f] = s.stencil_fail[f] = VX_OM_STENCIL_OP_KEEP;
    s.stencil_ref[f] = s.stencil_mask[f] = s.stencil_writemask[f] = 0;
  }
  s.blend_mode_rgb = s.blend_mode_a = VX_OM_BLEND_MODE_ADD;
  s.blend_src_rgb = s.blend_src_a = VX_OM_BLEND_FUNC_ONE;
  s.blend_dst_rgb = s.blend_dst_a = VX_OM_BLEND_FUNC_ZERO;
  s.blend_const = 0; s.logic_op = 0;
  s.zbuf_base = (uint64_t)(uintptr_t)depth; s.cbuf_base = (uint64_t)(uintptr_t)color;
  s.zbuf_pitch = W * S * 4; s.cbuf_pitch = W * S * 4;
  s.cbuf_writemask4 = 0xf;
  gfx_sw::resolve_om_state(s);
}

void clear(uint32_t* color, uint32_t* depth, uint32_t bg) {
  for (uint32_t i = 0; i < W * H * S; ++i) { color[i] = bg; depth[i] = 0x00ffffff; }
}

void render(const gfx_sw::om_state_t& s, const vec3e_t edges[3], uint32_t fg, uint32_t depth) {
  gfx_rast::RastConfig cfg{ 4, 0, 0, W, H };   // 16x16 tile == FB, scissor = FB
  gfx_rast::rast_walk_primitive_msaa(cfg, 0, 0, 1, edges,
    [&](uint32_t pos_mask, const vec3e_t*, const uint32_t* sample_masks, uint32_t) {
      uint32_t quad_x = (pos_mask >> 4) & ((1u << (VX_RASTER_DIM_BITS - 1)) - 1);
      uint32_t quad_y =  pos_mask >> (4 + VX_RASTER_DIM_BITS - 1);
      for (uint32_t p = 0; p < 4; ++p) {
        uint32_t px = quad_x * 2 + (p & 1), py = quad_y * 2 + (p >> 1);
        if (px >= W || py >= H) continue;
        uint32_t m = sample_masks[p];
        if (m) gfx_sw::om_fragment_msaa(s, S, px, py, 0, m, fg, depth);
      }
    });
}

void check_resolved(const gfx_sw::om_state_t& s, const vec3e_t edges[3],
                    uint32_t fg, uint32_t bg, const char* tag, uint32_t& interior, uint32_t& edge) {
  for (uint32_t y = 0; y < H; ++y) {
    for (uint32_t x = 0; x < W; ++x) {
      uint32_t cnt = coverage_count(edges, x, y);
      uint32_t want = box_avg(fg, bg, cnt);
      uint32_t got = gfx_sw::msaa_resolve_color(s, S, x, y);
      if (cnt == S) ++interior;
      else if (cnt > 0) ++edge;
      if (got != want) {
        if (g_fail < 8)
          printf("  [%s] (%u,%u) cnt=%u: want=%08x got=%08x\n", tag, x, y, cnt, want, got);
        ++g_fail;
      }
    }
  }
}

} // namespace

int main() {
  std::vector<uint32_t> color(W * H * S), depth(W * H * S);
  const uint32_t FG = 0xA0786428, BG = 0x10101010, OTHER = 0xFF00FF00;

  gfx_sw::om_state_t s;
  make_state_opaque(s, color.data(), depth.data());

  // A triangle with slanted edges → mix of interior, exterior, and partially
  // covered (anti-aliased) edge pixels.
  Tri tri{ 2, 2, 14, 5, 5, 14 };
  vec3e_t edges[3];
  if (!make_edges(tri, edges)) { printf("degenerate tri\n"); return 2; }

  // ── Test 1: coverage + box resolve (no depth/blend) ──────────────────────
  clear(color.data(), depth.data(), BG);
  render(s, edges, FG, 0);
  uint32_t interior = 0, edge = 0;
  check_resolved(s, edges, FG, BG, "opaque", interior, edge);

  // ── Test 2: per-sample depth test — occluded redraw must not change output ─
  gfx_sw::om_state_t sd;
  make_state_opaque(sd, color.data(), depth.data());
  sd.depth_func = VX_OM_DEPTH_FUNC_LESS; sd.depth_writemask = 1;
  gfx_sw::resolve_om_state(sd);
  clear(color.data(), depth.data(), BG);
  render(sd, edges, FG, 0x000000);          // near, writes depth 0
  render(sd, edges, OTHER, 0x800000);        // farther → LESS fails everywhere covered
  uint32_t i2 = 0, e2 = 0;
  check_resolved(sd, edges, FG, BG, "depth", i2, e2);   // must still be FG, not OTHER

  if (g_fail) {
    printf("\nGFX-MSAA: FAILED (%u pixel mismatches)\n", g_fail);
    return 1;
  }
  if (interior == 0 || edge == 0) {
    printf("\nGFX-MSAA: INCONCLUSIVE (interior=%u edge=%u — need both)\n", interior, edge);
    return 1;
  }
  printf("GFX-MSAA: PASSED (4x; %u interior + %u AA-edge px match box-avg oracle; "
         "per-sample depth test occludes correctly)\n", interior, edge);
  return 0;
}
