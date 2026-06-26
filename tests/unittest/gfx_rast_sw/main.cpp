// gfx_v2 SW rasterizer coverage unit test (§4.1).
//
// Proves the shared rasterizer coverage walk — rast_sw.h gfx_rast::
// rast_walk_primitive — emits EXACTLY the covered fragments: for a set of
// triangles, the per-pixel coverage produced by the recursive tile→quad walker
// is compared against an independent brute-force ground truth (a pixel is
// covered iff all three edge functions are >= 0 there and it is inside the
// scissor). This catches trivial-reject bugs (a too-aggressive tile/quad reject
// would drop covered pixels; a wrong leaf test would add spurious ones) and
// verifies the emitted bcoords equal the edge-function values at each fragment.
//
// rast_walk_primitive is the single source of truth shared by the FF RASTER
// model (gfx_render.cpp Rasterizer) and the device SW fallback (gfx_sw.h), so
// this also pins the contract both rely on. Self-contained: depends only on
// rast_sw.h (+ vx_gfx_abi.h / VX_types.h).
//
// Build/run: make -C tests/unittest/gfx_rast_sw run

#include "rast_sw.h"
#include <cstdio>
#include <cstdint>
#include <vector>
#include <set>
#include <utility>

using gfx_rast::vec3e_t;
using gfx_rast::FloatE;

namespace {

uint32_t g_fail = 0;

// Build the three CCW edge equations f_i(x,y) = a*x + b*y + c for a triangle.
// For a CCW winding the interior is f_i >= 0 for all i. Coefficients are exact
// integers carried in FloatE (fixed_t<16>).
struct Tri { int x0,y0, x1,y1, x2,y2; };

bool make_edges(const Tri& t, vec3e_t edges[3]) {
  // Signed area * 2; flip vertex order if clockwise so interior is f_i >= 0.
  long area2 = (long)(t.x1 - t.x0) * (t.y2 - t.y0) - (long)(t.x2 - t.x0) * (t.y1 - t.y0);
  if (area2 == 0) return false;          // degenerate
  int X[3] = {t.x0, t.x1, t.x2}, Y[3] = {t.y0, t.y1, t.y2};
  if (area2 < 0) { std::swap(X[1], X[2]); std::swap(Y[1], Y[2]); }
  for (int i = 0; i < 3; ++i) {
    int j = (i + 1) % 3;
    int a = Y[i] - Y[j];
    int b = X[j] - X[i];
    int c = X[i] * Y[j] - X[j] * Y[i];
    edges[i].x = FloatE(a);
    edges[i].y = FloatE(b);
    edges[i].z = FloatE(c);
  }
  return true;
}

void check_tri(const Tri& t, const gfx_rast::RastConfig& cfg, uint32_t ox, uint32_t oy) {
  vec3e_t edges[3];
  if (!make_edges(t, edges)) return;

  uint32_t tile = 1u << cfg.tile_logsize;

  // Walker coverage: collect (px,py) from emitted quads, and verify the emitted
  // edge values match EvalEdgeFunction at each covered fragment.
  std::set<std::pair<uint32_t,uint32_t>> walked;
  uint32_t bcoord_errs = 0;
  gfx_rast::rast_walk_primitive(cfg, ox, oy, /*pid*/1, edges,
    [&](uint32_t pos_mask, const vec3e_t* bcoords, uint32_t) {
      uint32_t mask   = pos_mask & 0xf;
      uint32_t quad_x = (pos_mask >> 4) & ((1u << (VX_RASTER_DIM_BITS - 1)) - 1);
      uint32_t quad_y =  pos_mask >> (4 + VX_RASTER_DIM_BITS - 1);
      for (uint32_t p = 0; p < 4; ++p) {
        if (!(mask & (1u << p))) continue;
        uint32_t i = p & 1, j = p >> 1;
        uint32_t px = quad_x * 2 + i, py = quad_y * 2 + j;
        walked.insert({px, py});
        // bcoords must equal the edge functions at this fragment.
        for (int e = 0; e < 3; ++e) {
          FloatE want = gfx_rast::EvalEdgeFunction(edges[e], (int)px, (int)py);
          FloatE got = (e == 0) ? bcoords[p].x : (e == 1) ? bcoords[p].y : bcoords[p].z;
          if (got != want) ++bcoord_errs;
        }
      }
    });

  // Ground truth: brute-force per-pixel inside test over the tile footprint.
  std::set<std::pair<uint32_t,uint32_t>> truth;
  FloatE z = gfx_rast::fx_zero();
  for (uint32_t py = oy; py < oy + tile; ++py) {
    for (uint32_t px = ox; px < ox + tile; ++px) {
      bool inside = gfx_rast::EvalEdgeFunction(edges[0], (int)px, (int)py) >= z
                 && gfx_rast::EvalEdgeFunction(edges[1], (int)px, (int)py) >= z
                 && gfx_rast::EvalEdgeFunction(edges[2], (int)px, (int)py) >= z
                 && px >= cfg.scissor_left && px < cfg.scissor_right
                 && py >= cfg.scissor_top  && py < cfg.scissor_bottom;
      if (inside) truth.insert({px, py});
    }
  }

  if (walked != truth || bcoord_errs) {
    ++g_fail;
    // Count the set difference for a concise diagnostic.
    uint32_t missing = 0, extra = 0;
    for (auto& p : truth)  if (!walked.count(p)) ++missing;
    for (auto& p : walked) if (!truth.count(p))  ++extra;
    printf("FAIL tri(%d,%d %d,%d %d,%d) scissor(%u,%u,%u,%u): "
           "missing=%u extra=%u bcoord_errs=%u (truth=%zu walked=%zu)\n",
           t.x0,t.y0,t.x1,t.y1,t.x2,t.y2,
           cfg.scissor_left,cfg.scissor_top,cfg.scissor_right,cfg.scissor_bottom,
           missing, extra, bcoord_errs, truth.size(), walked.size());
  }
}

// Independent double-precision edge value at a fractional sample position.
double edge_d(const vec3e_t& e, double x, double y) {
  double a = e.x.data() / 65536.0, b = e.y.data() / 65536.0, c = e.z.data() / 65536.0;
  return a * x + b * y + c;
}

// MSAA per-sample coverage: walk the primitive and, for every fragment of each
// emitted quad, compare rast_sample_mask against a double oracle evaluating the
// three edges at each sub-sample position. Disagreements are flagged only when
// the sample is clearly off the edge (|f| > tol) — samples sitting on the edge
// can differ by a fixed-point LSB and are not a correctness error.
void check_tri_msaa(const Tri& t, const gfx_rast::RastConfig& cfg, uint32_t ox, uint32_t oy) {
  vec3e_t edges[3];
  if (!make_edges(t, edges)) return;
  const double tol = 1e-2;
  const auto* off = gfx_rast::rast_msaa_offsets();
  uint32_t errs = 0, samples = 0;

  gfx_rast::rast_walk_primitive(cfg, ox, oy, 1, edges,
    [&](uint32_t pos_mask, const vec3e_t* bcoords, uint32_t) {
      uint32_t quad_x = (pos_mask >> 4) & ((1u << (VX_RASTER_DIM_BITS - 1)) - 1);
      uint32_t quad_y =  pos_mask >> (4 + VX_RASTER_DIM_BITS - 1);
      for (uint32_t p = 0; p < 4; ++p) {
        uint32_t px = quad_x * 2 + (p & 1), py = quad_y * 2 + (p >> 1);
        uint32_t m = gfx_rast::rast_sample_mask(edges, bcoords[p]);
        for (int k = 0; k < gfx_rast::RAST_MSAA_SAMPLES; ++k) {
          double sx = px + off[k].dx.data() / 65536.0;
          double sy = py + off[k].dy.data() / 65536.0;
          double f0 = edge_d(edges[0], sx, sy);
          double f1 = edge_d(edges[1], sx, sy);
          double f2 = edge_d(edges[2], sx, sy);
          bool oracle = (f0 >= 0 && f1 >= 0 && f2 >= 0);
          bool got = (m >> k) & 1;
          double margin = (f0 < f1 ? (f0 < f2 ? f0 : f2) : (f1 < f2 ? f1 : f2));
          if (got != oracle && margin < -tol) ++errs;   // clearly-outside but marked in
          if (got != oracle && margin >  tol) ++errs;   // clearly-inside but marked out
          ++samples;
        }
      }
    });

  if (errs) {
    ++g_fail;
    printf("FAIL(MSAA) tri(%d,%d %d,%d %d,%d): %u/%u sample errors\n",
           t.x0,t.y0,t.x1,t.y1,t.x2,t.y2, errs, samples);
  }
}

} // namespace

int main() {
  const uint32_t T = 5;               // 32x32 top-level tile
  const uint32_t tile = 1u << T;

  // A spread of triangles: corners, thin slivers, fully-covering, off-tile,
  // single-pixel — exercising every trivial-reject path and the leaf test.
  std::vector<Tri> tris = {
    {  2,  2, 28,  4, 10, 26 },   // general
    {  0,  0, 31,  0, 0, 31 },    // lower-left half, on tile edges
    {  0,  0, 31, 31, 0, 31 },    // upper-left half
    {  5,  5, 6,  5,  5, 6 },     // tiny ~1px
    { 15, 15, 17, 16, 16, 18 },   // small interior
    {  1, 30, 30, 29, 15, 1 },    // wide, near edges
    { 40, 40, 60, 41, 45, 60 },   // entirely off-tile (must emit nothing)
    {  3, 20, 25,  3, 30, 30 },   // slanted
    {  0, 16, 31, 15, 16, 31 },   // spanning
    {  8,  8, 24,  8, 16, 24 },   // centered
  };

  uint32_t cases = 0;

  // Full-open scissor (covers the whole tile).
  gfx_rast::RastConfig full{ T, 0, 0, 1u << 20, 1u << 20 };
  for (auto& t : tris) { check_tri(t, full, 0, 0); ++cases; }

  // Tight scissor box to exercise per-fragment scissor rejection.
  gfx_rast::RastConfig scis{ T, 6, 6, 20, 22 };
  for (auto& t : tris) { check_tri(t, scis, 0, 0); ++cases; }

  // Non-zero (quad-aligned) tile origin.
  gfx_rast::RastConfig off{ T, 0, 0, 1u << 20, 1u << 20 };
  for (auto& t : tris) { check_tri(t, off, tile, tile); ++cases; }

  // MSAA 4x per-sample coverage vs double oracle (§6).
  for (auto& t : tris) { check_tri_msaa(t, full, 0, 0); ++cases; }

  if (g_fail) {
    printf("\nRAST-SW COVERAGE: FAILED (%u/%u cases)\n", g_fail, cases);
    return 1;
  }
  printf("RAST-SW COVERAGE: PASSED (%u cases, walker == brute-force ground truth)\n", cases);
  return 0;
}
