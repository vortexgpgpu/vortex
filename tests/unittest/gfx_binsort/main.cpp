// gfx_v2 bin-sort binning — functional validation unit test.
// Build/run via the test harness: make -C tests/unittest/gfx_binsort run
//
// Proves the core design claim of docs/proposals/gfx_v2_tile_binning_redesign.md
// at GROUND TRUTH (per covered sample): for every covered pixel, the
// draw-ordered list of primitives covering it is IDENTICAL between the gfx-v1
// path (bbox binning, draw order) and the bin-sort pipeline (coarse 128px bins,
// count->prefix-sum->emit->sort->header-scan). Per-pixel ordered coverage is
// what determines the final image under depth/blend, so identical per-pixel
// lists ⟹ identical rendering. Also checks determinism and the coverage-key
// footprint advantage.
//
// Why per-pixel, not per-tile: bin-sort's coarser 128px descent conservatively
// visits a few tiles just outside a triangle's tight bbox, but those contain no
// covered SAMPLE (a covered sample is inside all edges ⟹ inside the bbox), so
// they emit no pixels. Tile-set comparison flags those as false diffs;
// sample-set comparison is exact.
//
// Self-contained: depends only on sw/common/vx_gfx_abi.h. Setup mirrors
// graphics.cpp (EdgeEquation/half-pixel/EdgeToFixed), specialized to screen
// space (w==1) since binning correctness is independent of the perspective
// divide. Sample test uses the same Q15.16 edge functions as gfx_ff_model.cpp.
//
// Standalone: g++ -std=c++17 -O2 -I <repo>/sw/common main.cpp -o gfx_binsort

#include "vx_gfx_abi.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

using FloatE  = vortex::graphics::fixed16_t;   // Q15.16, like RASTER edges
using vec3e_t = vortex::graphics::vec3e_t;

static constexpr int W         = 512;
static constexpr int H         = 512;
static constexpr int TILE_LOG  = 5;             // 32px gfx-v1 fine tile
static constexpr int BIN_LOG   = 7;             // 128px bin-sort coarse bin
static constexpr int BIN_COLS  = (W + (1 << BIN_LOG) - 1) >> BIN_LOG;
static constexpr int PRIM_BITS = 20;            // 32-bit composite key baseline
static constexpr uint32_t PRIM_MASK = (1u << PRIM_BITS) - 1;

static const FloatE fxZero = FloatE::make(0);

struct Vec2 { float x, y; };
struct Prim {
  vec3e_t edges[3];
  int bbL, bbR, bbT, bbB;
  bool valid;
};

// screen-space (w==1) edge setup, mirroring graphics.cpp.
static bool setupPrim(Prim& p, Vec2 v0, Vec2 v1, Vec2 v2) {
  float a0 = v1.y - v2.y, a1 = v2.y - v0.y, a2 = v0.y - v1.y;
  float b0 = v2.x - v1.x, b1 = v0.x - v2.x, b2 = v1.x - v0.x;
  float c0 = v1.x*v2.y - v2.x*v1.y;
  float c1 = v2.x*v0.y - v0.x*v2.y;
  float c2 = v0.x*v1.y - v1.x*v0.y;
  float det = c0 + c1 + c2;
  if (det == 0.0f) { p.valid = false; return false; }
  float s = (det < 0.0f) ? -1.0f : 1.0f;
  float E[3][3] = {{a0*s,b0*s,c0*s},{a1*s,b1*s,c1*s},{a2*s,b2*s,c2*s}};
  for (int i = 0; i < 3; ++i) E[i][2] += (E[i][0] + E[i][1]) * 0.5f;  // half-pixel
  float mx = 0.0f;
  for (int i = 0; i < 3; ++i) { mx = std::max(mx, std::fabs(E[i][0]));
                                mx = std::max(mx, std::fabs(E[i][1])); }
  float sc = (mx != 0.0f) ? (1.0f / mx) : 1.0f;
  for (int i = 0; i < 3; ++i)
    p.edges[i] = { FloatE(E[i][0]*sc), FloatE(E[i][1]*sc), FloatE(E[i][2]*sc) };
  float L = std::min({v0.x,v1.x,v2.x}), R = std::max({v0.x,v1.x,v2.x});
  float T = std::min({v0.y,v1.y,v2.y}), B = std::max({v0.y,v1.y,v2.y});
  p.bbL = std::max(0,(int)std::floor(L)); p.bbR = std::min(W,(int)std::ceil(R));
  p.bbT = std::max(0,(int)std::floor(T)); p.bbB = std::min(H,(int)std::ceil(B));
  p.valid = (p.bbR > p.bbL && p.bbB > p.bbT);
  return p.valid;
}

static inline FloatE evalEdge(const vec3e_t& e, int x, int y) {
  return (e.x * x) + (e.y * y) + e.z;
}

// ground-truth covered samples of a prim (pixel centers inside all 3 edges).
static std::vector<uint32_t> coveredSamples(const Prim& p) {
  std::vector<uint32_t> s;
  if (!p.valid) return s;
  for (int y = p.bbT; y < p.bbB; ++y)
    for (int x = p.bbL; x < p.bbR; ++x)
      if (evalEdge(p.edges[0], x, y) >= fxZero
       && evalEdge(p.edges[1], x, y) >= fxZero
       && evalEdge(p.edges[2], x, y) >= fxZero)
        s.push_back((uint32_t)y * W + (uint32_t)x);
  return s;
}

static inline int binOfPixel(uint32_t pix) {
  int x = (int)(pix % W), y = (int)(pix / W);
  return (y >> BIN_LOG) * BIN_COLS + (x >> BIN_LOG);
}

// stage 4: LSD radix sort (4 x 8-bit passes, stable) — the actual sort the
// design specifies. Stable + ascending ⟹ same total order as std::sort on the
// composite key, i.e. bin bucket then draw order.
static void radixLSD(std::vector<uint32_t>& a) {
  size_t n = a.size();
  if (n < 2) return;
  std::vector<uint32_t> tmp(n);
  uint32_t* src = a.data();
  uint32_t* dst = tmp.data();
  for (int shift = 0; shift < 32; shift += 8) {
    size_t cnt[256] = {0};
    for (size_t i = 0; i < n; ++i) ++cnt[(src[i] >> shift) & 0xFF];
    size_t sum = 0;
    for (int k = 0; k < 256; ++k) { size_t c = cnt[k]; cnt[k] = sum; sum += c; }
    for (size_t i = 0; i < n; ++i) { uint32_t v = src[i]; dst[cnt[(v >> shift) & 0xFF]++] = v; }
    std::swap(src, dst);
  }
  if (src != a.data()) std::copy(src, src + n, a.begin());  // 4 passes → src==a, guard anyway
}

// ---- bin-sort pipeline (the six stages) ----
struct BinSortOut {
  std::vector<uint32_t> keys;                  // sorted (bin<<PRIM)|prim
  std::map<int, std::vector<uint32_t>> bins;   // bin_id -> draw-ordered prim ids
};
static BinSortOut binSort(const std::vector<Prim>& prims) {
  BinSortOut o;
  std::vector<int> count(prims.size(), 0);
  for (size_t i = 0; i < prims.size(); ++i) {
    if (!prims[i].valid) continue;
    const Prim& p = prims[i];
    int bL = p.bbL>>BIN_LOG, bR = (p.bbR-1)>>BIN_LOG;
    int bT = p.bbT>>BIN_LOG, bB = (p.bbB-1)>>BIN_LOG;
    count[i] = (bR-bL+1)*(bB-bT+1);
  }
  std::vector<int> off(prims.size()+1, 0);
  for (size_t i = 0; i < prims.size(); ++i) off[i+1] = off[i] + count[i];
  o.keys.resize(off.back());
  for (size_t i = 0; i < prims.size(); ++i) {
    if (!prims[i].valid) continue;
    const Prim& p = prims[i];
    int bL = p.bbL>>BIN_LOG, bR = (p.bbR-1)>>BIN_LOG;
    int bT = p.bbT>>BIN_LOG, bB = (p.bbB-1)>>BIN_LOG;
    int w = off[i];
    for (int by = bT; by <= bB; ++by)
      for (int bx = bL; bx <= bR; ++bx)
        o.keys[w++] = ((uint32_t)(by*BIN_COLS+bx) << PRIM_BITS) | (uint32_t)i;
  }
  radixLSD(o.keys);
  for (uint32_t k : o.keys) o.bins[(int)(k>>PRIM_BITS)].push_back(k & PRIM_MASK);
  return o;
}

// per-pixel draw-ordered prim list, reference (gfx-v1: prims in submission order).
static std::map<uint32_t, std::vector<uint32_t>>
refPixelLists(const std::vector<std::vector<uint32_t>>& samples) {
  std::map<uint32_t, std::vector<uint32_t>> px;
  for (uint32_t pid = 0; pid < samples.size(); ++pid)
    for (uint32_t s : samples[pid]) px[s].push_back(pid);
  return px;
}

// per-pixel draw-ordered prim list, via the bin-sort output. A pixel lives in
// exactly one bin; every prim covering it is in that bin's list (covers pixel
// ⟹ covers bin), processed in the list's draw order.
static std::map<uint32_t, std::vector<uint32_t>>
binSortPixelLists(const BinSortOut& o,
                  const std::vector<std::vector<uint32_t>>& samples) {
  std::map<uint32_t, std::vector<uint32_t>> px;
  for (auto& [bin_id, plist] : o.bins)
    for (uint32_t pid : plist)
      for (uint32_t s : samples[pid])
        if (binOfPixel(s) == bin_id) px[s].push_back(pid);
  return px;
}

static std::vector<Prim> makeScene(unsigned seed, int n) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> px(-48, W+48), py(-48, H+48);
  std::uniform_real_distribution<float> sz(2, 150);
  std::vector<Prim> prims; prims.reserve(n);
  while ((int)prims.size() < n) {
    float cx = px(rng), cy = py(rng), s = sz(rng);
    std::uniform_real_distribution<float> j(-s, s);
    Prim p;
    setupPrim(p, {cx+j(rng),cy+j(rng)}, {cx+j(rng),cy+j(rng)}, {cx+j(rng),cy+j(rng)});
    prims.push_back(p);   // keep invalid (culled) ones too
  }
  return prims;
}

int main() {
  int scenes = 12, prims_per = 300;
  int t_cov = 0, t_det = 0, t_radix = 0;    // failure counters
  long binsortKeys = 0, hostPairs = 0, totalCovered = 0, totalPrims = 0, valid = 0;

  // validate the LSD radix primitive directly against std::sort on random data
  {
    std::mt19937 rng(99);
    for (int it = 0; it < 8; ++it) {
      int n = 1 + (int)(rng() % 5000);
      std::vector<uint32_t> a(n);
      for (auto& v : a) v = rng();
      auto b = a;
      radixLSD(a);
      std::sort(b.begin(), b.end());
      if (a != b) ++t_radix;
    }
  }

  for (int sc = 0; sc < scenes; ++sc) {
    auto prims = makeScene(1234 + sc, prims_per);
    totalPrims += (long)prims.size();
    std::vector<std::vector<uint32_t>> samples(prims.size());
    for (size_t i = 0; i < prims.size(); ++i) {
      samples[i] = coveredSamples(prims[i]);
      if (prims[i].valid) ++valid;
      totalCovered += (long)samples[i].size();
    }

    auto bs = binSort(prims);
    auto ref = refPixelLists(samples);
    auto got = binSortPixelLists(bs, samples);

    // Test: per-pixel ordered coverage identical (the rendering-equivalence proof)
    if (ref != got) {
      ++t_cov;
      if (t_cov <= 5) {
        size_t shown = 0;
        for (auto& [pix, l] : ref) {
          auto it = got.find(pix);
          if (it == got.end() || it->second != l) {
            printf("  [COV] scene %d pixel (%u,%u): ref=%zu binsort=%zu\n",
                   sc, pix % W, pix / W, l.size(),
                   it == got.end() ? 0 : it->second.size());
            if (++shown >= 3) break;
          }
        }
      }
    }

    // Determinism
    auto bs2 = binSort(prims);
    if (bs.keys != bs2.keys) ++t_det;

    // Footprint: 128px keys vs gfx-v1 32px (prim,tile) pairs
    binsortKeys += (long)bs.keys.size();
    for (auto& p : prims) {
      if (!p.valid) continue;
      int tL=p.bbL>>TILE_LOG, tR=(p.bbR-1)>>TILE_LOG, tT=p.bbT>>TILE_LOG, tB=(p.bbB-1)>>TILE_LOG;
      hostPairs += (long)(tR-tL+1)*(tB-tT+1);
    }
  }

  printf("\n=== gfx_v2 bin-sort validation (ground truth: per covered sample) ===\n");
  printf("scenes=%d  prims/scene=%d  total=%ld (valid=%ld)  covered samples=%ld\n",
         scenes, prims_per, totalPrims, valid, totalCovered);
  printf("Test COVERAGE  per-pixel draw-ordered lists (bin-sort == gfx-v1): %s (%d/%d scenes failed)\n",
         t_cov==0?"PASS":"FAIL", t_cov, scenes);
  printf("Test DETERMINISM  identical keys on re-run: %s (%d/%d failed)\n",
         t_det==0?"PASS":"FAIL", t_det, scenes);
  printf("Test RADIX        LSD radix == std::sort on random data: %s (%d/8 failed)\n",
         t_radix==0?"PASS":"FAIL", t_radix);
  printf("Footprint  bin-sort 128px keys=%ld  vs  gfx-v1 32px (prim,tile) pairs=%ld  -> %.2fx fewer\n",
         binsortKeys, hostPairs, binsortKeys ? (double)hostPairs/(double)binsortKeys : 0.0);
  int fails = t_cov + t_det + t_radix;
  printf("RESULT: %s\n", fails==0 ? "PASS" : "FAIL");
  return fails==0 ? 0 : 1;
}
