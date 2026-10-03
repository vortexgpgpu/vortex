// VX_rtu_tri_pe against SimX's rtu::ray_triangle: every verdict and, for a hit,
// t / u / v / back_facing must match bit for bit (any NaN matches any NaN).
//
// The PE's FP units flush subnormals (FTZ/DAZ) while SimX runs IEEE. A case
// whose RTL result differs from SimX but equals SimX evaluated under the host's
// FTZ/DAZ mode, and only there, is counted as a subnormal flush, not a mismatch.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <random>
#include <vector>
#include <cmath>
#include <xmmintrin.h>
#include "VVX_rtu_tri_pe.h"
#include "verilated.h"
#include "rtu_isect.h"

namespace {

struct Case {
  float o[3], d[3], v[3][3], tmin, tmax;
  int cat = 0;   // directed-case family (0: random)
};

const char* const kCatNames[] = {
  "random", "t==tmin", "t==tmax", "t==tmin==tmax", "t in (t-,t+)",
  "tmin=t+", "tmax=t-", "origin on triangle", "zero dir",
  "special origin", "special dir", "special vertex", "special tmin", "special tmax",
  "shared edge",
};
constexpr int kNumCats = int(sizeof(kCatNames) / sizeof(kCatNames[0]));

struct Result {
  bool hit = false;
  float t = 0, u = 0, v = 0;
  bool back = false;
};

struct Expect {
  uint32_t id;
  int cat;
  Result ieee, ftz;
};

uint32_t bits(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }

// bit-exact, except that any NaN matches any NaN (the payload is not specified)
bool same(uint32_t rtl, float ref) {
  const bool rtl_nan = ((rtl >> 23) & 0xff) == 0xff && (rtl & 0x7fffff) != 0;
  return std::isnan(ref) ? rtl_nan : rtl == bits(ref);
}

Result reference(const Case& c, bool ftz) {
  const unsigned csr = _mm_getcsr();
  if (ftz) _mm_setcsr(csr | 0x8040);   // FTZ | DAZ
  Result r;
  r.hit = vortex::rtu::ray_triangle(c.o, c.d, c.v[0], c.v[1], c.v[2],
                                    c.tmin, c.tmax, r.t, r.u, r.v, r.back);
  _mm_setcsr(csr);
  return r;
}

template <typename Dut>
bool matches(const Dut& dut, const Result& r) {
  if (bool(dut.hit) != r.hit) return false;
  return !r.hit || (same(dut.t, r.t) && same(dut.u, r.u) && same(dut.v, r.v)
                    && bool(dut.back_facing) == r.back);
}

bool same_result(const Result& a, const Result& b) {
  return a.hit == b.hit
      && (!a.hit || (same(bits(a.t), b.t) && same(bits(a.u), b.u)
                     && same(bits(a.v), b.v) && a.back == b.back));
}

std::mt19937 rng(7);
float uni(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng); }

// A ray aimed at a point inside (or just outside) the triangle, so hits,
// near-edge misses and shared-edge cases all get exercised.
Case make_case(uint32_t i, const Case* twin_of) {
  Case c;
  if (twin_of) {
    c = *twin_of;
    // coincident twin: rotate (even) or mirror (odd) the vertex order
    float v[3][3];
    std::memcpy(v, c.v, sizeof v);
    if (i & 1) { std::memcpy(c.v[1], v[2], 12); std::memcpy(c.v[2], v[1], 12); }
    else { std::memcpy(c.v[0], v[1], 12); std::memcpy(c.v[1], v[2], 12); std::memcpy(c.v[2], v[0], 12); }
    return c;
  }
  const float s = (i % 7 == 0) ? 1e3f : ((i % 5 == 0) ? 1e-2f : 10.f);
  for (auto& vv : c.v) for (float& x : vv) x = uni(-s, s);
  float a = uni(-0.2f, 1.2f), b = uni(-0.2f, 1.2f);
  if ((i % 3) == 0) b = 1.f - a;                      // on / near an edge
  float p[3];
  for (int k = 0; k < 3; ++k)
    p[k] = c.v[0][k] + a * (c.v[1][k] - c.v[0][k]) + b * (c.v[2][k] - c.v[0][k]);
  for (int k = 0; k < 3; ++k) c.o[k] = uni(-3 * s, 3 * s);
  for (int k = 0; k < 3; ++k) c.d[k] = p[k] - c.o[k];
  if ((i % 11) == 0) c.d[i % 3] = 0.f;                // axis-aligned component
  c.tmin = (i % 13 == 0) ? uni(0.f, 1.f) : 0.f;
  c.tmax = (i % 17 == 0) ? uni(0.f, 2.f) : 1e30f;
  return c;
}

// Quad (p0, p1, p2, p3) split along p0-p2 into (p0, p1, p2) and (p0, p2, p3),
// both wound alike; the ray aims at a point on the shared edge p0-p2.
float quad_pts[20000][4][3];

Case shared_edge_case(uint32_t i) {
  std::mt19937 g(1000 + i);
  auto u = [&](float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(g); };
  float (&p)[4][3] = quad_pts[i];
  const float s = (i % 3 == 0) ? 1e3f : ((i % 3 == 1) ? 1.f : 1e-2f);
  for (int k = 0; k < 3; ++k) { p[0][k] = u(-s, s); p[2][k] = u(-s, s); }
  float m[3], n[3];
  for (int k = 0; k < 3; ++k) { m[k] = 0.5f * (p[0][k] + p[2][k]); n[k] = u(-s, s); }
  for (int k = 0; k < 3; ++k) { p[1][k] = m[k] + n[k]; p[3][k] = m[k] - n[k]; }
  Case c;
  c.cat = 14;
  std::memcpy(c.v[0], p[0], 12);
  std::memcpy(c.v[1], p[1], 12);
  std::memcpy(c.v[2], p[2], 12);
  const float a = u(0.f, 1.f);
  float e[3];
  for (int k = 0; k < 3; ++k) e[k] = p[0][k] + a * (p[2][k] - p[0][k]);
  for (int k = 0; k < 3; ++k) c.o[k] = u(-3 * s, 3 * s);
  for (int k = 0; k < 3; ++k) c.d[k] = e[k] - c.o[k];
  c.tmin = 0.f;
  c.tmax = INFINITY;
  return c;
}

const float* shared_edge_far(uint32_t i) { return quad_pts[i][3]; }

// Directed cases: t landing exactly on tmin / tmax, a zero t from an origin on
// the triangle, and NaN / inf / -0 / subnormal operands.
std::vector<Case> directed_cases() {
  std::vector<Case> out;
  const float inf = INFINITY, nan = NAN;
  const float sub = 1e-40f;
  for (uint32_t i = 0; i < 4000; ++i) {
    Case c = make_case(i * 3 + 1, nullptr);   // edge-biased family excluded
    c.tmin = 0.f; c.tmax = 1e30f;
    float t, u, v; bool bf;
    if (!vortex::rtu::ray_triangle(c.o, c.d, c.v[0], c.v[1], c.v[2],
                                   -inf, inf, t, u, v, bf))
      continue;
    Case e = c;
    e.cat = 1; e.tmin = t; e.tmax = inf;              out.push_back(e);
    e.cat = 2; e.tmin = -inf; e.tmax = t;             out.push_back(e);
    e.cat = 3; e.tmin = t; e.tmax = t;                out.push_back(e);
    e.cat = 4; e.tmin = std::nextafter(t, -inf); e.tmax = std::nextafter(t, inf);
    out.push_back(e);
    e.cat = 5; e.tmin = std::nextafter(t, inf); e.tmax = inf;  out.push_back(e);
    e.cat = 6; e.tmin = -inf; e.tmax = std::nextafter(t, -inf); out.push_back(e);
  }
  // origin on the triangle's plane: every rz is 0, so t is exactly +-0
  for (uint32_t i = 0; i < 2000; ++i) {
    Case c;
    c.cat = 7;
    for (auto& vv : c.v) { vv[0] = uni(-5, 5); vv[1] = uni(-5, 5); vv[2] = 0.f; }
    const float a = uni(0.f, 0.5f), b = uni(0.f, 0.5f);
    for (int k = 0; k < 2; ++k)
      c.o[k] = c.v[0][k] + a * (c.v[1][k] - c.v[0][k]) + b * (c.v[2][k] - c.v[0][k]);
    c.o[2] = (i & 1) ? -0.f : 0.f;
    c.d[0] = uni(-1, 1); c.d[1] = uni(-1, 1); c.d[2] = (i & 2) ? uni(0.5f, 2) : -uni(0.5f, 2);
    const float tmins[] = { 0.f, -0.f, -1.f, sub, -sub };
    c.tmin = tmins[i % 5];
    c.tmax = (i % 7 == 0) ? 0.f : 1e30f;
    out.push_back(c);
  }
  // special operands in every input slot
  const float specials[] = { nan, -nan, inf, -inf, 0.f, -0.f, sub, -sub, 3.4e38f };
  for (uint32_t i = 0; i < 400; ++i) {
    const Case base = make_case(i * 5 + 2, nullptr);
    for (float sp : specials) {
      for (int slot = 0; slot < 17; ++slot) {
        Case c = base;
        c.cat = (slot < 3) ? 9 : (slot < 6) ? 10 : (slot < 15) ? 11 : (slot == 15) ? 12 : 13;
        if (slot < 3)       c.o[slot] = sp;
        else if (slot < 6)  c.d[slot - 3] = sp;
        else if (slot < 15) c.v[(slot - 6) / 3][(slot - 6) % 3] = sp;
        else if (slot == 15) c.tmin = sp;
        else                 c.tmax = sp;
        out.push_back(c);
      }
    }
    Case z = base;                                      // zero direction
    z.cat = 8;
    z.d[0] = z.d[1] = z.d[2] = (i & 1) ? -0.f : 0.f;
    out.push_back(z);
  }
  // rays at the shared edge of two triangles (a quad split along its diagonal),
  // both triangles traced
  for (uint32_t i = 0; i < 20000; ++i) {
    Case c = shared_edge_case(i);
    out.push_back(c);
    std::memcpy(c.v[1], c.v[2], 12);
    std::memcpy(c.v[2], shared_edge_far(i), 12);
    out.push_back(c);
  }
  return out;
}

// Watertightness: a ray through a shared edge hits at least one of the two
// triangles (SimX alone; the RTL then matches it case by case).
uint32_t shared_edge_leaks() {
  uint32_t leaks = 0;
  for (uint32_t i = 0; i < 20000; ++i) {
    Case a = shared_edge_case(i), b = a;
    std::memcpy(b.v[1], b.v[2], 12);
    std::memcpy(b.v[2], shared_edge_far(i), 12);
    float t, u, v; bool bf;
    const bool ha = vortex::rtu::ray_triangle(a.o, a.d, a.v[0], a.v[1], a.v[2],
                                              a.tmin, a.tmax, t, u, v, bf);
    const bool hb = vortex::rtu::ray_triangle(b.o, b.d, b.v[0], b.v[1], b.v[2],
                                              b.tmin, b.tmax, t, u, v, bf);
    leaks += !(ha || hb);
  }
  return leaks;
}

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  const uint32_t N = (argc > 1) ? uint32_t(std::atoi(argv[1])) : 1000000;
  VVX_rtu_tri_pe dut;
  dut.clk = 0; dut.reset = 1; dut.enable = 1; dut.valid_in = 0;
  auto tick = [&] { dut.clk = 0; dut.eval(); dut.clk = 1; dut.eval(); };
  for (int i = 0; i < 4; ++i) tick();
  dut.reset = 0;

  std::deque<Expect> exp;
  uint32_t sent = 0, checked = 0, errors = 0, hits = 0, twins = 0;
  uint32_t cat_cases[kNumCats] = {}, cat_errors[kNumCats] = {}, cat_ftz[kNumCats] = {};
  uint32_t flushed = 0;
  std::deque<Case> sent_cases;
  Case prev{};
  const std::vector<Case> directed = directed_cases();
  const uint32_t leaks = shared_edge_leaks();
  const uint32_t ND = uint32_t(directed.size());
  const uint32_t total = N + ND;
  while (checked < total) {
    if (sent < total) {
      Case c;
      if (sent < ND) {
        c = directed[sent];
      } else {
        const uint32_t r = sent - ND;
        const bool twin = (r > 0) && (r % 4 == 1);
        c = make_case(r, twin ? &prev : nullptr);
        if (!twin) prev = c; else ++twins;
      }
      for (int k = 0; k < 3; ++k) {
        dut.origin[k] = bits(c.o[k]);
        dut.dir[k]    = bits(c.d[k]);
        dut.v0[k] = bits(c.v[0][k]);
        dut.v1[k] = bits(c.v[1][k]);
        dut.v2[k] = bits(c.v[2][k]);
      }
      dut.t_min = bits(c.tmin);
      dut.t_max = bits(c.tmax);
      dut.tag_in = sent;
      dut.valid_in = 1;
      Expect e{sent, c.cat, reference(c, false), reference(c, true)};
      exp.push_back(e);
      sent_cases.push_back(c);
      ++sent;
    } else {
      dut.valid_in = 0;
    }
    tick();
    if (dut.valid_out) {
      const Expect e = exp.front();
      exp.pop_front();
      const Case c = sent_cases.front();
      sent_cases.pop_front();
      ++cat_cases[e.cat];
      bool ok = (dut.tag_out == e.id) && matches(dut, e.ieee);
      if (e.ieee.hit) ++hits;
      if (!ok && dut.tag_out == e.id && matches(dut, e.ftz) && !same_result(e.ieee, e.ftz)) {
        ++cat_ftz[e.cat];
        ++flushed;
        ok = true;
      }
      if (!ok) { ++cat_errors[e.cat]; ++errors; }
      if (!ok && cat_errors[e.cat] <= 3) {
        std::printf("  [%s] o=(%a %a %a) d=(%a %a %a) v0=(%a %a %a) v1=(%a %a %a) v2=(%a %a %a) tmin=%a tmax=%a\n",
                    kCatNames[e.cat], c.o[0], c.o[1], c.o[2], c.d[0], c.d[1], c.d[2],
                    c.v[0][0], c.v[0][1], c.v[0][2], c.v[1][0], c.v[1][1], c.v[1][2],
                    c.v[2][0], c.v[2][1], c.v[2][2], c.tmin, c.tmax);
        std::printf("MISMATCH #%u: ref hit=%d t=%08x u=%08x v=%08x bf=%d | rtl tag=%u hit=%d t=%08x u=%08x v=%08x bf=%d\n",
                    e.id, e.ieee.hit, bits(e.ieee.t), bits(e.ieee.u), bits(e.ieee.v), e.ieee.back,
                    dut.tag_out, dut.hit, dut.t, dut.u, dut.v, dut.back_facing);
      }
      ++checked;
    }
  }
  std::printf("rtu_tri_pe: %u cases (%u directed, %u hits, %u twins), %u mismatches, %u subnormal flushes\n",
              checked, ND, hits, twins, errors, flushed);
  for (int k = 0; k < kNumCats; ++k)
    std::printf("  %-20s %8u cases %6u mismatches %6u subnormal flushes\n",
                kCatNames[k], cat_cases[k], cat_errors[k], cat_ftz[k]);
  std::printf("  shared-edge rays through neither triangle: %u of 20000\n", leaks);
  const bool fail = errors || leaks;
  std::printf(fail ? "FAILED!\n" : "PASSED!\n");
  return fail ? 1 : 0;
}
