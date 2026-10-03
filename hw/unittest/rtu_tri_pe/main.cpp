// VX_rtu_tri_pe against SimX's rtu::ray_triangle: every verdict and, for a hit,
// t / u / v / back_facing must match bit for bit.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <random>
#include "VVX_rtu_tri_pe.h"
#include "verilated.h"
#include "rtu_isect.h"

namespace {

struct Case {
  float o[3], d[3], v[3][3], tmin, tmax;
};

struct Expect {
  uint32_t id;
  bool hit;
  float t, u, v;
  bool back;
};

uint32_t bits(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }

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
  Case prev{};
  while (checked < N) {
    if (sent < N) {
      const bool twin = (sent > 0) && (sent % 4 == 1);
      Case c = make_case(sent, twin ? &prev : nullptr);
      if (!twin) prev = c; else ++twins;
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
      Expect e{sent, false, 0, 0, 0, false};
      e.hit = vortex::rtu::ray_triangle(c.o, c.d, c.v[0], c.v[1], c.v[2],
                                        c.tmin, c.tmax, e.t, e.u, e.v, e.back);
      exp.push_back(e);
      ++sent;
    } else {
      dut.valid_in = 0;
    }
    tick();
    if (dut.valid_out) {
      const Expect e = exp.front();
      exp.pop_front();
      bool ok = (dut.tag_out == e.id) && (bool(dut.hit) == e.hit);
      if (ok && e.hit) {
        ok = dut.t == bits(e.t) && dut.u == bits(e.u) && dut.v == bits(e.v)
          && bool(dut.back_facing) == e.back;
        ++hits;
      }
      if (!ok && errors++ < 10) {
        std::printf("MISMATCH #%u: ref hit=%d t=%08x u=%08x v=%08x bf=%d | rtl tag=%u hit=%d t=%08x u=%08x v=%08x bf=%d\n",
                    e.id, e.hit, bits(e.t), bits(e.u), bits(e.v), e.back,
                    dut.tag_out, dut.hit, dut.t, dut.u, dut.v, dut.back_facing);
      }
      ++checked;
    }
  }
  std::printf("rtu_tri_pe: %u cases (%u hits, %u twins), %u mismatches\n", checked, hits, twins, errors);
  std::printf(errors ? "FAILED!\n" : "PASSED!\n");
  return errors ? 1 : 0;
}
