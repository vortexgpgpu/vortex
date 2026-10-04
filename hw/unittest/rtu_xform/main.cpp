// VX_rtu_xform against SimX's rtu::world_to_object_ray: the object-space origin
// and direction must match bit for bit (any NaN matches any NaN) for random
// affine instances -- rotation, non-uniform scale, shear, translation -- and
// special operands.
//
// The xform's FP units flush subnormals (FTZ/DAZ) while SimX runs IEEE. A case
// whose RTL result differs from SimX but equals SimX evaluated under the host's
// FTZ/DAZ mode, and only there, is counted as a subnormal flush, not a mismatch.

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <random>
#include <vector>
#include <xmmintrin.h>
#include "VVX_rtu_xform.h"
#include "verilated.h"
#include "rtu_isect.h"

namespace {

struct Case {
  float m[12], o[3], d[3];
  int cat = 0;
};

const char* const kCatNames[] = {
  "random affine", "rigid", "identity", "special matrix", "special ray",
};
constexpr int kNumCats = int(sizeof(kCatNames) / sizeof(kCatNames[0]));

struct Result { float o[3], d[3]; };

uint32_t bits(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }

bool same(uint32_t rtl, float ref) {
  const bool rtl_nan = ((rtl >> 23) & 0xff) == 0xff && (rtl & 0x7fffff) != 0;
  return std::isnan(ref) ? rtl_nan : rtl == bits(ref);
}

Result reference(const Case& c, bool ftz) {
  const unsigned csr = _mm_getcsr();
  if (ftz) _mm_setcsr(csr | 0x8040);   // FTZ | DAZ
  Result r;
  vortex::rtu::world_to_object_ray(c.m, c.o, c.d, r.o, r.d);
  _mm_setcsr(csr);
  return r;
}

template <typename Dut>
bool matches(const Dut& dut, const Result& r) {
  for (int a = 0; a < 3; ++a)
    if (!same(dut.obj_ro[a], r.o[a]) || !same(dut.obj_rd[a], r.d[a])) return false;
  return true;
}

bool same_result(const Result& a, const Result& b) {
  for (int k = 0; k < 3; ++k)
    if (!same(bits(a.o[k]), b.o[k]) || !same(bits(a.d[k]), b.d[k])) return false;
  return true;
}

std::mt19937 rng(5);
float uni(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng); }

// A random world->object matrix: rotation x scale x shear (or rigid), plus a
// translation, at assorted magnitudes.
Case make_case(uint32_t i) {
  Case c;
  const float ax = uni(-1, 1), ay = uni(-1, 1), az = uni(-1, 1);
  const float n = std::sqrt(ax * ax + ay * ay + az * az) + 1e-6f;
  const float x = ax / n, y = ay / n, z = az / n, th = uni(-3.2f, 3.2f);
  const float cs = std::cos(th), sn = std::sin(th), t = 1 - cs;
  float R[9] = { t * x * x + cs,     t * x * y - sn * z, t * x * z + sn * y,
                 t * x * y + sn * z, t * y * y + cs,     t * y * z - sn * x,
                 t * x * z - sn * y, t * y * z + sn * x, t * z * z + cs };
  c.cat = (i % 5 == 0) ? 1 : 0;
  float S[3] = { 1, 1, 1 }, H[3] = { 0, 0, 0 };
  if (c.cat == 0) {
    for (float& s : S) s = (i % 3 == 0) ? uni(1e-3f, 1e3f) : uni(0.1f, 10.f);
    for (float& h : H) h = uni(-2, 2);
  }
  const float M[9] = { S[0], H[0], H[1], 0, S[1], H[2], 0, 0, S[2] };
  for (int r = 0; r < 3; ++r)
    for (int k = 0; k < 3; ++k)
      c.m[r * 4 + k] = R[r * 3 + 0] * M[0 * 3 + k] + R[r * 3 + 1] * M[1 * 3 + k]
                     + R[r * 3 + 2] * M[2 * 3 + k];
  const float span = (i % 7 == 0) ? 1e4f : 50.f;
  for (int r = 0; r < 3; ++r) c.m[r * 4 + 3] = uni(-span, span);
  for (int a = 0; a < 3; ++a) { c.o[a] = uni(-span, span); c.d[a] = uni(-1, 1); }
  if (i % 11 == 0) c.d[i % 3] = (i & 1) ? -0.f : 0.f;
  return c;
}

std::vector<Case> directed_cases() {
  std::vector<Case> out;
  const float specials[] = { NAN, INFINITY, -INFINITY, 0.f, -0.f, 1e-40f, -1e-40f,
                             FLT_MAX, -FLT_MAX };
  for (uint32_t i = 0; i < 2000; ++i) {
    Case c = make_case(i * 3 + 1);
    Case id = c;
    id.cat = 2;
    for (int k = 0; k < 12; ++k) id.m[k] = (k % 5 == 0) ? 1.f : 0.f;
    out.push_back(id);
    for (float sp : specials) {
      Case s = c;
      s.cat = 3; s.m[i % 12] = sp; out.push_back(s);
      s = c; s.cat = 4;
      if (i & 1) s.o[i % 3] = sp; else s.d[i % 3] = sp;
      out.push_back(s);
    }
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  const uint32_t N = (argc > 1) ? uint32_t(std::atoi(argv[1])) : 1000000;
  VVX_rtu_xform dut;
  dut.clk = 0; dut.reset = 1; dut.enable = 1; dut.valid_in = 0;
  auto tick = [&] { dut.clk = 0; dut.eval(); dut.clk = 1; dut.eval(); };
  for (int i = 0; i < 4; ++i) tick();
  dut.reset = 0;

  const std::vector<Case> directed = directed_cases();
  const uint32_t ND = uint32_t(directed.size());
  const uint32_t total = N + ND;
  struct Expect { uint32_t id; int cat; Result ieee, ftz; };
  std::deque<Expect> exp;
  std::deque<Case> cases;
  uint32_t sent = 0, checked = 0, errors = 0, flushed = 0;
  uint32_t cat_cases[kNumCats] = {}, cat_err[kNumCats] = {}, cat_ftz[kNumCats] = {};
  while (checked < total) {
    if (sent < total) {
      const Case c = (sent < ND) ? directed[sent] : make_case(sent - ND);
      for (int k = 0; k < 12; ++k) dut.xform[k] = bits(c.m[k]);
      for (int a = 0; a < 3; ++a) { dut.ro[a] = bits(c.o[a]); dut.rd[a] = bits(c.d[a]); }
      dut.tag_in = sent;
      dut.valid_in = 1;
      exp.push_back({ sent, c.cat, reference(c, false), reference(c, true) });
      cases.push_back(c);
      ++sent;
    } else {
      dut.valid_in = 0;
    }
    tick();
    if (dut.valid_out) {
      const Expect e = exp.front(); exp.pop_front();
      const Case c = cases.front(); cases.pop_front();
      ++cat_cases[e.cat];
      bool ok = (dut.tag_out == e.id) && matches(dut, e.ieee);
      if (!ok && dut.tag_out == e.id && matches(dut, e.ftz) && !same_result(e.ieee, e.ftz)) {
        ok = true; ++flushed; ++cat_ftz[e.cat];
      }
      if (!ok) {
        ++errors;
        if (cat_err[e.cat]++ < 3) {
          std::printf("MISMATCH #%u [%s] m=(", e.id, kCatNames[e.cat]);
          for (float v : c.m) std::printf("%a ", v);
          std::printf(") o=(%a %a %a) d=(%a %a %a)\n", c.o[0], c.o[1], c.o[2], c.d[0], c.d[1], c.d[2]);
          std::printf("  ref o=(%08x %08x %08x) d=(%08x %08x %08x) | rtl o=(%08x %08x %08x) d=(%08x %08x %08x)\n",
                      bits(e.ieee.o[0]), bits(e.ieee.o[1]), bits(e.ieee.o[2]),
                      bits(e.ieee.d[0]), bits(e.ieee.d[1]), bits(e.ieee.d[2]),
                      dut.obj_ro[0], dut.obj_ro[1], dut.obj_ro[2],
                      dut.obj_rd[0], dut.obj_rd[1], dut.obj_rd[2]);
        }
      }
      ++checked;
    }
  }
  std::printf("rtu_xform: %u cases (%u directed), %u mismatches, %u subnormal flushes\n",
              checked, ND, errors, flushed);
  for (int k = 0; k < kNumCats; ++k)
    std::printf("  %-16s %8u cases %6u mismatches %6u subnormal flushes\n",
                kCatNames[k], cat_cases[k], cat_err[k], cat_ftz[k]);
  std::printf(errors ? "FAILED!\n" : "PASSED!\n");
  return errors ? 1 : 0;
}
