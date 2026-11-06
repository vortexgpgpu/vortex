// VX_tex_lerp against the sampler's C model (vx_gfx_abi.h Lerp8888 for the
// bilinear taps, gfx_frag_tex.h TexLodLerp for the level blend) and against the
// exact Vulkan blend a*(1-f/256) + b*f/256, over every (in1, in2, frac).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include "VVX_tex_lerp_top.h"
#include "verilated.h"
#include <gfx_frag_tex.h>

static VVX_tex_lerp_top* dut;

static void clock_cycle() {
  dut->clk = 0; dut->eval();
  dut->clk = 1; dut->eval();
}

struct Vec { uint32_t a, b, f; };

static int errors = 0;

static void check(const Vec& v, uint32_t rnd, uint32_t trn) {
  using vortex::graphics::Lerp8888;
  // the packed model, in both lanes it blends
  uint32_t m_lo = Lerp8888(v.a, v.b, v.f) & 0xff;
  uint32_t m_hi = (Lerp8888(v.a << 16, v.b << 16, v.f) >> 16) & 0xff;
  uint32_t m_lod = gfx_tex::TexLodLerp(v.a, v.b, v.f) & 0xff;
  // exact blend scaled by 256
  int32_t e256 = (int32_t)(v.a * (256 - v.f) + v.b * v.f);
  bool rnd_exact = 2 * std::abs((int32_t)rnd * 256 - e256) <= 256;
  bool trn_exact = (int32_t)trn * 256 <= e256 && e256 < ((int32_t)trn + 1) * 256;
  if (rnd != m_lo || rnd != m_hi || trn != m_lod || !rnd_exact || !trn_exact) {
    if (errors < 16)
      std::printf("MISMATCH a=%u b=%u f=%u: rtl round=%u trunc=%u, model lo=%u hi=%u lod=%u, exact=%.4f\n",
                  v.a, v.b, v.f, rnd, trn, m_lo, m_hi, m_lod, e256 / 256.0);
    ++errors;
  }
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  dut = new VVX_tex_lerp_top;
  dut->reset = 1;
  dut->enable = 1;
  dut->in1 = dut->in2 = dut->frac = 0;
  for (int i = 0; i < 4; ++i) clock_cycle();
  dut->reset = 0;

  const uint32_t LATENCY = 3;
  std::deque<Vec> inflight;
  uint64_t checked = 0;

  // A stall cycle (enable low) must hold every stage; the results then still
  // pair with their inputs after LATENCY enabled cycles.
  srand(1);
  auto step = [&](const Vec& v) {
    if ((rand() % 16) == 0) {
      dut->enable = 0;
      clock_cycle();
    }
    dut->enable = 1;
    dut->in1 = v.a; dut->in2 = v.b; dut->frac = v.f;
    inflight.push_back(v);
    clock_cycle();
    if (inflight.size() == LATENCY) {
      check(inflight.front(), dut->out_round, dut->out_trunc);
      inflight.pop_front();
      ++checked;
    }
  };

  // directed: pass-through at f=0, full-weight end of the range, the half
  // point where round and truncate split
  const Vec directed[] = {
    {0, 255, 0}, {255, 0, 0}, {0, 255, 255}, {255, 0, 255}, {255, 255, 255},
    {8, 10, 128}, {0, 1, 128}, {1, 0, 128}, {200, 100, 1}, {100, 200, 254},
  };
  for (auto& v : directed) step(v);

  // exhaustive
  for (uint32_t f = 0; f < 256; ++f)
    for (uint32_t a = 0; a < 256; ++a)
      for (uint32_t b = 0; b < 256; ++b) {
        step({a, b, f});
      }

  for (uint32_t i = 1; i < LATENCY; ++i) step({0, 0, 0});  // drain
  dut->final();
  delete dut;

  if (errors) {
    std::printf("FAILED: %d mismatches over %llu vectors\n", errors, (unsigned long long)checked);
    return 1;
  }
  std::printf("PASSED: %llu vectors\n", (unsigned long long)checked);
  return 0;
}
