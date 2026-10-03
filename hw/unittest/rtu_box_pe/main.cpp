// VX_rtu_recip + VX_rtu_box_pe against SimX's box test (reconstruct_child_aabb
// + rtu::ray_aabb_intersect): inv_d bit for bit, every accept decision, t_near
// by value, and the order the scheduler's insertion collector gives a node's
// accepted children against SimX's nearest-first sort.
//
// The PE's FP units flush subnormals (FTZ/DAZ) while SimX runs IEEE. A case
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
#include "VVX_rtu_box_pe_tb.h"
#include "verilated.h"
#include "rtu_isect.h"

namespace {

constexpr int kMaxChildren = 6;

struct Ray {
  float o[3], d[3], tmin, tmax;
};

struct Child {
  uint8_t qmin[3], qmax[3];
  float rmin[3], rmax[3];
};

struct Node {
  Ray ray;
  float origin[3];
  int8_t exp[3];
  bool raw = false;
  int n = 0;
  Child ch[kMaxChildren];
  int cat = 0;
};

const char* const kCatNames[] = {
  "random", "random raw", "zero dir on slab", "zero dir off slab", "flat box",
  "tmin past exit", "tmax == entry", "tmax < entry", "exit == 0", "-0 operand",
  "special origin", "special ro", "special dir", "special raw", "special tmin",
  "special tmax", "exp extremes",
};
constexpr int kNumCats = int(sizeof(kCatNames) / sizeof(kCatNames[0]));

uint32_t bits(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }
float fbits(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }

bool is_nan_bits(uint32_t b) { return ((b >> 23) & 0xff) == 0xff && (b & 0x7fffff) != 0; }

// bit-exact, except that any NaN matches any NaN (the payload is not specified)
bool same_bits(uint32_t rtl, float ref) {
  return std::isnan(ref) ? is_nan_bits(rtl) : rtl == bits(ref);
}

// IEEE value equality (+0 == -0), any NaN matches any NaN
bool same_value(uint32_t rtl, float ref) {
  return std::isnan(ref) ? is_nan_bits(rtl) : fbits(rtl) == ref;
}

// SimX rtu_walker.cpp reconstruct_child_aabb
void reconstruct_child_aabb(const float origin[3], const int8_t exp[3],
                            const uint8_t qmin[3], const uint8_t qmax[3],
                            float out_mn[3], float out_mx[3]) {
  for (int i = 0; i < 3; ++i) {
    float scale = std::ldexp(1.0f, exp[i]);
    out_mn[i] = origin[i] + static_cast<float>(qmin[i]) * scale;
    out_mx[i] = origin[i] + static_cast<float>(qmax[i]) * scale;
  }
}

struct BoxResult {
  bool hit = false;
  float t_near = 0;
};

struct Ref {
  float inv[3];
  BoxResult box;
};

Ref reference(const Node& nd, int c, bool ftz) {
  const unsigned csr = _mm_getcsr();
  if (ftz) _mm_setcsr(csr | 0x8040);   // FTZ | DAZ
  Ref r;
  for (int i = 0; i < 3; ++i)
    r.inv[i] = (nd.ray.d[i] == 0.0f) ? FLT_MAX : 1.0f / nd.ray.d[i];
  float mn[3], mx[3];
  if (nd.raw) {
    std::memcpy(mn, nd.ch[c].rmin, sizeof mn);
    std::memcpy(mx, nd.ch[c].rmax, sizeof mx);
  } else {
    reconstruct_child_aabb(nd.origin, nd.exp, nd.ch[c].qmin, nd.ch[c].qmax, mn, mx);
  }
  r.box.hit = vortex::rtu::ray_aabb_intersect(nd.ray.o, nd.ray.d, mn, mx,
                                              nd.ray.tmin, nd.ray.tmax, r.box.t_near);
  _mm_setcsr(csr);
  return r;
}

bool same_box(bool hit, uint32_t t_near, const BoxResult& r) {
  return hit == r.hit && (!r.hit || same_value(t_near, r.t_near));
}

bool same_ref(const Ref& a, const Ref& b) {
  for (int i = 0; i < 3; ++i)
    if (!same_bits(bits(a.inv[i]), b.inv[i])) return false;
  return same_box(a.box.hit, bits(a.box.t_near), b.box);
}

// SimX walker: accepted children, insertion-sorted by t_near (stable).
std::vector<int> simx_order(const std::vector<BoxResult>& r) {
  std::vector<int> o;
  for (int i = 0; i < int(r.size()); ++i) {
    if (!r[i].hit) continue;
    size_t j = o.size();
    o.push_back(i);
    while (j > 0 && r[o[j - 1]].t_near > r[i].t_near) { o[j] = o[j - 1]; --j; }
    o[j] = i;
  }
  return o;
}

// RTL scheduler collector: results arrive in child order; each goes after every
// collected entry whose t_near is <= its own as unsigned bits.
std::vector<int> rtl_order(const std::vector<bool>& hit, const std::vector<uint32_t>& t) {
  std::vector<int> o;
  for (int i = 0; i < int(hit.size()); ++i) {
    if (!hit[i]) continue;
    size_t j = 0;
    while (j < o.size() && t[o[j]] <= t[i]) ++j;
    o.insert(o.begin() + j, i);
  }
  return o;
}

std::mt19937 rng(11);
float uni(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng); }
int irand(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); }
bool chance(int n) { return irand(0, n - 1) == 0; }

void random_children(Node& nd, int n) {
  nd.n = n;
  for (int c = 0; c < n; ++c) {
    for (int a = 0; a < 3; ++a) {
      int lo = irand(0, 255), hi = chance(10) ? lo : irand(lo, 255);
      nd.ch[c].qmin[a] = uint8_t(lo);
      nd.ch[c].qmax[a] = uint8_t(hi);
      float r0 = uni(-100, 100), r1 = chance(10) ? r0 : r0 + uni(0, 50);
      nd.ch[c].rmin[a] = r0;
      nd.ch[c].rmax[a] = r1;
    }
  }
}

// node box corner along axis a, at quantized coordinate q
float node_coord(const Node& nd, int a, float q) {
  return nd.origin[a] + q * std::ldexp(1.0f, nd.exp[a]);
}

Node make_random(uint32_t i) {
  Node nd;
  nd.raw = (i % 10 == 0);
  nd.cat = nd.raw ? 1 : 0;
  const float S = (i % 3 == 0) ? 1.f : ((i % 3 == 1) ? 100.f : 1e4f);
  for (int a = 0; a < 3; ++a) {
    nd.origin[a] = uni(-S, S);
    nd.exp[a] = int8_t(chance(50) ? irand(-128, 127) : irand(-15, 5));
  }
  random_children(nd, irand(1, kMaxChildren));
  if (chance(3)) {   // overlapping siblings, so several are accepted together
    for (int c = 0; c < nd.n; ++c)
      for (int a = 0; a < 3; ++a) {
        nd.ch[c].qmin[a] = uint8_t(irand(0, 120));
        nd.ch[c].qmax[a] = uint8_t(irand(130, 255));
        nd.ch[c].rmin[a] = uni(-100, -10);
        nd.ch[c].rmax[a] = uni(10, 100);
      }
  }
  Ray& r = nd.ray;
  // half the rays aim into one child, the rest anywhere in the node
  float target[3];
  const int aim = chance(2) ? irand(0, nd.n - 1) : -1;
  for (int a = 0; a < 3; ++a) {
    const Child& ch = nd.ch[aim < 0 ? 0 : aim];
    const float f = uni(0, 1);
    if (aim < 0)
      target[a] = nd.raw ? uni(-100, 150) : node_coord(nd, a, uni(0, 255));
    else
      target[a] = nd.raw ? ch.rmin[a] + f * (ch.rmax[a] - ch.rmin[a])
                         : node_coord(nd, a, ch.qmin[a] + f * float(ch.qmax[a] - ch.qmin[a]));
    const float ext = nd.raw ? 150.f : 255.f * std::ldexp(1.0f, nd.exp[a]);
    r.o[a] = chance(3) ? target[a] + uni(-0.3f, 0.3f) * ext   // inside / near
                       : target[a] + uni(-3.f, 3.f) * ext;
  }
  for (int a = 0; a < 3; ++a) {
    r.d[a] = target[a] - r.o[a];
    if (chance(4)) r.d[a] = uni(-1, 1);
    if (chance(8)) r.d[a] = chance(2) ? 0.f : -0.f;
  }
  r.tmin = chance(4) ? uni(0, 2) : 0.f;
  const int tm = irand(0, 3);
  r.tmax = (tm == 0) ? INFINITY : (tm == 1) ? 1e30f : uni(0, 3);
  return nd;
}

std::vector<Node> directed_nodes() {
  std::vector<Node> out;
  const float inf = INFINITY, nan = NAN, sub = 1e-40f;
  auto one = [](Node nd, int cat) { nd.n = 1; nd.cat = cat; return nd; };
  for (uint32_t i = 0; i < 3000; ++i) {
    Node base = make_random(i * 7 + 1);
    base.raw = (i % 5 == 0);
    random_children(base, 1);
    const Child& ch = base.ch[0];
    float mn[3], mx[3];
    if (base.raw) {
      std::memcpy(mn, ch.rmin, sizeof mn); std::memcpy(mx, ch.rmax, sizeof mx);
    } else {
      reconstruct_child_aabb(base.origin, base.exp, ch.qmin, ch.qmax, mn, mx);
    }
    const int a = int(i % 3);
    // zero direction component with the origin exactly on / off a slab plane
    {
      Node nd = base;
      nd.ray.d[a] = (i & 8) ? -0.f : 0.f;
      nd.ray.o[a] = (i & 16) ? mx[a] : mn[a];
      out.push_back(one(nd, 2));
      nd.ray.o[a] = (i & 16) ? std::nextafter(mx[a], inf) : std::nextafter(mn[a], -inf);
      out.push_back(one(nd, 3));
    }
    // flat box crossed by the ray
    {
      Node nd = base;
      nd.ch[0].qmax[a] = nd.ch[0].qmin[a];
      nd.ch[0].rmax[a] = nd.ch[0].rmin[a];
      for (int k = 0; k < 3; ++k)
        nd.ray.d[k] = 0.5f * ((k == a ? mn[k] : 0.5f * (mn[k] + mx[k])) - nd.ray.o[k]);
      out.push_back(one(nd, 4));
    }
    // aimed at the box centre; tmin / tmax placed on the slab interval ends
    {
      Node nd = base;
      for (int k = 0; k < 3; ++k) nd.ray.d[k] = 0.5f * (mn[k] + mx[k]) - nd.ray.o[k];
      nd.ray.tmin = 0.f; nd.ray.tmax = inf;
      Ref r = reference(nd, 0, false);
      if (r.box.hit) {
        // the slab exit: shrink tmax until the box drops out
        Node e = nd;
        e.ray.tmin = 1e30f;                                     // past the exit
        out.push_back(one(e, 5));
        e = nd; e.ray.tmax = r.box.t_near;                       // entry, tmin = 0
        out.push_back(one(e, 6));
        e.ray.tmax = std::nextafter(r.box.t_near, -inf);
        out.push_back(one(e, 7));
      }
      // ray leaving the box through the face it starts on: exit at t == 0
      Node x = nd;
      x.ray.o[a] = mx[a];
      for (int k = 0; k < 3; ++k) if (k != a) x.ray.o[k] = 0.5f * (mn[k] + mx[k]);
      x.ray.d[a] = 1.f;
      out.push_back(one(x, 8));
      x.ray.d[a] = -1.f;
      out.push_back(one(x, 8));
    }
    // -0 in each operand
    for (int slot = 0; slot < 6; ++slot) {
      Node nd = base;
      if (slot == 0) nd.origin[a] = -0.f;
      if (slot == 1) nd.ray.o[a] = -0.f;
      if (slot == 2) nd.ray.d[a] = -0.f;
      if (slot == 3) { nd.ch[0].rmin[a] = -0.f; nd.ch[0].rmax[a] = 0.f; }
      if (slot == 4) nd.ray.tmin = -0.f;
      if (slot == 5) nd.ray.tmax = -0.f;
      out.push_back(one(nd, 9));
    }
  }
  // special operands in every slot
  const float specials[] = { nan, -nan, inf, -inf, 0.f, -0.f, sub, -sub, FLT_MAX, -FLT_MAX };
  for (uint32_t i = 0; i < 300; ++i) {
    Node base = make_random(i * 13 + 2);
    random_children(base, 1);
    for (float sp : specials) {
      for (int a = 0; a < 3; ++a) {
        Node nd = base;
        nd.raw = false; nd.origin[a] = sp;       out.push_back(one(nd, 10));
        nd = base; nd.ray.o[a] = sp;             out.push_back(one(nd, 11));
        nd = base; nd.ray.d[a] = sp;             out.push_back(one(nd, 12));
        nd = base; nd.raw = true; nd.ch[0].rmin[a] = sp; out.push_back(one(nd, 13));
        nd = base; nd.raw = true; nd.ch[0].rmax[a] = sp; out.push_back(one(nd, 13));
      }
      Node nd = base;
      nd.ray.tmin = sp;                          out.push_back(one(nd, 14));
      nd = base; nd.ray.tmax = sp;               out.push_back(one(nd, 15));
    }
  }
  // dequant exponents at and past the F32 range
  const int exps[] = { -128, -127, -126, -125, -120, 100, 120, 126, 127 };
  const int qs[] = { 0, 1, 2, 3, 128, 255 };
  for (uint32_t i = 0; i < 200; ++i) {
    Node base = make_random(i * 17 + 3);
    base.raw = false;
    random_children(base, 1);
    for (int e : exps) {
      for (int q : qs) {
        Node nd = base;
        const int a = int(i % 3);
        nd.exp[a] = int8_t(e);
        nd.ch[0].qmin[a] = uint8_t(i & 1 ? 0 : q);
        nd.ch[0].qmax[a] = uint8_t(q);
        if (i & 2) nd.origin[a] = (i & 4) ? -FLT_MAX : 0.f;
        out.push_back(one(nd, 16));
      }
    }
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  const uint32_t N = (argc > 1) ? uint32_t(std::atoi(argv[1])) : 1000000;
  VVX_rtu_box_pe_tb dut;
  dut.clk = 0; dut.reset = 1; dut.valid_in = 0;
  auto tick = [&] { dut.clk = 0; dut.eval(); dut.clk = 1; dut.eval(); };
  for (int i = 0; i < 4; ++i) tick();
  dut.reset = 0;

  std::vector<Node> nodes = directed_nodes();
  const uint32_t ND = uint32_t(nodes.size());
  uint32_t random_boxes = 0;
  for (uint32_t i = 0; random_boxes < N; ++i) {
    nodes.push_back(make_random(i));
    random_boxes += uint32_t(nodes.back().n);
  }

  struct Item { uint32_t node; int child; Ref ieee, ftz; };
  std::vector<Item> items;
  for (uint32_t k = 0; k < nodes.size(); ++k)
    for (int c = 0; c < nodes[k].n; ++c)
      items.push_back({k, c, reference(nodes[k], c, false), reference(nodes[k], c, true)});

  // per node: RTL results, and whether each child matched only the FTZ reference
  std::vector<std::vector<bool>>     rtl_hit(nodes.size());
  std::vector<std::vector<uint32_t>> rtl_t(nodes.size());
  std::vector<bool>                  node_ftz(nodes.size(), false);
  for (uint32_t k = 0; k < nodes.size(); ++k) {
    rtl_hit[k].assign(size_t(nodes[k].n), false);
    rtl_t[k].assign(size_t(nodes[k].n), 0);
  }

  uint32_t cat_cases[kNumCats] = {}, cat_err[kNumCats] = {}, cat_ftz[kNumCats] = {};
  uint32_t errors = 0, flushed = 0, inv_errors = 0, hits = 0;
  uint32_t order_multi = 0, order_checked = 0, order_errors = 0, order_ftz = 0;
  std::deque<uint32_t> inv_q, out_q;
  std::vector<uint8_t> inv_ok(items.size(), 0);
  uint32_t sent = 0, checked = 0;
  const uint32_t total = uint32_t(items.size());

  auto report_case = [&](const Item& it, const char* what) {
    const Node& nd = nodes[it.node];
    const Child& ch = nd.ch[it.child];
    std::printf("  [%s] %s raw=%d origin=(%a %a %a) exp=(%d %d %d) q=(%u %u %u)-(%u %u %u) "
                "rmin=(%a %a %a) rmax=(%a %a %a) o=(%a %a %a) d=(%a %a %a) tmin=%a tmax=%a\n",
                kCatNames[nd.cat], what, nd.raw, nd.origin[0], nd.origin[1], nd.origin[2],
                nd.exp[0], nd.exp[1], nd.exp[2], ch.qmin[0], ch.qmin[1], ch.qmin[2],
                ch.qmax[0], ch.qmax[1], ch.qmax[2], ch.rmin[0], ch.rmin[1], ch.rmin[2],
                ch.rmax[0], ch.rmax[1], ch.rmax[2], nd.ray.o[0], nd.ray.o[1], nd.ray.o[2],
                nd.ray.d[0], nd.ray.d[1], nd.ray.d[2], nd.ray.tmin, nd.ray.tmax);
  };

  while (checked < total) {
    if (sent < total) {
      const Item& it = items[sent];
      const Node& nd = nodes[it.node];
      const Child& ch = nd.ch[it.child];
      for (int a = 0; a < 3; ++a) {
        dut.origin[a]  = bits(nd.origin[a]);
        dut.raw_min[a] = bits(ch.rmin[a]);
        dut.raw_max[a] = bits(ch.rmax[a]);
        dut.ro[a]      = bits(nd.ray.o[a]);
        dut.dir[a]     = bits(nd.ray.d[a]);
      }
      dut.exp  = uint32_t(uint8_t(nd.exp[0])) | uint32_t(uint8_t(nd.exp[1])) << 8
               | uint32_t(uint8_t(nd.exp[2])) << 16;
      dut.qmin = uint32_t(ch.qmin[0]) | uint32_t(ch.qmin[1]) << 8 | uint32_t(ch.qmin[2]) << 16;
      dut.qmax = uint32_t(ch.qmax[0]) | uint32_t(ch.qmax[1]) << 8 | uint32_t(ch.qmax[2]) << 16;
      dut.raw   = nd.raw;
      dut.t_min = bits(nd.ray.tmin);
      dut.t_max = bits(nd.ray.tmax);
      dut.tag_in = sent;
      dut.valid_in = 1;
      inv_q.push_back(sent);
      out_q.push_back(sent);
      ++sent;
    } else {
      dut.valid_in = 0;
    }
    tick();
    if (dut.inv_valid) {
      const uint32_t id = inv_q.front();
      inv_q.pop_front();
      const Item& it = items[id];
      bool ok = (dut.inv_tag == id), ftz_ok = ok;
      for (int a = 0; a < 3; ++a) {
        ok     = ok     && same_bits(dut.inv_d[a], it.ieee.inv[a]);
        ftz_ok = ftz_ok && same_bits(dut.inv_d[a], it.ftz.inv[a]);
      }
      inv_ok[id] = ok ? 1 : (ftz_ok ? 2 : 0);
    }
    if (dut.valid_out) {
      const uint32_t id = out_q.front();
      out_q.pop_front();
      const Item& it = items[id];
      const int cat = nodes[it.node].cat;
      ++cat_cases[cat];
      bool ok = (dut.tag_out == id) && inv_ok[id] == 1
             && same_box(dut.hit, dut.t_near, it.ieee.box);
      if (it.ieee.box.hit) ++hits;
      if (!ok && dut.tag_out == id && inv_ok[id] != 0
          && same_box(dut.hit, dut.t_near, it.ftz.box) && !same_ref(it.ieee, it.ftz)) {
        ok = true;
        ++cat_ftz[cat];
        ++flushed;
        node_ftz[it.node] = true;
      }
      if (!ok) {
        ++errors;
        if (inv_ok[id] == 0) ++inv_errors;
        if (cat_err[cat]++ < 3) {
          report_case(it, "MISMATCH");
          std::printf("    ref hit=%d t_near=%08x inv=(%08x %08x %08x) | rtl tag=%u hit=%d t_near=%08x inv_ok=%d\n",
                      it.ieee.box.hit, bits(it.ieee.box.t_near), bits(it.ieee.inv[0]),
                      bits(it.ieee.inv[1]), bits(it.ieee.inv[2]), dut.tag_out, dut.hit,
                      dut.t_near, inv_ok[id]);
        }
      }
      rtl_hit[it.node][size_t(it.child)] = dut.hit;
      rtl_t[it.node][size_t(it.child)]   = dut.t_near;
      ++checked;
    }
  }

  // child visit order per node
  {
    size_t base = 0;
    for (uint32_t k = 0; k < nodes.size(); ++k) {
      const int n = nodes[k].n;
      if (n >= 2) {
        std::vector<BoxResult> ieee, ftz;
        for (int c = 0; c < n; ++c) {
          ieee.push_back(items[base + size_t(c)].ieee.box);
          ftz.push_back(items[base + size_t(c)].ftz.box);
        }
        ++order_checked;
        int acc = 0;
        for (const BoxResult& b : ieee) acc += b.hit;
        if (acc >= 2) ++order_multi;
        const std::vector<int> got = rtl_order(rtl_hit[k], rtl_t[k]);
        if (got != simx_order(ieee)) {
          if (node_ftz[k] && got == simx_order(ftz)) {
            ++order_ftz;
          } else {
            ++order_errors;
          }
        }
      }
      base += size_t(n);
    }
  }

  std::printf("rtu_box_pe: %u boxes (%u directed), %u accepted, %u mismatches "
              "(%u in inv_d), %u subnormal flushes\n",
              checked, ND, hits, errors, inv_errors, flushed);
  std::printf("  child order: %u nodes (%u with 2+ accepted), %u mismatches, %u subnormal flushes\n",
              order_checked, order_multi, order_errors, order_ftz);
  for (int k = 0; k < kNumCats; ++k)
    std::printf("  %-20s %8u boxes %6u mismatches %6u subnormal flushes\n",
                kCatNames[k], cat_cases[k], cat_err[k], cat_ftz[k]);
  const bool fail = errors || order_errors;
  std::printf(fail ? "FAILED!\n" : "PASSED!\n");
  return fail ? 1 : 0;
}
