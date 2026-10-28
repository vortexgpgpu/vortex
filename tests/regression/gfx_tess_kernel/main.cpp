// The SIMT tessellator on the device, bit for bit against the reference
// tessellator llvmpipe runs on the CPU.
//
// One batch holds patches of every domain (triangle, quad, isoline), spacing
// (equal, fractional odd, fractional even) and output (points, lines, both
// triangle windings): integer levels on all edges at once, each outer edge and
// the inside alone over a fixed base, -n random fractional and mixed level sets
// up to 16, and levels the reference culls or clamps (0, negative, NaN,
// infinite, denormal, past 64). The default levels are the ones where the
// tessellation's shape changes -- around each power of two, and the ends --
// which keeps the batch to a SimX run of minutes; -f takes every level 1..64
// and random levels up to 64, and -q only the smallest ones, for rtlsim. The
// batch runs through the kernels vortexpipe launches -- the control-stage
// epilogue, the device-wide scan, the evaluation prologue and the stitch kernel
// -- and every patch's counts, every domain point (u and v as bits) and every
// primitive (its point indices, offset by the patch's first point, and its
// gl_PrimitiveID) must equal the reference's, in its order. The reference runs
// with denormals flushed, as llvmpipe tessellates. The batch runs again in
// three ranges of its patches, and a sample of patches as batches of copies of
// one shared descriptor, the shape levels known on the host take.
//
// -b N checks against a deliberately wrong reference and passes only when the
// run reports that error and no other: 1 a count off by one, 2 a coordinate
// one ulp off, 3 two indices swapped.

#include <vortex.h>
#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#endif
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <random>
#include <vector>
#include "common.h"
#include "gfx_frontend_abi.h"
#include "gfx_tess.h"
#include "tessellator.hpp"

#define CHECK(expr)                                                      \
  do {                                                                   \
    int _r = (expr);                                                     \
    if (_r != 0) {                                                       \
      std::printf("Error: '%s' returned %d (%s:%d)\n", #expr, _r,        \
                  __FILE__, __LINE__);                                   \
      return -1;                                                         \
    }                                                                    \
  } while (0)

static uint32_t g_random = 300;
static uint32_t g_bite = 0;
static bool g_full = false;
static bool g_quick = false;
static uint32_t g_perf = 0;
static const char* g_kernel_file = "kernel.vxbin";

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:b:fqp:k:h")) != -1) {
    switch (c) {
    case 'n': g_random = (uint32_t)std::atoi(optarg); break;
    case 'b': g_bite = (uint32_t)std::atoi(optarg); break;
    case 'f': g_full = true; break;
    case 'q': g_quick = true; break;
    case 'p': g_perf = (uint32_t)std::atoi(optarg); break;
    case 'k': g_kernel_file = optarg; break;
    default: break;
    }
  }
}

static const uint32_t PATCH_BASE = 5, INST_PATCHES = 97;

struct patch_in {
  float lv[8];   // outer[4], inner[2], mode (bits), gl_PrimitiveID (bits)
};

static uint32_t fbits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}

static patch_in make_patch(uint32_t mode, const float* outer, const float* inner) {
  patch_in p{};
  std::memcpy(p.lv, outer, 16);
  std::memcpy(p.lv + 4, inner, 8);
  std::memcpy(&p.lv[6], &mode, 4);
  return p;
}

// The reference's output for one patch.
struct ref_patch {
  std::vector<DOMAIN_POINT> pts;
  std::vector<int> idx;
};

static ref_patch reference(uint32_t mode, const float* lv) {
  static const PIPE_TESSELLATOR_PARTITIONING part[3] = {
    PIPE_TESSELLATOR_PARTITIONING_INTEGER, PIPE_TESSELLATOR_PARTITIONING_FRACTIONAL_ODD,
    PIPE_TESSELLATOR_PARTITIONING_FRACTIONAL_EVEN };
  static const PIPE_TESSELLATOR_OUTPUT_PRIMITIVE prim[4] = {
    PIPE_TESSELLATOR_OUTPUT_POINT, PIPE_TESSELLATOR_OUTPUT_LINE,
    PIPE_TESSELLATOR_OUTPUT_TRIANGLE_CW, PIPE_TESSELLATOR_OUTPUT_TRIANGLE_CCW };
  static CHWTessellator ref;
#if defined(__x86_64__) || defined(__i386__)
  // FTZ (bit 15) and DAZ (bit 6), as llvmpipe tessellates.
  const unsigned csr = _mm_getcsr();
  _mm_setcsr(csr | 0x8040u);
#endif
  ref.Init(part[GFX_TESS_MODE_SPACING(mode)], prim[GFX_TESS_MODE_OUT(mode)]);
  switch (GFX_TESS_MODE_DOMAIN(mode)) {
  case GFX_TESS_DOMAIN_QUAD:
    ref.TessellateQuadDomain(lv[0], lv[1], lv[2], lv[3], lv[4], lv[5]);
    break;
  case GFX_TESS_DOMAIN_TRI:
    ref.TessellateTriDomain(lv[0], lv[1], lv[2], lv[4]);
    break;
  default:
    ref.TessellateIsoLineDomain(lv[0], lv[1]);
    break;
  }
#if defined(__x86_64__) || defined(__i386__)
  _mm_setcsr(csr);
#endif
  ref_patch r;
  r.pts.assign(ref.GetPoints(), ref.GetPoints() + ref.GetPointCount());
  r.idx.assign(ref.GetIndices(), ref.GetIndices() + ref.GetIndexCount());
  return r;
}

// -p N: N patches each of the level-64 quad, triangle and isoline, the shape a
// terrain or a close-up model gives, to time the tessellator's own kernels.
static void add_perf_patches(std::vector<patch_in>& ps) {
  const float o[4] = { 64.0f, 64.0f, 64.0f, 64.0f }, i[2] = { 64.0f, 64.0f };
  for (uint32_t n = 0; n < g_perf; n++) {
    ps.push_back(make_patch(GFX_TESS_MODE(GFX_TESS_DOMAIN_QUAD, GFX_TESS_SPACING_EVEN,
                                          GFX_TESS_OUT_TRI_CW), o, i));
    ps.push_back(make_patch(GFX_TESS_MODE(GFX_TESS_DOMAIN_TRI, GFX_TESS_SPACING_ODD,
                                          GFX_TESS_OUT_TRI_CCW), o, i));
    ps.push_back(make_patch(GFX_TESS_MODE(GFX_TESS_DOMAIN_ISOLINE, GFX_TESS_SPACING_EQUAL,
                                          GFX_TESS_OUT_LINE), o, i));
  }
}

static void add_patches(std::vector<patch_in>& ps) {
  if (g_perf) {
    add_perf_patches(ps);
    return;
  }
  const float specials[] = { 0.0f, -0.0f, -1.0f, NAN, INFINITY, 1e-40f, 0.5f, 1.0f,
                             1.0000076f, 1.5f, 63.5f, 64.5f, 1e9f };
  for (uint32_t d = 0; d < 3; d++) {
    for (uint32_t s = 0; s < 3; s++) {
      const uint32_t outs[3] = { GFX_TESS_OUT_POINT,
                                 d == GFX_TESS_DOMAIN_ISOLINE ? GFX_TESS_OUT_LINE : GFX_TESS_OUT_TRI_CW,
                                 d == GFX_TESS_DOMAIN_ISOLINE ? GFX_TESS_OUT_LINE : GFX_TESS_OUT_TRI_CCW };
      // integer levels on all edges at once, alternating the windings
      static const int SHAPES[] = { 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64 };
      const int nshapes = g_full ? 64 : g_quick ? 6 : (int)(sizeof(SHAPES) / sizeof(SHAPES[0]));
      for (int k = 0; k < nshapes; k++) {
        const int L = g_full ? k + 1 : SHAPES[k];
        const float o[4] = { (float)L, (float)L, (float)L, (float)L };
        const float i[2] = { (float)L, (float)L };
        ps.push_back(make_patch(GFX_TESS_MODE(d, s, outs[1 + (L & 1)]), o, i));
      }
      // each outer edge and the inside alone, over a fixed base
      static const int EDGE_LEVELS[] = { 1, 2, 3, 6, 11, 16, 33, 64 };
      const int nedge = g_full ? 64 : g_quick ? 3 : (int)(sizeof(EDGE_LEVELS) / sizeof(EDGE_LEVELS[0]));
      for (int e = 0; e < 6; e++) {
        for (int k = 0; k < nedge; k++) {
          const int L = g_full ? k + 1 : EDGE_LEVELS[k];
          float o[4] = { 5.0f, 3.0f, 7.0f, 4.0f };
          float i[2] = { 6.0f, 2.0f };
          if (e < 4) {
            o[e] = (float)L;
          } else {
            i[e - 4] = (float)L;
          }
          ps.push_back(make_patch(GFX_TESS_MODE(d, s, outs[(e + L) % 3]), o, i));
        }
      }
      // culled and clamped levels: an outer one, then the inner ones under
      // live outer levels
      for (float f : specials) {
        const float o[4] = { 3.0f, f, 4.0f, 2.0f };
        const float i[2] = { 2.5f, 3.5f };
        ps.push_back(make_patch(GFX_TESS_MODE(d, s, outs[1]), o, i));
        const float lo[4] = { 3.0f, 5.0f, 4.0f, 2.0f };
        const float li[2] = { f, f };
        ps.push_back(make_patch(GFX_TESS_MODE(d, s, outs[2]), lo, li));
      }
    }
  }
  std::mt19937 rng(7);
  for (uint32_t n = 0; n < g_random; n++) {
    const uint32_t d = rng() % 3, s = rng() % 3;
    const uint32_t out = d == GFX_TESS_DOMAIN_ISOLINE ? (rng() % 2 ? GFX_TESS_OUT_LINE : GFX_TESS_OUT_POINT)
                                                      : GFX_TESS_OUT_TRI_CW + (rng() % 2);
    float o[4], i[2];
    const uint32_t top = g_full ? 64u : 16u;
    for (float& v : o) {
      v = rng() % 3 ? 1.0f + (float)(rng() % (10000u * (top - 1u))) / 10000.0f
                    : (float)(1 + rng() % top);
    }
    for (float& v : i) {
      v = rng() % 3 ? 1.0f + (float)(rng() % (10000u * (top - 1u))) / 10000.0f
                    : (float)(1 + rng() % top);
    }
    ps.push_back(make_patch(GFX_TESS_MODE(d, s, out), o, i));
  }
}

int main(int argc, char** argv) {
  parse_args(argc, argv);
  std::vector<patch_in> ps;
  add_patches(ps);
  const uint32_t np = (uint32_t)ps.size();

  // The reference, and each patch's first point and primitive.
  std::vector<ref_patch> ref(np);
  std::vector<uint32_t> vbase(np + 1, 0), pbase(np + 1, 0);
  for (uint32_t p = 0; p < np; p++) {
    uint32_t mode;
    std::memcpy(&mode, &ps[p].lv[6], 4);
    ref[p] = reference(mode, ps[p].lv);
    const uint32_t out = GFX_TESS_MODE_OUT(mode);
    const uint32_t per = out == GFX_TESS_OUT_POINT ? 1 : out == GFX_TESS_OUT_LINE ? 2 : 3;
    vbase[p + 1] = vbase[p] + (uint32_t)ref[p].pts.size();
    pbase[p + 1] = pbase[p] + (uint32_t)ref[p].idx.size() / per;
  }
  const uint32_t V = vbase[np], P = pbase[np];
  std::printf("gfx_tess_kernel: %u patches, %u points, %u primitives\n", np, V, P);

  vx_device_h dev = nullptr;
  CHECK(vx_device_open(0, &dev));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  vx_queue_h q = nullptr;
  CHECK(vx_queue_create(dev, &qi, &q));
  uint32_t one = 1, grid[1], block[1];
  CHECK(vx_device_max_occupancy_grid(dev, 1, &one, grid, block));
  const uint32_t T = block[0] < 64u ? block[0] : 64u, G = grid[0];
  // The shapes vortexpipe launches the point and stitch kernels in: every
  // core's warps striding over the points, and as many 128-thread CTAs as the
  // cores hold striding over the stitch units.
  const uint32_t B = block[0], TS = B < 128u ? B : 128u, GS = G * (B / TS);

  vx_buffer_h lv_buf, desc_buf, cnt_buf, off_buf, part_buf, tot_buf, scr_buf, draw_buf,
              pt_buf, list_buf;
  CHECK(vx_buffer_create(dev, np * sizeof(patch_in), VX_MEM_READ, &lv_buf));
  CHECK(vx_buffer_create(dev, np * sizeof(gfx_tess_patch_t), 0, &desc_buf));
  CHECK(vx_buffer_create(dev, np * 12, 0, &cnt_buf));
  CHECK(vx_buffer_create(dev, np * 12, 0, &off_buf));
  CHECK(vx_buffer_create(dev, G * 12, 0, &part_buf));
  CHECK(vx_buffer_create(dev, 16, 0, &tot_buf));
  CHECK(vx_buffer_create(dev, G * 2 * T * 4, 0, &scr_buf));
  CHECK(vx_buffer_create(dev, sizeof(gfx_tess_draw_t), 0, &draw_buf));
  CHECK(vx_buffer_create(dev, (V ? V : 1) * 16, 0, &pt_buf));
  CHECK(vx_buffer_create(dev, (P ? P : 1) * sizeof(pipe_list_entry_t), 0, &list_buf));
  uint64_t lv_dev, desc_dev, cnt_dev, off_dev, part_dev, tot_dev, scr_dev, draw_dev, pt_dev,
           list_dev;
  CHECK(vx_buffer_address(lv_buf, &lv_dev));
  CHECK(vx_buffer_address(desc_buf, &desc_dev));
  CHECK(vx_buffer_address(cnt_buf, &cnt_dev));
  CHECK(vx_buffer_address(off_buf, &off_dev));
  CHECK(vx_buffer_address(part_buf, &part_dev));
  CHECK(vx_buffer_address(tot_buf, &tot_dev));
  CHECK(vx_buffer_address(scr_buf, &scr_dev));
  CHECK(vx_buffer_address(draw_buf, &draw_dev));
  CHECK(vx_buffer_address(pt_buf, &pt_dev));
  CHECK(vx_buffer_address(list_buf, &list_dev));
  // The batch's patches as patches PATCH_BASE.. of a draw of INST_PATCHES a
  // instance: the control stage gives each its gl_PrimitiveID.
  for (uint32_t p = 0; p < np; p++) {
    const uint32_t id = (PATCH_BASE + p) % INST_PATCHES;
    std::memcpy(&ps[p].lv[7], &id, 4);
  }
  CHECK(vx_enqueue_write(q, lv_buf, 0, ps.data(), np * sizeof(patch_in), 0, nullptr, nullptr));

  vx_module_h mod = nullptr;
  vx_kernel_h k_main = nullptr, k_scan = nullptr, k_stitch = nullptr;
  CHECK(vx_module_load_file(dev, g_kernel_file, &mod));
  CHECK(vx_module_get_kernel(mod, "main", &k_main));
  CHECK(vx_module_get_kernel(mod, "scan_k", &k_scan));
  CHECK(vx_module_get_kernel(mod, "tess_stitch_k", &k_stitch));

  // setup, then the scan of the counts: one batch, drained launch by launch.
  kernel_arg_t sarg{};
  sarg.stage = TESS_STAGE_SETUP;
  sarg.n = np;
  sarg.levels_addr = lv_dev;
  sarg.desc_addr = desc_dev;
  sarg.counts_addr = cnt_dev;
  std::vector<vx_launch_info_t> li;
  std::vector<pipe_scan_arg_t> scan(3);
  li.reserve(16);
  auto launch = [&](vx_kernel_h k, const void* a, uint32_t bytes, uint32_t g, uint32_t b) {
    vx_launch_info_t l{};
    l.struct_size = sizeof(l);
    l.kernel = k;
    l.args_host = a;
    l.args_size = bytes;
    l.ndim = 1;
    l.grid_dim[0] = g;
    l.block_dim[0] = b;
    li.push_back(l);
    return vx_enqueue_launch(q, &li.back(), 0, nullptr, nullptr);
  };
  CHECK(launch(k_main, &sarg, sizeof(sarg), G, T));
  for (uint32_t st = PIPE_SCAN_UP; st <= PIPE_SCAN_DOWN; st++) {
    pipe_scan_arg_t& sa = scan[st];
    sa = pipe_scan_arg_t{};
    sa.stage = st;
    sa.n = np;
    sa.lanes = 3;
    sa.chunk = (np + G - 1) / G;
    sa.parts = G;
    sa.in_addr = cnt_dev;
    sa.out_addr = off_dev;
    sa.part_addr = part_dev;
    sa.total_addr = tot_dev;
    sa.scratch_addr = scr_dev;
    sa.query_sum = PIPE_SCAN_QC_NONE;
    sa.query_lane[0] = sa.query_lane[1] = sa.query_lane[2] = 0xffffffffu;
    CHECK(launch(k_scan, &sa, sizeof(sa), st == PIPE_SCAN_TOP ? 1u : G, T));
  }
  uint32_t totals[4] = { 0, 0, 0, 0 };
  std::vector<uint32_t> counts(3 * np);
  CHECK(vx_enqueue_read(q, totals, tot_buf, 0, 12, 0, nullptr, nullptr));
  CHECK(vx_enqueue_read(q, counts.data(), cnt_buf, 0, 12 * np, 0, nullptr, nullptr));
  CHECK(vx_queue_finish(q, VX_TIMEOUT_INFINITE));

  // Errors by class: 1 counts, 2 domain points, 3 primitives, 0 anything else.
  int errors = 0;
  int by_class[4] = { 0, 0, 0, 0 };
  auto bad_in = [&](int cls, const char* what, uint32_t p, long got, long want) {
    by_class[cls]++;
    if (errors++ < 16) {
      std::printf("*** patch %u: %s %ld, reference %ld\n", p, what, got, want);
    }
  };
  auto bad = [&](const char* what, uint32_t p, long got, long want) {
    const int cls = !std::strcmp(what, "points") || !std::strcmp(what, "primitives") ? 1
                  : !std::strcmp(what, "point") ? 2 : !std::strcmp(what, "primitive") ? 3 : 0;
    bad_in(cls, what, p, got, want);
  };
  // The patches each bite changes: the first at or past a third of the batch
  // with a point, and with a triangle, past two thirds.
  uint32_t bite_v = np, bite_t = np;
  for (uint32_t p = np / 3; p < np && bite_v == np; p++) {
    bite_v = ref[p].pts.empty() ? np : p;
  }
  for (uint32_t p = 2 * np / 3; p < np && bite_t == np; p++) {
    uint32_t mode;
    std::memcpy(&mode, &ps[p].lv[6], 4);
    const uint32_t o = GFX_TESS_MODE_OUT(mode);
    bite_t = (o == GFX_TESS_OUT_TRI_CW || o == GFX_TESS_OUT_TRI_CCW) && !ref[p].idx.empty()
           ? p : np;
  }
  for (uint32_t p = 0; p < np; p++) {
    uint32_t mode;
    std::memcpy(&mode, &ps[p].lv[6], 4);
    const uint32_t out = GFX_TESS_MODE_OUT(mode);
    const uint32_t per = out == GFX_TESS_OUT_POINT ? 1 : out == GFX_TESS_OUT_LINE ? 2 : 3;
    const uint32_t want_v = (uint32_t)ref[p].pts.size() + (g_bite == 1 && p == np / 2 ? 1u : 0u);
    if (counts[3 * p] != want_v) {
      bad("points", p, counts[3 * p], want_v);
    }
    if (counts[3 * p + 1] != ref[p].idx.size() / per) {
      bad("primitives", p, counts[3 * p + 1], (long)(ref[p].idx.size() / per));
    }
  }
  if (totals[0] != V || totals[1] != P) {
    std::printf("*** totals: %u points, %u primitives; reference %u, %u\n", totals[0], totals[1],
                V, P);
    ++errors;
  }

  // Points, then the stitch, over the scanned batch.
  gfx_tess_draw_t draw{};
  draw.desc_addr = desc_dev;
  draw.scan_addr = off_dev;
  draw.list_addr = list_dev;
  draw.patches = np;
  draw.desc_stride = sizeof(gfx_tess_patch_t);
  draw.units = totals[2];
  draw.patch_base = PATCH_BASE;
  draw.inst_patches = INST_PATCHES;
  CHECK(vx_enqueue_write(q, draw_buf, 0, &draw, sizeof(draw), 0, nullptr, nullptr));
  kernel_arg_t parg{};
  parg.stage = TESS_STAGE_POINTS;
  parg.n = totals[0];
  parg.draw_addr = draw_dev;
  parg.points_addr = pt_dev;
  // Cycles and instructions of each kernel, all cores, from the counters.
  auto counters = [&](uint64_t* cyc, uint64_t* ins) {
    uint64_t nc = 1;
    vx_device_query(dev, VX_CAPS_NUM_CORES, &nc);
    *cyc = 0;
    *ins = 0;
    for (uint32_t c = 0; c < nc; c++) {
      uint64_t v = 0;
      vx_device_mpm_query(dev, VX_DCR_MPM_CLASS_CORE, VX_CSR_MCYCLE, c, &v);
      *cyc = v > *cyc ? v : *cyc;
    }
    vx_device_mpm_query(dev, VX_DCR_MPM_CLASS_CORE, VX_CSR_MINSTRET, 0xffffffffu, ins);
  };
  uint64_t c0, i0, c1, i1, c2, i2;
  counters(&c0, &i0);
  CHECK(launch(k_main, &parg, sizeof(parg), G, B));
  if (g_perf) {
    CHECK(vx_queue_finish(q, VX_TIMEOUT_INFINITE));
    counters(&c1, &i1);
  }
  CHECK(launch(k_stitch, &draw, sizeof(draw), GS, TS));
  if (g_perf) {
    CHECK(vx_queue_finish(q, VX_TIMEOUT_INFINITE));
    counters(&c2, &i2);
    std::printf("perf: points: %llu cycles, %llu instructions, %.1f instructions/point, "
                "%.3f points/cycle\n", (unsigned long long)(c1 - c0),
                (unsigned long long)(i1 - i0), (double)(i1 - i0) / V,
                (double)V / (double)(c1 - c0));
    std::printf("perf: stitch: %llu cycles, %llu instructions, %.1f instructions/primitive, "
                "%.3f primitives/cycle\n", (unsigned long long)(c2 - c1),
                (unsigned long long)(i2 - i1), (double)(i2 - i1) / P,
                (double)P / (double)(c2 - c1));
  }
  std::vector<uint32_t> pts(4 * (size_t)V);
  std::vector<pipe_list_entry_t> list(P);
  if (V) {
    CHECK(vx_enqueue_read(q, pts.data(), pt_buf, 0, 16 * (size_t)V, 0, nullptr, nullptr));
  }
  if (P) {
    CHECK(vx_enqueue_read(q, list.data(), list_buf, 0, P * sizeof(pipe_list_entry_t), 0, nullptr,
                          nullptr));
  }
  CHECK(vx_queue_finish(q, VX_TIMEOUT_INFINITE));

  for (uint32_t p = 0; p < np && errors < 16; p++) {
    uint32_t mode;
    std::memcpy(&mode, &ps[p].lv[6], 4);
    const uint32_t out = GFX_TESS_MODE_OUT(mode);
    const uint32_t id = (draw.patch_base + p) % draw.inst_patches;
    for (uint32_t i = 0; i < ref[p].pts.size(); i++) {
      const uint32_t* d = &pts[4 * (size_t)(vbase[p] + i)];
      float u = ref[p].pts[i].u;
      if (g_bite == 2 && p == bite_v && i == 0) {
        u = std::nextafter(u, 2.0f);
      }
      if (d[0] != p || d[1] != id || d[2] != fbits(u) || d[3] != fbits(ref[p].pts[i].v)) {
        bad("point", p, i, -1);
        break;
      }
    }
    if (out == GFX_TESS_OUT_POINT) {
      continue;
    }
    const uint32_t per = out == GFX_TESS_OUT_LINE ? 2 : 3;
    const uint32_t nprim = (uint32_t)ref[p].idx.size() / per;
    for (uint32_t k = 0; k < nprim; k++) {
      const pipe_list_entry_t& e = list[pbase[p] + k];
      uint32_t want[3];
      for (uint32_t c = 0; c < 3; c++) {
        want[c] = vbase[p] + (uint32_t)ref[p].idx[per * k + (c < per ? c : per - 1)];
      }
      if (g_bite == 3 && p == bite_t && k == 0) {
        std::swap(want[1], want[2]);
      }
      if (e.rec[0] != want[0] || e.rec[1] != want[1] || e.rec[2] != want[2] ||
          e.aux != (id & PIPE_LIST_AUX_MASK)) {
        bad("primitive", p, k, -1);
        break;
      }
    }
  }

  // The batch again as three ranges of its patches, each a batch of its own
  // whose points and primitives start at zero: every one must be the whole
  // batch's, renumbered from the range's first point and patch.
  std::vector<uint32_t> offs(3 * (size_t)np + 3);
  CHECK(vx_enqueue_read(q, offs.data(), off_buf, 0, 12 * np, 0, nullptr, nullptr));
  CHECK(vx_queue_finish(q, VX_TIMEOUT_INFINITE));
  offs[3 * np] = totals[0];
  offs[3 * np + 1] = totals[1];
  offs[3 * np + 2] = totals[2];
  const uint32_t cut[4] = { 0, np / 3, 2 * np / 3, np };
  for (uint32_t r = 0; r < 3 && errors < 16; r++) {
    const uint32_t a = cut[r], b = cut[r + 1];
    const uint32_t* fa = &offs[3 * a];
    const uint32_t* fb = &offs[3 * b];
    const uint32_t rv = fb[0] - fa[0], rp = fb[1] - fa[1];
    gfx_tess_draw_t rd = draw;
    rd.desc_addr = desc_dev + (uint64_t)a * sizeof(gfx_tess_patch_t);
    rd.scan_addr = off_dev + 12ull * a;
    rd.patches = b - a;
    rd.units = fb[2] - fa[2];
    rd.patch_base = draw.patch_base + a;
    rd.first_point = fa[0];
    rd.first_prim = fa[1];
    rd.first_unit = fa[2];
    if (b == a) {
      continue;
    }
    CHECK(vx_enqueue_write(q, draw_buf, 0, &rd, sizeof(rd), 0, nullptr, nullptr));
    kernel_arg_t rarg = parg;
    rarg.n = rv;
    CHECK(launch(k_main, &rarg, sizeof(rarg), G, B));
    CHECK(launch(k_stitch, &rd, sizeof(rd), GS, TS));
    std::vector<uint32_t> rpts(4 * (size_t)rv);
    std::vector<pipe_list_entry_t> rlist(rp);
    if (rv) {
      CHECK(vx_enqueue_read(q, rpts.data(), pt_buf, 0, 16 * (size_t)rv, 0, nullptr, nullptr));
    }
    if (rp) {
      CHECK(vx_enqueue_read(q, rlist.data(), list_buf, 0, rp * sizeof(pipe_list_entry_t), 0,
                            nullptr, nullptr));
    }
    CHECK(vx_queue_finish(q, VX_TIMEOUT_INFINITE));
    for (uint32_t k = 0; k < rv; k++) {
      const uint32_t* w = &pts[4 * ((size_t)fa[0] + k)];
      const uint32_t* g = &rpts[4 * (size_t)k];
      if (g[0] != w[0] - a || g[1] != w[1] || g[2] != w[2] || g[3] != w[3]) {
        bad("range point", a + g[0], k, -1);
        break;
      }
    }
    // Point-mode patches' entries are the evaluation stage's to write.
    for (uint32_t p = a; p < b; p++) {
      uint32_t mode;
      std::memcpy(&mode, &ps[p].lv[6], 4);
      if (GFX_TESS_MODE_OUT(mode) == GFX_TESS_OUT_POINT) {
        continue;
      }
      for (uint32_t k = pbase[p]; k < pbase[p + 1]; k++) {
        const pipe_list_entry_t& w = list[k];
        const pipe_list_entry_t& g = rlist[k - fa[1]];
        if (g.rec[0] != w.rec[0] - fa[0] || g.rec[1] != w.rec[1] - fa[0] ||
            g.rec[2] != w.rec[2] - fa[0] || g.aux != w.aux) {
          bad("range primitive", p, k - pbase[p], -1);
          break;
        }
      }
    }
  }

  // A sample of patches each as a batch of three copies of its one shared
  // descriptor, as with levels known on the host: copy c is the patch, its
  // points and primitives c times the patch's on from the first copy's.
  for (uint32_t p = 0; p < np && errors < 16; p += 29) {
    const uint32_t pv = (uint32_t)ref[p].pts.size();
    uint32_t mode;
    std::memcpy(&mode, &ps[p].lv[6], 4);
    const uint32_t out = GFX_TESS_MODE_OUT(mode);
    const uint32_t per = out == GFX_TESS_OUT_POINT ? 1 : out == GFX_TESS_OUT_LINE ? 2 : 3;
    const uint32_t pp = (uint32_t)ref[p].idx.size() / per;
    if (!pv) {
      continue;
    }
    const uint32_t copies = 3;
    gfx_tess_draw_t sd{};
    sd.desc_addr = desc_dev + (uint64_t)p * sizeof(gfx_tess_patch_t);
    sd.list_addr = list_dev;
    sd.patches = copies;
    sd.points = pv;
    sd.prims = counts[3 * p + 1];
    sd.units = offs[3 * (p + 1) + 2] - offs[3 * p + 2];
    sd.patch_base = draw.patch_base;
    sd.inst_patches = draw.inst_patches;
    CHECK(vx_enqueue_write(q, draw_buf, 0, &sd, sizeof(sd), 0, nullptr, nullptr));
    kernel_arg_t sparg = parg;
    sparg.n = copies * pv;
    CHECK(launch(k_main, &sparg, sizeof(sparg), G, B));
    if (out != GFX_TESS_OUT_POINT) {
      CHECK(launch(k_stitch, &sd, sizeof(sd), GS, TS));
    }
    std::vector<uint32_t> spts(4 * (size_t)copies * pv);
    std::vector<pipe_list_entry_t> slist((size_t)copies * pp + 1);
    CHECK(vx_enqueue_read(q, spts.data(), pt_buf, 0, spts.size() * 4, 0, nullptr, nullptr));
    if (pp && out != GFX_TESS_OUT_POINT) {
      CHECK(vx_enqueue_read(q, slist.data(), list_buf, 0, copies * pp * sizeof(pipe_list_entry_t),
                            0, nullptr, nullptr));
    }
    CHECK(vx_queue_finish(q, VX_TIMEOUT_INFINITE));
    for (uint32_t c = 0; c < copies; c++) {
      const uint32_t id = (draw.patch_base + c) % draw.inst_patches;
      for (uint32_t i = 0; i < pv; i++) {
        const uint32_t* d = &spts[4 * ((size_t)c * pv + i)];
        if (d[0] != c || d[1] != id || d[2] != fbits(ref[p].pts[i].u) ||
            d[3] != fbits(ref[p].pts[i].v)) {
          bad("shared point", p, c, i);
          break;
        }
      }
      if (out == GFX_TESS_OUT_POINT) {
        continue;
      }
      for (uint32_t k = 0; k < pp; k++) {
        const pipe_list_entry_t& e = slist[(size_t)c * pp + k];
        bool ok = e.aux == (id & PIPE_LIST_AUX_MASK);
        for (uint32_t v = 0; v < 3; v++) {
          ok = ok && e.rec[v] == c * pv + (uint32_t)ref[p].idx[per * k + (v < per ? v : per - 1)];
        }
        if (!ok) {
          bad("shared primitive", p, c, k);
          break;
        }
      }
    }
  }

  vx_buffer_h bufs[] = { lv_buf, desc_buf, cnt_buf, off_buf, part_buf, tot_buf, scr_buf,
                         draw_buf, pt_buf, list_buf };
  for (vx_buffer_h b : bufs) {
    vx_buffer_release(b);
  }
  vx_module_release(mod);
  vx_queue_release(q);
  vx_device_release(dev);

  if (g_bite) {
    const bool caught = g_bite < 4 && by_class[g_bite] > 0 && errors == by_class[g_bite];
    std::printf("%s (bite %u: %d errors, %d of its class)\n", caught ? "PASSED!" : "FAILED!",
                g_bite, errors, g_bite < 4 ? by_class[g_bite] : 0);
    return caught ? 0 : 1;
  }
  if (errors) {
    std::printf("FAILED! (%d errors)\n", errors);
    return 1;
  }
  std::printf("PASSED!\n");
  return 0;
}
