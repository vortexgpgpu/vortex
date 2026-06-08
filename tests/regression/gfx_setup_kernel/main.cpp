// gfx_v2 on-device triangle setup — SimX validation host.
// Generates a clip-space triangle list, runs the SIMT setup kernel (setup ->
// prefix-sum -> compact), and checks the dense rast_prim_t[] + bbox[] output.
// The reference is the shared setup math (setup_math.h), which is in turn
// anchored bit-for-bit against the real host Binning() oracle
// (sw/runtime/graphics.cpp) — so a device PASS proves parity with gfx-v1.

#include <vortex.h>
#include <graphics.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <vector>
#include <unordered_map>
#include "common.h"
#include "setup_math.h"

using vortex::graphics::rast_prim_t;

#define CHECK(expr)                                                      \
  do {                                                                   \
    int _r = (expr);                                                     \
    if (_r != 0) {                                                       \
      std::printf("Error: '%s' returned %d (%s:%d)\n", #expr, _r,        \
                  __FILE__, __LINE__);                                   \
      return -1;                                                         \
    }                                                                    \
  } while (0)

static uint32_t g_num_prims = 200;
static const char* g_kernel_file = "kernel.vxbin";

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:h")) != -1) {
    switch (c) {
      case 'n': g_num_prims = std::atoi(optarg); break;
      case 'k': g_kernel_file = optarg; break;
      default: break;
    }
  }
}

static float frand(float lo, float hi) {
  return lo + (hi - lo) * (float)std::rand() / (float)RAND_MAX;
}

// One clip-space vertex from NDC-ish coords scaled by w (>0, in front of near).
static setup_vertex_t make_vertex(float nx, float ny, float nz, float w) {
  setup_vertex_t v;
  v.pos[0] = nx * w; v.pos[1] = ny * w; v.pos[2] = nz * w; v.pos[3] = w;
  v.color[0] = frand(0, 1); v.color[1] = frand(0, 1);
  v.color[2] = frand(0, 1); v.color[3] = frand(0, 1);
  v.texcoord[0] = frand(0, 1); v.texcoord[1] = frand(0, 1);
  return v;
}

// Generate a triangle list mixing the common path with the two cull classes
// Binning() rejects: degenerate (collinear) and fully off-screen (empty bbox).
// All vertices have w > 0 (no near-plane crossing) so Binning() is a valid
// oracle — near-plane clipping is the next increment.
static std::vector<setup_vertex_t> gen_triangles(uint32_t n) {
  std::vector<setup_vertex_t> verts(3 * n);
  for (uint32_t t = 0; t < n; ++t) {
    int roll = std::rand() % 10;
    float w = frand(1.0f, 4.0f);
    if (roll == 0) {
      // degenerate: three coincident vertices (det == 0).
      setup_vertex_t v = make_vertex(frand(-0.8f, 0.8f), frand(-0.8f, 0.8f), frand(-1, 1), w);
      verts[3 * t + 0] = v; verts[3 * t + 1] = v; verts[3 * t + 2] = v;
    } else if (roll == 1) {
      // fully off-screen: bbox clamps to empty.
      float cx = frand(2.0f, 3.0f), cy = frand(2.0f, 3.0f);
      verts[3 * t + 0] = make_vertex(cx,        cy,        frand(-1, 1), w);
      verts[3 * t + 1] = make_vertex(cx + 0.2f, cy,        frand(-1, 1), w);
      verts[3 * t + 2] = make_vertex(cx,        cy + 0.2f, frand(-1, 1), w);
    } else {
      // common path: a small on-screen triangle (both windings occur).
      float cx = frand(-0.7f, 0.7f), cy = frand(-0.7f, 0.7f);
      for (int k = 0; k < 3; ++k)
        verts[3 * t + k] = make_vertex(cx + frand(-0.3f, 0.3f),
                                       cy + frand(-0.3f, 0.3f), frand(-1, 1), w);
    }
  }
  return verts;
}

// Host reference: the shared setup math, compacted in input order.
struct Golden {
  std::vector<rast_prim_t>  prim;
  std::vector<setup_bbox_t> bbox;
};

static Golden host_setup(const std::vector<setup_vertex_t>& verts, uint32_t n) {
  Golden g;
  for (uint32_t t = 0; t < n; ++t) {
    rast_prim_t p{};
    setup_bbox_t bb{};
    if (gfx_setup::setup_triangle(verts[3 * t + 0], verts[3 * t + 1], verts[3 * t + 2],
                                  SETUP_W, SETUP_H, SETUP_NEAR, SETUP_FAR, p, bb)) {
      g.prim.push_back(p);
      g.bbox.push_back(bb);
    }
  }
  return g;
}

// Anchor the shared math against the real Binning() oracle: identical inputs,
// compare the produced primbuf bit-for-bit. Proves host_setup() is faithful to
// gfx-v1 before we trust it as the device reference. Returns mismatch count.
static int anchor_against_binning(const std::vector<setup_vertex_t>& verts,
                                  uint32_t n, const Golden& g) {
  using namespace vortex;
  std::unordered_map<uint32_t, graphics::vertex_t> vmap;
  std::vector<graphics::primitive_t> prims;
  for (uint32_t i = 0; i < 3 * n; ++i) {
    graphics::vertex_t v;
    std::memcpy(&v, &verts[i], sizeof(v));  // setup_vertex_t == vertex_t layout
    vmap[i] = v;
  }
  for (uint32_t t = 0; t < n; ++t)
    prims.push_back({3 * t + 0, 3 * t + 1, 3 * t + 2});

  std::vector<uint8_t> tilebuf, primbuf;
  graphics::Binning(tilebuf, primbuf, vmap, prims, SETUP_W, SETUP_H,
                    SETUP_NEAR, SETUP_FAR, SETUP_BIN_LOG);

  size_t bp = primbuf.size() / sizeof(rast_prim_t);
  auto* bprim = reinterpret_cast<const rast_prim_t*>(primbuf.data());
  int errors = 0;
  if (bp != g.prim.size()) {
    std::printf("*** anchor: prim count host=%zu Binning=%zu\n", g.prim.size(), bp);
    ++errors;
  }
  size_t m = bp < g.prim.size() ? bp : g.prim.size();
  for (size_t i = 0; i < m && errors < 16; ++i)
    if (std::memcmp(&bprim[i], &g.prim[i], sizeof(rast_prim_t)) != 0) {
      std::printf("*** anchor: prim[%zu] shared-math != Binning()\n", i);
      ++errors;
    }
  return errors;
}

int main(int argc, char** argv) {
  parse_args(argc, argv);
  std::srand(50);
  const uint32_t n = g_num_prims;

  std::vector<setup_vertex_t> verts = gen_triangles(n);
  Golden ref = host_setup(verts, n);
  const uint32_t P = (uint32_t)ref.prim.size();
  std::printf("gfx_setup_kernel: n=%u  kept P=%u\n", n, P);

  int anchor_err = anchor_against_binning(verts, n, ref);
  if (anchor_err) {
    std::printf("RESULT: FAIL (reference diverges from Binning() oracle)\n");
    return 1;
  }
  std::printf("anchor: shared setup math matches Binning() oracle (P=%u)\n", P);

  vx_device_h dev = nullptr;
  CHECK(vx_device_open(0, &dev));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  vx_queue_h q = nullptr;
  CHECK(vx_queue_create(dev, &qi, &q));

  uint32_t one = 1, grid[1], block[1];
  CHECK(vx_device_max_occupancy_grid(dev, 1, &one, grid, block));
  const uint32_t T = block[0];
  const uint32_t G = grid[0];

  const uint32_t Pcap = P ? P : 1;
  const size_t   PRIM_SZ = sizeof(rast_prim_t);

  vx_buffer_h verts_buf, slot_prim_buf, slot_bbox_buf, keep_buf, offset_buf,
              tsum_buf, prim_buf, bbox_buf, meta_buf;
  CHECK(vx_buffer_create(dev, 3 * n * sizeof(setup_vertex_t), VX_MEM_READ,  &verts_buf));
  CHECK(vx_buffer_create(dev, n * PRIM_SZ,                    VX_MEM_WRITE, &slot_prim_buf));
  CHECK(vx_buffer_create(dev, n * sizeof(setup_bbox_t),       VX_MEM_WRITE, &slot_bbox_buf));
  CHECK(vx_buffer_create(dev, n * sizeof(uint32_t),           VX_MEM_WRITE, &keep_buf));
  CHECK(vx_buffer_create(dev, (n + 1) * sizeof(uint32_t),     VX_MEM_WRITE, &offset_buf));
  CHECK(vx_buffer_create(dev, T * sizeof(uint32_t),           VX_MEM_WRITE, &tsum_buf));
  CHECK(vx_buffer_create(dev, Pcap * PRIM_SZ,                 VX_MEM_WRITE, &prim_buf));
  CHECK(vx_buffer_create(dev, Pcap * sizeof(setup_bbox_t),    VX_MEM_WRITE, &bbox_buf));
  CHECK(vx_buffer_create(dev, sizeof(uint32_t),               VX_MEM_WRITE, &meta_buf));

  vx_module_h mod = nullptr;
  vx_kernel_h kern = nullptr;
  CHECK(vx_module_load_file(dev, g_kernel_file, &mod));
  CHECK(vx_module_get_kernel(mod, "main", &kern));

  kernel_arg_t karg{};
  karg.num_prims = n;
  karg.width = SETUP_W;
  karg.height = SETUP_H;
  CHECK(vx_buffer_address(verts_buf,     &karg.verts_addr));
  CHECK(vx_buffer_address(slot_prim_buf, &karg.slot_prim_addr));
  CHECK(vx_buffer_address(slot_bbox_buf, &karg.slot_bbox_addr));
  CHECK(vx_buffer_address(keep_buf,      &karg.keep_addr));
  CHECK(vx_buffer_address(offset_buf,    &karg.offset_addr));
  CHECK(vx_buffer_address(tsum_buf,      &karg.tsum_addr));
  CHECK(vx_buffer_address(prim_buf,      &karg.prim_addr));
  CHECK(vx_buffer_address(bbox_buf,      &karg.bbox_addr));
  CHECK(vx_buffer_address(meta_buf,      &karg.meta_addr));

  CHECK(vx_enqueue_write(q, verts_buf, 0, verts.data(), 3 * n * sizeof(setup_vertex_t), 0, nullptr, nullptr));

  // CP-sequenced: 3 chained launches. setup(0)/emit(2) multi-CTA (grid=G);
  // scan(1) single-CTA (grid=1). The launch-drain is the device barrier.
  const uint32_t NSTAGE = 3;
  const uint32_t sgrid[NSTAGE] = { G, 1, G };
  kernel_arg_t kargs[NSTAGE];
  vx_launch_info_t li[NSTAGE];
  vx_event_h ev[NSTAGE] = {};
  for (uint32_t s = 0; s < NSTAGE; ++s) {
    kargs[s] = karg; kargs[s].stage = s;
    li[s] = vx_launch_info_t{};
    li[s].struct_size = sizeof(li[s]);
    li[s].kernel      = kern;
    li[s].args_host   = &kargs[s];
    li[s].args_size   = sizeof(kernel_arg_t);
    li[s].ndim        = 1;
    li[s].grid_dim[0]  = sgrid[s];
    li[s].block_dim[0] = T;
    CHECK(vx_enqueue_launch(q, &li[s], s ? 1 : 0, s ? &ev[s - 1] : nullptr, &ev[s]));
  }

  std::vector<uint32_t> h_meta(1, 0);
  std::vector<rast_prim_t>  h_prim(Pcap);
  std::vector<setup_bbox_t> h_bbox(Pcap);
  vx_event_h last = ev[NSTAGE - 1], ev_m = nullptr, ev_p = nullptr, ev_b = nullptr;
  CHECK(vx_enqueue_read(q, h_meta.data(), meta_buf, 0, sizeof(uint32_t),   1, &last, &ev_m));
  CHECK(vx_enqueue_read(q, h_prim.data(), prim_buf, 0, P * PRIM_SZ,        1, &last, &ev_p));
  CHECK(vx_enqueue_read(q, h_bbox.data(), bbox_buf, 0, P * sizeof(setup_bbox_t), 1, &last, &ev_b));
  CHECK(vx_event_wait_value(ev_m, 1, VX_TIMEOUT_INFINITE));
  CHECK(vx_event_wait_value(ev_p, 1, VX_TIMEOUT_INFINITE));
  CHECK(vx_event_wait_value(ev_b, 1, VX_TIMEOUT_INFINITE));

  int errors = 0;
  if (h_meta[0] != P) { std::printf("*** P mismatch: dev=%u ref=%u\n", h_meta[0], P); ++errors; }
  for (uint32_t i = 0; i < P && errors < 16; ++i) {
    if (std::memcmp(&h_prim[i], &ref.prim[i], sizeof(rast_prim_t)) != 0) {
      std::printf("*** prim[%u] device != reference\n", i); ++errors;
    }
    const auto& a = h_bbox[i]; const auto& b = ref.bbox[i];
    if (a.bbL != b.bbL || a.bbR != b.bbR || a.bbT != b.bbT || a.bbB != b.bbB) {
      std::printf("*** bbox[%u] dev{%u,%u,%u,%u} != ref{%u,%u,%u,%u}\n",
                  i, a.bbL, a.bbR, a.bbT, a.bbB, b.bbL, b.bbR, b.bbT, b.bbB); ++errors;
    }
  }

  vx_event_release(ev_b); vx_event_release(ev_p); vx_event_release(ev_m);
  for (uint32_t s = 0; s < NSTAGE; ++s) vx_event_release(ev[s]);
  vx_buffer_release(verts_buf); vx_buffer_release(slot_prim_buf); vx_buffer_release(slot_bbox_buf);
  vx_buffer_release(keep_buf); vx_buffer_release(offset_buf); vx_buffer_release(tsum_buf);
  vx_buffer_release(prim_buf); vx_buffer_release(bbox_buf); vx_buffer_release(meta_buf);
  vx_module_release(mod);
  vx_queue_release(q);
  vx_device_release(dev);

  std::printf("RESULT: %s\n", errors == 0 ? "PASS" : "FAIL");
  return errors == 0 ? 0 : 1;
}
