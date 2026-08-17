// gfx_v2 on-device triangle setup + near-plane clip — SimX validation host.
// Generates a clip-space triangle list (common, near-crossing, behind-near,
// degenerate, off-screen), runs the SIMT setup+clip kernel (clip -> setup ->
// prefix-sum -> compact), and checks the dense output three ways:
//   1. device prim/bbox/vtx/pid == shared host setup math, bit-for-bit;
//   2. the shared math == real host Binning() oracle on the no-clip subset;
//   3. clipped subtriangles satisfy independent geometric invariants.
// (2) keeps the common path faithful to gfx-v1; (3) validates the new clip
// logic without reimplementing Sutherland-Hodgman.

#include <vortex.h>
#include <graphics.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <vector>
#include <unordered_map>
#include "common.h"
#include "gfx_setup.h"

using vortex::graphics::rast_prim_t;
namespace gs = gfx_setup;

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

// Vertex from NDC-ish coords scaled by w (>0). near_dist = z+w = w*(ndc_z+1),
// so ndc_z < -1 places the vertex behind the near plane (gets clipped).
static setup_vertex_t make_vertex(float ndc_x, float ndc_y, float ndc_z, float w) {
  setup_vertex_t v;
  v.pos[0] = ndc_x * w; v.pos[1] = ndc_y * w; v.pos[2] = ndc_z * w; v.pos[3] = w;
  v.color[0] = frand(0, 1); v.color[1] = frand(0, 1);
  v.color[2] = frand(0, 1); v.color[3] = frand(0, 1);
  v.texcoord[0] = frand(0, 1); v.texcoord[1] = frand(0, 1);
  return v;
}

// Triangle categories. CROSSING ones (>=1 vertex behind near) exercise the clip
// and are checked by the geometric-invariant oracle; the rest are non-crossing
// and anchored against Binning().
enum Cat { CAT_COMMON = 0, CAT_DEGEN, CAT_OFFSCREEN, CAT_CROSS1, CAT_CROSS2, CAT_BEHIND };
static inline bool is_crossing(int cat) { return cat >= CAT_CROSS1; }

struct Scene {
  std::vector<setup_vertex_t> verts;  // 3 per triangle
  std::vector<int> cat;               // per triangle
};

static Scene gen_scene(uint32_t n) {
  Scene s;
  s.verts.resize(3 * n);
  s.cat.resize(n);
  for (uint32_t t = 0; t < n; ++t) {
    int roll = std::rand() % 12;
    float w = frand(1.0f, 4.0f);
    int cat;
    if (roll == 0) {                       // degenerate (collinear, all in front)
      cat = CAT_DEGEN;
      setup_vertex_t v = make_vertex(frand(-0.7f, 0.7f), frand(-0.7f, 0.7f), frand(-0.8f, 0.8f), w);
      s.verts[3 * t + 0] = v; s.verts[3 * t + 1] = v; s.verts[3 * t + 2] = v;
    } else if (roll == 1) {                // fully off-screen (in front, empty bbox)
      cat = CAT_OFFSCREEN;
      float cx = frand(2.0f, 3.0f), cy = frand(2.0f, 3.0f);
      s.verts[3 * t + 0] = make_vertex(cx,        cy,        frand(-0.8f, 0.8f), w);
      s.verts[3 * t + 1] = make_vertex(cx + 0.2f, cy,        frand(-0.8f, 0.8f), w);
      s.verts[3 * t + 2] = make_vertex(cx,        cy + 0.2f, frand(-0.8f, 0.8f), w);
    } else if (roll == 2 || roll == 3) {   // crossing: 1 vertex in front, 2 behind
      cat = CAT_CROSS1;
      float cx = frand(-0.5f, 0.5f), cy = frand(-0.5f, 0.5f);
      s.verts[3 * t + 0] = make_vertex(cx,        cy,        frand(-0.8f, 0.5f), w);          // in front
      s.verts[3 * t + 1] = make_vertex(cx + 0.4f, cy + 0.1f, frand(-2.0f, -1.3f), frand(1, 4)); // behind
      s.verts[3 * t + 2] = make_vertex(cx + 0.1f, cy + 0.4f, frand(-2.0f, -1.3f), frand(1, 4)); // behind
    } else if (roll == 4 || roll == 5) {   // crossing: 2 vertices in front, 1 behind
      cat = CAT_CROSS2;
      float cx = frand(-0.5f, 0.5f), cy = frand(-0.5f, 0.5f);
      s.verts[3 * t + 0] = make_vertex(cx,        cy,        frand(-0.8f, 0.5f), w);          // in front
      s.verts[3 * t + 1] = make_vertex(cx + 0.4f, cy + 0.1f, frand(-0.8f, 0.5f), frand(1, 4)); // in front
      s.verts[3 * t + 2] = make_vertex(cx + 0.1f, cy + 0.4f, frand(-2.0f, -1.3f), frand(1, 4)); // behind
    } else if (roll == 6) {                // fully behind near (all clipped, K=0)
      cat = CAT_BEHIND;
      float cx = frand(-0.5f, 0.5f), cy = frand(-0.5f, 0.5f);
      for (int k = 0; k < 3; ++k)
        s.verts[3 * t + k] = make_vertex(cx + frand(-0.3f, 0.3f), cy + frand(-0.3f, 0.3f),
                                         frand(-2.0f, -1.3f), frand(1, 4));
    } else {                               // common: small on-screen, in front
      cat = CAT_COMMON;
      float cx = frand(-0.7f, 0.7f), cy = frand(-0.7f, 0.7f);
      for (int k = 0; k < 3; ++k)
        s.verts[3 * t + k] = make_vertex(cx + frand(-0.3f, 0.3f), cy + frand(-0.3f, 0.3f),
                                         frand(-0.8f, 0.8f), w);
    }
    s.cat[t] = cat;
  }
  return s;
}

// Host reference: the shared clip+setup math, mirroring the kernel exactly
// (per-tri clip -> per-subtri setup -> compact in input order).
struct Golden {
  std::vector<rast_prim_t>  prim;
  std::vector<setup_bbox_t> bbox;
  std::vector<clip_tri_t>   vtx;
  std::vector<uint32_t>     pid;
};

static Golden host_setup(const std::vector<setup_vertex_t>& verts, uint32_t n,
                         uint32_t cull_mode = SETUP_CULL_NONE) {
  Golden g;
  for (uint32_t t = 0; t < n; ++t) {
    clip_tri_t sub[SETUP_MAX_SUB];
    int ns = gs::clip_near(verts[3 * t + 0], verts[3 * t + 1], verts[3 * t + 2], sub);
    for (int s = 0; s < ns; ++s) {
      rast_prim_t p{};
      setup_bbox_t bb{};
      if (gs::setup_triangle(sub[s].v[0], sub[s].v[1], sub[s].v[2],
                             SETUP_W, SETUP_H, SETUP_NEAR, SETUP_FAR, p, bb, cull_mode)) {
        g.prim.push_back(p);
        g.bbox.push_back(bb);
        g.vtx.push_back(sub[s]);
        g.pid.push_back(t);
      }
    }
  }
  return g;
}

// Anchor the shared math against the real Binning() oracle on the NON-CROSSING
// subset (where clip is a passthrough, so setup must reproduce Binning()
// bit-for-bit). Returns mismatch count.
static int anchor_against_binning(const Scene& sc, uint32_t n) {
  using namespace vortex;
  std::unordered_map<uint32_t, graphics::vertex_t> vmap;
  std::vector<graphics::primitive_t> prims;
  std::vector<setup_vertex_t> sub;            // shared-math reference over the subset
  uint32_t vi = 0;
  for (uint32_t t = 0; t < n; ++t) {
    if (is_crossing(sc.cat[t])) continue;
    graphics::vertex_t gv;
    for (int k = 0; k < 3; ++k) {
      std::memcpy(&gv, &sc.verts[3 * t + k], sizeof(gv));
      vmap[vi + k] = gv;
      sub.push_back(sc.verts[3 * t + k]);
    }
    prims.push_back({vi + 0, vi + 1, vi + 2});
    vi += 3;
  }
  Golden ref = host_setup(sub, (uint32_t)prims.size());

  std::vector<uint8_t> tilebuf, primbuf;
  graphics::Binning(tilebuf, primbuf, vmap, prims, SETUP_W, SETUP_H,
                    SETUP_NEAR, SETUP_FAR, SETUP_BIN_LOG);

  size_t bp = primbuf.size() / sizeof(rast_prim_t);
  auto* bprim = reinterpret_cast<const rast_prim_t*>(primbuf.data());
  int errors = 0;
  if (bp != ref.prim.size()) {
    std::printf("*** anchor: prim count shared=%zu Binning=%zu\n", ref.prim.size(), bp);
    ++errors;
  }
  size_t m = bp < ref.prim.size() ? bp : ref.prim.size();
  for (size_t i = 0; i < m && errors < 16; ++i) {
    const rast_prim_t& a = ref.prim[i];   // shared math
    const rast_prim_t& b = bprim[i];      // Binning() oracle
    if (std::memcmp(&a, &b, sizeof(rast_prim_t)) == 0)
      continue;
    // Name the field that differs. A whole-struct compare reports every
    // primitive whenever either producer leaves any one field unset -- which
    // says nothing about the geometry and reads exactly like a math divergence.
    if (std::memcmp(a.edges, b.edges, sizeof(a.edges)) != 0)
      std::printf("*** anchor: prim[%zu] edges differ\n", i);
    if (std::memcmp(&a.attribs, &b.attribs, sizeof(a.attribs)) != 0)
      std::printf("*** anchor: prim[%zu] attribs differ\n", i);
    if (a.facing != b.facing)
      std::printf("*** anchor: prim[%zu] facing shared=%u Binning=%u\n",
                  i, a.facing, b.facing);
    if (std::memcmp(&a.rhw_scale, &b.rhw_scale, sizeof(a.rhw_scale)) != 0)
      std::printf("*** anchor: prim[%zu] rhw_scale shared=%g Binning=%g\n",
                  i, (double)a.rhw_scale, (double)b.rhw_scale);
    ++errors;
  }
  return errors;
}

// Independent geometric invariants on the clipped subtriangles of a crossing
// parent. Does not reimplement the clip: it checks properties every correct
// near-clip output must have. Returns mismatch count.
//   A  subtri count == near-clip multiplicity of the inside/outside split
//   B  every vertex in the near half-space (z+w >= 0)
//   C  every vertex inside the original triangle (barycentric in [0,1])
//   D  every introduced vertex lies on the near plane (z+w == 0) AND on an
//      original edge (one barycentric == 0) — i.e. at edge ∩ near-plane
static int check_invariants(const setup_vertex_t orig[3], const clip_tri_t* subs,
                            int nsub, uint32_t t) {
  const float TOLN = 1e-2f;  // near-dist (z+w ~ O(1))
  const float TOLB = 1e-2f;  // barycentric ratio
  int errors = 0;

  int inside = 0;
  for (int i = 0; i < 3; ++i) if (gs::near_dist(orig[i]) >= 0.0f) ++inside;
  int expect = (inside == 0) ? 0 : (inside == 2 ? 2 : 1);
  if (nsub != expect) {
    std::printf("*** tri %u: subtri count=%d expected=%d (inside=%d)\n", t, nsub, expect, inside);
    ++errors;
  }

  // Original-triangle edge functions in HDC (interior >= 0 after det-flip).
  gs::vec4f h[3];
  for (int i = 0; i < 3; ++i) {
    gs::vec4f c = { orig[i].pos[0], orig[i].pos[1], orig[i].pos[2], orig[i].pos[3] };
    h[i] = gs::ClipToHDC(c, 0, SETUP_W, 0, SETUP_H, SETUP_NEAR, SETUP_FAR);
  }
  gs::vec3f E[3];
  bool nondegen = gs::EdgeEquation(E, h[0], h[1], h[2]);

  for (int s = 0; s < nsub && errors < 16; ++s) {
    for (int j = 0; j < 3; ++j) {
      const setup_vertex_t& p = subs[s].v[j];
      // B: near half-space
      if (gs::near_dist(p) < -TOLN) {
        std::printf("*** tri %u sub %d v%d: near_dist=%g < 0\n", t, s, j, gs::near_dist(p));
        ++errors;
      }
      if (!nondegen) continue;
      // barycentric of p wrt original tri via edge functions (scale-free)
      gs::vec4f pc = { p.pos[0], p.pos[1], p.pos[2], p.pos[3] };
      gs::vec4f ph = gs::ClipToHDC(pc, 0, SETUP_W, 0, SETUP_H, SETUP_NEAR, SETUP_FAR);
      float e0 = E[0].x * ph.x + E[0].y * ph.y + E[0].z * ph.w;
      float e1 = E[1].x * ph.x + E[1].y * ph.y + E[1].z * ph.w;
      float e2 = E[2].x * ph.x + E[2].y * ph.y + E[2].z * ph.w;
      float sum = e0 + e1 + e2;
      float l0 = e0 / sum, l1 = e1 / sum, l2 = e2 / sum;
      // C: inside the original triangle
      float lmin = l0 < l1 ? (l0 < l2 ? l0 : l2) : (l1 < l2 ? l1 : l2);
      float lmax = l0 > l1 ? (l0 > l2 ? l0 : l2) : (l1 > l2 ? l1 : l2);
      if (lmin < -TOLB || lmax > 1.0f + TOLB) {
        std::printf("*** tri %u sub %d v%d: bary {%g,%g,%g} outside original\n", t, s, j, l0, l1, l2);
        ++errors;
      }
      // D: introduced (non-original) vertices sit at edge ∩ near-plane
      bool original = false;
      for (int k = 0; k < 3; ++k) {
        const setup_vertex_t& o = orig[k];
        if (std::fabs(p.pos[0] - o.pos[0]) < 1e-4f && std::fabs(p.pos[1] - o.pos[1]) < 1e-4f &&
            std::fabs(p.pos[2] - o.pos[2]) < 1e-4f && std::fabs(p.pos[3] - o.pos[3]) < 1e-4f)
          original = true;
      }
      if (!original) {
        if (std::fabs(gs::near_dist(p)) > TOLN) {
          std::printf("*** tri %u sub %d v%d: introduced but off near plane (z+w=%g)\n",
                      t, s, j, gs::near_dist(p)); ++errors;
        }
        if (lmin > TOLB) {  // not on any original edge
          std::printf("*** tri %u sub %d v%d: introduced but not on an original edge (bary min=%g)\n",
                      t, s, j, lmin); ++errors;
        }
      }
    }
  }
  return errors;
}

int main(int argc, char** argv) {
  parse_args(argc, argv);
  std::srand(50);
  const uint32_t n = g_num_prims;

  Scene sc = gen_scene(n);
  Golden ref = host_setup(sc.verts, n);
  const uint32_t P = (uint32_t)ref.prim.size();
  uint32_t ncross = 0;
  for (uint32_t t = 0; t < n; ++t) ncross += is_crossing(sc.cat[t]);
  std::printf("gfx_setup_kernel: n=%u  crossing=%u  kept P=%u\n", n, ncross, P);

  if (anchor_against_binning(sc, n)) {
    std::printf("RESULT: FAIL (reference diverges from Binning() oracle)\n");
    return 1;
  }
  std::printf("anchor: shared setup math matches Binning() oracle (no-clip subset)\n");

  vx_device_h dev = nullptr;
  CHECK(vx_device_open(0, &dev));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  vx_queue_h q = nullptr;
  CHECK(vx_queue_create(dev, &qi, &q));

  uint32_t one = 1, grid[1], block[1];
  CHECK(vx_device_max_occupancy_grid(dev, 1, &one, grid, block));
  const uint32_t T = block[0];
  const uint32_t G = grid[0];

  const uint32_t MS = SETUP_MAX_SUB;
  const uint32_t Pcap = P ? P : 1;
  const size_t   PRIM_SZ = sizeof(rast_prim_t);

  vx_buffer_h verts_buf, slot_prim_buf, slot_bbox_buf, slot_vtx_buf, keep_buf,
              offset_buf, tsum_buf, prim_buf, bbox_buf, vtx_buf, pid_buf, meta_buf;
  CHECK(vx_buffer_create(dev, 3 * n * sizeof(setup_vertex_t), VX_MEM_READ,  &verts_buf));
  CHECK(vx_buffer_create(dev, n * MS * PRIM_SZ,               VX_MEM_WRITE, &slot_prim_buf));
  CHECK(vx_buffer_create(dev, n * MS * sizeof(setup_bbox_t),  VX_MEM_WRITE, &slot_bbox_buf));
  CHECK(vx_buffer_create(dev, n * MS * sizeof(clip_tri_t),    VX_MEM_WRITE, &slot_vtx_buf));
  CHECK(vx_buffer_create(dev, n * sizeof(uint32_t),           VX_MEM_WRITE, &keep_buf));
  CHECK(vx_buffer_create(dev, (n + 1) * sizeof(uint32_t),     VX_MEM_WRITE, &offset_buf));
  CHECK(vx_buffer_create(dev, T * sizeof(uint32_t),           VX_MEM_WRITE, &tsum_buf));
  CHECK(vx_buffer_create(dev, Pcap * PRIM_SZ,                 VX_MEM_WRITE, &prim_buf));
  CHECK(vx_buffer_create(dev, Pcap * sizeof(setup_bbox_t),    VX_MEM_WRITE, &bbox_buf));
  CHECK(vx_buffer_create(dev, Pcap * sizeof(clip_tri_t),      VX_MEM_WRITE, &vtx_buf));
  CHECK(vx_buffer_create(dev, Pcap * sizeof(uint32_t),        VX_MEM_WRITE, &pid_buf));
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
  CHECK(vx_buffer_address(slot_vtx_buf,  &karg.slot_vtx_addr));
  CHECK(vx_buffer_address(keep_buf,      &karg.keep_addr));
  CHECK(vx_buffer_address(offset_buf,    &karg.offset_addr));
  CHECK(vx_buffer_address(tsum_buf,      &karg.tsum_addr));
  CHECK(vx_buffer_address(prim_buf,      &karg.prim_addr));
  CHECK(vx_buffer_address(bbox_buf,      &karg.bbox_addr));
  CHECK(vx_buffer_address(vtx_buf,       &karg.vtx_addr));
  CHECK(vx_buffer_address(pid_buf,       &karg.pid_addr));
  CHECK(vx_buffer_address(meta_buf,      &karg.meta_addr));

  CHECK(vx_enqueue_write(q, verts_buf, 0, sc.verts.data(), 3 * n * sizeof(setup_vertex_t), 0, nullptr, nullptr));

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
  std::vector<clip_tri_t>   h_vtx(Pcap);
  std::vector<uint32_t>     h_pid(Pcap);
  vx_event_h last = ev[NSTAGE - 1];
  vx_event_h ev_m = nullptr, ev_p = nullptr, ev_b = nullptr, ev_v = nullptr, ev_i = nullptr;
  CHECK(vx_enqueue_read(q, h_meta.data(), meta_buf, 0, sizeof(uint32_t),            1, &last, &ev_m));
  CHECK(vx_enqueue_read(q, h_prim.data(), prim_buf, 0, P * PRIM_SZ,                 1, &last, &ev_p));
  CHECK(vx_enqueue_read(q, h_bbox.data(), bbox_buf, 0, P * sizeof(setup_bbox_t),    1, &last, &ev_b));
  CHECK(vx_enqueue_read(q, h_vtx.data(),  vtx_buf,  0, P * sizeof(clip_tri_t),      1, &last, &ev_v));
  CHECK(vx_enqueue_read(q, h_pid.data(),  pid_buf,  0, P * sizeof(uint32_t),        1, &last, &ev_i));
  CHECK(vx_event_wait_value(ev_m, 1, VX_TIMEOUT_INFINITE));
  CHECK(vx_event_wait_value(ev_p, 1, VX_TIMEOUT_INFINITE));
  CHECK(vx_event_wait_value(ev_b, 1, VX_TIMEOUT_INFINITE));
  CHECK(vx_event_wait_value(ev_v, 1, VX_TIMEOUT_INFINITE));
  CHECK(vx_event_wait_value(ev_i, 1, VX_TIMEOUT_INFINITE));

  int errors = 0;
  if (h_meta[0] != P) { std::printf("*** P mismatch: dev=%u ref=%u\n", h_meta[0], P); ++errors; }

  // (1) device == shared host math, bit-for-bit.
  for (uint32_t i = 0; i < P && errors < 16; ++i) {
    if (std::memcmp(&h_prim[i], &ref.prim[i], sizeof(rast_prim_t)) != 0) {
      std::printf("*** prim[%u] device != reference\n", i); ++errors;
    }
    if (std::memcmp(&h_vtx[i], &ref.vtx[i], sizeof(clip_tri_t)) != 0) {
      std::printf("*** vtx[%u] device != reference\n", i); ++errors;
    }
    if (h_pid[i] != ref.pid[i]) {
      std::printf("*** pid[%u] dev=%u ref=%u\n", i, h_pid[i], ref.pid[i]); ++errors;
    }
    const auto& a = h_bbox[i]; const auto& b = ref.bbox[i];
    if (a.bbL != b.bbL || a.bbR != b.bbR || a.bbT != b.bbT || a.bbB != b.bbB) {
      std::printf("*** bbox[%u] dev{%u,%u,%u,%u} != ref{%u,%u,%u,%u}\n",
                  i, a.bbL, a.bbR, a.bbT, a.bbB, b.bbL, b.bbR, b.bbT, b.bbB); ++errors;
    }
  }

  // (3) geometric invariants on each crossing parent's clipped subtriangles
  // (run on the DEVICE output, grouped by parent pid).
  for (uint32_t i = 0; i < P && errors < 16; ) {
    uint32_t t = h_pid[i], j = i;
    while (j < P && h_pid[j] == t) ++j;
    if (is_crossing(sc.cat[t]))
      errors += check_invariants(&sc.verts[3 * t], &h_vtx[i], (int)(j - i), t);
    i = j;
  }
  // crossing parents that produced zero subtris (fully behind) — verify count.
  for (uint32_t t = 0; t < n && errors < 16; ++t) {
    if (sc.cat[t] != CAT_BEHIND) continue;
    bool present = false;
    for (uint32_t i = 0; i < P; ++i) if (h_pid[i] == t) { present = true; break; }
    if (present) { std::printf("*** behind-near tri %u emitted subtris\n", t); ++errors; }
  }

  // (4) Back-face culling: re-run the device with SETUP_CULL_BACK and
  // validate it matches the reference at CULL_BACK bit-for-bit, and that
  // culling actually removed the negative-area winding (kept_back < kept_none).
  Golden ref_back = host_setup(sc.verts, n, SETUP_CULL_BACK);
  const uint32_t Pb = (uint32_t)ref_back.prim.size();
  {
    kernel_arg_t kb[NSTAGE]; vx_launch_info_t lib[NSTAGE]; vx_event_h evb[NSTAGE] = {};
    for (uint32_t s = 0; s < NSTAGE; ++s) {
      kb[s] = karg; kb[s].stage = s; kb[s].cull_mode = SETUP_CULL_BACK;
      lib[s] = vx_launch_info_t{}; lib[s].struct_size = sizeof(lib[s]);
      lib[s].kernel = kern; lib[s].args_host = &kb[s]; lib[s].args_size = sizeof(kernel_arg_t);
      lib[s].ndim = 1; lib[s].grid_dim[0] = sgrid[s]; lib[s].block_dim[0] = T;
      CHECK(vx_enqueue_launch(q, &lib[s], s ? 1 : 0, s ? &evb[s - 1] : nullptr, &evb[s]));
    }
    std::vector<uint32_t>    hb_meta(1, 0);
    std::vector<rast_prim_t> hb_prim(Pb ? Pb : 1);
    vx_event_h lb = evb[NSTAGE - 1], em = nullptr, ep = nullptr;
    CHECK(vx_enqueue_read(q, hb_meta.data(), meta_buf, 0, sizeof(uint32_t), 1, &lb, &em));
    CHECK(vx_enqueue_read(q, hb_prim.data(), prim_buf, 0, Pb * PRIM_SZ,     1, &lb, &ep));
    CHECK(vx_event_wait_value(em, 1, VX_TIMEOUT_INFINITE));
    CHECK(vx_event_wait_value(ep, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(em); vx_event_release(ep);
    for (uint32_t s = 0; s < NSTAGE; ++s) vx_event_release(evb[s]);

    if (hb_meta[0] != Pb) { std::printf("*** cull P mismatch: dev=%u ref=%u\n", hb_meta[0], Pb); ++errors; }
    for (uint32_t i = 0; i < Pb && errors < 16; ++i)
      if (std::memcmp(&hb_prim[i], &ref_back.prim[i], sizeof(rast_prim_t)) != 0) {
        std::printf("*** cull prim[%u] device != reference\n", i); ++errors;
      }
    if (Pb >= P) { std::printf("*** CULL_BACK removed nothing (Pb=%u >= P=%u)\n", Pb, P); ++errors; }
    std::printf("cull: CULL_BACK kept P=%u of %u (device == reference, back faces culled)\n", Pb, P);
  }

  vx_event_release(ev_i); vx_event_release(ev_v); vx_event_release(ev_b);
  vx_event_release(ev_p); vx_event_release(ev_m);
  for (uint32_t s = 0; s < NSTAGE; ++s) vx_event_release(ev[s]);
  vx_buffer_release(verts_buf); vx_buffer_release(slot_prim_buf); vx_buffer_release(slot_bbox_buf);
  vx_buffer_release(slot_vtx_buf); vx_buffer_release(keep_buf); vx_buffer_release(offset_buf);
  vx_buffer_release(tsum_buf); vx_buffer_release(prim_buf); vx_buffer_release(bbox_buf);
  vx_buffer_release(vtx_buf); vx_buffer_release(pid_buf); vx_buffer_release(meta_buf);
  vx_module_release(mod);
  vx_queue_release(q);
  vx_device_release(dev);

  std::printf("RESULT: %s\n", errors == 0 ? "PASS" : "FAIL");
  return errors == 0 ? 0 : 1;
}
