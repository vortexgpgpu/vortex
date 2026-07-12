// gfx_v2 fused setup -> binning front end — SimX validation host.
// The CP runs the whole front end as nine chained device launches over resident
// buffers (setup: clip+setup -> dense bbox; binning: bin-sort -> per-bin prim
// lists), host-untouched between stages. The device's sorted per-bin prim lists
// are checked against the host Binning() oracle's tilebuf, end-to-end.
//
// Scene is non-crossing (all triangles in front of the near plane), so the clip
// is a passthrough and Binning() is a valid oracle for the full chain; clip
// correctness is covered separately by the gfx_setup_kernel test.

#include <vortex.h>
#include <graphics.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <map>
#include <utility>
#include <vector>
#include <unordered_map>
#include "common.h"

using vortex::graphics::rast_prim_t;
using vortex::graphics::rast_tile_header_t;

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

// Vertex in front of the near plane (ndc_z in [-0.8,0.8] -> z+w > 0).
static setup_vertex_t make_vertex(float ndc_x, float ndc_y, float w) {
  setup_vertex_t v;
  float ndc_z = frand(-0.8f, 0.8f);
  v.pos[0] = ndc_x * w; v.pos[1] = ndc_y * w; v.pos[2] = ndc_z * w; v.pos[3] = w;
  v.color[0] = frand(0, 1); v.color[1] = frand(0, 1);
  v.color[2] = frand(0, 1); v.color[3] = frand(0, 1);
  v.texcoord[0] = frand(0, 1); v.texcoord[1] = frand(0, 1);
  return v;
}

// Non-crossing scene: mostly on-screen triangles, plus degenerate and
// off-screen ones (both Binning() and setup cull these identically).
static std::vector<setup_vertex_t> gen_scene(uint32_t n) {
  std::vector<setup_vertex_t> verts(3 * n);
  for (uint32_t t = 0; t < n; ++t) {
    int roll = std::rand() % 10;
    float w = frand(1.0f, 4.0f);
    if (roll == 0) {                       // degenerate (collinear)
      setup_vertex_t v = make_vertex(frand(-0.7f, 0.7f), frand(-0.7f, 0.7f), w);
      verts[3 * t + 0] = v; verts[3 * t + 1] = v; verts[3 * t + 2] = v;
    } else if (roll == 1) {                // off-screen
      float cx = frand(2.0f, 3.0f), cy = frand(2.0f, 3.0f);
      verts[3 * t + 0] = make_vertex(cx,        cy,        w);
      verts[3 * t + 1] = make_vertex(cx + 0.2f, cy,        w);
      verts[3 * t + 2] = make_vertex(cx,        cy + 0.2f, w);
    } else {                               // common on-screen
      float cx = frand(-0.7f, 0.7f), cy = frand(-0.7f, 0.7f);
      for (int k = 0; k < 3; ++k)
        verts[3 * t + k] = make_vertex(cx + frand(-0.3f, 0.3f), cy + frand(-0.3f, 0.3f), w);
    }
  }
  return verts;
}

using TileMap = std::map<std::pair<uint16_t, uint16_t>, std::vector<uint32_t>>;

int main(int argc, char** argv) {
  parse_args(argc, argv);
  std::srand(50);
  const uint32_t n = g_num_prims;

  std::vector<setup_vertex_t> verts = gen_scene(n);

  // Host oracle: Binning() over the same triangles, parsed into a (tile -> pids)
  // map. Walk the tilebuf in tile order using pids_count (contiguous lists).
  using namespace vortex;
  std::unordered_map<uint32_t, graphics::vertex_t> vmap;
  std::vector<graphics::primitive_t> prims;
  for (uint32_t i = 0; i < 3 * n; ++i) {
    graphics::vertex_t gv;
    std::memcpy(&gv, &verts[i], sizeof(gv));
    vmap[i] = gv;
  }
  for (uint32_t t = 0; t < n; ++t) prims.push_back({3 * t + 0, 3 * t + 1, 3 * t + 2});

  std::vector<uint8_t> tilebuf, primbuf;
  uint32_t ntiles = graphics::Binning(tilebuf, primbuf, vmap, prims, SETUP_W, SETUP_H,
                                      SETUP_NEAR, SETUP_FAR, SETUP_BIN_LOG);
  const uint32_t P_ref = (uint32_t)(primbuf.size() / sizeof(rast_prim_t));

  TileMap gold;
  uint32_t Pbin_ref = 0;
  {
    auto* hdr = reinterpret_cast<const rast_tile_header_t*>(tilebuf.data());
    const uint8_t* pp = tilebuf.data() + (size_t)ntiles * sizeof(rast_tile_header_t);
    for (uint32_t i = 0; i < ntiles; ++i) {
      uint32_t cnt = hdr[i].pids_count;
      std::vector<uint32_t> v(cnt);
      std::memcpy(v.data(), pp, cnt * sizeof(uint32_t));
      pp += cnt * sizeof(uint32_t);
      gold[{hdr[i].tile_x, hdr[i].tile_y}] = std::move(v);
      Pbin_ref += cnt;
    }
  }
  std::printf("gfx_pipeline_kernel: n=%u  P=%u  tiles=%u  keys=%u\n", n, P_ref, ntiles, Pbin_ref);

  vx_device_h dev = nullptr;
  CHECK(vx_device_open(0, &dev));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  vx_queue_h q = nullptr;
  CHECK(vx_queue_create(dev, &qi, &q));

  uint32_t one = 1, grid[1], block[1];
  CHECK(vx_device_max_occupancy_grid(dev, 1, &one, grid, block));
  const uint32_t T = block[0];
  const uint32_t G = grid[0];
  const uint32_t B = PIPE_NUM_BINS;
  const uint32_t MS = SETUP_MAX_SUB;
  const uint32_t Kcap = Pbin_ref ? Pbin_ref : 1;

  const size_t PRIM_SZ = sizeof(rast_prim_t);
  const size_t HDR_SZ  = sizeof(rast_tile_header_t);
  const size_t TILEBUF_SZ = B * HDR_SZ + (size_t)Kcap * sizeof(uint32_t);

  vx_buffer_h verts_buf, slot_prim_buf, slot_bbox_buf, keep_buf, offset_buf, tsum_buf,
              prim_buf, bbox_buf, bcount_buf, boffset_buf, keys_buf, btsum_buf,
              thist_buf, bincount_buf, binbase_buf, tilebuf_buf, meta_buf;
  CHECK(vx_buffer_create(dev, 3 * n * sizeof(setup_vertex_t), VX_MEM_READ,  &verts_buf));
  CHECK(vx_buffer_create(dev, n * MS * PRIM_SZ,              VX_MEM_WRITE, &slot_prim_buf));
  CHECK(vx_buffer_create(dev, n * MS * sizeof(setup_bbox_t),  VX_MEM_WRITE, &slot_bbox_buf));
  CHECK(vx_buffer_create(dev, n * sizeof(uint32_t),           VX_MEM_WRITE, &keep_buf));
  CHECK(vx_buffer_create(dev, (n + 1) * sizeof(uint32_t),     VX_MEM_WRITE, &offset_buf));
  CHECK(vx_buffer_create(dev, T * sizeof(uint32_t),           VX_MEM_WRITE, &tsum_buf));
  CHECK(vx_buffer_create(dev, n * MS * PRIM_SZ,              VX_MEM_WRITE, &prim_buf));
  CHECK(vx_buffer_create(dev, n * MS * sizeof(setup_bbox_t),  VX_MEM_WRITE, &bbox_buf));
  CHECK(vx_buffer_create(dev, n * MS * sizeof(uint32_t),      VX_MEM_WRITE, &bcount_buf));
  CHECK(vx_buffer_create(dev, (n * MS + 1) * sizeof(uint32_t),VX_MEM_WRITE, &boffset_buf));
  CHECK(vx_buffer_create(dev, Kcap * sizeof(uint32_t),        VX_MEM_WRITE, &keys_buf));
  CHECK(vx_buffer_create(dev, T * sizeof(uint32_t),           VX_MEM_WRITE, &btsum_buf));
  CHECK(vx_buffer_create(dev, T * B * sizeof(uint32_t),       VX_MEM_WRITE, &thist_buf));
  CHECK(vx_buffer_create(dev, B * sizeof(uint32_t),           VX_MEM_WRITE, &bincount_buf));
  CHECK(vx_buffer_create(dev, B * sizeof(uint32_t),           VX_MEM_WRITE, &binbase_buf));
  CHECK(vx_buffer_create(dev, TILEBUF_SZ,                     VX_MEM_WRITE, &tilebuf_buf));
  CHECK(vx_buffer_create(dev, 3 * sizeof(uint32_t),           VX_MEM_WRITE, &meta_buf));

  vx_module_h mod = nullptr;
  vx_kernel_h k_setup = nullptr, k_binning = nullptr;
  CHECK(vx_module_load_file(dev, g_kernel_file, &mod));
  CHECK(vx_module_get_kernel(mod, "setup_k", &k_setup));
  CHECK(vx_module_get_kernel(mod, "binning_k", &k_binning));

  kernel_arg_t karg{};
  karg.num_tris = n;
  karg.width = SETUP_W;
  karg.height = SETUP_H;
  karg.bin_stripe = (B + G - 1) / G;
  CHECK(vx_buffer_address(verts_buf,    &karg.verts_addr));
  CHECK(vx_buffer_address(slot_prim_buf,&karg.slot_prim_addr));
  CHECK(vx_buffer_address(slot_bbox_buf,&karg.slot_bbox_addr));
  CHECK(vx_buffer_address(keep_buf,     &karg.keep_addr));
  CHECK(vx_buffer_address(offset_buf,   &karg.offset_addr));
  CHECK(vx_buffer_address(tsum_buf,     &karg.tsum_addr));
  CHECK(vx_buffer_address(prim_buf,     &karg.prim_addr));
  CHECK(vx_buffer_address(bbox_buf,     &karg.bbox_addr));
  CHECK(vx_buffer_address(bcount_buf,   &karg.bcount_addr));
  CHECK(vx_buffer_address(boffset_buf,  &karg.boffset_addr));
  CHECK(vx_buffer_address(keys_buf,     &karg.keys_addr));
  CHECK(vx_buffer_address(btsum_buf,    &karg.btsum_addr));
  CHECK(vx_buffer_address(thist_buf,    &karg.thist_addr));
  CHECK(vx_buffer_address(bincount_buf, &karg.bincount_addr));
  CHECK(vx_buffer_address(binbase_buf,  &karg.binbase_addr));
  CHECK(vx_buffer_address(tilebuf_buf,  &karg.tilebuf_addr));
  CHECK(vx_buffer_address(meta_buf,     &karg.meta_addr));

  CHECK(vx_enqueue_write(q, verts_buf, 0, verts.data(), 3 * n * sizeof(setup_vertex_t), 0, nullptr, nullptr));

  // CP-sequenced: 9 chained launches. multi-CTA (grid=G) except the three
  // single-CTA scans (grid=1). Each launch's drain is the device barrier.
  const uint32_t NSTAGE = 9;
  const uint32_t sgrid[NSTAGE] = { G, 1, G,  G, 1, G,  G, 1, G };
  kernel_arg_t kargs[NSTAGE];
  vx_launch_info_t li[NSTAGE];
  vx_event_h ev[NSTAGE] = {};
  for (uint32_t s = 0; s < NSTAGE; ++s) {
    kargs[s] = karg; kargs[s].stage = s;
    li[s] = vx_launch_info_t{};
    li[s].struct_size = sizeof(li[s]);
    li[s].kernel      = (s < PIPE_STAGE_BCOUNT) ? k_setup : k_binning;
    li[s].args_host   = &kargs[s];
    li[s].args_size   = sizeof(kernel_arg_t);
    li[s].ndim        = 1;
    li[s].grid_dim[0]  = sgrid[s];
    li[s].block_dim[0] = T;
    CHECK(vx_enqueue_launch(q, &li[s], s ? 1 : 0, s ? &ev[s - 1] : nullptr, &ev[s]));
  }

  std::vector<uint32_t> h_meta(3, 0);
  std::vector<rast_prim_t> h_prim(P_ref ? P_ref : 1);
  std::vector<uint8_t> h_tilebuf(TILEBUF_SZ);
  vx_event_h last = ev[NSTAGE - 1], ev_m = nullptr, ev_p = nullptr, ev_t = nullptr;
  CHECK(vx_enqueue_read(q, h_meta.data(),    meta_buf,    0, 3 * sizeof(uint32_t), 1, &last, &ev_m));
  CHECK(vx_enqueue_read(q, h_prim.data(),    prim_buf,    0, P_ref * PRIM_SZ,      1, &last, &ev_p));
  CHECK(vx_enqueue_read(q, h_tilebuf.data(), tilebuf_buf, 0, TILEBUF_SZ,           1, &last, &ev_t));
  CHECK(vx_event_wait_value(ev_m, 1, VX_TIMEOUT_INFINITE));
  CHECK(vx_event_wait_value(ev_p, 1, VX_TIMEOUT_INFINITE));
  CHECK(vx_event_wait_value(ev_t, 1, VX_TIMEOUT_INFINITE));

  int errors = 0;
  if (h_meta[0] != P_ref)    { std::printf("*** P mismatch: dev=%u ref=%u\n", h_meta[0], P_ref); ++errors; }
  if (h_meta[1] != Pbin_ref) { std::printf("*** keys mismatch: dev=%u ref=%u\n", h_meta[1], Pbin_ref); ++errors; }
  if (h_meta[2] != ntiles)   { std::printf("*** tile count mismatch: dev=%u ref=%u\n", h_meta[2], ntiles); ++errors; }

  // (1) primbuf bit-for-bit vs Binning() (same dense input order).
  auto* bprim = reinterpret_cast<const rast_prim_t*>(primbuf.data());
  for (uint32_t i = 0; i < P_ref && errors < 16; ++i)
    if (std::memcmp(&h_prim[i], &bprim[i], sizeof(rast_prim_t)) != 0) {
      std::printf("*** primbuf[%u] device != Binning()\n", i); ++errors;
    }

  // (2) Parse the device tilebuf exactly as RASTER does (pid block at
  // header_addr + 8 + pids_offset*4) into a (tile -> pids) map, vs the oracle.
  TileMap devmap;
  {
    auto* hdr = reinterpret_cast<const rast_tile_header_t*>(h_tilebuf.data());
    for (uint32_t i = 0; i < h_meta[2]; ++i) {
      size_t pid_byte = (size_t)i * HDR_SZ + HDR_SZ + (size_t)hdr[i].pids_offset * sizeof(uint32_t);
      auto* pp = reinterpret_cast<const uint32_t*>(h_tilebuf.data() + pid_byte);
      std::vector<uint32_t> v(hdr[i].pids_count);
      for (uint32_t j = 0; j < hdr[i].pids_count; ++j) v[j] = pp[j];
      devmap[{hdr[i].tile_x, hdr[i].tile_y}] = std::move(v);
    }
  }
  for (auto it = gold.begin(); it != gold.end() && errors < 16; ++it) {
    auto dit = devmap.find(it->first);
    if (dit == devmap.end()) {
      std::printf("*** tile (%u,%u) missing on device\n", it->first.first, it->first.second); ++errors; continue;
    }
    if (dit->second != it->second) {
      std::printf("*** tile (%u,%u) pid list differs (ref %zu, dev %zu prims)\n",
                  it->first.first, it->first.second, it->second.size(), dit->second.size()); ++errors;
    }
  }
  if (devmap.size() != gold.size() && errors < 16) {
    std::printf("*** tile-set size: dev=%zu ref=%zu\n", devmap.size(), gold.size()); ++errors;
  }

  vx_event_release(ev_t); vx_event_release(ev_p); vx_event_release(ev_m);
  for (uint32_t s = 0; s < NSTAGE; ++s) vx_event_release(ev[s]);
  vx_buffer_release(verts_buf); vx_buffer_release(slot_prim_buf); vx_buffer_release(slot_bbox_buf);
  vx_buffer_release(keep_buf); vx_buffer_release(offset_buf); vx_buffer_release(tsum_buf);
  vx_buffer_release(prim_buf); vx_buffer_release(bbox_buf); vx_buffer_release(bcount_buf);
  vx_buffer_release(boffset_buf); vx_buffer_release(keys_buf); vx_buffer_release(btsum_buf);
  vx_buffer_release(thist_buf); vx_buffer_release(bincount_buf); vx_buffer_release(binbase_buf);
  vx_buffer_release(tilebuf_buf); vx_buffer_release(meta_buf);
  vx_module_release(mod);
  vx_queue_release(q);
  vx_device_release(dev);

  std::printf("RESULT: %s\n", errors == 0 ? "PASS" : "FAIL");
  return errors == 0 ? 0 : 1;
}
