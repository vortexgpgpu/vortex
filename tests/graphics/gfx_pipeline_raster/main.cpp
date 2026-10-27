// gfx_v2 device front end -> RASTER, end to end.
//
// The fused setup+binning pipeline runs on the SIMT cores to produce RASTER's
// tilebuf + primbuf (pinned), which are bound to the RASTER unit and rasterised
// by a trivial fragment kernel. Two validations: (1) the device buffers match
// host Binning() (localises any front-end bug), and (2) the rendered image
// matches the gfx-v1 reference PNG (proves the device front end drives the FF
// unit to a pixel-correct result, no host Binning() in the loop).

#include <iostream>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <map>
#include <utility>
#include <algorithm>
#include <unordered_map>
#include <unistd.h>
#include <getopt.h>
#include <vortex2.h>
#include <graphics.h>
#include "common.h"
#include <cocogfx/include/cgltrace.hpp>
#include <cocogfx/include/imageutil.hpp>

using namespace cocogfx;
using namespace vortex;

#ifndef ASSETS_PATHS
#define ASSETS_PATHS ""
#endif

static std::string resolve_path(const std::string& filename, const std::string& searchPaths) {
  std::ifstream ifs(filename);
  if (!ifs) {
    std::stringstream ss(searchPaths);
    std::string path;
    while (std::getline(ss, path, ',')) {
      if (!path.empty()) {
        std::string fp = path + "/" + filename;
        std::ifstream t(fp);
        if (t) return fp;
      }
    }
  }
  return filename;
}

#define RT_CHECK(_expr)                                         \
  do {                                                          \
    int _ret = _expr;                                           \
    if (0 == _ret) break;                                       \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);    \
    exit(-1);                                                   \
  } while (false)

using rast_prim_t = graphics::rast_prim_t;
using rast_tile_header_t = graphics::rast_tile_header_t;  // host Binning() oracle (8 B)
using rast_bin_header_t  = graphics::rast_bin_header_t;   // device gfx_v2 bins (12 B)
using TileMap = std::map<std::pair<uint16_t, uint16_t>, std::vector<uint32_t>>;

static const char* kernel_file = "kernel.vxbin";
static const char* trace_file  = "triangle.cgltrace";
static const char* output_file = "output.png";
static const char* reference_file = nullptr;
static uint32_t dst_width = 128, dst_height = 128;
static uint32_t pass_prims = 0;   // -P: run the pass scenes, this many primitives per pass

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "t:o:r:w:h:k:P:?")) != -1) {
    switch (c) {
      case 't': trace_file = optarg; break;
      case 'o': output_file = optarg; break;
      case 'r': reference_file = optarg; break;
      case 'w': dst_width = std::atoi(optarg); break;
      case 'h': dst_height = std::atoi(optarg); break;
      case 'P': pass_prims = std::atoi(optarg); break;
      default: break;
    }
  }
}

// Build a non-indexed triangle list (the pipeline's assembly baseline) from the
// drawcall's indexed primitives; setup_vertex_t == graphics::vertex_t layout.
static std::vector<setup_vertex_t> expand_vertices(const CGLTrace::drawcall_t& dc) {
  std::vector<setup_vertex_t> out(3 * dc.primitives.size());
  for (size_t t = 0; t < dc.primitives.size(); ++t) {
    const auto& p = dc.primitives[t];
    uint32_t idx[3] = { p.i0, p.i1, p.i2 };
    for (int k = 0; k < 3; ++k) {
      const auto& v = dc.vertices.at(idx[k]);
      setup_vertex_t s{};   // varying2 feeds the w0..w5 planes; an
                            // indeterminate one reaches setup as an
                            // attribute plane.
      s.pos[0] = v.pos.x; s.pos[1] = v.pos.y; s.pos[2] = v.pos.z; s.pos[3] = v.pos.w;
      s.color[0] = v.color.r; s.color[1] = v.color.g; s.color[2] = v.color.b; s.color[3] = v.color.a;
      s.texcoord[0] = v.texcoord.u; s.texcoord[1] = v.texcoord.v;
      out[3 * t + k] = s;
    }
  }
  return out;
}

// Host Binning() oracle at the RASTER tile log, parsed into (tile -> pids).
static TileMap binning_oracle(const std::vector<setup_vertex_t>& verts, uint32_t ntri,
                              std::vector<uint8_t>& primbuf_out, uint32_t& nb, uint32_t& keys) {
  std::unordered_map<uint32_t, graphics::vertex_t> vmap;
  std::vector<graphics::primitive_t> prims;
  for (uint32_t i = 0; i < 3 * ntri; ++i) {
    graphics::vertex_t gv; std::memcpy(&gv, &verts[i], sizeof(gv)); vmap[i] = gv;
  }
  for (uint32_t t = 0; t < ntri; ++t) prims.push_back({3 * t, 3 * t + 1, 3 * t + 2});
  std::vector<uint8_t> tilebuf;
  nb = graphics::Binning(tilebuf, primbuf_out, vmap, prims, dst_width, dst_height,
                         SETUP_NEAR, SETUP_FAR, PIPE_BIN_LOG);
  TileMap m;
  // Host Binning() emits the coarse-bin layout (matching the
  // device front end): a dense rast_bin_header_t block followed by a sorted-pid
  // array, each bin's pids_offset an absolute index into it.
  auto* hdr = reinterpret_cast<const rast_bin_header_t*>(tilebuf.data());
  const uint32_t* pids = reinterpret_cast<const uint32_t*>(
      tilebuf.data() + (size_t)nb * sizeof(rast_bin_header_t));
  keys = 0;
  for (uint32_t i = 0; i < nb; ++i) {
    uint32_t cnt = hdr[i].pids_count;
    std::vector<uint32_t> v(pids + hdr[i].pids_offset, pids + hdr[i].pids_offset + cnt);
    m[{hdr[i].bin_x, hdr[i].bin_y}] = std::move(v); keys += cnt;
  }
  return m;
}

// ---- Pass scenes (-P): one draw run as one pass and as several --------------
//
// A draw larger than a RASTER pass runs as consecutive front-end passes over
// primitive ranges (pipe_arg_t::prim_base), each rebinning into the same buffers
// and followed by a fragment launch, all in one device-orchestrated draw. Each
// scene draws once as a single pass and once in passes of pass_prims, and the
// two must agree: the image (each pixel names its primitive's first VS record),
// the last pass's records and varying records against the single pass's tail,
// and the stipple counter left in meta[3]. The primitives cover disjoint pixels,
// so the image does not depend on fragment order. A stippled strip's counter
// must also be the one each of its segments starts at, in both.

struct PassScene {
  const char* name;
  uint32_t num_prims;                // primitives the launches are sized for
  std::vector<float> recs;           // clip-space positions, one vec4 per VS record
  // A draw with an input assembly: its 16-bit index positions, which ia_k cuts
  // into runs at kRestart and setup_ia_k takes the primitives of.
  std::vector<uint16_t> indices;
  uint32_t instances;
  std::vector<uint32_t> stipple_start;  // the counter each made segment starts at
  uint32_t stipple_end;                 // and the one the draw leaves
  pipe_arg_t pa;                     // the scene's draw state
};

static const uint16_t kRestart = 0xffff;

static void scene_rec(std::vector<float>& recs, float sx, float sy) {
  recs.push_back(2.0f * sx / dst_width - 1.0f);
  recs.push_back(2.0f * sy / dst_height - 1.0f);
  recs.push_back(0.5f);
  recs.push_back(1.0f);
}

// Instanced triangles: 7 instances of 3, so instance boundaries fall inside
// passes and pass boundaries inside instances.
static PassScene scene_instanced() {
  PassScene sc{};
  sc.name = "instanced";
  const uint32_t inst = 7, per = 3;
  for (uint32_t i = 0; i < inst * per; ++i) {
    const float x = float(2 * (i % 32)), y = float(2 * (i / 32));
    scene_rec(sc.recs, x + 0.1f, y + 0.1f);
    scene_rec(sc.recs, x + 0.95f, y + 0.1f);
    scene_rec(sc.recs, x + 0.1f, y + 0.95f);
  }
  sc.num_prims = inst * per;
  sc.pa.prim_type  = SETUP_PRIM_TRIANGLE;
  sc.pa.inst_prims = per;
  sc.pa.inst_verts = 3 * per;
  return sc;
}

// Triangles drawn as points (polygon mode): primitive t is vertex t % 3 of
// triangle t / 3, so pass boundaries fall inside triangles.
static PassScene scene_polygon_points() {
  PassScene sc{};
  sc.name = "polygon-points";
  const uint32_t tris = 7;
  for (uint32_t t = 0; t < tris; ++t) {
    const float x = float(4 * t) + 0.5f, y = 10.5f;
    scene_rec(sc.recs, x, y);
    scene_rec(sc.recs, x + 2.0f, y);
    scene_rec(sc.recs, x, y + 2.0f);
  }
  sc.num_prims = 3 * tris;
  sc.pa.prim_type  = SETUP_PRIM_POLYGON;
  sc.pa.poly_front = SETUP_POLY_POINT;
  sc.pa.poly_back  = SETUP_POLY_POINT;
  sc.pa.point_size = 1.0f;
  return sc;
}

// A stippled line strip of horizontal segments, one row each, cut by restart
// indices into runs that begin at segments 5 and 12, drawn `instances` times:
// the stipple counter restarts at each run and each instance, and carries
// across every other pass boundary through meta[3]. The launches are sized for
// the strip's closed form, which counts primitives the restarts take away, or
// with `exact` for just the ones it makes.
static PassScene scene_stipple_restart(const char* name, uint32_t instances, bool exact) {
  PassScene sc{};
  sc.name = name;
  sc.instances = instances;
  const uint32_t runs[] = { 5, 7, 9 };
  uint32_t segs = 0;
  for (uint32_t r : runs) {
    if (!sc.indices.empty()) {
      sc.indices.push_back(kRestart);
    }
    for (uint32_t v = 0; v <= r; ++v) {
      sc.indices.push_back((uint16_t)sc.indices.size());
    }
    segs += r;
  }
  for (uint32_t i = 0; i < instances; ++i) {
    uint32_t t = 0;
    for (uint32_t r : runs) {
      uint32_t ctr = 0;
      for (uint32_t k = 0; k < r; ++k, ++t) {
        const uint32_t len = 3 + t % 5;
        const float y = float(20 + i * segs + t) + 0.5f;
        scene_rec(sc.recs, 2.5f, y);
        scene_rec(sc.recs, 2.5f + float(len), y);
        sc.stipple_start.push_back(ctr);
        ctr += len;
      }
      sc.stipple_end = ctr;
    }
  }
  const uint32_t bound = (uint32_t)sc.indices.size() - 1;
  sc.num_prims = instances * (exact ? segs : bound);
  sc.pa.prim_type     = SETUP_PRIM_LINE;
  sc.pa.line_mode     = SETUP_LINE_RECT;
  sc.pa.line_width    = 1.0f;
  sc.pa.stipple_strip = 1;
  sc.pa.inst_prims    = instances > 1 ? bound : 0;
  sc.pa.inst_verts    = instances > 1 ? 2 * bound : 0;
  return sc;
}

struct PassResult {
  std::vector<uint32_t> image;
  std::vector<rast_prim_t> prims;    // the last pass's records
  std::vector<rast_vary_t> vary;
  std::vector<uint32_t> stipple;     // the last pass's segment counters
  uint32_t last_base;                // the last pass's first primitive
  uint32_t meta[PIPE_META_WORDS];
};

struct SceneKernels {
  vx_kernel_h setup, setup_ia, ia, binning, frag;
  uint32_t frag_lanes;
};

static int run_scene_draw(vx_device_h dev, vx_queue_h q, const SceneKernels& k,
                          const PassScene& sc, uint32_t per_pass, PassResult& out) {
  uint32_t one = 1, grid[1], block[1];
  RT_CHECK(vx_device_max_occupancy_grid(dev, 1, &one, grid, block));
  const uint32_t T = block[0], G = grid[0], MS = SETUP_MAX_SUB;
  const uint32_t ts = 1u << PIPE_BIN_LOG;
  const uint32_t bin_cols = (dst_width + ts - 1) / ts;
  const uint32_t B = bin_cols * ((dst_height + ts - 1) / ts);
  const uint32_t NT = sc.num_prims, P_max = NT * MS, K = P_max * B;
  const size_t PRIM_SZ = sizeof(rast_prim_t), HDR_SZ = sizeof(rast_bin_header_t);
  const uint32_t cbuf_size = dst_width * dst_height * 4;

  // One working set for the whole draw: every pass rebins into it.
  struct Buf { uint64_t bytes; uint32_t flags; uint64_t* addr; vx_buffer_h h; };
  pipe_arg_t pa = sc.pa;
  const bool assembled = !sc.indices.empty();
  const uint32_t ia_bound = assembled ? (uint32_t)sc.indices.size() - 1 : 0;
  uint64_t recs_addr = 0, index_addr = 0, cbuf_addr = 0, fa_addr = 0;
  uint64_t ia_addr = 0, ia_part_addr = 0, ia_tsum_addr = 0, ia_map_addr = 0;
  const uint32_t RWP = VX_MEM_READ | VX_MEM_WRITE | VX_MEM_PHYS;
  std::vector<Buf> bufs = {
    { sc.recs.size() * sizeof(float),          VX_MEM_READ,  &recs_addr,         nullptr },
    { std::max<size_t>(sc.indices.size(), 2) * 2, VX_MEM_READ, &index_addr,      nullptr },
    { (uint64_t)NT * 4,                        VX_MEM_WRITE, &pa.keep_addr,      nullptr },
    { (uint64_t)(NT + 1) * 4,                  VX_MEM_WRITE, &pa.offset_addr,    nullptr },
    { (uint64_t)T * 4,                         VX_MEM_WRITE, &pa.tsum_addr,      nullptr },
    { (uint64_t)NT * 4,                        VX_MEM_WRITE, &pa.stipple_addr,   nullptr },
    { (uint64_t)P_max * PRIM_SZ,               RWP,          &pa.prim_addr,      nullptr },
    { (uint64_t)P_max * sizeof(setup_bbox_t),  VX_MEM_WRITE, &pa.bbox_addr,      nullptr },
    { (uint64_t)P_max * 4,                     VX_MEM_WRITE, &pa.bcount_addr,    nullptr },
    { (uint64_t)(P_max + 1) * 4,               VX_MEM_WRITE, &pa.boffset_addr,   nullptr },
    { (uint64_t)K * 4,                         VX_MEM_WRITE, &pa.keys_addr,      nullptr },
    { (uint64_t)T * 4,                         VX_MEM_WRITE, &pa.btsum_addr,     nullptr },
    { (uint64_t)T * B * 4,                     VX_MEM_WRITE, &pa.thist_addr,     nullptr },
    { (uint64_t)B * 4,                         VX_MEM_WRITE, &pa.bincount_addr,  nullptr },
    { (uint64_t)B * 4,                         VX_MEM_WRITE, &pa.binbase_addr,   nullptr },
    { (uint64_t)B * HDR_SZ + (uint64_t)K * 4,  RWP,          &pa.tilebuf_addr,   nullptr },
    { (uint64_t)PIPE_META_WORDS * 4,           VX_MEM_WRITE, &pa.meta_addr,      nullptr },
    { (uint64_t)P_max * sizeof(rast_vary_t),   VX_MEM_WRITE, &pa.vary_addr,      nullptr },
    { cbuf_size,                               VX_MEM_WRITE, &cbuf_addr,         nullptr },
    { sizeof(frag_arg_t),                      VX_MEM_READ,  &fa_addr,           nullptr },
    { sizeof(gfx_ia_t),                        VX_MEM_READ | VX_MEM_WRITE, &ia_addr,          nullptr },
    { std::max<uint64_t>(ia_bound, 1) * sizeof(gfx_ia_prim_t), VX_MEM_WRITE, &ia_map_addr, nullptr },
    { (uint64_t)T * 3 * 4,                     VX_MEM_WRITE, &ia_part_addr,      nullptr },
    { (uint64_t)T * 2 * 4,                     VX_MEM_WRITE, &ia_tsum_addr,      nullptr },
  };
  for (auto& b : bufs) {
    RT_CHECK(vx_buffer_create(dev, b.bytes, b.flags, &b.h));
    RT_CHECK(vx_buffer_address(b.h, b.addr));
  }
  vx_buffer_h recs_b = bufs[0].h, index_b = bufs[1].h, stipple_b = bufs[5].h, prim_b = bufs[6].h;
  vx_buffer_h meta_b = bufs[16].h, vary_b = bufs[17].h, cbuf_b = bufs[18].h, fa_b = bufs[19].h;
  vx_buffer_h ia_b = bufs[20].h;

  pa.width = dst_width; pa.height = dst_height;
  pa.bin_cols = bin_cols; pa.num_bins = B;
  pa.bin_stripe = (B + G - 1) / G;
  pa.vsrec_addr = recs_addr;
  pa.vstride = 4 * sizeof(float);
  pa.ia_addr = assembled ? ia_addr : 0;

  // The draw's input assembly as the driver records it: the primitives the
  // strip's closed form bounds, which ia_k replaces with the ones the restarts
  // leave, mapping each to its run.
  gfx_ia_t ia{};
  ia.index_addr    = index_addr;
  ia.map_addr      = ia_map_addr;
  ia.index_size    = 2;
  ia.index_limit   = (uint32_t)sc.indices.size();
  ia.count         = (uint32_t)sc.indices.size();
  ia.instances     = sc.instances;
  ia.topology      = GFX_IA_LINE_STRIP;
  ia.per           = 2;
  ia.restart       = 1;
  ia.restart_index = kRestart;
  ia.prims         = ia_bound;
  ia_arg_t iarg{};
  iarg.ia_addr   = ia_addr;
  iarg.part_addr = ia_part_addr;
  iarg.tsum_addr = ia_tsum_addr;
  iarg.stage     = IA_STAGE_ALL;
  iarg.threads   = T;

  frag_arg_t fa = {};
  fa.dst_width = dst_width; fa.dst_height = dst_height;
  fa.cbuf_addr = cbuf_addr; fa.cbuf_stride = 4; fa.cbuf_pitch = dst_width * 4;
  fa.prim_addr = pa.prim_addr;
  fa.vary_addr = pa.vary_addr;

  std::vector<uint32_t> clear(cbuf_size / 4, 0xff000000);
  std::vector<uint32_t> zero_meta(PIPE_META_WORDS, 0);
  RT_CHECK(vx_enqueue_write(q, recs_b, 0, sc.recs.data(), sc.recs.size() * sizeof(float), 0, nullptr, nullptr));
  if (assembled) {
    RT_CHECK(vx_enqueue_write(q, index_b, 0, sc.indices.data(), sc.indices.size() * 2, 0, nullptr, nullptr));
    RT_CHECK(vx_enqueue_write(q, ia_b, 0, &ia, sizeof(ia), 0, nullptr, nullptr));
  }
  RT_CHECK(vx_enqueue_write(q, cbuf_b, 0, clear.data(), cbuf_size, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(q, meta_b, 0, zero_meta.data(), PIPE_META_WORDS * 4, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(q, fa_b, 0, &fa, sizeof(fa), 0, nullptr, nullptr));

  // The whole draw as one command batch, the way the driver issues it: ia_k
  // first when the draw has an input assembly, then per pass the nine
  // front-end launches, the RASTER configuration before the first fragment
  // launch, then the fragment launch.
  const uint32_t NSTAGE = 9;
  const uint32_t sgrid[NSTAGE] = { G, 1, G,  G, 1, G,  G, 1, G };
  const uint32_t passes = (NT + per_pass - 1) / per_pass;
  std::vector<pipe_arg_t> pargs(passes * NSTAGE);
  std::vector<vx_launch_info_t> pli(passes * NSTAGE);
  uint64_t frag_entry = 0;
  RT_CHECK(vx_kernel_address(k.frag, &frag_entry));
  vx_launch_info_t fli = {}; fli.struct_size = sizeof(fli); fli.kernel = k.frag;
  fli.args_host = &fa; fli.args_size = sizeof(fa); fli.ndim = 1;
  fli.grid_dim[0] = 0; fli.block_dim[0] = k.frag_lanes;
  std::vector<vx_command_t> cmds;
  auto launch = [&](const vx_launch_info_t* l) {
    vx_command_t c{}; c.type = VX_COMMAND_LAUNCH; c.data.launch = l; cmds.push_back(c);
  };
  auto dcr = [&](uint32_t a, uint32_t v) {
    vx_command_t c{}; c.type = VX_COMMAND_DCR_WRITE; c.data.dcr.addr = a; c.data.dcr.value = v;
    cmds.push_back(c);
  };
  vx_launch_info_t ili = {}; ili.struct_size = sizeof(ili); ili.kernel = k.ia;
  ili.args_host = &iarg; ili.args_size = sizeof(iarg); ili.ndim = 1;
  ili.grid_dim[0] = 1; ili.block_dim[0] = T;
  if (assembled) {
    launch(&ili);
  }
  for (uint32_t p = 0; p < passes; ++p) {
    for (uint32_t s = 0; s < NSTAGE; ++s) {
      const uint32_t i = p * NSTAGE + s;
      pargs[i] = pa;
      pargs[i].stage     = s;
      pargs[i].prim_base = p * per_pass;
      pargs[i].num_tris  = std::min(per_pass, NT - p * per_pass);
      pli[i] = vx_launch_info_t{}; pli[i].struct_size = sizeof(pli[i]);
      pli[i].kernel = s >= PIPE_STAGE_BCOUNT ? k.binning : assembled ? k.setup_ia : k.setup;
      pli[i].args_host = &pargs[i]; pli[i].args_size = sizeof(pipe_arg_t);
      pli[i].ndim = 1; pli[i].grid_dim[0] = sgrid[s]; pli[i].block_dim[0] = T;
      launch(&pli[i]);
    }
    if (p == 0) {
      dcr(VX_DCR_RASTER_TBUF_ADDR, (uint32_t)(pa.tilebuf_addr / 64));
      dcr(VX_DCR_RASTER_TILE_COUNT, B);
      dcr(VX_DCR_RASTER_PBUF_ADDR, (uint32_t)(pa.prim_addr / 64));
      dcr(VX_DCR_RASTER_PBUF_STRIDE, (uint32_t)PRIM_SZ);
      dcr(VX_DCR_RASTER_SCISSOR_X, dst_width << 16);
      dcr(VX_DCR_RASTER_SCISSOR_Y, dst_height << 16);
      dcr(VX_DCR_RASTER_FRAG_ENTRY_LO, (uint32_t)(frag_entry & 0xffffffff));
      dcr(VX_DCR_RASTER_FRAG_ENTRY_HI, (uint32_t)(frag_entry >> 32));
      dcr(VX_DCR_RASTER_FRAG_PARAM_LO, (uint32_t)(fa_addr & 0xffffffff));
      dcr(VX_DCR_RASTER_FRAG_PARAM_HI, (uint32_t)(fa_addr >> 32));
    }
    launch(&fli);
  }
  vx_event_h ev = nullptr;
  RT_CHECK(vx_enqueue_draw(q, cmds.data(), (uint32_t)cmds.size(), 0, nullptr, &ev));
  RT_CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(ev);

  out.image.assign(cbuf_size / 4, 0);
  vx_event_h e[3] = {};
  RT_CHECK(vx_enqueue_read(q, out.image.data(), cbuf_b, 0, cbuf_size, 0, nullptr, &e[0]));
  RT_CHECK(vx_enqueue_read(q, out.meta, meta_b, 0, PIPE_META_WORDS * 4, 0, nullptr, &e[1]));
  RT_CHECK(vx_event_wait_value(e[1], 1, VX_TIMEOUT_INFINITE));
  RT_CHECK(vx_event_wait_value(e[0], 1, VX_TIMEOUT_INFINITE));
  vx_event_release(e[0]); vx_event_release(e[1]);
  const uint32_t P = out.meta[0];
  out.prims.resize(P ? P : 1);
  out.vary.resize(P ? P : 1);
  if (P) {
    RT_CHECK(vx_enqueue_read(q, out.prims.data(), prim_b, 0, P * PRIM_SZ, 0, nullptr, &e[0]));
    RT_CHECK(vx_enqueue_read(q, out.vary.data(), vary_b, 0, P * sizeof(rast_vary_t), 0, nullptr, &e[1]));
    RT_CHECK(vx_event_wait_value(e[0], 1, VX_TIMEOUT_INFINITE));
    RT_CHECK(vx_event_wait_value(e[1], 1, VX_TIMEOUT_INFINITE));
    vx_event_release(e[0]); vx_event_release(e[1]);
  }
  out.prims.resize(P);
  out.vary.resize(P);
  out.last_base = (passes - 1) * per_pass;
  if (sc.pa.stipple_strip) {
    out.stipple.resize(NT - out.last_base);
    RT_CHECK(vx_enqueue_read(q, out.stipple.data(), stipple_b, 0, out.stipple.size() * 4, 0, nullptr, &e[0]));
    RT_CHECK(vx_event_wait_value(e[0], 1, VX_TIMEOUT_INFINITE));
    vx_event_release(e[0]);
  }
  for (auto& b : bufs) {
    vx_buffer_release(b.h);
  }
  return 0;
}

static int run_pass_scenes(vx_device_h dev, vx_queue_h q, const SceneKernels& k) {
  int errors = 0;
  for (const PassScene& sc : { scene_instanced(), scene_polygon_points(),
                               scene_stipple_restart("stipple-restart", 1, false),
                               scene_stipple_restart("stipple-restart-instanced", 2, true) }) {
    PassResult one, many;
    RT_CHECK(run_scene_draw(dev, q, k, sc, sc.num_prims, one));
    RT_CHECK(run_scene_draw(dev, q, k, sc, pass_prims, many));
    int scene_errors = 0;
    uint32_t covered = 0;
    for (size_t i = 0; i < one.image.size(); ++i) {
      covered += (one.image[i] != 0xff000000);
      if (one.image[i] != many.image[i]) {
        if (scene_errors < 4) {
          std::printf("*** %s: pixel (%zu, %zu) = 0x%08x in passes, 0x%08x in one\n", sc.name,
                      i % dst_width, i / dst_width, many.image[i], one.image[i]);
        }
        ++scene_errors;
      }
    }
    if (covered == 0) {
      std::printf("*** %s: the single pass covers no pixel\n", sc.name);
      ++scene_errors;
    }
    // The last pass's records are the single pass's tail.
    const uint32_t P_one = one.meta[0], P_last = many.meta[0];
    if (P_last == 0 || P_last > P_one) {
      std::printf("*** %s: last pass kept %u records of %u\n", sc.name, P_last, P_one);
      ++scene_errors;
    } else {
      const uint32_t tail = P_one - P_last;
      for (uint32_t i = 0; i < P_last; ++i) {
        if (std::memcmp(&many.prims[i], &one.prims[tail + i], sizeof(rast_prim_t)) != 0
         || std::memcmp(&many.vary[i], &one.vary[tail + i], sizeof(rast_vary_t)) != 0) {
          std::printf("*** %s: last-pass record %u differs from record %u of the single pass\n",
                      sc.name, i, tail + i);
          ++scene_errors;
          break;
        }
      }
    }
    if (sc.pa.stipple_strip && one.meta[3] != many.meta[3]) {
      std::printf("*** %s: stipple counter %u after the passes, %u after one\n",
                  sc.name, many.meta[3], one.meta[3]);
      ++scene_errors;
    }
    if (sc.pa.stipple_strip && one.meta[3] != sc.stipple_end) {
      std::printf("*** %s: stipple counter %u after the draw, expected %u\n",
                  sc.name, one.meta[3], sc.stipple_end);
      ++scene_errors;
    }
    for (const PassResult* r : { &one, &many }) {
      for (uint32_t i = 0; i < r->stipple.size(); ++i) {
        const uint32_t t = r->last_base + i;
        if (t < sc.stipple_start.size() && r->stipple[i] != sc.stipple_start[t]) {
          std::printf("*** %s: segment %u starts at stipple counter %u, expected %u%s\n",
                      sc.name, t, r->stipple[i], sc.stipple_start[t],
                      r == &one ? "" : " (passes)");
          ++scene_errors;
          break;
        }
      }
    }
    std::printf("%s: %u primitives, passes of %u, %u pixels covered, %u + %u records: %s\n",
                sc.name, sc.num_prims, pass_prims, covered, P_one - P_last, P_last,
                scene_errors ? "FAIL" : "PASS");
    errors += scene_errors;
  }
  return errors;
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

  CGLTrace trace;
  RT_CHECK(trace.load(resolve_path(trace_file, ASSETS_PATHS).c_str()));

  vx_device_h dev = nullptr;
  RT_CHECK(vx_device_open(0, &dev));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  vx_queue_h q = nullptr;
  RT_CHECK(vx_queue_create(dev, &qi, &q));

  uint64_t isa_flags = 0;
  RT_CHECK(vx_device_query(dev, VX_CAPS_ISA_FLAGS, &isa_flags));
  if (!(isa_flags & VX_ISA_EXT_RASTER)) { std::cout << "RASTER not supported!\n"; return -1; }
  uint64_t num_cores = 0, num_warps = 0, num_threads = 0;
  RT_CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS, &num_threads));

  vx_module_h mod = nullptr;
  vx_kernel_h k_setup = nullptr, k_binning = nullptr, k_frag = nullptr;
  RT_CHECK(vx_module_load_file(dev, kernel_file, &mod));
  RT_CHECK(vx_module_get_kernel(mod, "setup_k", &k_setup));
  RT_CHECK(vx_module_get_kernel(mod, "binning_k", &k_binning));
  RT_CHECK(vx_module_get_kernel(mod, "main", &k_frag));

  if (pass_prims) {
    SceneKernels k{ k_setup, nullptr, nullptr, k_binning, k_frag,
                    (uint32_t)(num_threads * num_warps) };
    RT_CHECK(vx_module_get_kernel(mod, "setup_ia_k", &k.setup_ia));
    RT_CHECK(vx_module_get_kernel(mod, "ia_k", &k.ia));
    const int errors = run_pass_scenes(dev, q, k);
    vx_module_release(mod); vx_queue_release(q); vx_device_release(dev);
    std::cout << (errors ? "FAILED!" : "PASSED!") << std::endl;
    return errors ? 1 : 0;
  }

  // color buffer (kernel-written via LSU; not pinned)
  uint32_t cbuf_stride = 4, cbuf_pitch = dst_width * cbuf_stride, cbuf_size = dst_height * cbuf_pitch;
  vx_buffer_h color_buffer = nullptr;
  uint64_t cbuf_addr = 0;
  RT_CHECK(vx_buffer_create(dev, cbuf_size, VX_MEM_WRITE, &color_buffer));
  RT_CHECK(vx_buffer_address(color_buffer, &cbuf_addr));
  { std::vector<uint32_t> clr(cbuf_size / 4, 0xff000000); vx_event_h e = nullptr;
    RT_CHECK(vx_enqueue_write(q, color_buffer, 0, clr.data(), cbuf_size, 0, nullptr, &e));
    RT_CHECK(vx_event_wait_value(e, 1, VX_TIMEOUT_INFINITE)); vx_event_release(e); }

  int total_errors = 0;
  for (auto& dc : trace.drawcalls) {
    const uint32_t ntri = (uint32_t)dc.primitives.size();
    if (ntri == 0) continue;
    std::vector<setup_vertex_t> verts = expand_vertices(dc);

    // Host oracle (sizes + cross-check).
    std::vector<uint8_t> ref_primbuf;
    uint32_t nb_ref = 0, keys_ref = 0;
    TileMap gold = binning_oracle(verts, ntri, ref_primbuf, nb_ref, keys_ref);
    const uint32_t P_ref = (uint32_t)(ref_primbuf.size() / sizeof(rast_prim_t));
    std::cout << "drawcall: tris=" << ntri << " P=" << P_ref << " tiles=" << nb_ref
              << " keys=" << keys_ref << std::endl;
    if (nb_ref == 0) continue;

    // Pipeline grid/block.
    uint32_t one = 1, grid[1], block[1];
    RT_CHECK(vx_device_max_occupancy_grid(dev, 1, &one, grid, block));
    const uint32_t T = block[0], G = grid[0], MS = SETUP_MAX_SUB;
    // Dense tile grid sized to the render target — RASTER's tile count is then a
    // host-known function of the framebuffer (no device readback of num_tiles).
    const uint32_t ts = 1u << PIPE_BIN_LOG;
    const uint32_t bin_cols = (dst_width + ts - 1) / ts;
    const uint32_t bin_rows = (dst_height + ts - 1) / ts;
    const uint32_t B = bin_cols * bin_rows;
    const uint32_t Kcap = keys_ref ? keys_ref : 1;
    const size_t PRIM_SZ = sizeof(rast_prim_t), HDR_SZ = sizeof(rast_bin_header_t);
    const size_t TILEBUF_SZ = (size_t)B * HDR_SZ + (size_t)Kcap * sizeof(uint32_t);

    // Buffers. prim/tilebuf are pinned (RASTER's AXI master reads them).
    vx_buffer_h verts_b, slot_prim_b, slot_bbox_b, keep_b, offset_b, tsum_b, prim_b, bbox_b,
                bcount_b, boffset_b, keys_b, btsum_b, thist_b, bincount_b, binbase_b, tilebuf_b, meta_b;
    RT_CHECK(vx_buffer_create(dev, 3 * ntri * sizeof(setup_vertex_t), VX_MEM_READ,  &verts_b));
    RT_CHECK(vx_buffer_create(dev, ntri * MS * PRIM_SZ,             VX_MEM_WRITE, &slot_prim_b));
    RT_CHECK(vx_buffer_create(dev, ntri * MS * sizeof(setup_bbox_t), VX_MEM_WRITE, &slot_bbox_b));
    RT_CHECK(vx_buffer_create(dev, ntri * sizeof(uint32_t),          VX_MEM_WRITE, &keep_b));
    RT_CHECK(vx_buffer_create(dev, (ntri + 1) * sizeof(uint32_t),    VX_MEM_WRITE, &offset_b));
    RT_CHECK(vx_buffer_create(dev, T * sizeof(uint32_t),            VX_MEM_WRITE, &tsum_b));
    RT_CHECK(vx_buffer_create(dev, ntri * MS * PRIM_SZ, VX_MEM_READ | VX_MEM_WRITE | VX_MEM_PHYS, &prim_b));
    RT_CHECK(vx_buffer_create(dev, ntri * MS * sizeof(setup_bbox_t), VX_MEM_WRITE, &bbox_b));
    RT_CHECK(vx_buffer_create(dev, ntri * MS * sizeof(uint32_t),     VX_MEM_WRITE, &bcount_b));
    RT_CHECK(vx_buffer_create(dev, (ntri * MS + 1) * sizeof(uint32_t), VX_MEM_WRITE, &boffset_b));
    RT_CHECK(vx_buffer_create(dev, Kcap * sizeof(uint32_t),          VX_MEM_WRITE, &keys_b));
    RT_CHECK(vx_buffer_create(dev, T * sizeof(uint32_t),            VX_MEM_WRITE, &btsum_b));
    RT_CHECK(vx_buffer_create(dev, T * B * sizeof(uint32_t),        VX_MEM_WRITE, &thist_b));
    RT_CHECK(vx_buffer_create(dev, B * sizeof(uint32_t),           VX_MEM_WRITE, &bincount_b));
    RT_CHECK(vx_buffer_create(dev, B * sizeof(uint32_t),           VX_MEM_WRITE, &binbase_b));
    RT_CHECK(vx_buffer_create(dev, TILEBUF_SZ, VX_MEM_READ | VX_MEM_WRITE | VX_MEM_PHYS, &tilebuf_b));
    RT_CHECK(vx_buffer_create(dev, PIPE_META_WORDS * sizeof(uint32_t), VX_MEM_WRITE, &meta_b));

    pipe_arg_t pa = {};
    pa.num_tris = ntri; pa.width = dst_width; pa.height = dst_height;
    pa.bin_cols = bin_cols; pa.num_bins = B;
    pa.bin_stripe = (B + G - 1) / G;
    uint64_t prim_addr = 0, tilebuf_addr = 0;
    RT_CHECK(vx_buffer_address(verts_b,     &pa.verts_addr));
    RT_CHECK(vx_buffer_address(slot_prim_b, &pa.slot_prim_addr));
    RT_CHECK(vx_buffer_address(slot_bbox_b, &pa.slot_bbox_addr));
    RT_CHECK(vx_buffer_address(keep_b,      &pa.keep_addr));
    RT_CHECK(vx_buffer_address(offset_b,    &pa.offset_addr));
    RT_CHECK(vx_buffer_address(tsum_b,      &pa.tsum_addr));
    RT_CHECK(vx_buffer_address(prim_b,      &pa.prim_addr));    prim_addr = pa.prim_addr;
    RT_CHECK(vx_buffer_address(bbox_b,      &pa.bbox_addr));
    RT_CHECK(vx_buffer_address(bcount_b,    &pa.bcount_addr));
    RT_CHECK(vx_buffer_address(boffset_b,   &pa.boffset_addr));
    RT_CHECK(vx_buffer_address(keys_b,      &pa.keys_addr));
    RT_CHECK(vx_buffer_address(btsum_b,     &pa.btsum_addr));
    RT_CHECK(vx_buffer_address(thist_b,     &pa.thist_addr));
    RT_CHECK(vx_buffer_address(bincount_b,  &pa.bincount_addr));
    RT_CHECK(vx_buffer_address(binbase_b,   &pa.binbase_addr));
    RT_CHECK(vx_buffer_address(tilebuf_b,   &pa.tilebuf_addr)); tilebuf_addr = pa.tilebuf_addr;
    RT_CHECK(vx_buffer_address(meta_b,      &pa.meta_addr));

    RT_CHECK(vx_enqueue_write(q, verts_b, 0, verts.data(), 3 * ntri * sizeof(setup_vertex_t), 0, nullptr, nullptr));

    // Nine CP-sequenced launches: setup_k (0-2), binning_k (3-8).
    const uint32_t NSTAGE = 9;
    const uint32_t sgrid[NSTAGE] = { G, 1, G,  G, 1, G,  G, 1, G };
    pipe_arg_t pargs[NSTAGE];
    vx_launch_info_t pli[NSTAGE];
    vx_event_h pev[NSTAGE] = {};
    for (uint32_t s = 0; s < NSTAGE; ++s) {
      pargs[s] = pa; pargs[s].stage = s;
      pli[s] = vx_launch_info_t{}; pli[s].struct_size = sizeof(pli[s]);
      pli[s].kernel = (s < PIPE_STAGE_BCOUNT) ? k_setup : k_binning;
      pli[s].args_host = &pargs[s]; pli[s].args_size = sizeof(pipe_arg_t);
      pli[s].ndim = 1; pli[s].grid_dim[0] = sgrid[s]; pli[s].block_dim[0] = T;
      RT_CHECK(vx_enqueue_launch(q, &pli[s], s ? 1 : 0, s ? &pev[s - 1] : nullptr, &pev[s]));
    }
    vx_event_h plast = pev[NSTAGE - 1];

    // Host-free renderpass: the binning emits a dense tile grid, so RASTER's
    // tile count is B = bin_cols*bin_rows — a host-known function of the
    // framebuffer, NOT the geometry. So the host programs RASTER and launches the
    // fragment kernel as one submitted command stream (chained on the pipeline's
    // drain) with no mid-renderpass readback, and waits only for the result.
    frag_arg_t fa = {};
    fa.dst_width = dst_width; fa.dst_height = dst_height;
    fa.cbuf_addr = cbuf_addr; fa.cbuf_stride = (uint8_t)cbuf_stride; fa.cbuf_pitch = cbuf_pitch;
    fa.prim_addr = prim_addr;
    // RASTER dispatch v2 (push): the raster engine launches the FS on-device.
    // Stage the FS args in device memory and program the fragment-dispatch
    // descriptor (FS entry PC + args pointer) in the RASTER DCR block.
    vx_buffer_h fa_b = nullptr; uint64_t fa_addr = 0;
    RT_CHECK(vx_buffer_create(dev, sizeof(fa), VX_MEM_READ, &fa_b));
    RT_CHECK(vx_buffer_address(fa_b, &fa_addr));
    RT_CHECK(vx_enqueue_write(q, fa_b, 0, &fa, sizeof(fa), 1, &plast, nullptr));
    uint64_t frag_entry = 0;
    RT_CHECK(vx_kernel_address(k_frag, &frag_entry));
    vx_enqueue_dcr_write(q, VX_DCR_RASTER_TBUF_ADDR, tilebuf_addr / 64, 1, &plast, nullptr);
    vx_enqueue_dcr_write(q, VX_DCR_RASTER_TILE_COUNT, B, 0, nullptr, nullptr);
    vx_enqueue_dcr_write(q, VX_DCR_RASTER_PBUF_ADDR, prim_addr / 64, 0, nullptr, nullptr);
    vx_enqueue_dcr_write(q, VX_DCR_RASTER_PBUF_STRIDE, (uint32_t)PRIM_SZ, 0, nullptr, nullptr);
    vx_enqueue_dcr_write(q, VX_DCR_RASTER_SCISSOR_X, (dst_width << 16) | 0, 0, nullptr, nullptr);
    vx_enqueue_dcr_write(q, VX_DCR_RASTER_SCISSOR_Y, (dst_height << 16) | 0, 0, nullptr, nullptr);
    vx_enqueue_dcr_write(q, VX_DCR_RASTER_FRAG_ENTRY_LO, (uint32_t)(frag_entry & 0xffffffff), 0, nullptr, nullptr);
    vx_enqueue_dcr_write(q, VX_DCR_RASTER_FRAG_ENTRY_HI, (uint32_t)(frag_entry >> 32), 0, nullptr, nullptr);
    vx_enqueue_dcr_write(q, VX_DCR_RASTER_FRAG_PARAM_LO, (uint32_t)(fa_addr & 0xffffffff), 0, nullptr, nullptr);
    vx_enqueue_dcr_write(q, VX_DCR_RASTER_FRAG_PARAM_HI, (uint32_t)(fa_addr >> 32), 0, nullptr, nullptr);
    // Grid-less kick: no host fragment grid (grid_dim=0); the armed raster work
    // distributor injects the fragment warps and sustains the run.
    vx_launch_info_t fli = {}; fli.struct_size = sizeof(fli); fli.kernel = k_frag;
    fli.args_host = &fa; fli.args_size = sizeof(fa); fli.ndim = 1;
    fli.grid_dim[0] = 0; fli.block_dim[0] = (uint32_t)(num_threads * num_warps);
    vx_event_h fev = nullptr;
    RT_CHECK(vx_enqueue_launch(q, &fli, 1, &plast, &fev));
    RT_CHECK(vx_event_wait_value(fev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(fev);
    vx_buffer_release(fa_b);

    // Post-hoc cross-check (reads buffers AFTER rasterise — does NOT gate the
    // renderpass): device tilebuf+primbuf == host Binning(), to localise bugs.
    std::vector<uint32_t> h_meta(3, 0);
    std::vector<rast_prim_t> h_prim(P_ref ? P_ref : 1);
    std::vector<uint8_t> h_tilebuf(TILEBUF_SZ);
    vx_event_h em = nullptr, ep = nullptr, et = nullptr;
    RT_CHECK(vx_enqueue_read(q, h_meta.data(),    meta_b,    0, 3 * sizeof(uint32_t), 1, &plast, &em));
    RT_CHECK(vx_enqueue_read(q, h_prim.data(),    prim_b,    0, P_ref * PRIM_SZ,      1, &plast, &ep));
    RT_CHECK(vx_enqueue_read(q, h_tilebuf.data(), tilebuf_b, 0, TILEBUF_SZ,           1, &plast, &et));
    RT_CHECK(vx_event_wait_value(em, 1, VX_TIMEOUT_INFINITE));
    RT_CHECK(vx_event_wait_value(ep, 1, VX_TIMEOUT_INFINITE));
    RT_CHECK(vx_event_wait_value(et, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(em); vx_event_release(ep); vx_event_release(et);

    int errors = 0;
    if (h_meta[0] != P_ref)   { std::printf("*** P dev=%u ref=%u\n", h_meta[0], P_ref); ++errors; }
    if (h_meta[1] != keys_ref){ std::printf("*** keys dev=%u ref=%u\n", h_meta[1], keys_ref); ++errors; }
    if (h_meta[2] != nb_ref)  { std::printf("*** tiles dev=%u ref=%u\n", h_meta[2], nb_ref); ++errors; }
    auto* bprim = reinterpret_cast<const rast_prim_t*>(ref_primbuf.data());
    for (uint32_t i = 0; i < P_ref && errors < 8; ++i)
      if (std::memcmp(&h_prim[i], &bprim[i], sizeof(rast_prim_t)) != 0) { std::printf("*** primbuf[%u] mismatch\n", i); ++errors; }
    {
      TileMap devmap;
      auto* hdr = reinterpret_cast<const rast_bin_header_t*>(h_tilebuf.data());
      // sorted_pids follows the dense bin-header block; pids_offset is absolute.
      auto* pids = reinterpret_cast<const uint32_t*>(h_tilebuf.data() + (size_t)B * HDR_SZ);
      for (uint32_t b = 0; b < B; ++b) {   // dense grid: skip empty bins
        if (hdr[b].pids_count == 0) continue;
        const uint32_t* pp = pids + hdr[b].pids_offset;
        std::vector<uint32_t> v(hdr[b].pids_count);
        for (uint32_t j = 0; j < hdr[b].pids_count; ++j) v[j] = pp[j];
        devmap[{hdr[b].bin_x, hdr[b].bin_y}] = std::move(v);
      }
      if (devmap != gold) { std::printf("*** tilebuf bin->pid map != Binning()\n"); ++errors; }
    }
    std::cout << (errors ? "buffer cross-check: FAIL" : "buffer cross-check: PASS (device == Binning)") << std::endl;
    total_errors += errors;
    for (uint32_t s = 0; s < NSTAGE; ++s) vx_event_release(pev[s]);

    vx_buffer_release(verts_b); vx_buffer_release(slot_prim_b); vx_buffer_release(slot_bbox_b);
    vx_buffer_release(keep_b); vx_buffer_release(offset_b); vx_buffer_release(tsum_b);
    vx_buffer_release(prim_b); vx_buffer_release(bbox_b); vx_buffer_release(bcount_b);
    vx_buffer_release(boffset_b); vx_buffer_release(keys_b); vx_buffer_release(btsum_b);
    vx_buffer_release(thist_b); vx_buffer_release(bincount_b); vx_buffer_release(binbase_b);
    vx_buffer_release(tilebuf_b); vx_buffer_release(meta_b);
  }

  // Save + compare image.
  if (strcmp(output_file, "null") != 0) {
    std::vector<uint8_t> px(cbuf_size);
    vx_event_h e = nullptr;
    RT_CHECK(vx_enqueue_read(q, px.data(), color_buffer, 0, cbuf_size, 0, nullptr, &e));
    RT_CHECK(vx_event_wait_value(e, 1, VX_TIMEOUT_INFINITE)); vx_event_release(e);
    auto bits = px.data() + (dst_height - 1) * cbuf_pitch;
    RT_CHECK(SaveImage(output_file, FORMAT_A8R8G8B8, bits, dst_width, dst_height, -cbuf_pitch));
  }

  vx_buffer_release(color_buffer);
  vx_module_release(mod); vx_queue_release(q); vx_device_release(dev);

  int img_errors = 0;
  if (reference_file && strcmp(output_file, "null") != 0) {
    auto ref = resolve_path(reference_file, ASSETS_PATHS);
    img_errors = CompareImages(output_file, ref.c_str(), FORMAT_A8R8G8B8);
    std::cout << "image vs reference: " << (img_errors ? "FAIL" : "PASS")
              << " (" << img_errors << " errors)" << std::endl;
  }

  bool ok = (total_errors == 0) && (img_errors == 0);
  std::cout << (ok ? "PASSED!" : "FAILED!") << std::endl;
  return ok ? 0 : 1;
}
