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

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "t:o:r:w:h:k:?")) != -1) {
    switch (c) {
      case 't': trace_file = optarg; break;
      case 'o': output_file = optarg; break;
      case 'r': reference_file = optarg; break;
      case 'w': dst_width = std::atoi(optarg); break;
      case 'h': dst_height = std::atoi(optarg); break;
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
      setup_vertex_t s;
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
  // Host Binning() now emits the gfx_v2 §6.3 coarse-bin layout (dense
  // rast_bin_header_t block + absolute-indexed sorted-pid array).
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

  // OM writes the color buffer via its AXI master (bypasses the MMU) -> pinned.
  uint32_t cbuf_stride = 4, cbuf_pitch = dst_width * cbuf_stride, cbuf_size = dst_height * cbuf_pitch;
  vx_buffer_h color_buffer = nullptr;
  uint64_t cbuf_addr = 0;
  RT_CHECK(vx_buffer_create(dev, cbuf_size, VX_MEM_READ | VX_MEM_WRITE | VX_MEM_PHYS, &color_buffer));
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
    RT_CHECK(vx_buffer_create(dev, 3 * sizeof(uint32_t),           VX_MEM_WRITE, &meta_b));

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
    fa.prim_addr = prim_addr;
    fa.depth_enabled = dc.states.depth_test;
    fa.color_enabled = dc.states.color_enabled;
    fa.tex_enabled = 0;        // this test targets the (non-textured) colour path
    fa.tex_modulate = 0;
    // RASTER dispatch v2 (push): stage the FS args in device memory + program the
    // fragment-dispatch descriptor (FS entry PC + args pointer).
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
    // configure OM: colour write; depth/stencil/blend disabled (the triangle's
    // state — matches gfx_draw3d's disabled-feature branches).
    vx_enqueue_dcr_write(q, VX_DCR_OM_CBUF_ADDR, cbuf_addr / 64, 0, nullptr, nullptr);
    vx_enqueue_dcr_write(q, VX_DCR_OM_CBUF_PITCH, cbuf_pitch, 0, nullptr, nullptr);
    vx_enqueue_dcr_write(q, VX_DCR_OM_CBUF_WRITEMASK, dc.states.color_writemask, 0, nullptr, nullptr);
    vx_enqueue_dcr_write(q, VX_DCR_OM_DEPTH_FUNC, VX_OM_DEPTH_FUNC_ALWAYS, 0, nullptr, nullptr);
    vx_enqueue_dcr_write(q, VX_DCR_OM_DEPTH_WRITEMASK, 0, 0, nullptr, nullptr);
    vx_enqueue_dcr_write(q, VX_DCR_OM_STENCIL_FUNC, VX_OM_DEPTH_FUNC_ALWAYS, 0, nullptr, nullptr);
    vx_enqueue_dcr_write(q, VX_DCR_OM_STENCIL_ZPASS, VX_OM_STENCIL_OP_KEEP, 0, nullptr, nullptr);
    vx_enqueue_dcr_write(q, VX_DCR_OM_STENCIL_FAIL, VX_OM_STENCIL_OP_KEEP, 0, nullptr, nullptr);
    vx_enqueue_dcr_write(q, VX_DCR_OM_STENCIL_REF, 0, 0, nullptr, nullptr);
    vx_enqueue_dcr_write(q, VX_DCR_OM_STENCIL_MASK, VX_OM_STENCIL_MASK, 0, nullptr, nullptr);
    vx_enqueue_dcr_write(q, VX_DCR_OM_STENCIL_WRITEMASK, 0, 0, nullptr, nullptr);
    vx_enqueue_dcr_write(q, VX_DCR_OM_BLEND_MODE, (VX_OM_BLEND_MODE_ADD << 16) | (VX_OM_BLEND_MODE_ADD << 0), 0, nullptr, nullptr);
    vx_enqueue_dcr_write(q, VX_DCR_OM_BLEND_FUNC,
        (VX_OM_BLEND_FUNC_ZERO << 24) | (VX_OM_BLEND_FUNC_ZERO << 16)
      | (VX_OM_BLEND_FUNC_ONE << 8)   | (VX_OM_BLEND_FUNC_ONE << 0), 0, nullptr, nullptr);
    vx_enqueue_dcr_write(q, VX_DCR_RASTER_FRAG_ENTRY_LO, (uint32_t)(frag_entry & 0xffffffff), 0, nullptr, nullptr);
    vx_enqueue_dcr_write(q, VX_DCR_RASTER_FRAG_ENTRY_HI, (uint32_t)(frag_entry >> 32), 0, nullptr, nullptr);
    vx_enqueue_dcr_write(q, VX_DCR_RASTER_FRAG_PARAM_LO, (uint32_t)(fa_addr & 0xffffffff), 0, nullptr, nullptr);
    vx_enqueue_dcr_write(q, VX_DCR_RASTER_FRAG_PARAM_HI, (uint32_t)(fa_addr >> 32), 0, nullptr, nullptr);
    // Grid-less kick: no host fragment grid; the armed raster work distributor
    // injects the fragment warps and sustains the run.
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
  std::cout << "RESULT: " << (ok ? "PASS" : "FAIL") << std::endl;
  return ok ? 0 : 1;
}
