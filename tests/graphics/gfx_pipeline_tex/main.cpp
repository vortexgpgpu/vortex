// gfx_v2 device front end -> RASTER -> TEX -> OM, end to end.
//
// A synthetic full-screen textured quad (in front of the near plane, so clip is
// a passthrough) is rendered through the identical RASTER + TEX + OM back end
// from two front ends: the device setup+binning pipeline, and host Binning().
// The two images must match (the device front end is a drop-in that drives the
// TEX fixed-function unit correctly), and the device tilebuf+primbuf must match
// Binning() bit-for-bit (cross-check). No host Binning() in the device path.

#include <iostream>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <utility>
#include <unordered_map>
#include <unistd.h>
#include <getopt.h>
#include <vortex2.h>
#include <graphics.h>
#include <cocogfx/include/format.hpp>
#include <cocogfx/include/blitter.hpp>
#include <cocogfx/include/imageutil.hpp>
#include "common.h"

using namespace cocogfx;
using namespace vortex;

using rast_prim_t = graphics::rast_prim_t;
using rast_tile_header_t = graphics::rast_tile_header_t;
using TileMap = std::map<std::pair<uint16_t, uint16_t>, std::vector<uint32_t>>;

#define RT_CHECK(_expr) do { int _r = (_expr); if (_r) { \
  printf("Error: '%s' returned %d (%s:%d)\n", #_expr, _r, __FILE__, __LINE__); exit(-1); } } while(0)

static uint32_t dst_width = 128, dst_height = 128;
static const char* output_file = "output.png";

static uint32_t log2ceil(uint32_t n) { uint32_t l = 0; while ((1u << l) < n) ++l; return l; }

// Synthetic quad: two triangles, NDC [-0.9,0.9], uv [0,1], w=1 (in front of near).
static std::vector<setup_vertex_t> gen_quad() {
  auto V = [](float nx, float ny, float u, float v) {
    setup_vertex_t s{};
    s.pos[0] = nx; s.pos[1] = ny; s.pos[2] = 0.0f; s.pos[3] = 1.0f;
    s.color[0] = s.color[1] = s.color[2] = s.color[3] = 1.0f;
    s.texcoord[0] = u; s.texcoord[1] = v;
    return s;
  };
  setup_vertex_t tl = V(-0.9f, -0.9f, 0, 0), tr = V(0.9f, -0.9f, 1, 0);
  setup_vertex_t br = V(0.9f, 0.9f, 1, 1),  bl = V(-0.9f, 0.9f, 0, 1);
  return { tl, tr, br,  tl, br, bl };   // two triangles, non-indexed
}

// A smaller centered quad (NDC [-e,e], uv [0,1]) — the second draw of the
// multi-draw frame, overwriting the center of the full quad.
static std::vector<setup_vertex_t> gen_quad_centered(float e) {
  auto V = [](float nx, float ny, float u, float v) {
    setup_vertex_t s{};
    s.pos[0] = nx; s.pos[1] = ny; s.pos[2] = 0.0f; s.pos[3] = 1.0f;
    s.color[0] = s.color[1] = s.color[2] = s.color[3] = 1.0f;
    s.texcoord[0] = u; s.texcoord[1] = v;
    return s;
  };
  setup_vertex_t tl = V(-e, -e, 0, 0), tr = V(e, -e, 1, 0);
  setup_vertex_t br = V(e, e, 1, 1),  bl = V(-e, e, 0, 1);
  return { tl, tr, br,  tl, br, bl };
}

// 64x64 ARGB8 checkerboard.
static std::vector<uint8_t> gen_texture(uint32_t tw, uint32_t th) {
  std::vector<uint8_t> px(tw * th * 4);
  auto* p = reinterpret_cast<uint32_t*>(px.data());
  for (uint32_t y = 0; y < th; ++y)
    for (uint32_t x = 0; x < tw; ++x) {
      bool c = ((x >> 3) ^ (y >> 3)) & 1;
      p[y * tw + x] = c ? 0xffff8000u : 0xff0040ffu;  // orange / blue
    }
  return px;
}

// Host Binning() oracle at the RASTER tile log -> (tile -> pids) + primbuf.
static TileMap binning_oracle(const std::vector<setup_vertex_t>& verts, uint32_t ntri,
                              std::vector<uint8_t>& tilebuf, std::vector<uint8_t>& primbuf,
                              uint32_t& num_tiles) {
  std::unordered_map<uint32_t, graphics::vertex_t> vmap;
  std::vector<graphics::primitive_t> prims;
  for (uint32_t i = 0; i < 3 * ntri; ++i) { graphics::vertex_t gv; std::memcpy(&gv, &verts[i], sizeof(gv)); vmap[i] = gv; }
  for (uint32_t t = 0; t < ntri; ++t) prims.push_back({3 * t, 3 * t + 1, 3 * t + 2});
  num_tiles = graphics::Binning(tilebuf, primbuf, vmap, prims, dst_width, dst_height,
                                SETUP_NEAR, SETUP_FAR, PIPE_BIN_LOG);
  TileMap m;
  auto* hdr = reinterpret_cast<const rast_tile_header_t*>(tilebuf.data());
  const uint8_t* pp = tilebuf.data() + (size_t)num_tiles * sizeof(rast_tile_header_t);
  for (uint32_t i = 0; i < num_tiles; ++i) {
    uint32_t cnt = hdr[i].pids_count;
    std::vector<uint32_t> v(cnt);
    std::memcpy(v.data(), pp, cnt * sizeof(uint32_t)); pp += cnt * sizeof(uint32_t);
    m[{hdr[i].tile_x, hdr[i].tile_y}] = std::move(v);
  }
  return m;
}

int main(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "w:h:o:")) != -1) {
    if (c == 'w') dst_width = atoi(optarg);
    else if (c == 'h') dst_height = atoi(optarg);
    else if (c == 'o') output_file = optarg;
  }

  const std::vector<setup_vertex_t> verts = gen_quad();
  const uint32_t ntri = (uint32_t)verts.size() / 3;

  vx_device_h dev = nullptr;
  RT_CHECK(vx_device_open(0, &dev));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  vx_queue_h q = nullptr;
  RT_CHECK(vx_queue_create(dev, &qi, &q));
  uint64_t isa = 0; RT_CHECK(vx_device_query(dev, VX_CAPS_ISA_FLAGS, &isa));
  if (!(isa & VX_ISA_EXT_TEX) || !(isa & VX_ISA_EXT_OM) || !(isa & VX_ISA_EXT_RASTER)) {
    std::cout << "TEX/OM/RASTER not all supported!\n"; return -1;
  }
  uint64_t num_cores = 0, num_warps = 0, num_threads = 0;
  RT_CHECK(vx_device_query(dev, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_device_query(dev, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_device_query(dev, VX_CAPS_NUM_THREADS, &num_threads));

  vx_module_h mod = nullptr;
  vx_kernel_h k_setup = nullptr, k_binning = nullptr, k_frag = nullptr;
  RT_CHECK(vx_module_load_file(dev, "kernel.vxbin", &mod));
  RT_CHECK(vx_module_get_kernel(mod, "setup_k", &k_setup));
  RT_CHECK(vx_module_get_kernel(mod, "binning_k", &k_binning));
  RT_CHECK(vx_module_get_kernel(mod, "main", &k_frag));

  // OM color buffer (pinned: OM's AXI master writes it).
  const uint32_t cbuf_stride = 4, cbuf_pitch = dst_width * cbuf_stride, cbuf_size = dst_height * cbuf_pitch;
  vx_buffer_h cbuf_b = nullptr; uint64_t cbuf_addr = 0;
  RT_CHECK(vx_buffer_create(dev, cbuf_size, VX_MEM_READ | VX_MEM_WRITE | VX_MEM_PHYS, &cbuf_b));
  RT_CHECK(vx_buffer_address(cbuf_b, &cbuf_addr));

  // Texture: checkerboard -> mipmaps -> pinned tex buffer (TEX's AXI master).
  const uint32_t tw = 64, th = 64;
  std::vector<uint8_t> tex_pixels = gen_texture(tw, th);
  std::vector<uint8_t> texbuf; std::vector<uint32_t> mip_offsets;
  RT_CHECK(GenerateMipmaps(texbuf, mip_offsets, tex_pixels.data(), FORMAT_A8R8G8B8, tw, th, tw * 4));
  vx_buffer_h tex_b = nullptr; uint64_t tex_addr = 0;
  RT_CHECK(vx_buffer_create(dev, texbuf.size(), VX_MEM_READ | VX_MEM_PHYS, &tex_b));
  RT_CHECK(vx_buffer_address(tex_b, &tex_addr));
  RT_CHECK(vx_enqueue_write(q, tex_b, 0, texbuf.data(), texbuf.size(), 0, nullptr, nullptr));

  // Configure TEX + OM once (shared by both render paths).
  vx_enqueue_dcr_write(q, VX_DCR_TEX_STAGE, 0, 0, nullptr, nullptr);
  vx_enqueue_dcr_write(q, VX_DCR_TEX_LOGDIM, (log2ceil(th) << 16) | log2ceil(tw), 0, nullptr, nullptr);
  vx_enqueue_dcr_write(q, VX_DCR_TEX_FORMAT, VX_TEX_FORMAT_A8R8G8B8, 0, nullptr, nullptr);
  vx_enqueue_dcr_write(q, VX_DCR_TEX_WRAP, (1 << 16) | 1, 0, nullptr, nullptr);   // wrap U/V
  vx_enqueue_dcr_write(q, VX_DCR_TEX_FILTER, VX_TEX_FILTER_POINT, 0, nullptr, nullptr);
  vx_enqueue_dcr_write(q, VX_DCR_TEX_ADDR, tex_addr / 64, 0, nullptr, nullptr);
  for (uint32_t i = 0; i < mip_offsets.size(); ++i)
    vx_enqueue_dcr_write(q, VX_DCR_TEX_MIPOFF(i), mip_offsets[i], 0, nullptr, nullptr);

  vx_enqueue_dcr_write(q, VX_DCR_OM_CBUF_ADDR, cbuf_addr / 64, 0, nullptr, nullptr);
  vx_enqueue_dcr_write(q, VX_DCR_OM_CBUF_PITCH, cbuf_pitch, 0, nullptr, nullptr);
  vx_enqueue_dcr_write(q, VX_DCR_OM_CBUF_WRITEMASK, 0xffffffff, 0, nullptr, nullptr);
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

  frag_arg_t fa = {};
  fa.depth_enabled = 0; fa.color_enabled = 0; fa.tex_enabled = 1; fa.tex_modulate = 0;

  // RASTER config + fragment launch into the (cleared) color buffer; reads the
  // shaded image back. tbuf/pbuf select the front end's buffers.
  auto render = [&](uint64_t tbuf, uint64_t pbuf, uint32_t ntiles, std::vector<uint8_t>& out) {
    { std::vector<uint32_t> clr(cbuf_size / 4, 0xff000000); vx_event_h e = nullptr;
      RT_CHECK(vx_enqueue_write(q, cbuf_b, 0, clr.data(), cbuf_size, 0, nullptr, &e));
      RT_CHECK(vx_event_wait_value(e, 1, VX_TIMEOUT_INFINITE)); vx_event_release(e); }
    fa.prim_addr = pbuf;
    vx_enqueue_dcr_write(q, VX_DCR_RASTER_TBUF_ADDR, tbuf / 64, 0, nullptr, nullptr);
    vx_enqueue_dcr_write(q, VX_DCR_RASTER_TILE_COUNT, ntiles, 0, nullptr, nullptr);
    vx_enqueue_dcr_write(q, VX_DCR_RASTER_PBUF_ADDR, pbuf / 64, 0, nullptr, nullptr);
    vx_enqueue_dcr_write(q, VX_DCR_RASTER_PBUF_STRIDE, (uint32_t)sizeof(rast_prim_t), 0, nullptr, nullptr);
    vx_enqueue_dcr_write(q, VX_DCR_RASTER_SCISSOR_X, (dst_width << 16) | 0, 0, nullptr, nullptr);
    vx_enqueue_dcr_write(q, VX_DCR_RASTER_SCISSOR_Y, (dst_height << 16) | 0, 0, nullptr, nullptr);
    vx_launch_info_t li = {}; li.struct_size = sizeof(li); li.kernel = k_frag;
    li.args_host = &fa; li.args_size = sizeof(fa); li.ndim = 1;
    li.grid_dim[0] = (uint32_t)num_cores; li.block_dim[0] = (uint32_t)(num_threads * num_warps);
    vx_event_h ev = nullptr;
    RT_CHECK(vx_enqueue_launch(q, &li, 0, nullptr, &ev));
    RT_CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE)); vx_event_release(ev);
    out.resize(cbuf_size); vx_event_h er = nullptr;
    RT_CHECK(vx_enqueue_read(q, out.data(), cbuf_b, 0, cbuf_size, 0, nullptr, &er));
    RT_CHECK(vx_event_wait_value(er, 1, VX_TIMEOUT_INFINITE)); vx_event_release(er);
  };

  // ---- Path A: host Binning() front end ------------------------------------
  std::vector<uint8_t> a_tilebuf, a_primbuf;
  uint32_t a_tiles = 0;
  TileMap gold = binning_oracle(verts, ntri, a_tilebuf, a_primbuf, a_tiles);
  const uint32_t P_ref = (uint32_t)(a_primbuf.size() / sizeof(rast_prim_t));
  std::cout << "quad: tris=" << ntri << " P=" << P_ref << " tiles=" << a_tiles << std::endl;

  vx_buffer_h a_tb = nullptr, a_pb = nullptr; uint64_t a_tb_addr = 0, a_pb_addr = 0;
  RT_CHECK(vx_buffer_create(dev, a_tilebuf.size(), VX_MEM_READ | VX_MEM_PHYS, &a_tb));
  RT_CHECK(vx_buffer_create(dev, a_primbuf.size(), VX_MEM_READ | VX_MEM_PHYS, &a_pb));
  RT_CHECK(vx_buffer_address(a_tb, &a_tb_addr));
  RT_CHECK(vx_buffer_address(a_pb, &a_pb_addr));
  RT_CHECK(vx_enqueue_write(q, a_tb, 0, a_tilebuf.data(), a_tilebuf.size(), 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(q, a_pb, 0, a_primbuf.data(), a_primbuf.size(), 0, nullptr, nullptr));
  std::vector<uint8_t> img_ref;
  render(a_tb_addr, a_pb_addr, a_tiles, img_ref);

  // ---- Path B: device setup+binning pipeline -------------------------------
  uint32_t one = 1, grid[1], block[1];
  RT_CHECK(vx_device_max_occupancy_grid(dev, 1, &one, grid, block));
  const uint32_t T = block[0], G = grid[0], MS = SETUP_MAX_SUB;
  const uint32_t ts = 1u << PIPE_BIN_LOG;
  const uint32_t bin_cols = (dst_width + ts - 1) / ts, bin_rows = (dst_height + ts - 1) / ts;
  const uint32_t B = bin_cols * bin_rows;
  const size_t PRIM_SZ = sizeof(rast_prim_t), HDR_SZ = sizeof(rast_tile_header_t);
  uint32_t keys_ref = 0; for (auto& kv : gold) keys_ref += (uint32_t)kv.second.size();
  const uint32_t Kcap = keys_ref ? keys_ref : 1;
  const size_t TILEBUF_SZ = (size_t)B * HDR_SZ + (size_t)Kcap * sizeof(uint32_t);

  vx_buffer_h verts_b, slot_prim_b, slot_bbox_b, keep_b, offset_b, tsum_b, prim_b, bbox_b,
              bcount_b, boffset_b, keys_b, btsum_b, thist_b, bincount_b, binbase_b, tilebuf_b, meta_b;
  RT_CHECK(vx_buffer_create(dev, 3 * ntri * sizeof(setup_vertex_t), VX_MEM_READ, &verts_b));
  RT_CHECK(vx_buffer_create(dev, ntri * MS * PRIM_SZ, VX_MEM_WRITE, &slot_prim_b));
  RT_CHECK(vx_buffer_create(dev, ntri * MS * sizeof(setup_bbox_t), VX_MEM_WRITE, &slot_bbox_b));
  RT_CHECK(vx_buffer_create(dev, ntri * sizeof(uint32_t), VX_MEM_WRITE, &keep_b));
  RT_CHECK(vx_buffer_create(dev, (ntri + 1) * sizeof(uint32_t), VX_MEM_WRITE, &offset_b));
  RT_CHECK(vx_buffer_create(dev, T * sizeof(uint32_t), VX_MEM_WRITE, &tsum_b));
  RT_CHECK(vx_buffer_create(dev, ntri * MS * PRIM_SZ, VX_MEM_READ | VX_MEM_WRITE | VX_MEM_PHYS, &prim_b));
  RT_CHECK(vx_buffer_create(dev, ntri * MS * sizeof(setup_bbox_t), VX_MEM_WRITE, &bbox_b));
  RT_CHECK(vx_buffer_create(dev, ntri * MS * sizeof(uint32_t), VX_MEM_WRITE, &bcount_b));
  RT_CHECK(vx_buffer_create(dev, (ntri * MS + 1) * sizeof(uint32_t), VX_MEM_WRITE, &boffset_b));
  RT_CHECK(vx_buffer_create(dev, Kcap * sizeof(uint32_t), VX_MEM_WRITE, &keys_b));
  RT_CHECK(vx_buffer_create(dev, T * sizeof(uint32_t), VX_MEM_WRITE, &btsum_b));
  RT_CHECK(vx_buffer_create(dev, T * B * sizeof(uint32_t), VX_MEM_WRITE, &thist_b));
  RT_CHECK(vx_buffer_create(dev, B * sizeof(uint32_t), VX_MEM_WRITE, &bincount_b));
  RT_CHECK(vx_buffer_create(dev, B * sizeof(uint32_t), VX_MEM_WRITE, &binbase_b));
  RT_CHECK(vx_buffer_create(dev, TILEBUF_SZ, VX_MEM_READ | VX_MEM_WRITE | VX_MEM_PHYS, &tilebuf_b));
  RT_CHECK(vx_buffer_create(dev, 3 * sizeof(uint32_t), VX_MEM_WRITE, &meta_b));

  pipe_arg_t pa = {};
  pa.num_tris = ntri; pa.width = dst_width; pa.height = dst_height;
  pa.bin_cols = bin_cols; pa.num_bins = B; pa.bin_stripe = (B + G - 1) / G;
  uint64_t prim_addr = 0, tilebuf_addr = 0;
  RT_CHECK(vx_buffer_address(verts_b, &pa.verts_addr));
  RT_CHECK(vx_buffer_address(slot_prim_b, &pa.slot_prim_addr));
  RT_CHECK(vx_buffer_address(slot_bbox_b, &pa.slot_bbox_addr));
  RT_CHECK(vx_buffer_address(keep_b, &pa.keep_addr));
  RT_CHECK(vx_buffer_address(offset_b, &pa.offset_addr));
  RT_CHECK(vx_buffer_address(tsum_b, &pa.tsum_addr));
  RT_CHECK(vx_buffer_address(prim_b, &pa.prim_addr)); prim_addr = pa.prim_addr;
  RT_CHECK(vx_buffer_address(bbox_b, &pa.bbox_addr));
  RT_CHECK(vx_buffer_address(bcount_b, &pa.bcount_addr));
  RT_CHECK(vx_buffer_address(boffset_b, &pa.boffset_addr));
  RT_CHECK(vx_buffer_address(keys_b, &pa.keys_addr));
  RT_CHECK(vx_buffer_address(btsum_b, &pa.btsum_addr));
  RT_CHECK(vx_buffer_address(thist_b, &pa.thist_addr));
  RT_CHECK(vx_buffer_address(bincount_b, &pa.bincount_addr));
  RT_CHECK(vx_buffer_address(binbase_b, &pa.binbase_addr));
  RT_CHECK(vx_buffer_address(tilebuf_b, &pa.tilebuf_addr)); tilebuf_addr = pa.tilebuf_addr;
  RT_CHECK(vx_buffer_address(meta_b, &pa.meta_addr));
  RT_CHECK(vx_enqueue_write(q, verts_b, 0, verts.data(), 3 * ntri * sizeof(setup_vertex_t), 0, nullptr, nullptr));

  const uint32_t NSTAGE = 9;
  const uint32_t sgrid[NSTAGE] = { G, 1, G, G, 1, G, G, 1, G };
  pipe_arg_t pargs[NSTAGE]; vx_launch_info_t pli[NSTAGE]; vx_event_h pev[NSTAGE] = {};
  for (uint32_t s = 0; s < NSTAGE; ++s) {
    pargs[s] = pa; pargs[s].stage = s;
    pli[s] = vx_launch_info_t{}; pli[s].struct_size = sizeof(pli[s]);
    pli[s].kernel = (s < PIPE_STAGE_BCOUNT) ? k_setup : k_binning;
    pli[s].args_host = &pargs[s]; pli[s].args_size = sizeof(pipe_arg_t);
    pli[s].ndim = 1; pli[s].grid_dim[0] = sgrid[s]; pli[s].block_dim[0] = T;
    RT_CHECK(vx_enqueue_launch(q, &pli[s], s ? 1 : 0, s ? &pev[s - 1] : nullptr, &pev[s]));
  }
  RT_CHECK(vx_event_wait_value(pev[NSTAGE - 1], 1, VX_TIMEOUT_INFINITE));
  for (uint32_t s = 0; s < NSTAGE; ++s) vx_event_release(pev[s]);

  std::vector<uint8_t> img_dev;
  render(tilebuf_addr, prim_addr, B, img_dev);

  // ---- Path C: the whole draw as ONE CP command batch (charter §6.4) -------
  // Front-end 9 stages + RASTER config + fragment launch are submitted as a
  // single vx_enqueue_commands: the host writes the whole sequence into the CP
  // ring, rings the doorbell once, and polls completion once. The CP runs every
  // stage to completion in order — each launch drains before the next, which is
  // the device-wide inter-stage barrier — with the host untouched between
  // submit and the final read. Must be bit-identical to the host-Binning render.
  std::vector<uint8_t> img_batched;
  {
    // Clear the color buffer (a transfer, not part of the command batch).
    std::vector<uint32_t> clr(cbuf_size / 4, 0xff000000);
    vx_event_h ec = nullptr;
    RT_CHECK(vx_enqueue_write(q, cbuf_b, 0, clr.data(), cbuf_size, 0, nullptr, &ec));
    RT_CHECK(vx_event_wait_value(ec, 1, VX_TIMEOUT_INFINITE)); vx_event_release(ec);

    // Assemble the whole draw through the graphics command-sequence builder:
    // front-end 9 stages, RASTER config, fragment launch. The builder copies
    // args, so pargs[]/fa_c need only live across the launch() calls.
    frag_arg_t fa_c = fa; fa_c.prim_addr = prim_addr;
    graphics::DrawCommands dc;
    for (uint32_t s = 0; s < NSTAGE; ++s) {
      const uint32_t fg[3] = { sgrid[s], 1, 1 }, fb[3] = { T, 1, 1 };
      dc.launch((s < PIPE_STAGE_BCOUNT) ? k_setup : k_binning,
                &pargs[s], sizeof(pipe_arg_t), 1, fg, fb);
    }
    dc.dcr_write(VX_DCR_RASTER_TBUF_ADDR,  (uint32_t)(tilebuf_addr / 64));
    dc.dcr_write(VX_DCR_RASTER_TILE_COUNT, B);
    dc.dcr_write(VX_DCR_RASTER_PBUF_ADDR,  (uint32_t)(prim_addr / 64));
    dc.dcr_write(VX_DCR_RASTER_PBUF_STRIDE, (uint32_t)sizeof(rast_prim_t));
    dc.dcr_write(VX_DCR_RASTER_SCISSOR_X,  (dst_width  << 16) | 0);
    dc.dcr_write(VX_DCR_RASTER_SCISSOR_Y,  (dst_height << 16) | 0);
    const uint32_t cg[3] = { (uint32_t)num_cores, 1, 1 };
    const uint32_t cb[3] = { (uint32_t)(num_threads * num_warps), 1, 1 };
    dc.launch(k_frag, &fa_c, sizeof(fa_c), 1, cg, cb);

    vx_event_h ev = nullptr;
    RT_CHECK(dc.submit(q, 0, nullptr, &ev));
    RT_CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE)); vx_event_release(ev);

    img_batched.resize(cbuf_size); vx_event_h er = nullptr;
    RT_CHECK(vx_enqueue_read(q, img_batched.data(), cbuf_b, 0, cbuf_size, 0, nullptr, &er));
    RT_CHECK(vx_event_wait_value(er, 1, VX_TIMEOUT_INFINITE)); vx_event_release(er);
  }

  // Buffer cross-check: device primbuf bit-exact + dense tilebuf -> same map.
  // Read here, while prim_b/tilebuf_b still hold the full-quad front-end result
  // (Path D below reuses these buffers and leaves the second draw's data).
  int errors = 0;
  std::vector<rast_prim_t> h_prim(P_ref ? P_ref : 1);
  std::vector<uint8_t> h_tilebuf(TILEBUF_SZ);
  { vx_event_h e1 = nullptr, e2 = nullptr;
    RT_CHECK(vx_enqueue_read(q, h_prim.data(), prim_b, 0, P_ref * PRIM_SZ, 0, nullptr, &e1));
    RT_CHECK(vx_enqueue_read(q, h_tilebuf.data(), tilebuf_b, 0, TILEBUF_SZ, 0, nullptr, &e2));
    RT_CHECK(vx_event_wait_value(e1, 1, VX_TIMEOUT_INFINITE));
    RT_CHECK(vx_event_wait_value(e2, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(e1); vx_event_release(e2); }
  auto* bprim = reinterpret_cast<const rast_prim_t*>(a_primbuf.data());
  for (uint32_t i = 0; i < P_ref && errors < 8; ++i)
    if (std::memcmp(&h_prim[i], &bprim[i], sizeof(rast_prim_t)) != 0) { std::printf("*** primbuf[%u] mismatch\n", i); ++errors; }
  { TileMap devmap; auto* hdr = reinterpret_cast<const rast_tile_header_t*>(h_tilebuf.data());
    for (uint32_t b = 0; b < B; ++b) {
      if (hdr[b].pids_count == 0) continue;
      size_t pb = (size_t)b * HDR_SZ + HDR_SZ + (size_t)hdr[b].pids_offset * sizeof(uint32_t);
      auto* pp = reinterpret_cast<const uint32_t*>(h_tilebuf.data() + pb);
      std::vector<uint32_t> v(hdr[b].pids_count);
      for (uint32_t j = 0; j < hdr[b].pids_count; ++j) v[j] = pp[j];
      devmap[{hdr[b].tile_x, hdr[b].tile_y}] = std::move(v);
    }
    if (devmap != gold) { std::printf("*** tilebuf tile->pid map != Binning()\n"); ++errors; } }
  std::cout << (errors ? "buffer cross-check: FAIL" : "buffer cross-check: PASS (device == Binning)") << std::endl;

  // ---- Path D: a multi-draw frame in ONE batch (charter §8 / pillar 4) -----
  // Two draws — the full quad then a smaller centered quad — concatenated into
  // a single CP batch. The color attachment accumulates across draws (the
  // center is overwritten), and the front-end scratch pool is REUSED by both
  // draws: draw 0's render fully drains before draw 1's front-end overwrites
  // the shared primbuf/tilebuf (launch-drain serialization), so the pool is
  // reset-by-reuse between draws (§4.1) — which also proves the front-end
  // self-initializes per run. Validated against the same two draws issued as
  // two separate host-submitted batches (must be bit-identical), and shown to
  // differ from the single-draw image (the second draw really composited).
  const std::vector<setup_vertex_t> verts2 = gen_quad_centered(0.5f);
  vx_buffer_h verts2_b = nullptr; uint64_t verts2_addr = 0;
  RT_CHECK(vx_buffer_create(dev, 3 * ntri * sizeof(setup_vertex_t), VX_MEM_READ, &verts2_b));
  RT_CHECK(vx_buffer_address(verts2_b, &verts2_addr));
  RT_CHECK(vx_enqueue_write(q, verts2_b, 0, verts2.data(), 3 * ntri * sizeof(setup_vertex_t), 0, nullptr, nullptr));

  // Per-draw front-end args. Draw 0 reuses pa (verts_b); draw 1 differs only in
  // the vertex source — same scratch pool, same render config.
  pipe_arg_t pargs0[NSTAGE], pargs1[NSTAGE];
  for (uint32_t s = 0; s < NSTAGE; ++s) {
    pargs0[s] = pa; pargs0[s].stage = s;
    pargs1[s] = pa; pargs1[s].stage = s; pargs1[s].verts_addr = verts2_addr;
  }
  frag_arg_t fa_d = fa; fa_d.prim_addr = prim_addr;

  // Append one draw (front-end 9 stages + RASTER config + fragment) to dc.
  auto append_draw = [&](graphics::DrawCommands& dc, const pipe_arg_t pgs[NSTAGE]) {
    for (uint32_t s = 0; s < NSTAGE; ++s) {
      const uint32_t dg[3] = { sgrid[s], 1, 1 }, db[3] = { T, 1, 1 };
      dc.launch((s < PIPE_STAGE_BCOUNT) ? k_setup : k_binning,
                &pgs[s], sizeof(pipe_arg_t), 1, dg, db);
    }
    dc.dcr_write(VX_DCR_RASTER_TBUF_ADDR,  (uint32_t)(tilebuf_addr / 64));
    dc.dcr_write(VX_DCR_RASTER_TILE_COUNT, B);
    dc.dcr_write(VX_DCR_RASTER_PBUF_ADDR,  (uint32_t)(prim_addr / 64));
    dc.dcr_write(VX_DCR_RASTER_PBUF_STRIDE, (uint32_t)sizeof(rast_prim_t));
    dc.dcr_write(VX_DCR_RASTER_SCISSOR_X,  (dst_width  << 16) | 0);
    dc.dcr_write(VX_DCR_RASTER_SCISSOR_Y,  (dst_height << 16) | 0);
    const uint32_t cg[3] = { (uint32_t)num_cores, 1, 1 };
    const uint32_t cb[3] = { (uint32_t)(num_threads * num_warps), 1, 1 };
    dc.launch(k_frag, &fa_d, sizeof(fa_d), 1, cg, cb);
  };
  auto clear_cbuf = [&]() {
    std::vector<uint32_t> clr(cbuf_size / 4, 0xff000000); vx_event_h e = nullptr;
    RT_CHECK(vx_enqueue_write(q, cbuf_b, 0, clr.data(), cbuf_size, 0, nullptr, &e));
    RT_CHECK(vx_event_wait_value(e, 1, VX_TIMEOUT_INFINITE)); vx_event_release(e);
  };
  auto read_cbuf = [&](std::vector<uint8_t>& out) {
    out.resize(cbuf_size); vx_event_h e = nullptr;
    RT_CHECK(vx_enqueue_read(q, out.data(), cbuf_b, 0, cbuf_size, 0, nullptr, &e));
    RT_CHECK(vx_event_wait_value(e, 1, VX_TIMEOUT_INFINITE)); vx_event_release(e);
  };

  // Reference: the two draws as two separate host-submitted batches.
  std::vector<uint8_t> img_frame_seq;
  { clear_cbuf();
    graphics::DrawCommands d0; append_draw(d0, pargs0);
    vx_event_h e0 = nullptr; RT_CHECK(d0.submit(q, 0, nullptr, &e0));
    RT_CHECK(vx_event_wait_value(e0, 1, VX_TIMEOUT_INFINITE)); vx_event_release(e0);
    graphics::DrawCommands d1; append_draw(d1, pargs1);
    vx_event_h e1 = nullptr; RT_CHECK(d1.submit(q, 0, nullptr, &e1));
    RT_CHECK(vx_event_wait_value(e1, 1, VX_TIMEOUT_INFINITE)); vx_event_release(e1);
    read_cbuf(img_frame_seq); }

  // Frame: both draws in ONE batch — host submits the whole frame once.
  std::vector<uint8_t> img_frame_one;
  { clear_cbuf();
    graphics::DrawCommands frame; append_draw(frame, pargs0); append_draw(frame, pargs1);
    vx_event_h ef = nullptr; RT_CHECK(frame.submit(q, 0, nullptr, &ef));
    RT_CHECK(vx_event_wait_value(ef, 1, VX_TIMEOUT_INFINITE)); vx_event_release(ef);
    read_cbuf(img_frame_one); }

  // ---- Validation ----------------------------------------------------------
  // (buffer cross-check ran above, before Path D reused the shared front-end
  // buffers.) Image cross-check: device render == host-Binning render (same TEX+OM).
  int img_diff = (img_dev == img_ref) ? 0 : 1;
  if (img_diff) { size_t n = 0; for (size_t i = 0; i < img_dev.size(); ++i) if (img_dev[i] != img_ref[i]) ++n;
    std::printf("*** image device vs host-Binning differs in %zu bytes\n", n); }
  std::cout << "image device vs host-Binning: " << (img_diff ? "FAIL" : "PASS") << std::endl;

  // Batched-draw cross-check: the whole draw as one CP command batch must
  // render bit-identically to the host-Binning reference (§6.4).
  int batch_diff = (img_batched == img_ref) ? 0 : 1;
  if (batch_diff) { size_t n = 0; for (size_t i = 0; i < img_batched.size(); ++i) if (img_batched[i] != img_ref[i]) ++n;
    std::printf("*** image batched-draw vs host-Binning differs in %zu bytes\n", n); }
  std::cout << "image batched-draw vs host-Binning: " << (batch_diff ? "FAIL" : "PASS") << std::endl;

  // Multi-draw frame: one-batch frame == two-batch frame, and the second draw
  // actually composited (frame != single-draw image).
  int frame_diff = (img_frame_one == img_frame_seq) ? 0 : 1;
  if (frame_diff) { size_t n = 0; for (size_t i = 0; i < img_frame_one.size(); ++i) if (img_frame_one[i] != img_frame_seq[i]) ++n;
    std::printf("*** one-batch frame vs two-batch frame differs in %zu bytes\n", n); }
  int frame_composited = (img_frame_seq != img_dev) ? 1 : 0;
  std::cout << "multi-draw frame one-batch vs two-batch: " << (frame_diff ? "FAIL" : "PASS")
            << " (second draw composited: " << (frame_composited ? "yes" : "NO") << ")" << std::endl;

  // Save the device render.
  auto bits = img_dev.data() + (dst_height - 1) * cbuf_pitch;
  RT_CHECK(SaveImage(output_file, FORMAT_A8R8G8B8, bits, dst_width, dst_height, -cbuf_pitch));

  bool ok = (errors == 0) && (img_diff == 0) && (batch_diff == 0)
         && (frame_diff == 0) && (frame_composited == 1);
  std::cout << "RESULT: " << (ok ? "PASS" : "FAIL") << std::endl;
  return ok ? 0 : 1;
}
