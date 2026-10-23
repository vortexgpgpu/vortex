// Vortex rasterizer regression test.
//
// Pinned-buffer layout:
//   tile_buffer   → RASTER HW (VX_DCR_RASTER_TBUF_ADDR) → pinned
//   prim_buffer   → RASTER HW (VX_DCR_RASTER_PBUF_ADDR) → pinned
//   color_buffer  → kernel-only (LSU stores from shaded quads) → not pinned
// The raster fixed-function block reads tile_buffer + prim_buffer via its
// own AXI master (no per-core MMU). The kernel writes color_buffer via
// the LSU, which goes through the MMU — no pinning needed.

#include <iostream>
#include <vector>
#include <unistd.h>
#include <cstring>
#include <chrono>
#include <cmath>
#include <array>
#include <assert.h>
#include <vortex2.h>
#include <graphics.h>
#include <gfx_ff_model.h>
#include <gfx_frag_rast.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <unordered_map>
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
        std::string filePath = path + "/" + filename;
        std::ifstream ifs(filePath);
        if (ifs)
          return filePath;
      }
    }
  }
  return filename;
}

// CGLTrace (test-input format) -> vortex::graphics input types for Binning.
static std::unordered_map<uint32_t, graphics::vertex_t>
cgl_to_gfx_vertices(const std::unordered_map<uint32_t, CGLTrace::vertex_t>& cgl) {
  std::unordered_map<uint32_t, graphics::vertex_t> out;
  out.reserve(cgl.size());
  for (auto& kv : cgl) {
    const auto& v = kv.second;
    graphics::vertex_t vx{};   // varying2 feeds the w0..w5 planes; an
                               // indeterminate one reaches setup as an
                               // attribute plane.
    vx.pos[0] = v.pos.x;       vx.pos[1] = v.pos.y;
    vx.pos[2] = v.pos.z;       vx.pos[3] = v.pos.w;
    vx.color[0] = v.color.r;   vx.color[1] = v.color.g;
    vx.color[2] = v.color.b;   vx.color[3] = v.color.a;
    vx.texcoord[0] = v.texcoord.u;
    vx.texcoord[1] = v.texcoord.v;
    out[kv.first] = vx;
  }
  return out;
}

static std::vector<graphics::primitive_t>
cgl_to_gfx_primitives(const std::vector<CGLTrace::primitive_t>& cgl) {
  std::vector<graphics::primitive_t> out;
  out.reserve(cgl.size());
  for (auto& p : cgl) {
    out.push_back({p.i0, p.i1, p.i2});
  }
  return out;
}

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
	 cleanup();			                                              \
     exit(-1);                                                  \
   } while (false)

///////////////////////////////////////////////////////////////////////////////

const char* kernel_file = "kernel.vxbin";
const char* trace_file  = "triangle.cgltrace";
const char* output_file = "output.png";
const char* reference_file  = nullptr;

uint32_t clear_color = 0xff000000;

uint32_t dst_width  = 128;
uint32_t dst_height = 128;

uint32_t cbuf_stride;
uint32_t cbuf_pitch;
uint32_t cbuf_size;

uint64_t cbuf_addr;
uint64_t tilebuf_addr;
uint64_t primbuf_addr;

vx_device_h device      = nullptr;
vx_queue_h  queue       = nullptr;
vx_module_h module_     = nullptr;
vx_kernel_h kernel      = nullptr;
vx_buffer_h color_buffer= nullptr;
vx_buffer_h tile_buffer = nullptr;
vx_buffer_h prim_buffer = nullptr;
vx_buffer_h frag_arg_buffer = nullptr;   // FS args (RASTER frag-dispatch descriptor)
uint64_t    frag_arg_addr = 0;

bool use_sw = false;
bool prim_scissor = false;   // -s: a distinct scissor rectangle in every primitive record
uint32_t num_generated = 0;  // -n: draw this many generated triangles instead of the trace
bool relaunch = false;       // -L: launch every pass twice, with no DCR write in between
// -M: 4x multisampling. The colour buffer holds every sample, interleaved within
// its pixel, and the image is checked sample by sample against the shared
// multisample coverage walk.
uint32_t num_samples = 1;
// -E: raster early-Z against depth images that start with every sample's own
// depth (the build needs OM, which brings early-Z). Nothing writes depth, so a
// pixel is drawn, with every sample the walk covers, exactly when one of those
// samples has a plane depth not strictly behind the stored one.
bool early_z = false;
// -S: draw a sliver per row, thin across one sample row and steep in depth, in
// place of the trace: the plane moves by many depth ranges between a pixel's
// centre and its samples, so a sample's depth tests the offset's full range.
bool slivers = false;
uint64_t zbuf_addr = 0;
vx_buffer_h depth_buffer = nullptr;
// -L: the second launch's records take colours this far past the first's.
constexpr uint32_t kRelaunchIdOffset = 1u << 20;
std::vector<uint32_t> ref_pixels;
uint32_t exact_errors = 0;
uint64_t num_threads = 0;  // populated in main, read by render()
uint64_t num_warps   = 0;  // populated in main, read by render()
uint64_t num_cores   = 0;  // populated in main, read by render()

kernel_arg_t kernel_arg = {};

// Host Binning() must emit coarse-bin headers at the granularity the RASTER
// walker descends by (1 << VX_CFG_RASTER_BIN_LOG_SIZE) — see graphics.cpp
// Binning(). Binning at VX_CFG_RASTER_TILE_LOG_SIZE instead lets the
// BIN_LOGSIZE walker over-cover smaller tiles (dropped/corrupt quads in
// non-origin tiles).
uint32_t tileLogSize = VX_CFG_RASTER_BIN_LOG_SIZE;

static void show_usage() {
   std::cout << "Vortex rasterizer Test." << std::endl;
   std::cout << "Usage: [-t trace] [-o output] [-r reference] [-w width] [-h height] [-z no_hw] [-k tilelogsize] [-s prim_scissor] [-n generated_triangles] [-L relaunch] [-M 4x multisampling] [-E early-Z] [-S slivers]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "t:i:o:r:w:h:t:k:n:zsLMES?")) != -1) {
    switch (c) {
    case 't':
      trace_file = optarg;
      break;
    case 'o':
      output_file = optarg;
      break;
    case 'r':
      reference_file = optarg;
      break;
    case 'w':
      dst_width = std::atoi(optarg);
      break;
    case 'h':
      dst_height = std::atoi(optarg);
      break;
    case 'k':
      tileLogSize = std::atoi(optarg);
      break;
    case 'z':
      use_sw = true;
      break;
    case 's':
      prim_scissor = true;
      break;
    case 'n':
      num_generated = std::atoi(optarg);
      break;
    case 'L':
      relaunch = true;
      break;
    case 'M':
      num_samples = gfx_rast::RAST_MSAA_SAMPLES;
      break;
    case 'E':
      early_z = true;
      break;
    case 'S':
      slivers = true;
      break;
    case '?': {
      show_usage();
      exit(0);
    } break;
    default:
      show_usage();
      exit(-1);
    }
  }
  if (relaunch && num_generated == 0) {
    std::cout << "Error: -L needs -n, whose colours tell the two launches apart!" << std::endl;
    exit(1);
  }
  if (strcmp (output_file, "null") == 0 && reference_file) {
    std::cout << "Error: the output file is missing for reference validation!" << std::endl;
    exit(1);
  }
  if (num_samples > 1 && reference_file) {
    std::cout << "Error: -M checks every sample against the coverage walk and takes no -r!" << std::endl;
    exit(1);
  }
  if (early_z && (reference_file || use_sw)) {
    std::cout << "Error: -E checks the early-Z outcome itself, on the hardware rasterizer; it takes no -r or -z!" << std::endl;
    exit(1);
  }
}

void cleanup() {
  if (color_buffer) vx_buffer_release(color_buffer);
  if (tile_buffer)  vx_buffer_release(tile_buffer);
  if (prim_buffer)  vx_buffer_release(prim_buffer);
  if (frag_arg_buffer) vx_buffer_release(frag_arg_buffer);
  if (depth_buffer) vx_buffer_release(depth_buffer);
  if (kernel)  vx_kernel_release(kernel);
  if (module_) vx_module_release(module_);
  if (queue)   vx_queue_release(queue);
  if (device) {
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

// The draw-wide scissor the DCRs program. Under -s it is narrower than the
// render target on every side, so the rasterizer's intersection of the DCR and
// record rectangles is what decides the outer pixels.
static void draw_scissor(uint32_t& left, uint32_t& right, uint32_t& top, uint32_t& bottom) {
  left = 0; right = dst_width; top = 0; bottom = dst_height;
  if (prim_scissor) {
    left = 3; right = dst_width - 2; top = 1; bottom = dst_height - 5;
  }
}

// Give primitive i its own rectangle: odd origins and extents, so the edges fall
// mid-quad and mid-tile, and some reach past the draw rectangle.
static void patch_prim_scissors(std::vector<uint8_t>& primbuf) {
  auto* prims = reinterpret_cast<graphics::rast_prim_t*>(primbuf.data());
  const uint32_t n = (uint32_t)(primbuf.size() / sizeof(graphics::rast_prim_t));
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t x0 = (dst_width / 4 + 5 + 13 * i) % (dst_width / 2);
    const uint32_t y0 = (dst_height / 4 + 9 + 11 * i) % (dst_height / 2);
    const uint32_t x1 = dst_width - (3 + 7 * i) % (dst_width / 4) + (i & 1) * 8;
    const uint32_t y1 = dst_height - (dst_height / 4 + 1 + 3 * i) % (dst_height / 2);
    prims[i].scissor_x = RAST_SCISSOR_PACK(x0, x1);
    prims[i].scissor_y = RAST_SCISSOR_PACK(y0, y1);
  }
}

// Whether the image is checked pixel-exactly against reference_draw.
static bool exact_check() {
  return prim_scissor || num_generated != 0 || num_samples > 1 || early_z || slivers;
}

// -E: the D24 depth each sample's slot starts with.
static uint32_t initial_depth(uint32_t x, uint32_t y, uint32_t k) {
  return ((x * 7u + y * 13u + k * 5u) & 0xffu) << 16;
}

// Whether early-Z keeps sample k of (x, y) under LESS: its plane depth, as the
// fragment shader evaluates it at the sample, is not strictly behind the stored
// one.
static bool early_z_keeps(const graphics::rast_attrib_t& zp, uint32_t x, uint32_t y, uint32_t k) {
  static constexpr int dx16[4] = GFX_MSAA4_SAMPLE_DX16;
  static constexpr int dy16[4] = GFX_MSAA4_SAMPLE_DY16;
  // As the fragment shader evaluates it: in 64 bits, saturated.
  int64_t z = int64_t(zp.x.data()) * x + int64_t(zp.y.data()) * y + int64_t(zp.z.data());
  if (num_samples > 1) {
    z += (int64_t(zp.x.data()) * dx16[k] + int64_t(zp.y.data()) * dy16[k]) >> 4;
  }
  const uint32_t cand = z < 0 ? 0u : uint32_t(std::min<int64_t>(z, OM_DEPTH_MASK));
  return cand <= initial_depth(x, y, k);
}

// The colour the kernel writes for the record at draw-wide index `id`.
static uint32_t record_color(uint32_t id) {
  return num_generated ? prim_id_color(id) : 0xffffffff;
}

// The image the device must produce: every primitive walked by the shared
// coverage core under its record's rectangle narrowed to the draw's. The pass's
// records are draw-wide indices pid_base onward.
static void reference_draw(const std::vector<uint8_t>& primbuf, uint32_t pid_base) {
  uint32_t left, right, top, bottom;
  draw_scissor(left, right, top, bottom);
  const auto* prims = reinterpret_cast<const graphics::rast_prim_t*>(primbuf.data());
  const uint32_t n = (uint32_t)(primbuf.size() / sizeof(graphics::rast_prim_t));
  const uint32_t tile = 1u << tileLogSize;
  for (uint32_t pid = 0; pid < n; ++pid) {
    const gfx_rast::RastConfig cfg = gfx_rast::rast_prim_config(
        { tileLogSize, left, top, right, bottom }, prims[pid].scissor_x, prims[pid].scissor_y);
    for (uint32_t ty = 0; ty < dst_height; ty += tile) {
      for (uint32_t tx = 0; tx < dst_width; tx += tile) {
        if (num_samples > 1) {
          gfx_rast::rast_walk_primitive_msaa(cfg, tx, ty, pid, prims[pid].edges,
            [&](uint32_t pos_mask, const graphics::vec3e_t*, const uint32_t* sample_masks, uint32_t) {
              const uint32_t qx = (pos_mask >> 4) & ((1u << (VX_RASTER_DIM_BITS - 1)) - 1);
              const uint32_t qy = pos_mask >> (4 + (VX_RASTER_DIM_BITS - 1));
              for (uint32_t c = 0; c < 4; ++c) {
                const uint32_t px = (qx << 1) + (c & 1), py = (qy << 1) + (c >> 1);
                if (px >= dst_width || py >= dst_height) {
                  continue;
                }
                bool kept = !early_z;
                for (uint32_t k = 0; k < num_samples; ++k) {
                  kept |= ((sample_masks[c] >> k) & 1) && early_z_keeps(prims[pid].attribs.z, px, py, k);
                }
                for (uint32_t k = 0; k < num_samples; ++k) {
                  if (kept && ((sample_masks[c] >> k) & 1)) {
                    ref_pixels[(py * dst_width + px) * num_samples + k] = record_color(pid_base + pid);
                  }
                }
              }
            });
          continue;
        }
        gfx_rast::rast_walk_primitive(cfg, tx, ty, pid, prims[pid].edges,
          [&](uint32_t pos_mask, const graphics::vec3e_t*, uint32_t) {
            const uint32_t qx = (pos_mask >> 4) & ((1u << (VX_RASTER_DIM_BITS - 1)) - 1);
            const uint32_t qy = pos_mask >> (4 + (VX_RASTER_DIM_BITS - 1));
            for (uint32_t k = 0; k < 4; ++k) {
              const uint32_t px = (qx << 1) + (k & 1), py = (qy << 1) + (k >> 1);
              if (((pos_mask >> k) & 1) && px < dst_width && py < dst_height
               && (!early_z || early_z_keeps(prims[pid].attribs.z, px, py, 0))) {
                ref_pixels[py * dst_width + px] = record_color(pid_base + pid);
              }
            }
          });
      }
    }
  }
}

// A draw of `count` triangles, each inside one pixel. Every 32nd covers its
// pixel's centre, at a pixel of its own; the rest cover no sample. The draw is
// large while its fragments stay few, and the covering ones include the last pid
// of every full RASTER pass.
static void generated_draw(uint32_t count,
                           std::unordered_map<uint32_t, graphics::vertex_t>& verts,
                           std::vector<graphics::primitive_t>& prims) {
  constexpr uint32_t kCoverStride = 32;
  if (count / kCoverStride > dst_width * dst_height) {
    std::cout << "Error: " << count << " triangles need more than "
              << dst_width << "x" << dst_height << " pixels!" << std::endl;
    cleanup();
    exit(1);
  }
  auto vertex = [&](float sx, float sy) {
    graphics::vertex_t v{};
    v.pos[0] = 2.0f * sx / dst_width - 1.0f;
    v.pos[1] = 2.0f * sy / dst_height - 1.0f;
    v.pos[2] = 0.5f;
    v.pos[3] = 1.0f;
    return v;
  };
  verts.clear();
  prims.clear();
  for (uint32_t i = 0; i < count; ++i) {
    const bool covers = (i % kCoverStride) == kCoverStride - 1;
    const uint32_t pixel = covers ? i / kCoverStride : i;
    const float x = float(pixel % dst_width);
    const float y = float((pixel / dst_width) % dst_height);
    // Around the pixel centre (0.5, 0.5), or in its lower-right corner past it.
    const float lo = covers ? 0.1f : 0.6f, hi = 0.95f;
    verts[3 * i + 0] = vertex(x + lo, y + lo);
    verts[3 * i + 1] = vertex(x + hi, y + lo);
    verts[3 * i + 2] = vertex(x + lo, y + hi);
    prims.push_back({3 * i, 3 * i + 1, 3 * i + 2});
  }
}

// Slivers along the diagonal, each 0.008 px across around the line where
// x - y = m + o_k, the offset o_k of sample k from its pixel's centre (m in
// -1..1), so it covers sample k of the pixels it crosses. The depth rises 0.4
// across a sliver's width: 50 depth ranges a pixel, while near the origin the
// plane stays within its range.
static void sliver_draw(std::unordered_map<uint32_t, graphics::vertex_t>& verts,
                        std::vector<graphics::primitive_t>& prims) {
  static constexpr int dx16[4] = GFX_MSAA4_SAMPLE_DX16;
  static constexpr int dy16[4] = GFX_MSAA4_SAMPLE_DY16;
  constexpr float kHalfWidth = 0.004f;
  auto vertex = [&](float sx, float sy, float z) {
    graphics::vertex_t v{};
    v.pos[0] = 2.0f * sx / dst_width - 1.0f;
    v.pos[1] = 2.0f * sy / dst_height - 1.0f;
    v.pos[2] = z;
    v.pos[3] = 1.0f;
    return v;
  };
  verts.clear();
  prims.clear();
  const float t0 = 2.0f, t1 = float(std::min(dst_width, dst_height)) - 2.0f;
  uint32_t n = 0;
  for (int m = -1; m <= 1; ++m) {
    for (uint32_t k = 0; k < 4; ++k) {
      // The pixel's centre (t + 0.5, t + 0.5) moved to the line through sample k.
      const float d = float(m) + float(dx16[k] - dy16[k]) / 16.0f;
      auto on_line = [&](float t, float e, float z) {
        return vertex(t + 0.5f + 0.5f * (d + e), t + 0.5f - 0.5f * (d + e), z);
      };
      verts[3 * n + 0] = on_line(t0, -kHalfWidth, -0.2f);
      verts[3 * n + 1] = on_line(t1, -kHalfWidth, -0.2f);
      verts[3 * n + 2] = on_line(t0,  kHalfWidth,  0.6f);
      prims.push_back({3 * n, 3 * n + 1, 3 * n + 2});
      ++n;
    }
  }
}

// One RASTER pass: bin `prims`, upload the pass's buffers and run it. Its
// records are draw-wide indices pid_base onward; `num_records` returns how many
// it kept.
static int draw_pass(const std::unordered_map<uint32_t, graphics::vertex_t>& verts,
                     const std::vector<graphics::primitive_t>& prims,
                     float near, float far, uint32_t pid_base, uint32_t* num_records) {
  std::vector<uint8_t> tilebuf;
  std::vector<uint8_t> primbuf;

  // Perform tile binning
  auto num_tiles = graphics::Binning(tilebuf, primbuf, verts, prims, dst_width, dst_height, near, far, tileLogSize);
  *num_records = (uint32_t)(primbuf.size() / sizeof(graphics::rast_prim_t));
  std::cout << "Binning allocated " << std::dec << num_tiles << " tiles with " << *num_records << " primitives." << std::endl;
  if (0 == num_tiles) {
    // Every input of this test has visible primitives, so an empty bin set is
    // Binning refusing the pass.
    if (!prims.empty()) {
      std::cout << "Error: Binning produced no bins for " << prims.size() << " primitives!" << std::endl;
      return -1;
    }
    return 0;
  }
  if (prim_scissor) {
    patch_prim_scissors(primbuf);
  }
  if (exact_check()) {
    reference_draw(primbuf, pid_base + (relaunch ? kRelaunchIdOffset : 0));
  }

  // allocate tile memory
  if (tile_buffer != nullptr) { vx_buffer_release(tile_buffer); tile_buffer = nullptr; }
  if (prim_buffer != nullptr) { vx_buffer_release(prim_buffer); prim_buffer = nullptr; }
  // HW path: tile_buffer / prim_buffer are bound to the raster unit (via
  // VX_DCR_RASTER_T/PBUF_ADDR), which reads them through its own AXI master
  // bypassing the per-core MMU — both need physical (pinned) addresses. SW path
  // (-z): the kernel reads prim_buffer through the LSU, so allocate it as a
  // normal kernel-read buffer (tile_buffer is unused by the SW walk).
  uint32_t prim_flags = VX_MEM_READ | (use_sw ? 0u : (uint32_t)VX_MEM_PHYS);
  RT_CHECK(vx_buffer_create(device, tilebuf.size(), VX_MEM_READ | VX_MEM_PHYS, &tile_buffer));
  RT_CHECK(vx_buffer_address(tile_buffer, &tilebuf_addr));
  RT_CHECK(vx_buffer_create(device, primbuf.size(), prim_flags, &prim_buffer));
  RT_CHECK(vx_buffer_address(prim_buffer, &primbuf_addr));
  std::cout << "tile_buffer=0x" << std::hex << tilebuf_addr << std::dec << std::endl;
  std::cout << "prim_buffer=0x" << std::hex << primbuf_addr << std::dec << std::endl;

  // upload tiles buffer
  std::cout << "upload tile buffer" << std::endl;
  RT_CHECK(vx_enqueue_write(queue, tile_buffer, 0, tilebuf.data(), tilebuf.size(), 0, nullptr, nullptr));

  // upload primitives buffer
  std::cout << "upload primitive buffer" << std::endl;
  RT_CHECK(vx_enqueue_write(queue, prim_buffer, 0, primbuf.data(), primbuf.size(), 0, nullptr, nullptr));

  // prepare kernel argument
  std::cout << "prepare kernel argument" << std::endl;
  {
    kernel_arg.prim_addr   = primbuf_addr;
    kernel_arg.dst_width   = dst_width;
    kernel_arg.dst_height  = dst_height;
    kernel_arg.cbuf_addr   = cbuf_addr;
    kernel_arg.cbuf_stride = cbuf_stride;
    kernel_arg.cbuf_pitch  = cbuf_pitch;
    // SW raster routing: dense visible-prim count + the walk tile size.
    kernel_arg.sw_path      = use_sw ? 1u : 0u;
    kernel_arg.num_prims    = (uint32_t)(primbuf.size() / sizeof(graphics::rast_prim_t));
    kernel_arg.tile_logsize = tileLogSize;
    uint32_t l, r, t, b;
    draw_scissor(l, r, t, b);
    kernel_arg.scissor_x = RAST_SCISSOR_PACK(l, r);
    kernel_arg.scissor_y = RAST_SCISSOR_PACK(t, b);
    kernel_arg.pid_color = num_generated ? 1u : 0u;
    kernel_arg.pid_base  = pid_base;
    kernel_arg.samples   = num_samples;
  }
  if (use_sw)
    std::cout << "[gfx_raster] software fine-rasterizer path "
              << "(gfx_rast::rast_walk_primitive, " << kernel_arg.num_prims
              << " prims)" << std::endl;

  uint32_t primbuf_stride = sizeof(graphics::rast_prim_t);

  // configure raster units
  vx_enqueue_dcr_write(queue, VX_DCR_RASTER_TBUF_ADDR, tilebuf_addr / 64, 0, nullptr, nullptr);  // block address
  vx_enqueue_dcr_write(queue, VX_DCR_RASTER_TILE_COUNT, num_tiles, 0, nullptr, nullptr);
  vx_enqueue_dcr_write(queue, VX_DCR_RASTER_PBUF_ADDR, primbuf_addr / 64, 0, nullptr, nullptr);  // block address
  vx_enqueue_dcr_write(queue, VX_DCR_RASTER_PBUF_STRIDE, primbuf_stride, 0, nullptr, nullptr);
  uint32_t sc_left, sc_right, sc_top, sc_bottom;
  draw_scissor(sc_left, sc_right, sc_top, sc_bottom);
  vx_enqueue_dcr_write(queue, VX_DCR_RASTER_SCISSOR_X, RAST_SCISSOR_PACK(sc_left, sc_right), 0, nullptr, nullptr);
  vx_enqueue_dcr_write(queue, VX_DCR_RASTER_SCISSOR_Y, RAST_SCISSOR_PACK(sc_top, sc_bottom), 0, nullptr, nullptr);
  if (num_samples > 1) {
    vx_enqueue_dcr_write(queue, VX_DCR_RASTER_SAMPLES, num_samples, 0, nullptr, nullptr);
  }
  if (early_z) {
    vx_enqueue_dcr_write(queue, VX_DCR_OM_ZBUF_ADDR, zbuf_addr / 64, 0, nullptr, nullptr);
    vx_enqueue_dcr_write(queue, VX_DCR_OM_ZBUF_PITCH, dst_width * 4, 0, nullptr, nullptr);
    vx_enqueue_dcr_write(queue, VX_DCR_OM_ZBUF_SAMPLE_STRIDE, dst_width * dst_height * 4, 0, nullptr, nullptr);
    vx_enqueue_dcr_write(queue, VX_DCR_OM_DEPTH_FORMAT, VX_OM_DEPTH_FORMAT_D24S8, 0, nullptr, nullptr);
    vx_enqueue_dcr_write(queue, VX_DCR_OM_DEPTH_FUNC, VX_OM_DEPTH_FUNC_LESS, 0, nullptr, nullptr);
    vx_enqueue_dcr_write(queue, VX_DCR_OM_SAMPLES, num_samples, 0, nullptr, nullptr);
    vx_enqueue_dcr_write(queue, VX_DCR_OM_EARLYZ_SAFE, 1, 0, nullptr, nullptr);
  }

  // RASTER dispatch v2 (push, HW path only): stage the FS args and program the
  // fragment-dispatch descriptor (FS entry PC + args pointer). The work
  // distributor launches one fragment warp per covered-quad wave on-device.
  if (!use_sw) {
    if (frag_arg_buffer == nullptr) {
      RT_CHECK(vx_buffer_create(device, sizeof(kernel_arg), VX_MEM_READ, &frag_arg_buffer));
      RT_CHECK(vx_buffer_address(frag_arg_buffer, &frag_arg_addr));
    }
    RT_CHECK(vx_enqueue_write(queue, frag_arg_buffer, 0, &kernel_arg, sizeof(kernel_arg), 0, nullptr, nullptr));
    uint64_t frag_entry = 0;
    RT_CHECK(vx_kernel_address(kernel, &frag_entry));
    vx_enqueue_dcr_write(queue, VX_DCR_RASTER_FRAG_ENTRY_LO, (uint32_t)(frag_entry & 0xffffffff), 0, nullptr, nullptr);
    vx_enqueue_dcr_write(queue, VX_DCR_RASTER_FRAG_ENTRY_HI, (uint32_t)(frag_entry >> 32), 0, nullptr, nullptr);
    vx_enqueue_dcr_write(queue, VX_DCR_RASTER_FRAG_PARAM_LO, (uint32_t)(frag_arg_addr & 0xffffffff), 0, nullptr, nullptr);
    vx_enqueue_dcr_write(queue, VX_DCR_RASTER_FRAG_PARAM_HI, (uint32_t)(frag_arg_addr >> 32), 0, nullptr, nullptr);
  }

  auto time_start = std::chrono::high_resolution_clock::now();

  // HW path: grid-less kick (grid_dim=0 → no host warps; the armed raster work
  // distributor injects the fragment warps and sustains the run). SW path: one
  // lane is one pixel, so a quad needs four adjacent lanes — give each primitive
  // a group of VX_FRAG_QUAD_LANES threads and launch enough CTAs to cover them.
  auto launch = [&]() {
    uint32_t block_sz = (uint32_t)(num_threads * num_warps);
    uint32_t grid0;
    if (use_sw) {
      uint32_t sw_threads = kernel_arg.num_prims * VX_FRAG_QUAD_LANES;
      grid0 = (sw_threads + block_sz - 1) / block_sz;
      if (grid0 == 0) {
        grid0 = 1;
      }
    } else {
      grid0 = 0;   // grid-less kick (push dispatch)
    }
    std::cout << "start device (grid=" << grid0 << ", block=" << block_sz << ")" << std::endl;
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &kernel_arg;
    li.args_size    = sizeof(kernel_arg);
    li.ndim         = 1;
    li.grid_dim[0]  = grid0;
    li.block_dim[0] = block_sz;
    vx_event_h launch_ev = nullptr;
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
    std::cout << "wait for completion" << std::endl;
    RT_CHECK(vx_event_wait_value(launch_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(launch_ev);
  };
  launch();

  // The same bins again, with only the fragment arguments changed (a memory
  // write, not a DCR write): the raster unit re-walks them on every launch, so
  // the second launch repaints every covered pixel in its own colours.
  if (relaunch) {
    kernel_arg.pid_base += kRelaunchIdOffset;
    if (!use_sw) {
      RT_CHECK(vx_enqueue_write(queue, frag_arg_buffer, 0, &kernel_arg, sizeof(kernel_arg), 0, nullptr, nullptr));
    }
    launch();
  }

  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  return 0;
}

// Draw `prims` as consecutive RASTER passes of at most GFX_RASTER_MAX_PRIMS
// primitives, each binned over its own records. Host Binning keeps at most one
// record per triangle (it does not clip), so chunking the triangles by the pass
// limit keeps every pass within it.
static int draw(const std::unordered_map<uint32_t, graphics::vertex_t>& verts,
                const std::vector<graphics::primitive_t>& prims, float near, float far) {
  uint32_t pid_base = 0;
  for (size_t first = 0; first < prims.size(); first += GFX_RASTER_MAX_PRIMS) {
    const size_t last = std::min(prims.size(), first + GFX_RASTER_MAX_PRIMS);
    const std::vector<graphics::primitive_t> pass(prims.begin() + first, prims.begin() + last);
    uint32_t num_records = 0;
    RT_CHECK(draw_pass(verts, pass, near, far, pid_base, &num_records));
    pid_base += num_records;
  }
  return 0;
}

int render(const CGLTrace& trace) {
  if (slivers) {
    std::unordered_map<uint32_t, graphics::vertex_t> verts;
    std::vector<graphics::primitive_t> prims;
    sliver_draw(verts, prims);
    RT_CHECK(draw(verts, prims, 0.0f, 1.0f));
  } else if (num_generated) {
    std::unordered_map<uint32_t, graphics::vertex_t> verts;
    std::vector<graphics::primitive_t> prims;
    generated_draw(num_generated, verts, prims);
    RT_CHECK(draw(verts, prims, 0.0f, 1.0f));
  } else {
    for (auto& drawcall : trace.drawcalls) {
      RT_CHECK(draw(cgl_to_gfx_vertices(drawcall.vertices),
                    cgl_to_gfx_primitives(drawcall.primitives),
                    drawcall.viewport.near, drawcall.viewport.far));
    }
  }

  // save output image
  if (strcmp(output_file, "null") != 0 || exact_check()) {
    std::vector<uint8_t> dst_pixels(cbuf_size);
    {
      vx_event_h read_ev = nullptr;
      RT_CHECK(vx_enqueue_read(queue, dst_pixels.data(), color_buffer, 0, cbuf_size, 0, nullptr, &read_ev));
      RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
      vx_event_release(read_ev);
    }
    if (strcmp(output_file, "null") != 0) {
      std::cout << "save output image" << std::endl;
      // A multisample buffer is saved as its sample-0 plane.
      std::vector<uint32_t> plane(dst_width * dst_height);
      for (uint32_t y = 0; y < dst_height; ++y) {
        for (uint32_t x = 0; x < dst_width; ++x) {
          std::memcpy(&plane[y * dst_width + x], dst_pixels.data() + y * cbuf_pitch + x * cbuf_stride, 4);
        }
      }
      auto bits = reinterpret_cast<const uint8_t*>(plane.data()) + (dst_height-1) * dst_width * 4;
      RT_CHECK(SaveImage(output_file, FORMAT_A8R8G8B8, bits, dst_width, dst_height, -int(dst_width * 4)));
    }
    if (exact_check()) {
      uint32_t errors = 0, covered = 0;
      for (uint32_t y = 0; y < dst_height; ++y) {
        for (uint32_t x = 0; x < dst_width; ++x) {
          for (uint32_t k = 0; k < num_samples; ++k) {
            uint32_t got;
            std::memcpy(&got, dst_pixels.data() + y * cbuf_pitch + x * cbuf_stride + k * 4, 4);
            const uint32_t want = ref_pixels[(y * dst_width + x) * num_samples + k];
            covered += (want != clear_color);
            if (got != want) {
              if (errors < 10) {
                printf("*** error: pixel (%u, %u) sample %u = 0x%08x, expected 0x%08x\n", x, y, k, got, want);
              }
              ++errors;
            }
          }
        }
      }
      printf("exact check: %u of %u samples covered, %u mismatches\n",
             covered, dst_width * dst_height * num_samples, errors);
      exact_errors = errors;
    }
  }

  return 0;
}

int main(int argc, char *argv[]) {
  // parse command arguments
  parse_args(argc, argv);

  // open device connection
  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t isa_flags;
  RT_CHECK(vx_device_query(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  bool has_ext = (isa_flags & VX_ISA_EXT_RASTER) != 0;
  if (!has_ext) {
    std::cout << "RASTER extensions not supported!" << std::endl;
    cleanup();
    return -1;
  }

  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));
  std::cout << "device: " << num_cores << " cores, " << num_warps
            << " warps, " << num_threads << " threads" << std::endl;

  CGLTrace trace;
  auto trace_file_s = resolve_path(trace_file, ASSETS_PATHS);
  RT_CHECK(trace.load(trace_file_s.c_str()));

  // load kernel module
  std::cout << "load kernel module" << std::endl;
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, num_samples > 1 ? "kernel_main_msaa" : "main", &kernel));

  cbuf_stride = 4 * num_samples;
  cbuf_pitch  = dst_width * cbuf_stride;
  cbuf_size   = dst_height * cbuf_pitch;

  // allocate device memory
  // color_buffer is kernel-written via the LSU only — no HW-block reader
  // (the kernel does scalar stores from its rasterised quads), so no
  // pinning needed. The per-core MMU translates the VA for the kernel.
  RT_CHECK(vx_buffer_create(device, cbuf_size, VX_MEM_WRITE, &color_buffer));
  RT_CHECK(vx_buffer_address(color_buffer, &cbuf_addr));
  std::cout << "color_buffer=0x" << std::hex << cbuf_addr << std::dec << std::endl;

  // clear destination buffer
  std::cout << "clear destination buffer" << std::endl;
  {
    std::vector<uint32_t> staging_buf(cbuf_size / 4, clear_color);
    vx_event_h ev = nullptr;
    RT_CHECK(vx_enqueue_write(queue, color_buffer, 0, staging_buf.data(), cbuf_size, 0, nullptr, &ev));
    RT_CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(ev);
  }

  // The depth images early-Z reads, through the OM's cache: one per sample.
  if (early_z) {
    const uint32_t zsize = dst_width * dst_height * num_samples * 4;
    RT_CHECK(vx_buffer_create(device, zsize, VX_MEM_READ_WRITE | VX_MEM_PHYS, &depth_buffer));
    RT_CHECK(vx_buffer_address(depth_buffer, &zbuf_addr));
    std::vector<uint32_t> zinit(zsize / 4);
    for (uint32_t y = 0; y < dst_height; ++y) {
      for (uint32_t x = 0; x < dst_width; ++x) {
        for (uint32_t k = 0; k < num_samples; ++k) {
          zinit[(k * dst_height + y) * dst_width + x] = initial_depth(x, y, k);
        }
      }
    }
    vx_event_h ev = nullptr;
    RT_CHECK(vx_enqueue_write(queue, depth_buffer, 0, zinit.data(), zsize, 0, nullptr, &ev));
    RT_CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(ev);
  }

  // update kernel arguments
  kernel_arg.dst_width  = dst_width;
  kernel_arg.dst_height = dst_height;

  kernel_arg.cbuf_stride= cbuf_stride;
  kernel_arg.cbuf_pitch = cbuf_pitch;
  kernel_arg.cbuf_addr  = cbuf_addr;

  // run tests
  std::cout << "render" << std::endl;
  ref_pixels.assign(dst_width * dst_height * num_samples, clear_color);
  RT_CHECK(render(trace));

  // cleanup
  std::cout << "cleanup" << std::endl;
  cleanup();

  if (exact_check()) {
    if (exact_errors != 0) {
      std::cout << "FAILED! " << exact_errors << " errors." << std::endl;
      return 1;
    }
    std::cout << "PASSED!" << std::endl;
    return 0;
  }

  if (strcmp (output_file, "") != 0 && reference_file) {
    auto reference_file_s = resolve_path(reference_file, ASSETS_PATHS);
    auto errors = CompareImages(output_file, reference_file_s.c_str(), FORMAT_A8R8G8B8);
    if (0 == errors) {
      std::cout << "PASSED!" << std::endl;
    } else {
      std::cout << "FAILED! " << errors << " errors." << std::endl;
      return 1;  // non-zero exit on mismatch (error count truncates mod 256 as a code)
    }
  }

  return 0;
}