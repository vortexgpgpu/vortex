// 3D rendering regression test: full TEX + RASTER + OM pipeline
// against a CGLTrace replay.
//
// Pinned-buffer contract:
//   tile_buffer   → RASTER HW (VX_DCR_RASTER_TBUF_ADDR) → pinned
//   prim_buffer   → RASTER HW (VX_DCR_RASTER_PBUF_ADDR) → pinned
//   tex_buffer    → TEX    HW (VX_DCR_TEX_ADDR)         → pinned
//   depth_buffer  → OM     HW (VX_DCR_OM_ZBUF_ADDR)     → pinned
//   color_buffer  → OM     HW (VX_DCR_OM_CBUF_ADDR)     → pinned
// All five buffers are accessed by fixed-function HW whose AXI master
// bypasses the per-core MMU, so under VM they must be
// identity-mapped (allocated with VX_MEM_PHYS).

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
#include <bitmanip.h>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include "common.h"
#include <cocogfx/include/cgltrace.hpp>
#include <cocogfx/include/format.hpp>
#include <cocogfx/include/blitter.hpp>
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

// Local helpers: CGLTrace -> VX_OM / VX_TEX state translation.
static uint32_t toVXFormat(ePixelFormat format) {
  switch (format) {
  case FORMAT_A8R8G8B8: return VX_TEX_FORMAT_A8R8G8B8;
  case FORMAT_R5G6B5:   return VX_TEX_FORMAT_R5G6B5;
  case FORMAT_A1R5G5B5: return VX_TEX_FORMAT_A1R5G5B5;
  case FORMAT_A4R4G4B4: return VX_TEX_FORMAT_A4R4G4B4;
  case FORMAT_A8L8:     return VX_TEX_FORMAT_A8L8;
  case FORMAT_L8:       return VX_TEX_FORMAT_L8;
  case FORMAT_A8:       return VX_TEX_FORMAT_A8;
  default:
    std::cout << "Error: invalid format: " << format << std::endl;
    exit(1);
  }
}

static uint32_t toVXCompare(CGLTrace::ecompare compare) {
  switch (compare) {
  case CGLTrace::COMPARE_NEVER:    return VX_OM_DEPTH_FUNC_NEVER;
  case CGLTrace::COMPARE_LESS:     return VX_OM_DEPTH_FUNC_LESS;
  case CGLTrace::COMPARE_EQUAL:    return VX_OM_DEPTH_FUNC_EQUAL;
  case CGLTrace::COMPARE_LEQUAL:   return VX_OM_DEPTH_FUNC_LEQUAL;
  case CGLTrace::COMPARE_GREATER:  return VX_OM_DEPTH_FUNC_GREATER;
  case CGLTrace::COMPARE_NOTEQUAL: return VX_OM_DEPTH_FUNC_NOTEQUAL;
  case CGLTrace::COMPARE_GEQUAL:   return VX_OM_DEPTH_FUNC_GEQUAL;
  case CGLTrace::COMPARE_ALWAYS:   return VX_OM_DEPTH_FUNC_ALWAYS;
  default:
    std::cout << "Error: invalid compare function: " << compare << std::endl;
    exit(1);
  }
}

static uint32_t toVXStencilOp(CGLTrace::eStencilOp op) {
  switch (op) {
  case CGLTrace::STENCIL_KEEP:    return VX_OM_STENCIL_OP_KEEP;
  case CGLTrace::STENCIL_REPLACE: return VX_OM_STENCIL_OP_REPLACE;
  case CGLTrace::STENCIL_INCR:    return VX_OM_STENCIL_OP_INCR;
  case CGLTrace::STENCIL_DECR:    return VX_OM_STENCIL_OP_DECR;
  case CGLTrace::STENCIL_ZERO:    return VX_OM_STENCIL_OP_ZERO;
  case CGLTrace::STENCIL_INVERT:  return VX_OM_STENCIL_OP_INVERT;
  default:
    std::cout << "Error: invalid stencil operation: " << op << std::endl;
    exit(1);
  }
}

static uint32_t toVXBlendFunc(CGLTrace::eBlendOp op) {
  switch (op) {
  case CGLTrace::BLEND_ZERO:                return VX_OM_BLEND_FUNC_ZERO;
  case CGLTrace::BLEND_ONE:                 return VX_OM_BLEND_FUNC_ONE;
  case CGLTrace::BLEND_SRC_COLOR:           return VX_OM_BLEND_FUNC_SRC_RGB;
  case CGLTrace::BLEND_ONE_MINUS_SRC_COLOR: return VX_OM_BLEND_FUNC_ONE_MINUS_SRC_RGB;
  case CGLTrace::BLEND_SRC_ALPHA:           return VX_OM_BLEND_FUNC_SRC_A;
  case CGLTrace::BLEND_ONE_MINUS_SRC_ALPHA: return VX_OM_BLEND_FUNC_ONE_MINUS_SRC_A;
  case CGLTrace::BLEND_DST_ALPHA:           return VX_OM_BLEND_FUNC_DST_A;
  case CGLTrace::BLEND_ONE_MINUS_DST_ALPHA: return VX_OM_BLEND_FUNC_ONE_MINUS_DST_A;
  case CGLTrace::BLEND_DST_COLOR:           return VX_OM_BLEND_FUNC_DST_RGB;
  case CGLTrace::BLEND_ONE_MINUS_DST_COLOR: return VX_OM_BLEND_FUNC_ONE_MINUS_DST_RGB;
  case CGLTrace::BLEND_SRC_ALPHA_SATURATE:  return VX_OM_BLEND_FUNC_ALPHA_SAT;
  default:
    std::cout << "Error: invalid blend function: " << op << std::endl;
    exit(1);
  }
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
const char* reference_file = nullptr;

// FF/SW routing is a build property (NO_TEX / NO_OM / NO_RASTER), not a runtime
// flag: a stage cannot take the FF path on a device whose FF unit was not built.
// Host and kernel both key off VX_CFG_EXT_*_ENABLED so they cannot disagree.
static constexpr bool ff_tex    = (VX_CFG_EXT_TEX_ENABLED != 0);
static constexpr bool ff_om     = (VX_CFG_EXT_OM_ENABLED != 0);
static constexpr bool ff_raster = (VX_CFG_EXT_RASTER_ENABLED != 0);

uint64_t num_threads = 0;  // populated in main, read by render()
uint64_t num_warps   = 0;  // populated in main, read by render()
uint64_t num_cores   = 0;  // populated in main, read by render()

uint32_t start_draw = 0;
uint32_t end_draw = -1;

uint32_t clear_color = 0xff000000;
uint32_t clear_depth = 0xffffffff;

uint32_t dst_width  = 128;
uint32_t dst_height = 128;

uint32_t zbuf_stride;
uint32_t zbuf_pitch;
uint32_t zbuf_size;

uint32_t cbuf_stride;
uint32_t cbuf_pitch;
uint32_t cbuf_size;

uint64_t cbuf_addr;
uint64_t zbuf_addr;
uint64_t texbuf_addr;
uint64_t tilebuf_addr;
uint64_t primbuf_addr;

vx_device_h device      = nullptr;
vx_queue_h  queue       = nullptr;
vx_module_h module_     = nullptr;
vx_kernel_h kernel      = nullptr;
vx_buffer_h depth_buffer= nullptr;
vx_buffer_h color_buffer= nullptr;
vx_buffer_h tex_buffer  = nullptr;
vx_buffer_h tile_buffer = nullptr;
vx_buffer_h prim_buffer = nullptr;
vx_buffer_h frag_arg_buffer = nullptr;   // FS args (RASTER frag-dispatch descriptor)
uint64_t    frag_arg_addr = 0;

kernel_arg_t kernel_arg = {};

uint32_t tileLogSize = VX_CFG_RASTER_BIN_LOG_SIZE;   // host Binning() emits coarse-bin headers

static void show_usage() {
   std::cout << "Vortex 3D Rendering Test." << std::endl;
   std::cout << "Usage: [-t trace] [-s startdraw] [-e enddraw] [-o output] [-r reference] [-w width] [-h height] [-k tilelogsize]" << std::endl;
   std::cout << "  FF/SW routing is a build knob: make NO_TEX=1 / NO_OM=1 / NO_RASTER=1" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "t:s:e:i:o:r:w:h:t:k:?")) != -1) {
    switch (c) {
    case 't':
      trace_file = optarg;
      break;
    case 's':
      start_draw = std::atoi(optarg);
      break;
    case 'e':
      end_draw = std::atoi(optarg);
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
    case '?': {
      show_usage();
      exit(0);
    } break;
    default:
      show_usage();
      exit(-1);
    }
  }
  if (strcmp (output_file, "null") == 0 && reference_file) {
    std::cout << "Error: the output file is missing for reference validation!" << std::endl;
    exit(1);
  }
}

void cleanup() {
  if (depth_buffer) vx_buffer_release(depth_buffer);
  if (color_buffer) vx_buffer_release(color_buffer);
  if (tex_buffer)   vx_buffer_release(tex_buffer);
  if (tile_buffer)  vx_buffer_release(tile_buffer);
  if (prim_buffer)  vx_buffer_release(prim_buffer);
  if (frag_arg_buffer) vx_buffer_release(frag_arg_buffer);
  if (kernel)  vx_kernel_release(kernel);
  if (module_) vx_module_release(module_);
  if (queue)   vx_queue_release(queue);
  if (device) {
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

// A DCR write only reaches a unit that exists. When a stage is software-routed
// its state travels in kernel_arg instead (kernel_arg.tex / .om), so the DCR
// write is skipped rather than issued into the void.
#define RASTER_DCR_WRITE(addr, value)  \
  do { if (ff_raster) vx_enqueue_dcr_write(queue, addr, value, 0, nullptr, nullptr); } while (0)

#define OM_DCR_WRITE(addr, value)  \
  do { if (ff_om) vx_enqueue_dcr_write(queue, addr, value, 0, nullptr, nullptr); } while (0)

#define TEX_DCR_WRITE(addr, value)  \
  do { if (ff_tex) vx_enqueue_dcr_write(queue, addr, value, 0, nullptr, nullptr); } while (0)

int render(const CGLTrace& trace) {
  std::cout << "render" << std::endl;
  auto time_begin = std::chrono::high_resolution_clock::now();

  // render each draw call
  for (uint32_t d = 0, nd = trace.drawcalls.size(); d < nd; ++d) {
    if (d < start_draw || d > end_draw)
      continue;

    auto& drawcall = trace.drawcalls.at(d);
    auto& states = drawcall.states;

    std::vector<uint8_t> tilebuf;
    std::vector<uint8_t> primbuf;
    // texbuf is hoisted to drawcall-loop scope so the host data passed to
    // vx_enqueue_write stays alive until the launch completion is waited.
    std::vector<uint8_t> texbuf;

    // Perform tile binning
    auto verts = cgl_to_gfx_vertices(drawcall.vertices);
    auto prims = cgl_to_gfx_primitives(drawcall.primitives);
    auto num_tiles = graphics::Binning(tilebuf, primbuf, verts, prims, dst_width, dst_height, drawcall.viewport.near, drawcall.viewport.far, tileLogSize);
    std::cout << "Binning allocated " << std::dec << num_tiles << " tiles with " << (primbuf.size() / sizeof(graphics::rast_prim_t)) << " total primitives." << std::endl;
    if (0 == num_tiles)
      continue;

    // allocate tile memory
    if (tile_buffer != nullptr) { vx_buffer_release(tile_buffer); tile_buffer = nullptr; }
    if (prim_buffer != nullptr) { vx_buffer_release(prim_buffer); prim_buffer = nullptr; }
    // With the FF unit present, tile_buffer / prim_buffer are bound to the raster
    // unit (via VX_DCR_RASTER_T/PBUF_ADDR) which bypasses the per-core MMU.
    // Software-routed, the shader walks the primitives through the LSU instead
    // and the tile buffer goes unused (the SW walk is binning-independent).
    uint32_t rast_flags = VX_MEM_READ | (ff_raster ? (uint32_t)VX_MEM_PHYS : 0u);
    RT_CHECK(vx_buffer_create(device, tilebuf.size(), rast_flags, &tile_buffer));
    RT_CHECK(vx_buffer_address(tile_buffer, &tilebuf_addr));
    RT_CHECK(vx_buffer_create(device, primbuf.size(), rast_flags, &prim_buffer));
    RT_CHECK(vx_buffer_address(prim_buffer, &primbuf_addr));
    std::cout << "tile_buffer=0x" << std::hex << tilebuf_addr << std::dec << std::endl;
    std::cout << "prim_buffer=0x" << std::hex << primbuf_addr << std::dec << std::endl;

    // upload tiles buffer
    std::cout << "upload tile buffer" << std::endl;
    RT_CHECK(vx_enqueue_write(queue, tile_buffer, 0, tilebuf.data(), tilebuf.size(), 0, nullptr, nullptr));

    // upload primitives buffer
    std::cout << "upload primitive buffer" << std::endl;
    RT_CHECK(vx_enqueue_write(queue, prim_buffer, 0, primbuf.data(), primbuf.size(), 0, nullptr, nullptr));

    uint32_t primbuf_stride = sizeof(graphics::rast_prim_t);

    // configure raster units
    RASTER_DCR_WRITE(VX_DCR_RASTER_TBUF_ADDR,   tilebuf_addr / 64); // block address
    RASTER_DCR_WRITE(VX_DCR_RASTER_TILE_COUNT,  num_tiles);
    RASTER_DCR_WRITE(VX_DCR_RASTER_PBUF_ADDR,   primbuf_addr / 64); // block address
    RASTER_DCR_WRITE(VX_DCR_RASTER_PBUF_STRIDE, primbuf_stride);
    RASTER_DCR_WRITE(VX_DCR_RASTER_SCISSOR_X, (dst_width << 16) | 0);
    RASTER_DCR_WRITE(VX_DCR_RASTER_SCISSOR_Y, (dst_height << 16) | 0);

    // configure om color buffer
    OM_DCR_WRITE(VX_DCR_OM_CBUF_ADDR,  cbuf_addr / 64); // block address
    OM_DCR_WRITE(VX_DCR_OM_CBUF_PITCH, cbuf_pitch);
    OM_DCR_WRITE(VX_DCR_OM_CBUF_WRITEMASK, states.color_writemask);

    if (states.depth_test || states.stencil_test) {
      // configure om depth buffer
      OM_DCR_WRITE(VX_DCR_OM_ZBUF_ADDR,  zbuf_addr / 64); // block address
      OM_DCR_WRITE(VX_DCR_OM_ZBUF_PITCH, zbuf_pitch);
    }

    uint32_t earlyz_safe = 0;
    if (states.depth_test) {
      // configure om depth states
      auto depth_func = toVXCompare(states.depth_func);
      OM_DCR_WRITE(VX_DCR_OM_DEPTH_FUNC, depth_func);
      OM_DCR_WRITE(VX_DCR_OM_DEPTH_WRITEMASK, states.depth_writemask);
      // P3 early-Z: safe to cull occluded fragments before shading when the
      // depth func is monotonic, no stencil test is in play (the FS emits the
      // interpolated plane depth, so early-Z == late-Z bit-for-bit), and
      // blending is off. Early-Z reads the depth buffer out of OM order, so it
      // can observe a nearer write that lands after this fragment's OM slot;
      // with replace-mode color the dropped fragment was overwritten anyway,
      // but with blending its color contribution is legitimate and lost.
      earlyz_safe = (!states.stencil_test
                  && !states.blend_enabled
                  && (depth_func == VX_OM_DEPTH_FUNC_LESS
                   || depth_func == VX_OM_DEPTH_FUNC_LEQUAL)) ? 1u : 0u;
    } else {
      OM_DCR_WRITE(VX_DCR_OM_DEPTH_FUNC, VX_OM_DEPTH_FUNC_ALWAYS);
      OM_DCR_WRITE(VX_DCR_OM_DEPTH_WRITEMASK, 0);
    }
    OM_DCR_WRITE(VX_DCR_OM_EARLYZ_SAFE, earlyz_safe);

    // Fragment-export aperture. The shader stores its fragments here and the OM
    // ingress decodes the offset back into (x, y, face) by bit-slicing, which is
    // why the pitch is padded to a power of two. The aperture is virtual, so the
    // padding costs address space and nothing else.
    //
    // The shader builds the SAME address from kernel_arg, so these two must agree
    // exactly -- derive both from one place and never hand-write either.
    uint32_t aperture_xbits = log2ceil(dst_width);
    uint32_t aperture_ybits = log2ceil(dst_height);
    uint32_t aperture_record_shift = 3;   // this FS exports colour AND depth
    OM_DCR_WRITE(VX_DCR_OM_APERTURE_XBITS,        aperture_xbits);
    OM_DCR_WRITE(VX_DCR_OM_APERTURE_YBITS,        aperture_ybits);
    OM_DCR_WRITE(VX_DCR_OM_APERTURE_RECORD_SHIFT, aperture_record_shift);
    OM_DCR_WRITE(VX_DCR_OM_APERTURE_DEPTH_ONLY,   0);
    kernel_arg.aperture_xbits        = aperture_xbits;
    kernel_arg.aperture_ybits        = aperture_ybits;
    kernel_arg.aperture_record_shift = aperture_record_shift;

    if (states.stencil_test) {
      // configure om stencil states
      auto stencil_func  = toVXCompare(states.stencil_func);
      auto stencil_zpass = toVXStencilOp(states.stencil_zpass);
      auto stencil_zfail = toVXStencilOp(states.stencil_zfail);
      auto stencil_fail  = toVXStencilOp(states.stencil_fail);
      OM_DCR_WRITE(VX_DCR_OM_STENCIL_FUNC, stencil_func);
      OM_DCR_WRITE(VX_DCR_OM_STENCIL_ZPASS, stencil_zpass);
      OM_DCR_WRITE(VX_DCR_OM_STENCIL_ZPASS, stencil_zfail);
      OM_DCR_WRITE(VX_DCR_OM_STENCIL_FAIL, stencil_fail);
      OM_DCR_WRITE(VX_DCR_OM_STENCIL_REF, states.stencil_ref);
      OM_DCR_WRITE(VX_DCR_OM_STENCIL_MASK, states.stencil_mask);
      OM_DCR_WRITE(VX_DCR_OM_STENCIL_WRITEMASK, states.stencil_writemask);
    } else {
      OM_DCR_WRITE(VX_DCR_OM_STENCIL_FUNC, VX_OM_DEPTH_FUNC_ALWAYS);
      OM_DCR_WRITE(VX_DCR_OM_STENCIL_ZPASS, VX_OM_STENCIL_OP_KEEP);
      OM_DCR_WRITE(VX_DCR_OM_STENCIL_ZPASS, VX_OM_STENCIL_OP_KEEP);
      OM_DCR_WRITE(VX_DCR_OM_STENCIL_FAIL, VX_OM_STENCIL_OP_KEEP);
      OM_DCR_WRITE(VX_DCR_OM_STENCIL_REF, 0);
      OM_DCR_WRITE(VX_DCR_OM_STENCIL_MASK, OM_STENCIL_MASK);
      OM_DCR_WRITE(VX_DCR_OM_STENCIL_WRITEMASK, 0);
    }

    if (states.blend_enabled) {
      // configure om blend states
      auto blend_src = toVXBlendFunc(states.blend_src);
      auto blend_dst = toVXBlendFunc(states.blend_dst);
      OM_DCR_WRITE(VX_DCR_OM_BLEND_MODE, (VX_OM_BLEND_MODE_ADD << 16)   // DST
                                         | (VX_OM_BLEND_MODE_ADD << 0));  // SRC
      OM_DCR_WRITE(VX_DCR_OM_BLEND_FUNC, (blend_dst << 24)            // DST_A
                                         | (blend_dst << 16)            // DST_RGB
                                         | (blend_src << 8)             // SRC_A
                                         | (blend_src << 0));           // SRC_RGB
    } else {
      OM_DCR_WRITE(VX_DCR_OM_BLEND_MODE, (VX_OM_BLEND_MODE_ADD << 16)   // DST
                                         | (VX_OM_BLEND_MODE_ADD << 0));  // SRC
      OM_DCR_WRITE(VX_DCR_OM_BLEND_FUNC, (VX_OM_BLEND_FUNC_ZERO << 24)  // DST_A
                                         | (VX_OM_BLEND_FUNC_ZERO << 16)  // DST_RGB
                                         | (VX_OM_BLEND_FUNC_ONE << 8)    // SRC_A
                                         | (VX_OM_BLEND_FUNC_ONE << 0));  // SRC_RGB
    }

    if (states.texture_enabled) {
      // configure texture states
      std::vector<uint32_t> mip_offsets;

      auto& texture = trace.textures.at(drawcall.texture_id);

      auto tex_bpp = Format::GetInfo(texture.format).BytePerPixel;
      auto tex_pitch = texture.width * tex_bpp;

      // generate mipmaps
      RT_CHECK(GenerateMipmaps(texbuf, mip_offsets, texture.pixels.data(), texture.format, texture.width, texture.height, tex_pitch));

      uint32_t tex_logwidth = log2ceil(texture.width);
      uint32_t tex_logheight = log2ceil(texture.height);

      int tex_format = toVXFormat(texture.format);

      int tex_filter = (states.texture_magfilter != CGLTrace::FILTER_NEAREST)
                    || (states.texture_magfilter != CGLTrace::FILTER_NEAREST);

      int tex_wrapU = (states.texture_addressU == CGLTrace::ADDRESS_WRAP);
      int tex_wrapV = (states.texture_addressU == CGLTrace::ADDRESS_WRAP);

      // allocate texture memory
      if (tex_buffer != nullptr) { vx_buffer_release(tex_buffer); tex_buffer = nullptr; }
      // With the FF unit present, tex_buffer is bound to it (VX_DCR_TEX_ADDR) and
      // bypasses the MMU; software-routed, the shader loads it through the LSU.
      uint32_t tex_flags = VX_MEM_READ | (ff_tex ? (uint32_t)VX_MEM_PHYS : 0u);
      RT_CHECK(vx_buffer_create(device, texbuf.size(), tex_flags, &tex_buffer));
      RT_CHECK(vx_buffer_address(tex_buffer, &texbuf_addr));
      std::cout << "tex_buffer=0x" << std::hex << texbuf_addr << std::dec << std::endl;

      // upload texture data
      std::cout << "upload texture buffer" << std::endl;
      RT_CHECK(vx_enqueue_write(queue, tex_buffer, 0, texbuf.data(), texbuf.size(), 0, nullptr, nullptr));

      // configure texture units
      TEX_DCR_WRITE(VX_DCR_TEX_STAGE,  0);
      TEX_DCR_WRITE(VX_DCR_TEX_LOGDIM, (tex_logheight << 16) | tex_logwidth);
      TEX_DCR_WRITE(VX_DCR_TEX_FORMAT, tex_format);
      TEX_DCR_WRITE(VX_DCR_TEX_WRAP,   (tex_wrapV << 16) | tex_wrapU);
      TEX_DCR_WRITE(VX_DCR_TEX_FILTER, tex_filter ? VX_TEX_FILTER_BILINEAR : VX_TEX_FILTER_POINT);
      TEX_DCR_WRITE(VX_DCR_TEX_ADDR,   texbuf_addr / 64); // block address
      for (uint32_t i = 0; i < mip_offsets.size(); ++i) {
        assert(i < VX_TEX_LOD_MAX);
        TEX_DCR_WRITE(VX_DCR_TEX_MIPOFF(i), mip_offsets.at(i));
      };

      if (!ff_tex) {
        gfx_sw::TexState& ts = kernel_arg.tex;
        ts = {};
        ts.base   = texbuf_addr;   // TexState takes the byte address, not the block
        ts.logdim = (tex_logheight << 16) | tex_logwidth;
        ts.format = tex_format;
        ts.filter = tex_filter ? VX_TEX_FILTER_BILINEAR : VX_TEX_FILTER_POINT;
        ts.wrap   = (tex_wrapV << 16) | tex_wrapU;
        for (uint32_t i = 0; i < mip_offsets.size() && i <= (uint32_t)VX_TEX_LOD_MAX; ++i)
          ts.mip_off[i] = mip_offsets.at(i);
      }
    }

    // Software-routing state: mirror the DCR configuration above into the
    // gfx_sw state structs so a software-routed stage merges/samples identically
    // to its FF unit. resolve_om_state() derives the enable flags and the expanded
    // write mask exactly as VX_om_core does.
    if (!ff_om) {
      gfx_sw::om_state_t& om = kernel_arg.om;
      om = {};
      om.cbuf_base       = cbuf_addr;
      om.cbuf_pitch      = cbuf_pitch;
      om.zbuf_base       = zbuf_addr;
      om.zbuf_pitch      = zbuf_pitch;
      om.cbuf_writemask4 = states.color_writemask;
      if (states.depth_test) {
        om.depth_func      = toVXCompare(states.depth_func);
        om.depth_writemask = states.depth_writemask;
      } else {
        om.depth_func      = VX_OM_DEPTH_FUNC_ALWAYS;
        om.depth_writemask = 0;
      }
      for (int f = 0; f < 2; ++f) {
        if (states.stencil_test) {
          om.stencil_func[f]      = toVXCompare(states.stencil_func);
          om.stencil_zpass[f]     = toVXStencilOp(states.stencil_zpass);
          om.stencil_zfail[f]     = toVXStencilOp(states.stencil_zfail);
          om.stencil_fail[f]      = toVXStencilOp(states.stencil_fail);
          om.stencil_ref[f]       = states.stencil_ref;
          om.stencil_mask[f]      = states.stencil_mask;
          om.stencil_writemask[f] = states.stencil_writemask;
        } else {
          om.stencil_func[f]      = VX_OM_DEPTH_FUNC_ALWAYS;
          om.stencil_zpass[f]     = VX_OM_STENCIL_OP_KEEP;
          om.stencil_zfail[f]     = VX_OM_STENCIL_OP_KEEP;
          om.stencil_fail[f]      = VX_OM_STENCIL_OP_KEEP;
          om.stencil_ref[f]       = 0;
          om.stencil_mask[f]      = OM_STENCIL_MASK;
          om.stencil_writemask[f] = 0;
        }
      }
      om.blend_mode_rgb = VX_OM_BLEND_MODE_ADD;
      om.blend_mode_a   = VX_OM_BLEND_MODE_ADD;
      if (states.blend_enabled) {
        auto blend_src = toVXBlendFunc(states.blend_src);
        auto blend_dst = toVXBlendFunc(states.blend_dst);
        om.blend_src_rgb = blend_src;
        om.blend_src_a   = blend_src;
        om.blend_dst_rgb = blend_dst;
        om.blend_dst_a   = blend_dst;
      } else {
        om.blend_src_rgb = VX_OM_BLEND_FUNC_ONE;
        om.blend_src_a   = VX_OM_BLEND_FUNC_ONE;
        om.blend_dst_rgb = VX_OM_BLEND_FUNC_ZERO;
        om.blend_dst_a   = VX_OM_BLEND_FUNC_ZERO;
      }
      om.blend_const = 0;
      om.logic_op    = VX_OM_LOGIC_OP_COPY;
      gfx_sw::resolve_om_state(om);
    }

    // prepare kernel argument
    std::cout << "prepare kernel argument" << std::endl;
    {
      kernel_arg.dst_width   = dst_width;
      kernel_arg.dst_height  = dst_height;
      // SW RASTER walks the screen itself: one thread per resident primitive.
      kernel_arg.num_prims    = primbuf.size() / sizeof(graphics::rast_prim_t);
      kernel_arg.tile_logsize = tileLogSize;
      kernel_arg.depth_enabled = states.depth_test;
      kernel_arg.color_enabled = states.color_enabled;
      kernel_arg.tex_enabled   = states.texture_enabled;
      kernel_arg.tex_modulate  = (states.texture_enabled && states.texture_envmode == CGLTrace::ENVMODE_MODULATE);
      kernel_arg.prim_addr     = primbuf_addr;
      if (kernel_arg.tex_modulate && !kernel_arg.color_enabled)
        kernel_arg.tex_modulate = false;
      if (kernel_arg.tex_enabled && kernel_arg.color_enabled && !kernel_arg.tex_modulate)
        kernel_arg.color_enabled = false;
    }

    // RASTER dispatch v2 (push): the raster engine launches the fragment shader
    // on-device — no host fragment grid. Stage the FS args in device memory and
    // program the fragment-dispatch descriptor (FS entry PC + args pointer) into
    // the RASTER DCR block; the work distributor launches one fragment warp per
    // covered-quad wave at frag_entry with mscratch = frag_param.
    if (ff_raster) {
      if (frag_arg_buffer == nullptr) {
        RT_CHECK(vx_buffer_create(device, sizeof(kernel_arg), VX_MEM_READ, &frag_arg_buffer));
        RT_CHECK(vx_buffer_address(frag_arg_buffer, &frag_arg_addr));
      }
      RT_CHECK(vx_enqueue_write(queue, frag_arg_buffer, 0, &kernel_arg, sizeof(kernel_arg), 0, nullptr, nullptr));
      uint64_t frag_entry = 0;
      RT_CHECK(vx_kernel_address(kernel, &frag_entry));
      RASTER_DCR_WRITE(VX_DCR_RASTER_FRAG_ENTRY_LO, (uint32_t)(frag_entry & 0xffffffff));
      RASTER_DCR_WRITE(VX_DCR_RASTER_FRAG_ENTRY_HI, (uint32_t)(frag_entry >> 32));
      RASTER_DCR_WRITE(VX_DCR_RASTER_FRAG_PARAM_LO, (uint32_t)(frag_arg_addr & 0xffffffff));
      RASTER_DCR_WRITE(VX_DCR_RASTER_FRAG_PARAM_HI, (uint32_t)(frag_arg_addr >> 32));
    }

    auto time_start = std::chrono::high_resolution_clock::now();

    // start device
    std::cout << "start device" << std::endl;
    vx_event_h launch_ev = nullptr;
    {
      // FF RASTER — grid-less kick: no host fragment grid (grid_dim=0 → the KMU
      // produces no host warps). The launch still pulses vortex_start, sets the
      // program image base (warp launch PC) and stages the args, while the armed
      // raster work distributor injects the fragment warps and sustains the
      // device run until it drains.
      //
      // SW RASTER — there is no work distributor, so this is an ordinary grid. One
      // lane is one pixel, so a quad needs four adjacent lanes: the grid gives each
      // resident primitive a group of VX_FRAG_QUAD_LANES threads, which walk it in
      // lockstep and split its quads' sub-pixels between them.
      uint32_t block_x = (uint32_t)(num_threads * num_warps);
      // grid_dim 0 is the grid-less kick, so an empty SW draw still needs one CTA:
      // with no raster armed there would be nothing to sustain the run.
      uint32_t sw_blocks = (kernel_arg.num_prims * VX_FRAG_QUAD_LANES + block_x - 1) / block_x;
      if (sw_blocks == 0) sw_blocks = 1;
      vx_launch_info_t li = {};
      li.struct_size  = sizeof(li);
      li.kernel       = kernel;
      li.args_host    = &kernel_arg;
      li.args_size    = sizeof(kernel_arg);
      li.ndim         = 1;
      li.grid_dim[0]  = ff_raster ? 0 : sw_blocks;
      li.block_dim[0] = block_x;
      RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
    }

    // wait for completion
    std::cout << "wait for completion" << std::endl;
    RT_CHECK(vx_event_wait_value(launch_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(launch_ev);

    auto time_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
    printf("Elapsed time: %lg ms\n", elapsed);

    if (d < trace.drawcalls.size()-1) {
      vx_device_dump_perf(device, stdout);
    }
  }

  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_begin).count();
  printf("Total elapsed time: %lg ms\n", elapsed);

  if (strcmp(output_file, "null") != 0) {
    std::cout << "save output image" << std::endl;
    std::vector<uint8_t> dst_pixels(cbuf_size);
    {
      vx_event_h read_ev = nullptr;
      RT_CHECK(vx_enqueue_read(queue, dst_pixels.data(), color_buffer, 0, cbuf_size, 0, nullptr, &read_ev));
      RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
      vx_event_release(read_ev);
    }
    auto bits = dst_pixels.data() + (dst_height-1) * cbuf_pitch;
    RT_CHECK(SaveImage(output_file, FORMAT_A8R8G8B8, bits, dst_width, dst_height, -cbuf_pitch));
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

  // A stage needs its ISA extension only when it is FF-routed. Assert the device
  // agrees with the build, so a mismatched pair fails loudly here instead of
  // trapping on an illegal instruction inside the shader.
  uint64_t isa_flags;
  RT_CHECK(vx_device_query(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  if (ff_raster && 0 == (isa_flags & VX_ISA_EXT_RASTER)) {
    std::cout << "RASTER ISA extension is needed!" << std::endl;
    cleanup();
    return -1;
  }
  if (ff_tex && 0 == (isa_flags & VX_ISA_EXT_TEX)) {
    std::cout << "TEX ISA extension is needed!" << std::endl;
    cleanup();
    return -1;
  }
  if (ff_om && 0 == (isa_flags & VX_ISA_EXT_OM)) {
    std::cout << "OM ISA extension is needed!" << std::endl;
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

  uint64_t total_drawcalls  = trace.drawcalls.size();
  uint64_t total_textures   = trace.textures.size();
  uint64_t total_vertices   = 0;
  uint64_t total_primitives = 0;
  bool depth_test    = false;
  bool stencil_test  = false;
  bool blend_enabled = false;
  for (auto& drawcall : trace.drawcalls) {
    if (drawcall.states.depth_test)
      depth_test = true;
    if (drawcall.states.stencil_test)
      stencil_test = true;
    if (drawcall.states.blend_enabled)
      blend_enabled = true;
    total_vertices += drawcall.vertices.size();
    total_primitives += drawcall.primitives.size();
  }
  std::cout << "CGL Trace: drawcalls=" << std::dec << total_drawcalls
            << ", vertices=" << total_vertices
            << ", primitives=" << total_primitives
            << ", textures=" << total_textures
            << ", depth=" << depth_test
            << ", stencil=" << stencil_test
            << ", blend=" << blend_enabled << std::endl;

  // load kernel module
  std::cout << "load kernel module" << std::endl;
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  zbuf_stride = 4;
  zbuf_pitch  = dst_width * zbuf_stride;
  zbuf_size   = dst_height * zbuf_pitch;

  cbuf_stride = 4;
  cbuf_pitch  = dst_width * cbuf_stride;
  cbuf_size   = dst_width * cbuf_pitch;

  // With the FF unit present, depth_buffer / color_buffer are bound to the OM
  // unit (via VX_DCR_OM_Z/CBUF_ADDR) and bypass the MMU; software-routed, the
  // shader read-modify-writes them through the LSU.
  uint32_t om_flags = VX_MEM_READ_WRITE | (ff_om ? (uint32_t)VX_MEM_PHYS : 0u);
  RT_CHECK(vx_buffer_create(device, zbuf_size, om_flags, &depth_buffer));
  RT_CHECK(vx_buffer_address(depth_buffer, &zbuf_addr));
  RT_CHECK(vx_buffer_create(device, cbuf_size, om_flags, &color_buffer));
  RT_CHECK(vx_buffer_address(color_buffer, &cbuf_addr));

  std::cout << "depth_buffer=0x" << std::hex << zbuf_addr << std::dec << std::endl;
  std::cout << "color_buffer=0x" << std::hex << cbuf_addr << std::dec << std::endl;

  // clear depth buffer
  std::cout << "clear depth buffer" << std::endl;
  {
    std::vector<uint32_t> staging_buf(zbuf_size / zbuf_stride, clear_depth);
    vx_event_h ev = nullptr;
    RT_CHECK(vx_enqueue_write(queue, depth_buffer, 0, staging_buf.data(), zbuf_size, 0, nullptr, &ev));
    RT_CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(ev);
  }

  // clear destination buffer
  std::cout << "clear destination buffer" << std::endl;
  {
    std::vector<uint32_t> staging_buf(cbuf_size / cbuf_stride, clear_color);
    vx_event_h ev = nullptr;
    RT_CHECK(vx_enqueue_write(queue, color_buffer, 0, staging_buf.data(), cbuf_size, 0, nullptr, &ev));
    RT_CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(ev);
  }

  std::cout << "routing: RASTER=" << (ff_raster ? "FF" : "SW")
            << " TEX="            << (ff_tex    ? "FF" : "SW")
            << " OM="             << (ff_om     ? "FF" : "SW") << std::endl;

  // run tests
  RT_CHECK(render(trace));

  // cleanup
  std::cout << "cleanup" << std::endl;
  cleanup();

  if (reference_file) {
    auto reference_file_s = resolve_path(reference_file, ASSETS_PATHS);
    auto errors = CompareImages(output_file, reference_file_s.c_str(), FORMAT_A8R8G8B8);
    if (0 == errors) {
      std::cout << "PASSED!" << std::endl;
    } else {
      std::cout << "FAILED! " << errors << " errors." << std::endl;
      return 1;  // non-zero exit on mismatch (error count truncates mod 256 as a code)
    }
  } else {
    // No reference image; run-without-crash passes.
    std::cout << "PASSED!" << std::endl;
  }

  return 0;
}
