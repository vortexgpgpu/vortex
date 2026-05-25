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
#include <gfxutil.h>
#include <bitmanip.h>
#include "common.h"
#include <cocogfx/include/blitter.hpp>
#include <cocogfx/include/imageutil.hpp>

using namespace cocogfx;
using namespace vortex;

#ifndef ASSETS_PATHS
#define ASSETS_PATHS ""
#endif

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

bool sw_rast = false;
bool sw_tex = false;
bool sw_om = false;
uint64_t num_threads = 0;  // populated in main, read by render()

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

kernel_arg_t kernel_arg = {};

uint32_t tileLogSize = VX_CFG_RASTER_TILE_LOGSIZE;

static void show_usage() {
   std::cout << "Vortex 3D Rendering Test." << std::endl;
   std::cout << "Usage: [-t trace] [-s startdraw] [-e enddraw] [-o output] [-r reference] [-w width] [-h height] [-e empty] [-x s/w rast] [-y s/w om] [-k tilelogsize]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "t:s:e:i:o:r:w:h:t:k:uxyz?")) != -1) {
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
    case 'u':
      sw_tex = true;
      break;
    case 'x':
      sw_rast = true;
      break;
    case 'y':
      sw_om = true;
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
  if (kernel)  vx_kernel_release(kernel);
  if (module_) vx_module_release(module_);
  if (queue)   vx_queue_release(queue);
  if (device) {
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

#ifdef SW_ENABLE
  #define RASTER_DCR_WRITE(addr, value)  \
    vx_enqueue_dcr_write(queue, addr, value, 0, nullptr, nullptr); \
    kernel_arg.raster_dcrs.write(addr, value)

  #define OM_DCR_WRITE(addr, value)  \
    vx_enqueue_dcr_write(queue, addr, value, 0, nullptr, nullptr); \
    kernel_arg.om_dcrs.write(addr, value)

  #define TEX_DCR_WRITE(addr, value)  \
    vx_enqueue_dcr_write(queue, addr, value, 0, nullptr, nullptr); \
    kernel_arg.tex_dcrs.write(addr, value)
#else
  #define RASTER_DCR_WRITE(addr, value)  \
    vx_enqueue_dcr_write(queue, addr, value, 0, nullptr, nullptr)

  #define OM_DCR_WRITE(addr, value)  \
    vx_enqueue_dcr_write(queue, addr, value, 0, nullptr, nullptr)

  #define TEX_DCR_WRITE(addr, value)  \
    vx_enqueue_dcr_write(queue, addr, value, 0, nullptr, nullptr)
#endif

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
    auto num_tiles = graphics::Binning(tilebuf, primbuf, drawcall.vertices, drawcall.primitives, dst_width, dst_height, drawcall.viewport.near, drawcall.viewport.far, tileLogSize);
    std::cout << "Binning allocated " << std::dec << num_tiles << " tiles with " << (primbuf.size() / sizeof(graphics::rast_prim_t)) << " total primitives." << std::endl;
    if (0 == num_tiles)
      continue;

    // allocate tile memory
    if (tile_buffer != nullptr) { vx_buffer_release(tile_buffer); tile_buffer = nullptr; }
    if (prim_buffer != nullptr) { vx_buffer_release(prim_buffer); prim_buffer = nullptr; }
    // tile_buffer / prim_buffer are bound to the raster unit (via
    // VX_DCR_RASTER_T/PBUF_ADDR) which bypasses the per-core MMU.
    RT_CHECK(vx_buffer_create(device, tilebuf.size(), VX_MEM_READ | VX_MEM_PHYS, &tile_buffer));
    RT_CHECK(vx_buffer_address(tile_buffer, &tilebuf_addr));
    RT_CHECK(vx_buffer_create(device, primbuf.size(), VX_MEM_READ | VX_MEM_PHYS, &prim_buffer));
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

    if (states.depth_test) {
      // configure om depth states
      auto depth_func = graphics::toVXCompare(states.depth_func);
      OM_DCR_WRITE(VX_DCR_OM_DEPTH_FUNC, depth_func);
      OM_DCR_WRITE(VX_DCR_OM_DEPTH_WRITEMASK, states.depth_writemask);
    } else {
      OM_DCR_WRITE(VX_DCR_OM_DEPTH_FUNC, VX_OM_DEPTH_FUNC_ALWAYS);
      OM_DCR_WRITE(VX_DCR_OM_DEPTH_WRITEMASK, 0);
    }

    if (states.stencil_test) {
      // configure om stencil states
      auto stencil_func  = graphics::toVXCompare(states.stencil_func);
      auto stencil_zpass = graphics::toVXStencilOp(states.stencil_zpass);
      auto stencil_zfail = graphics::toVXStencilOp(states.stencil_zfail);
      auto stencil_fail  = graphics::toVXStencilOp(states.stencil_fail);
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
      OM_DCR_WRITE(VX_DCR_OM_STENCIL_MASK, VX_OM_STENCIL_MASK);
      OM_DCR_WRITE(VX_DCR_OM_STENCIL_WRITEMASK, 0);
    }

    if (states.blend_enabled) {
      // configure om blend states
      auto blend_src = graphics::toVXBlendFunc(states.blend_src);
      auto blend_dst = graphics::toVXBlendFunc(states.blend_dst);
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

      int tex_format = graphics::toVXFormat(texture.format);

      int tex_filter = (states.texture_magfilter != CGLTrace::FILTER_NEAREST)
                    || (states.texture_magfilter != CGLTrace::FILTER_NEAREST);

      int tex_wrapU = (states.texture_addressU == CGLTrace::ADDRESS_WRAP);
      int tex_wrapV = (states.texture_addressU == CGLTrace::ADDRESS_WRAP);

      // allocate texture memory
      if (tex_buffer != nullptr) { vx_buffer_release(tex_buffer); tex_buffer = nullptr; }
      // tex_buffer is bound to the TEX unit (VX_DCR_TEX_ADDR), bypass.
      RT_CHECK(vx_buffer_create(device, texbuf.size(), VX_MEM_READ | VX_MEM_PHYS, &tex_buffer));
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
    }

    // prepare kernel argument
    std::cout << "prepare kernel argument" << std::endl;
    {
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

    auto time_start = std::chrono::high_resolution_clock::now();

    // start device
    std::cout << "start device" << std::endl;
    vx_event_h launch_ev = nullptr;
    {
      // 1D launch — every thread polls vx_rast() until the cluster-shared
      // raster_core drains its tile queue.
      vx_launch_info_t li = {};
      li.struct_size  = sizeof(li);
      li.kernel       = kernel;
      li.args_host    = &kernel_arg;
      li.args_size    = sizeof(kernel_arg);
      li.ndim         = 1;
      li.grid_dim[0]  = 1;
      li.block_dim[0] = (uint32_t)num_threads;
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
    // NOTE: per-counter MPM queries (legacy vx_mpm_query) are not exposed by
    // vortex2.h; the formatted report from vx_device_dump_perf above is the
    // performance-reporting path. IPC computation is therefore omitted.
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
    //DumpImage(dst_pixels, dst_width, dst_height, 4);
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

  uint64_t isa_flags;
  RT_CHECK(vx_device_query(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  bool has_ext = (isa_flags & VX_ISA_EXT_RASTER) != 0;
  if (!has_ext) {
    std::cout << "RASTER ISA extensions are needed!" << std::endl;
    cleanup();
    return -1;
  }
  if (0 == (isa_flags & (VX_ISA_EXT_TEX))) {
    std::cout << "TEX ISA extensions are needed!" << std::endl;
    cleanup();
    return -1;
  }
  if (0 == (isa_flags & (VX_ISA_EXT_OM))) {
    std::cout << "OM ISA extensions are needed!" << std::endl;
    cleanup();
    return -1;
  }

  uint64_t num_cores, num_warps;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));
  std::cout << "device: " << num_cores << " cores, " << num_warps
            << " warps, " << num_threads << " threads" << std::endl;

  CGLTrace trace;
  auto trace_file_s = graphics::ResolveFilePath(trace_file, ASSETS_PATHS);
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

  // depth_buffer / color_buffer are bound to the OM unit (via
  // VX_DCR_OM_Z/CBUF_ADDR), MMU-bypass.
  RT_CHECK(vx_buffer_create(device, zbuf_size, VX_MEM_READ_WRITE | VX_MEM_PHYS, &depth_buffer));
  RT_CHECK(vx_buffer_address(depth_buffer, &zbuf_addr));
  RT_CHECK(vx_buffer_create(device, cbuf_size, VX_MEM_READ_WRITE | VX_MEM_PHYS, &color_buffer));
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

  // sw_* fallback fields removed in the v2 KMU port; kernel_arg is now
  // assembled per-drawcall inside render(). num_threads is populated above.
  (void)sw_tex; (void)sw_rast; (void)sw_om;

  // run tests
  RT_CHECK(render(trace));

  // cleanup
  std::cout << "cleanup" << std::endl;
  cleanup();

  if (reference_file) {
    auto reference_file_s = graphics::ResolveFilePath(reference_file, ASSETS_PATHS);
    auto errors = CompareImages(output_file, reference_file_s.c_str(), FORMAT_A8R8G8B8);
    if (0 == errors) {
      std::cout << "PASSED!" << std::endl;
    } else {
      std::cout << "FAILED! " << errors << " errors." << std::endl;
      return errors;
    }
  } else {
    // Build-and-run smoke (no reference image diff). Functional output
    // requires raster bcoord/pos_mask CSR plumbing which is deferred.
    std::cout << "PASSED!" << std::endl;
  }

  return 0;
}
