// Texture regression test using the KMU launch API.
//
// Pinned-buffer contract:
//   src_buffer  → TEX HW (VX_DCR_TEX_ADDR) → VX_MEM_PHYS (identity-mapped)
//   dst_buffer  → kernel-only (LSU stores)  → not pinned
// Only src_buffer is read by the TEX fixed-function block. dst_buffer is
// written by kernel code through the per-core MMU, so it gets a regular
// minted VA under VM.
//
// Host loads a PNG, generates a mipmap chain, configures TEX stage 0
// DCRs, then launches a 2D grid where each thread samples one output
// pixel via vx_tex(stage, u, v, lod). Output is saved to a PNG and
// optionally diffed against a reference image.
// Per-pixel delta is precomputed on host and passed in kernel_arg.

#include <iostream>
#include <vector>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <cmath>
#include <assert.h>
#include <vortex2.h>
#include <VX_types.h>
#include <vx_gfx_abi.h>
#include "common.h"
#include <bitmanip.h>
#include <fstream>
#include <sstream>
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

#ifndef VX_DCR_TEX_MIPOFF
#define VX_DCR_TEX_MIPOFF(lod) (VX_DCR_TEX_MIPOFF_BASE + (lod))
#endif

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret) break;                                      \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
     cleanup();                                                 \
     exit(-1);                                                  \
   } while (false)

const char* kernel_file    = "kernel.vxbin";
const char* input_file     = "palette64.png";
const char* output_file    = "output.png";
const char* reference_file = nullptr;
int         wrap           = VX_TEX_WRAP_CLAMP;
int         filter         = VX_TEX_FILTER_POINT;
float       scale          = 1.0f;
// Sampled level, as a fixed-point lambda, overriding the one the source/dest
// ratio implies. A sampler's lod clamp is independent of that ratio, so a level
// past the end of the mip chain is reachable in a real draw and unreachable
// from the ratio alone -- there is no destination size that asks a texture for
// a level it does not have. Negative means "use the ratio".
float       lod_override   = -1.0f;
int         format         = VX_TEX_FORMAT_A8R8G8B8;
ePixelFormat eformat       = FORMAT_A8R8G8B8;
// -B: sample a span wider than the texture with a CLAMP_TO_BORDER wrap, so the
// outer margin lands where a border and a clamp disagree. The colour is one no
// texel of the source image carries, so finding it anywhere is proof the unit
// produced it rather than fetched it.
bool        border_mode    = false;
const uint32_t kBorderColor = 0xff7f00ff;

vx_device_h device      = nullptr;
vx_queue_h  queue       = nullptr;
vx_module_h module_     = nullptr;
vx_kernel_h kernel      = nullptr;
vx_buffer_h dst_buffer  = nullptr;
vx_buffer_h src_buffer  = nullptr;

static void show_usage() {
   std::cout << "Vortex Texture Test (v2 KMU)." << std::endl;
   std::cout << "Usage: [-k: kernel] [-i image] [-o image] [-r reference] [-s scale] [-l lod] [-w wrap] [-f format] [-g filter] [-B border] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "i:o:k:w:f:g:s:l:r:Bh?")) != -1) {
    switch (c) {
    case 'i': input_file = optarg; break;
    case 'o': output_file = optarg; break;
    case 'k': kernel_file = optarg; break;
    case 'w': wrap = atoi(optarg); break;
    case 'f': format = atoi(optarg);
      switch (format) {
      case VX_TEX_FORMAT_A8R8G8B8: eformat = FORMAT_A8R8G8B8; break;
      case VX_TEX_FORMAT_R5G6B5:   eformat = FORMAT_R5G6B5;   break;
      case VX_TEX_FORMAT_A1R5G5B5: eformat = FORMAT_A1R5G5B5; break;
      case VX_TEX_FORMAT_A4R4G4B4: eformat = FORMAT_A4R4G4B4; break;
      case VX_TEX_FORMAT_A8L8:     eformat = FORMAT_A8L8;     break;
      case VX_TEX_FORMAT_L8:       eformat = FORMAT_L8;       break;
      case VX_TEX_FORMAT_A8:       eformat = FORMAT_A8;       break;
      }
      break;
    case 'g': filter = atoi(optarg); break;
    case 's': scale = atof(optarg); break;
    case 'l': lod_override = atof(optarg); break;
    case 'r': reference_file = optarg; break;
    case 'B': border_mode = true; wrap = VX_TEX_WRAP_BORDER; break;
    case 'h': case '?': show_usage(); exit(0);
    default:  show_usage(); exit(-1);
    }
  }
  if (output_file && strcmp(output_file, "null") == 0 && reference_file) {
    std::cout << "Error: output file is missing for reference validation" << std::endl;
    exit(1);
  }
}

void cleanup() {
  if (device) {
    if (src_buffer) vx_buffer_release(src_buffer);
    if (dst_buffer) vx_buffer_release(dst_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

int main(int argc, char *argv[]) {
  uint64_t src_addr;
  uint64_t dst_addr;
  std::vector<uint8_t> src_pixels;
  std::vector<uint32_t> mip_offsets;
  uint32_t src_width  = 0;
  uint32_t src_height = 0;

  parse_args(argc, argv);

  // ---- load PNG + generate mipmap chain --------------------------------
  {
    std::vector<uint8_t> staging;
    auto input_file_s = resolve_path(input_file, ASSETS_PATHS);
    RT_CHECK(LoadImage(input_file_s.c_str(), eformat, staging, &src_width, &src_height));
    if (!ispow2(src_width) || !ispow2(src_height)) {
      std::cout << "Error: only power-of-two textures supported (got "
                << src_width << "x" << src_height << ")" << std::endl;
      return -1;
    }
    uint32_t src_bpp   = Format::GetInfo(eformat).BytePerPixel;
    uint32_t src_pitch = src_width * src_bpp;
    RT_CHECK(GenerateMipmaps(src_pixels, mip_offsets, staging.data(), eformat,
                             src_width, src_height, src_pitch));
  }

  uint32_t src_logwidth  = log2ceil(src_width);
  uint32_t src_logheight = log2ceil(src_height);
  uint32_t src_bufsize   = src_pixels.size();

  uint32_t dst_width   = (uint32_t)(src_width * scale);
  uint32_t dst_height  = (uint32_t)(src_height * scale);
  uint32_t dst_bpp     = 4;
  uint32_t dst_pitch   = dst_bpp * dst_width;
  uint32_t dst_bufsize = dst_pitch * dst_height;

  // ---- open device + sanity check --------------------------------------
  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t isa_flags;
  RT_CHECK(vx_device_query(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  if ((isa_flags & VX_ISA_EXT_TEX) == 0) {
    std::cout << "tex extension not enabled (build with -DEXT_TEX_ENABLE)" << std::endl;
    cleanup();
    return -1;
  }

  uint64_t num_threads, num_warps;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS,   &num_warps));

  std::cout << "src: " << src_width << "x" << src_height << " (" << src_bufsize << " bytes incl. mipmaps)" << std::endl;
  std::cout << "dst: " << dst_width << "x" << dst_height << " (" << dst_bufsize << " bytes)" << std::endl;

  // ---- load kernel module ----------------------------------------------
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, border_mode ? "kernel_main_border" : "main", &kernel));

  // ---- allocate device buffers -----------------------------------------
  // src_buffer is bound to the TEX unit (VX_DCR_TEX_ADDR) which
  // bypasses the per-core MMU — needs a physical address.
  RT_CHECK(vx_buffer_create(device, src_bufsize, VX_MEM_READ | VX_MEM_PHYS, &src_buffer));
  RT_CHECK(vx_buffer_address(src_buffer, &src_addr));
  // dst_buffer is kernel-written via the LSU only — no HW-block reader,
  // no need to pin (the per-core MMU translates the VA for the kernel).
  RT_CHECK(vx_buffer_create(device, dst_bufsize, VX_MEM_WRITE, &dst_buffer));
  RT_CHECK(vx_buffer_address(dst_buffer, &dst_addr));
  RT_CHECK(vx_enqueue_write(queue, src_buffer, 0, src_pixels.data(), src_bufsize, 0, nullptr, nullptr));

  // ---- pre-compute LOD + per-pixel delta -------------------------------
  // Match skybox kernel's lod selection: minification = max(width_ratio, height_ratio)
  // expressed as 16.16 fixed-point.
  uint64_t width_ratio_q16  = ((uint64_t)src_width  << 16) / dst_width;
  uint64_t height_ratio_q16 = ((uint64_t)src_height << 16) / dst_height;
  uint64_t minif_q16 = std::max(width_ratio_q16, height_ratio_q16);
  if (minif_q16 < (1ull << 16)) minif_q16 = 1ull << 16;
  // log2floor(minif_q16) - 16 = integer log2 of the minification ratio.
  int lod = 0;
  for (uint64_t v = minif_q16; v > (1ull << 16); v >>= 1) ++lod;
  if (lod > VX_TEX_LOD_MAX) lod = VX_TEX_LOD_MAX;
  uint32_t frac_q8 = (uint32_t)((minif_q16 - ((uint64_t)1 << (lod + 16))) >> (lod + 16 - 8));
  if (lod_override >= 0.0f) {
    float lam = lod_override > (float)VX_TEX_LOD_MAX ? (float)VX_TEX_LOD_MAX : lod_override;
    lod = (int)lam;
    frac_q8 = (uint32_t)((lam - (float)lod) * 256.0f) & 0xff;
  }

  uint32_t deltaX = ((uint32_t)1 << TEX_FXD_FRAC) / dst_width;
  uint32_t deltaY = ((uint32_t)1 << TEX_FXD_FRAC) / dst_height;

  // ---- configure TEX DCRs ----------------------------------------------
  RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_TEX_STAGE,  0, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_TEX_LOGDIM, (src_logheight << 16) | src_logwidth, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_TEX_FORMAT, format, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_TEX_WRAP,   (wrap << 16) | wrap, 0, nullptr, nullptr));
  // filter: 0=POINT, 1=BILINEAR, 2=two-level blend composed in the shader (two
  // samples and a lerp) over point taps, 3=the same asked of the texture unit,
  // 4=composed in the shader over bilinear taps, 5=the same asked of the unit.
  //
  // Each pair must produce the identical image: the unit blends the two levels
  // with the same weights over the same texels, so the only thing that differs
  // is who does it. That is why 3 and 5 have no reference of their own and are
  // checked against 2's and 4's.
  //
  // 4/5 are the combination a real trilinear sampler uses -- bilinear taps at
  // each of two levels -- and are the only modes where the per-level tap
  // addressing has to be right at both levels rather than only the lower one.
  const bool bilinear_taps = (filter == VX_TEX_FILTER_BILINEAR || filter == 4 || filter == 5);
  const bool unit_mip      = (filter == 3 || filter == 5);
  const bool shader_mip    = (filter == 2 || filter == 4);
  uint32_t filter_dcr = bilinear_taps ? VX_TEX_FILTER_BILINEAR : VX_TEX_FILTER_POINT;
  if (unit_mip) {
    filter_dcr |= VX_TEX_FILTER_MIP_LINEAR;
  }
  RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_TEX_FILTER, filter_dcr, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_TEX_ADDR,   src_addr / 64, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_TEX_BORDER, kBorderColor, 0, nullptr, nullptr));
  // Every entry, not only the levels the chain has: a sampler's lod clamp is
  // independent of the chain length, so the unit can be asked for a level past
  // the end, and an entry left unwritten sends it at an address nothing owns.
  // Levels past the last repeat it, which is the offset table the driver builds.
  const uint32_t last_lod = (uint32_t)mip_offsets.size() - 1;
  for (uint32_t i = 0; i <= (uint32_t)VX_TEX_LOD_MAX; ++i) {
    RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_TEX_MIPOFF(i),
                                  mip_offsets[i <= last_lod ? i : last_lod], 0, nullptr, nullptr));
  }

  // ---- pack kernel arg + launch ----------------------------------------
  kernel_arg_t kernel_arg = {};
  kernel_arg.dst_addr      = dst_addr;
  kernel_arg.dst_width     = dst_width;
  kernel_arg.dst_height    = dst_height;
  kernel_arg.dst_pitch     = dst_pitch;
  kernel_arg.dst_stride    = (uint8_t)dst_bpp;
  kernel_arg.filter           = (uint8_t)filter;
  kernel_arg.use_trilinear    = shader_mip ? 1 : 0;
  kernel_arg.deltaX        = deltaX;
  kernel_arg.deltaY        = deltaY;
  // Asking the unit for the blend means handing it the level and the weight in
  // one operand: an integer level with the weight in its low bits. The shader
  // path keeps them apart because it applies the weight itself.
  kernel_arg.lod           = unit_mip
                           ? (((uint32_t)lod << VX_TEX_LOD_FRAC_BITS) | (frac_q8 & 0xff))
                           : (uint32_t)lod;
  kernel_arg.frac          = frac_q8;
  // Twice the texture across the destination, centred: a quarter of the width
  // falls off each side.
  const int32_t kOne = (int32_t)1 << TEX_FXD_FRAC;
  kernel_arg.uv_delta = (int32_t)(deltaX * 2);
  kernel_arg.uv_bias  = (kernel_arg.uv_delta >> 1) - (kOne / 2);

  // 2D launch: gx ranges [0, dst_width), gy ranges [0, dst_height).
  // block_x fills one CTA: num_threads × num_warps, capped at dst_width.
  uint32_t block_x = std::min<uint32_t>((uint32_t)(num_threads * num_warps), dst_width);
  uint32_t block_y = 1;
  uint32_t grid_x  = (dst_width  + block_x - 1) / block_x;
  uint32_t grid_y  = (dst_height + block_y - 1) / block_y;
  uint32_t grid[2]  = { grid_x, grid_y };
  uint32_t block[2] = { block_x, block_y };
  std::cout << "launch grid=" << grid_x << "x" << grid_y
            << ", block=" << block_x << "x" << block_y
            << ", lod=" << lod << std::endl;

  auto t0 = std::chrono::high_resolution_clock::now();
  vx_event_h launch_ev = nullptr;
  {
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &kernel_arg;
    li.args_size    = sizeof(kernel_arg);
    li.ndim         = 2;
    li.grid_dim[0]  = grid[0];
    li.grid_dim[1]  = grid[1];
    li.block_dim[0] = block[0];
    li.block_dim[1] = block[1];
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }
  RT_CHECK(vx_event_wait_value(launch_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(launch_ev);
  auto t1 = std::chrono::high_resolution_clock::now();
  printf("Elapsed time: %.2f ms\n",
         std::chrono::duration<double, std::milli>(t1 - t0).count());

  // ---- save output PNG --------------------------------------------------
  if (output_file && strcmp(output_file, "null") != 0) {
    std::vector<uint8_t> dst_pixels(dst_bufsize);
    {
      vx_event_h read_ev = nullptr;
      RT_CHECK(vx_enqueue_read(queue, dst_pixels.data(), dst_buffer, 0, dst_bufsize, 0, nullptr, &read_ev));
      RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
      vx_event_release(read_ev);
    }
    RT_CHECK(SaveImage(output_file, FORMAT_A8R8G8B8, dst_pixels.data(),
                       dst_width, dst_height, dst_pitch));
  }

  // ---- border check -----------------------------------------------------
  // Taken where a border and a clamp disagree: outside the texture. A pixel
  // every one of whose taps left [0,1) must be exactly the border colour, and
  // the middle of the image must still be texture. The half-texel band either
  // side of the edge is skipped -- there a bilinear tap set straddles the
  // boundary and the answer is a blend, which this check does not model.
  int border_errors = 0;
  if (border_mode) {
    std::vector<uint32_t> dst(dst_bufsize / 4);
    {
      vx_event_h read_ev = nullptr;
      RT_CHECK(vx_enqueue_read(queue, dst.data(), dst_buffer, 0, dst_bufsize, 0, nullptr, &read_ev));
      RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
      vx_event_release(read_ev);
    }
    const int32_t one  = (int32_t)1 << TEX_FXD_FRAC;
    const int32_t half = one >> (src_logwidth + 1);   // one half-texel
    uint32_t outside = 0, inside = 0;
    for (uint32_t y = 0; y < dst_height && border_errors < 16; ++y) {
      for (uint32_t x = 0; x < dst_width && border_errors < 16; ++x) {
        int32_t u = kernel_arg.uv_bias + kernel_arg.uv_delta * (int32_t)x;
        int32_t v = kernel_arg.uv_bias + kernel_arg.uv_delta * (int32_t)y;
        bool all_out = (u < -half || u >= one + half) || (v < -half || v >= one + half);
        bool all_in  = (u >= half && u < one - half) && (v >= half && v < one - half);
        uint32_t got = dst[x + y * (dst_pitch / 4)];
        if (all_out) {
          ++outside;
          if (got != kBorderColor) {
            std::cout << "FAILED: outside the texture at (" << x << "," << y
                      << "): got 0x" << std::hex << got << ", expected the border 0x"
                      << kBorderColor << std::dec << std::endl;
            ++border_errors;
          }
        } else if (all_in) {
          ++inside;
          if (got == kBorderColor) {
            std::cout << "FAILED: inside the texture at (" << x << "," << y
                      << ") returned the border colour" << std::endl;
            ++border_errors;
          }
        }
      }
    }
    // A span that never left the texture, or never entered it, would pass the
    // two checks above vacuously.
    if (outside == 0 || inside == 0) {
      std::cout << "FAILED: the sampled span did not straddle the texture edge ("
                << outside << " outside, " << inside << " inside)" << std::endl;
      ++border_errors;
    } else {
      std::cout << "border: " << outside << " px outside took the border colour, "
                << inside << " px inside kept their texels" << std::endl;
    }
  }

  cleanup();

  if (border_errors != 0) {
    return 1;
  }

  // ---- compare to reference --------------------------------------------
  if (reference_file) {
    auto reference_file_s = resolve_path(reference_file, ASSETS_PATHS);
    auto errors = CompareImages(output_file, reference_file_s.c_str(), FORMAT_A8R8G8B8);
    if (0 == errors) {
      std::cout << "PASSED!" << std::endl;
    } else {
      std::cout << "FAILED: " << errors << " errors against reference" << std::endl;
      return 1;  // non-zero exit on mismatch (error count truncates mod 256 as a code)
    }
  } else {
    std::cout << "PASSED!" << std::endl;
  }

  return 0;
}
