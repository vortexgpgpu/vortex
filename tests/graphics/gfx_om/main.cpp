// Output-merger regression test.
//
// Pinned-buffer contract:
//   depth_buffer  → OM HW (VX_DCR_OM_ZBUF_ADDR)  → VX_MEM_PHYS (identity-mapped)
//   color_buffer  → OM HW (VX_DCR_OM_CBUF_ADDR)  → VX_MEM_PHYS (identity-mapped)
// Both are read+written by the OM fixed-function block, which bypasses
// the per-core MMU; under VM the buffers must be identity-mapped.

#include <iostream>
#include <vector>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <vortex.h>
#include <VX_types.h>
#include <vx_gfx_abi.h>
#include "common.h"
#include <fstream>
#include <sstream>
#include <cocogfx/include/fixed.hpp>
#include <cocogfx/include/imageutil.hpp>

using namespace cocogfx;

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

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret) break;                                      \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
     cleanup();                                                 \
     exit(-1);                                                  \
   } while (false)

const char* kernel_file    = "kernel.vxbin";
const char* output_file    = "output.png";
const char* reference_file = nullptr;

uint32_t color = 0xffffffff;
uint32_t depth = TFixed<24>(0.5f).data();

bool blend_enable = false;
bool depth_enable = false;
bool stencil_enable = false;   // -s : stencil test + ops on a striped stencil
bool backface     = false;
bool sw_path      = false;   // -S : merge via gfx_sw::om_fragment
bool mrt_enable   = false;   // -m : also render to a second colour attachment
bool float_depth  = false;   // -z : D32F depth target (ordered keys, face in the address)
bool stencil_word = false;   // -Z : D32F_S8, the stencil in the next word of an 8-byte texel
bool depth16      = false;   // -y : D16, a 2-byte texel, checked texel by texel
uint32_t mrt_rt   = 0;       // -n : which attachment that is (default 1)
// -q: count the draw's passing fragments with the OM occlusion counter and check
// the count against the test outcome of every pixel. -Q counts a draw that
// writes nothing, as an occlusion-culling proxy does: depth tested, neither
// depth nor colour written.
bool occ_query    = false;
bool occ_proxy    = false;
// -F fmt: the colour attachment's VX_OM_COLOR_FORMAT_*, checked texel by texel
// against the blend of the kernel's colour with the packed clear.
uint32_t cbuf_format = VX_OM_COLOR_FORMAT_A8R8G8B8;
const uint32_t kFormatClear = 0x80c04020;

// Second colour attachment, cleared to a value no fragment produces so that a
// channel the writemask should have protected is visible if it is overwritten.
const uint32_t kRt1Clear = 0x11223344;
const uint32_t kRt1Mask  = 0x0000ff00;   // CBUF_WRITEMASK 0b0010: green only

uint32_t clear_color = 0x00000000;
uint32_t clear_depth = TFixed<24>(0.5f).data();

// Stencil state under -s: EQUAL to 1, with a different op for each outcome so
// the result names the path a fragment took. Without -s the test is off.
struct stencil_cfg_t { uint32_t func, ref, writemask, fail, zfail, zpass; };
static stencil_cfg_t kStencil = { VX_OM_DEPTH_FUNC_ALWAYS, 0, 0,
                                  VX_OM_STENCIL_OP_KEEP, VX_OM_STENCIL_OP_KEEP,
                                  VX_OM_STENCIL_OP_KEEP };
// The back face compares against 0 instead, so -f admits the other rows: a
// merger that misread the face would fail every pixel.
static const uint32_t kStencilBackRef = 0;

static uint32_t float_bits(float f) {
  uint32_t b;
  memcpy(&b, &f, 4);
  return b;
}

// Under -z the fragment depth is -1.0, against stored depths on both sides of
// it, either zero, and outside [0, 1]: an unsigned compare of the raw float bits
// orders every negative value after every positive one and gets most of these
// wrong, which is what the ordered key exists to prevent.
static const float kFloatDepths[6] = { -2.0f, -0.0f, 0.0f, 0.5f, 1.0f, 3.0f };
const float kFloatFragDepth = -1.0f;

// The depth/stencil word a pixel starts with: a depth checkerboard of 0.0 and
// 0.99, and under -s stencil 1 on odd rows; under -z the D32F keys of
// kFloatDepths in a 6-cycle. Under -Z the stencil is a word of its own
// (initial_stencil).
static uint32_t initial_depthstencil(uint32_t x, uint32_t y) {
  uint32_t s = stencil_enable ? (y & 1) : 0;
  if (float_depth) {
    return gfx_sw::om_float_depth_key(float_bits(kFloatDepths[(x + 3 * y) % 6]));
  }
  uint32_t z = ((x & 1) == (y & 1)) ? TFixed<24>(0.0f).data() : TFixed<24>(0.99f).data();
  return (s << VX_OM_DEPTH_BITS) | z;
}

// Under -y a D16 checkerboard of 0 and 0xfd70 (0.99), and the fragment's depth
// as the 24-bit key the merger compares: a D16 value with its high byte
// replicated, which is what a stored texel widens to.
static uint16_t initial_d16(uint32_t x, uint32_t y) {
  return ((x & 1) == (y & 1)) ? 0x0000 : 0xfd70;
}
const uint16_t kD16Frag = 0x8000;
static uint32_t d16_key(uint16_t d) {
  return ((uint32_t)d << 8) | (d >> 8);
}

static uint32_t initial_stencil(uint32_t y) {
  return stencil_enable ? (y & 1) : 0;
}

uint32_t dst_width  = 128;
uint32_t dst_height = 128;

uint32_t cbuf_pitch, cbuf_size;
uint32_t zbuf_pitch, zbuf_size;
uint64_t cbuf_addr, zbuf_addr;

vx_device_h device       = nullptr;
vx_queue_h  queue        = nullptr;
vx_module_h module_      = nullptr;
vx_kernel_h kernel       = nullptr;
vx_buffer_h depth_buffer = nullptr;
vx_buffer_h color_buffer = nullptr;
vx_buffer_h color_buffer1 = nullptr;
uint64_t    cbuf1_addr = 0;

static void show_usage() {
   std::cout << "Vortex Render Output Test (v2 KMU)." << std::endl;
   std::cout << "Usage: [-c color] [-d depth] [-s stencil] [-z float depth] [-Z float depth + stencil] [-y D16 depth] [-b blend] [-m mrt] [-n mrt attachment] [-f face] [-q occlusion count] [-Q occlusion count, no writes] [-k kernel] [-o image] [-r reference] [-w width] [-h height] [-F colour format]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "o:r:k:w:h:c:n:F:bdfmszZySqQ?")) != -1) {
    switch (c) {
    case 'o': output_file = optarg; break;
    case 'r': reference_file = optarg; break;
    case 'k': kernel_file = optarg; break;
    case 'w': dst_width = std::atoi(optarg); break;
    case 'h': dst_height = std::atoi(optarg); break;
    case 'f': backface = true; break;
    case 'c': color = std::atoi(optarg); break;
    case 'd': depth_enable = true; break;
    case 's': stencil_enable = depth_enable = true; break;
    case 'b': blend_enable = true; break;
    case 'm': mrt_enable = true; break;
    case 'z': float_depth = true; break;
    case 'Z': float_depth = stencil_word = true; break;
    case 'y': depth16 = depth_enable = true; break;
    case 'n': mrt_rt = std::atoi(optarg); break;
    case 'S': sw_path = true; break;
    case 'q': occ_query = true; break;
    case 'Q': occ_query = occ_proxy = depth_enable = true; break;
    case 'F': cbuf_format = (uint32_t)std::atoi(optarg); break;
    case '?': show_usage(); exit(0);
    default:  show_usage(); exit(-1);
    }
  }
  if (output_file && strcmp(output_file, "null") == 0 && reference_file) {
    std::cout << "Error: output file is missing for reference validation" << std::endl;
    exit(1);
  }
  // The attachments share one depth buffer, so a fragment covering several of
  // them runs the depth test once per attachment. That only agrees with itself
  // if the test writes nothing back.
  if (mrt_enable && depth_enable) {
    std::cout << "Error: -m and -d cannot be combined: several colour attachments "
                 "share the depth buffer, which the depth write would then change "
                 "between one attachment and the next" << std::endl;
    exit(1);
  }
  if (stencil_enable && (mrt_enable || blend_enable)) {
    std::cout << "Error: -s checks every pixel against the unblended single-attachment "
                 "colour, so it takes neither -m nor -b" << std::endl;
    exit(1);
  }
  if (float_depth && blend_enable) {
    std::cout << "Error: -z checks every pixel against the unblended colour, so it "
                 "takes no -b" << std::endl;
    exit(1);
  }
  if (float_depth && stencil_enable && !stencil_word) {
    std::cout << "Error: a D32F target has no stencil; -Z is D32F_S8" << std::endl;
    exit(1);
  }
  if (depth16 && (float_depth || stencil_enable || mrt_enable || blend_enable || sw_path ||
                  reference_file || cbuf_format != VX_OM_COLOR_FORMAT_A8R8G8B8)) {
    std::cout << "Error: -y checks a D16 target's every texel against the unblended colour, "
                 "so it takes none of -z, -Z, -s, -m, -b, -S, -r or -F" << std::endl;
    exit(1);
  }
  if (float_depth && depth_enable && reference_file) {
    std::cout << "Error: -z -d checks the buffers itself and has no reference image" << std::endl;
    exit(1);
  }
  if (mrt_rt != 0 && !mrt_enable) {
    std::cout << "Error: -n names the attachment -m renders to" << std::endl;
    exit(1);
  }
  if (mrt_rt == 0) {
    mrt_rt = 1;
  }
  if (mrt_rt >= VX_OM_MAX_RT) {
    std::cout << "Error: -n names attachment 1.." << (VX_OM_MAX_RT - 1) << std::endl;
    exit(1);
  }
  if (cbuf_format != VX_OM_COLOR_FORMAT_A8R8G8B8 &&
      (!gfx_sw::om_ff_color_format(cbuf_format) || reference_file || mrt_enable || depth_enable)) {
    std::cout << "Error: -F takes a format the merger stores, checks the buffer itself, and "
                 "runs without -r, -m and depth" << std::endl;
    exit(1);
  }
  if (mrt_enable && sw_path) {
    std::cout << "Error: -m is a fixed-function path; -S merges in software" << std::endl;
    exit(1);
  }
  if (occ_query && sw_path) {
    std::cout << "Error: -q reads the fixed-function merger's counter; -S merges in software" << std::endl;
    exit(1);
  }
  if (occ_proxy && (mrt_enable || blend_enable || stencil_enable || float_depth || depth16 ||
                    reference_file || cbuf_format != VX_OM_COLOR_FORMAT_A8R8G8B8)) {
    std::cout << "Error: -Q checks that a D24S8 draw writing nothing leaves both buffers as "
                 "they were, so it takes none of -m, -b, -s, -z, -Z, -y, -r or -F" << std::endl;
    exit(1);
  }
}

// Whether the fragment at (x, y) passes the depth and stencil tests, from the
// state the buffers start with.
static bool ref_frag_passes(uint32_t x, uint32_t y, uint32_t frag_z) {
  if (depth16) {
    return kD16Frag < initial_d16(x, y);
  }
  const uint32_t init = initial_depthstencil(x, y);
  const uint32_t s0 = stencil_word ? initial_stencil(y)
                    : float_depth  ? 0 : (init >> VX_OM_DEPTH_BITS);
  const bool spass = !stencil_enable || s0 == (backface ? kStencilBackRef : kStencil.ref);
  const bool dpass = !depth_enable
                  || (float_depth ? (kFloatFragDepth < kFloatDepths[(x + 3 * y) % 6])
                                  : (frag_z < (init & OM_DEPTH_MASK)));
  return spass && dpass;
}

void cleanup() {
  if (device) {
    if (depth_buffer) vx_buffer_release(depth_buffer);
    if (color_buffer) vx_buffer_release(color_buffer);
    if (color_buffer1) vx_buffer_release(color_buffer1);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

// One read per cluster, each answering with the sum of its OM cores' counters.
static uint32_t read_occ_count(uint32_t num_clusters) {
  uint32_t sum = 0;
  for (uint32_t c = 0; c < num_clusters; ++c) {
    uint32_t v = 0;
    RT_CHECK(vx_dcr_read(device, VX_DCR_OM_OCC_COUNT, c, &v));
    sum += v;
  }
  return sum;
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  if (stencil_enable) {
    kStencil = { VX_OM_DEPTH_FUNC_EQUAL, 1, OM_STENCIL_MASK, VX_OM_STENCIL_OP_INCR,
                 VX_OM_STENCIL_OP_ZERO, VX_OM_STENCIL_OP_INVERT };
  }

  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t isa_flags;
  RT_CHECK(vx_device_query(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  if ((isa_flags & VX_ISA_EXT_OM) == 0) {
    std::cout << "om extension not enabled (build with -DEXT_OM_ENABLE)" << std::endl;
    cleanup(); return -1;
  }

  uint64_t num_threads, num_warps, num_clusters;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS,  &num_threads));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS,    &num_warps));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CLUSTERS, &num_clusters));

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, float_depth ? (mrt_enable ? "kernel_main_mrt_float"
                                                                   : "kernel_main_float")
                                                     : (mrt_enable ? "kernel_main_mrt" : "main"),
                                &kernel));

  // ---- allocate cbuf + zbuf --------------------------------------------
  // The merger addresses depth a word at a time, so a D16 row starts on one.
  zbuf_pitch = depth16 ? ((dst_width * 2 + 3) & ~3u) : dst_width * (stencil_word ? 8 : 4);
  zbuf_size  = dst_height * zbuf_pitch;
  const uint32_t cbuf_bpp = gfx_sw::om_color_bpp(cbuf_format);
  cbuf_pitch = dst_width * cbuf_bpp;
  cbuf_size  = dst_height * cbuf_pitch;
  if (cbuf_format != VX_OM_COLOR_FORMAT_A8R8G8B8) {
    clear_color = kFormatClear;
  }

  // HW path: depth_buffer / color_buffer are bound to the OM unit (via
  // VX_DCR_OM_ZBUF_ADDR / VX_DCR_OM_CBUF_ADDR), which reads/writes them through
  // the OM/ocache path bypassing the per-core MMU — both need physical (pinned)
  // addresses. SW path (-S): the kernel does the merge through the LSU/dcache,
  // so allocate them as normal kernel-accessed buffers.
  uint32_t om_flags = VX_MEM_READ_WRITE | (sw_path ? 0u : (uint32_t)VX_MEM_PHYS);
  RT_CHECK(vx_buffer_create(device, zbuf_size, om_flags, &depth_buffer));
  RT_CHECK(vx_buffer_address(depth_buffer, &zbuf_addr));
  RT_CHECK(vx_buffer_create(device, cbuf_size, om_flags, &color_buffer));
  RT_CHECK(vx_buffer_address(color_buffer, &cbuf_addr));
  if (mrt_enable) {
    RT_CHECK(vx_buffer_create(device, cbuf_size, om_flags, &color_buffer1));
    RT_CHECK(vx_buffer_address(color_buffer1, &cbuf1_addr));
  }

  // depth checkerboard prefill; under -s, stencil 1 on odd rows.
  if (depth16) {
    std::vector<uint8_t> staging(zbuf_size, 0);
    for (uint32_t y = 0; y < dst_height; ++y) {
      for (uint32_t x = 0; x < dst_width; ++x) {
        const uint16_t d = initial_d16(x, y);
        memcpy(&staging[y * zbuf_pitch + x * 2], &d, 2);
      }
    }
    vx_event_h ev = nullptr;
    RT_CHECK(vx_enqueue_write(queue, depth_buffer, 0, staging.data(), zbuf_size, 0, nullptr, &ev));
    RT_CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(ev);
  } else {
    const uint32_t zw = stencil_word ? 2 : 1;
    std::vector<uint32_t> staging(dst_width * dst_height * zw);
    for (uint32_t y = 0; y < dst_height; ++y) {
      for (uint32_t x = 0; x < dst_width; ++x) {
        staging[(x + y * dst_width) * zw] = initial_depthstencil(x, y);
        if (stencil_word) {
          staging[(x + y * dst_width) * zw + 1] = initial_stencil(y);
        }
      }
    }
    vx_event_h ev = nullptr;
    RT_CHECK(vx_enqueue_write(queue, depth_buffer, 0, staging.data(), zbuf_size, 0, nullptr, &ev));
    RT_CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(ev);
  }
  // clear cbuf to clear_color, as the attachment stores it.
  {
    std::vector<uint8_t> staging(cbuf_size);
    const uint32_t texel = gfx_sw::om_encode_color(cbuf_format, clear_color);
    for (uint32_t i = 0; i < cbuf_size; i += cbuf_bpp) {
      memcpy(&staging[i], &texel, cbuf_bpp);
    }
    vx_event_h ev = nullptr;
    RT_CHECK(vx_enqueue_write(queue, color_buffer, 0, staging.data(), cbuf_size, 0, nullptr, &ev));
    RT_CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(ev);
  }
  if (mrt_enable) {
    std::vector<uint32_t> staging(cbuf_size / 4, kRt1Clear);
    vx_event_h ev = nullptr;
    RT_CHECK(vx_enqueue_write(queue, color_buffer1, 0, staging.data(), cbuf_size, 0, nullptr, &ev));
    RT_CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(ev);
  }

  // ---- configure OM DCRs -----------------------------------------------
  RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_CBUF_ADDR,      cbuf_addr / 64, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_CBUF_PITCH,     cbuf_pitch, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_CBUF_WRITEMASK, occ_proxy ? 0x0 : 0xf, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_CBUF_FORMAT,    cbuf_format, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_ZBUF_ADDR,      zbuf_addr / 64, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_ZBUF_PITCH,     zbuf_pitch, 0, nullptr, nullptr));

  // Fragment-export aperture. The kernel stores its fragments here; the OM
  // ingress bit-slices the offset back into (x, y, face), which is why the pitch
  // is padded to a power of two. Derived once and passed to the kernel arg too --
  // if the two ever disagree the ingress silently misreads every record.
  auto ceil_log2 = [](uint32_t v) { uint32_t b = 0; while ((1u << b) < v) ++b; return b; };
  uint32_t ap_xbits = ceil_log2(dst_width);
  uint32_t ap_ybits = ceil_log2(dst_height);
  uint32_t ap_shift = 3;   // this kernel exports colour AND depth
  RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_APERTURE_XBITS,        ap_xbits, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_APERTURE_YBITS,        ap_ybits, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_APERTURE_RECORD_SHIFT, ap_shift, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_APERTURE_DEPTH_ONLY,   0, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_DEPTH_FORMAT,
                                stencil_word ? VX_OM_DEPTH_FORMAT_D32FS8
                                : float_depth ? VX_OM_DEPTH_FORMAT_D32F
                                : depth16 ? VX_OM_DEPTH_FORMAT_D16 : VX_OM_DEPTH_FORMAT_D24S8,
                                0, nullptr, nullptr));

  if (depth_enable) {
    RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_DEPTH_FUNC,      VX_OM_DEPTH_FUNC_LESS, 0, nullptr, nullptr));
    RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_DEPTH_WRITEMASK, occ_proxy ? 0 : 1, 0, nullptr, nullptr));
  } else {
    RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_DEPTH_FUNC,      VX_OM_DEPTH_FUNC_ALWAYS, 0, nullptr, nullptr));
    RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_DEPTH_WRITEMASK, 0, 0, nullptr, nullptr));
  }

  // Both faces take the same stencil state.
  auto both = [](uint32_t v) { return (v << 16) | v; };
  RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_STENCIL_FUNC,  both(kStencil.func), 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_STENCIL_ZPASS, both(kStencil.zpass), 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_STENCIL_ZFAIL, both(kStencil.zfail), 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_STENCIL_FAIL,  both(kStencil.fail), 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_STENCIL_REF,
                                (kStencilBackRef << 16) | kStencil.ref, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_STENCIL_MASK,  both(OM_STENCIL_MASK), 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_STENCIL_WRITEMASK, both(kStencil.writemask), 0, nullptr, nullptr));

  RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_BLEND_MODE,
                        (VX_OM_BLEND_MODE_ADD << 16) | VX_OM_BLEND_MODE_ADD, 0, nullptr, nullptr));
  if (blend_enable) {
    RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_BLEND_FUNC,
                            (VX_OM_BLEND_FUNC_ONE_MINUS_SRC_A << 24)   // dst_a
                          | (VX_OM_BLEND_FUNC_ONE_MINUS_SRC_A << 16)   // dst_rgb
                          | (VX_OM_BLEND_FUNC_ONE             <<  8)   // src_a
                          |  VX_OM_BLEND_FUNC_ONE, 0, nullptr, nullptr));   // src_rgb
  } else {
    RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_BLEND_FUNC,
                            (VX_OM_BLEND_FUNC_ZERO << 24)
                          | (VX_OM_BLEND_FUNC_ZERO << 16)
                          | (VX_OM_BLEND_FUNC_ONE  <<  8)
                          |  VX_OM_BLEND_FUNC_ONE, 0, nullptr, nullptr));
  }
  RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_BLEND_CONST, 0, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_LOGIC_OP,    VX_OM_LOGIC_OP_COPY, 0, nullptr, nullptr));

  // The second colour attachment (-n). Everything above landed on attachment 0, which the
  // select register names by default. Attachment 1 differs from it in exactly
  // the two things that are per-attachment -- the buffer it writes and the
  // channels it writes there -- and, under -b, in whether it blends at all.
  if (mrt_enable) {
    RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_RT_SELECT,      mrt_rt, 0, nullptr, nullptr));
    RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_CBUF_ADDR,      cbuf1_addr / 64, 0, nullptr, nullptr));
    RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_CBUF_PITCH,     cbuf_pitch, 0, nullptr, nullptr));
    RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_CBUF_WRITEMASK, 0x2, 0, nullptr, nullptr));
    RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_BLEND_MODE,
                          (VX_OM_BLEND_MODE_ADD << 16) | VX_OM_BLEND_MODE_ADD, 0, nullptr, nullptr));
    RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_BLEND_FUNC,
                            (VX_OM_BLEND_FUNC_ZERO << 24)
                          | (VX_OM_BLEND_FUNC_ZERO << 16)
                          | (VX_OM_BLEND_FUNC_ONE  <<  8)
                          |  VX_OM_BLEND_FUNC_ONE, 0, nullptr, nullptr));
    RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_BLEND_CONST, 0, 0, nullptr, nullptr));
    RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_LOGIC_OP,    VX_OM_LOGIC_OP_COPY, 0, nullptr, nullptr));
    // Leave the select where the next caller expects to find it.
    RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_RT_SELECT,   0, 0, nullptr, nullptr));
  }

  // The counter never clears, so a query is the difference of two reads. The
  // first one waits for the enable, which the queue may still hold.
  uint32_t occ_before = 0;
  if (occ_query) {
    vx_event_h ev = nullptr;
    RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_OCC_CTRL, 1, 0, nullptr, &ev));
    RT_CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(ev);
    occ_before = read_occ_count((uint32_t)num_clusters);
  }

  // ---- pack kernel arg + launch ----------------------------------------
  kernel_arg_t kernel_arg = {};
  kernel_arg.dst_width    = dst_width;
  kernel_arg.dst_height   = dst_height;
  kernel_arg.color        = color;
  kernel_arg.depth        = depth;
  kernel_arg.backface     = backface ? 1 : 0;
  kernel_arg.rt2          = mrt_rt;
  kernel_arg.blend_enable = blend_enable ? 1 : 0;
  kernel_arg.aperture_xbits        = ap_xbits;
  kernel_arg.aperture_ybits        = ap_ybits;
  kernel_arg.aperture_record_shift = ap_shift;
  if (float_depth) {
    kernel_arg.depth = gfx_sw::om_float_depth_key(float_bits(kFloatFragDepth));
  }
  if (depth16) {
    kernel_arg.depth = d16_key(kD16Frag);
  }
  uint32_t r = (color >> 16) & 0xff;
  uint32_t g = (color >>  8) & 0xff;
  uint32_t b = (color      ) & 0xff;
  kernel_arg.a_scale_q16 = (255u << 16) / dst_height;
  kernel_arg.r_scale_q16 = (r    << 16) / dst_width;
  kernel_arg.g_scale_q16 = (g    << 16) / dst_height;
  kernel_arg.b_scale_q16 = (b    << 16) / (dst_width + dst_height);

  // Software OM state: mirror the OM DCR config above into om_state_t so the
  // kernel's gfx_sw::om_fragment merges identically to the FF OM unit. The enable
  // flags + expanded color mask are derived exactly as the FF unit does, via
  // resolve_om_state() (the same logic om_core/DepthTencil/Blender::configure use).
  kernel_arg.sw_path = sw_path ? 1u : 0u;
  gfx_sw::om_state_t& om = kernel_arg.om;
  om.depth_func      = depth_enable ? VX_OM_DEPTH_FUNC_LESS : VX_OM_DEPTH_FUNC_ALWAYS;
  om.depth_writemask = depth_enable ? 1u : 0u;
  for (int f = 0; f < 2; ++f) {
    om.stencil_func[f]      = kStencil.func;
    om.stencil_zpass[f]     = kStencil.zpass;
    om.stencil_zfail[f]     = kStencil.zfail;
    om.stencil_fail[f]      = kStencil.fail;
    om.stencil_ref[f]       = f ? kStencilBackRef : kStencil.ref;
    om.stencil_mask[f]      = OM_STENCIL_MASK;
    om.stencil_writemask[f] = kStencil.writemask;
  }
  om.blend_mode_rgb = VX_OM_BLEND_MODE_ADD;
  om.blend_mode_a   = VX_OM_BLEND_MODE_ADD;
  om.blend_src_rgb  = VX_OM_BLEND_FUNC_ONE;
  om.blend_src_a    = VX_OM_BLEND_FUNC_ONE;
  om.blend_dst_rgb  = blend_enable ? VX_OM_BLEND_FUNC_ONE_MINUS_SRC_A : VX_OM_BLEND_FUNC_ZERO;
  om.blend_dst_a    = blend_enable ? VX_OM_BLEND_FUNC_ONE_MINUS_SRC_A : VX_OM_BLEND_FUNC_ZERO;
  om.blend_const    = 0;
  om.logic_op       = VX_OM_LOGIC_OP_COPY;
  om.zbuf_base      = zbuf_addr;
  om.cbuf_base      = cbuf_addr;
  om.zbuf_pitch     = zbuf_pitch;
  om.cbuf_pitch     = cbuf_pitch;
  om.cbuf_writemask4 = 0xf;
  om.color_format    = cbuf_format;
  om.depth_format    = stencil_word ? VX_OM_DEPTH_FORMAT_D32FS8
                     : float_depth ? VX_OM_DEPTH_FORMAT_D32F : VX_OM_DEPTH_FORMAT_D24S8;
  gfx_sw::resolve_om_state(om);
  if (sw_path) std::cout << "[gfx_om] software output-merger path (gfx_sw::om_fragment)\n";

  // block_x fills one CTA: every thread × every warp = num_threads ×
  // num_warps, capped at dst_width.
  uint32_t block_x = std::min<uint32_t>((uint32_t)(num_threads * num_warps), dst_width);
  uint32_t block_y = 1;
  uint32_t grid_x  = (dst_width  + block_x - 1) / block_x;
  uint32_t grid_y  = (dst_height + block_y - 1) / block_y;
  uint32_t grid[2]  = { grid_x, grid_y };
  uint32_t block[2] = { block_x, block_y };

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

  // -q: one count per fragment however many attachments it exports to.
  int occ_errors = 0;
  if (occ_query) {
    const uint32_t occ_count = read_occ_count((uint32_t)num_clusters) - occ_before;
    vx_event_h ev = nullptr;
    RT_CHECK(vx_enqueue_dcr_write(queue, VX_DCR_OM_OCC_CTRL, 0, 0, nullptr, &ev));
    RT_CHECK(vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(ev);
    uint32_t want = 0;
    for (uint32_t y = 0; y < dst_height; ++y) {
      for (uint32_t x = 0; x < dst_width; ++x) {
        want += ref_frag_passes(x, y, kernel_arg.depth) ? 1 : 0;
      }
    }
    if (occ_count != want) {
      std::cout << "FAILED: occlusion count " << occ_count << ", expected " << want << std::endl;
      ++occ_errors;
    } else {
      std::cout << "occlusion: " << occ_count << " of " << dst_width * dst_height
                << " fragments passed" << std::endl;
    }
  }

  // -Q: the draw wrote nothing, so both buffers still hold what they started with.
  if (occ_proxy) {
    std::vector<uint32_t> zs(zbuf_size / 4), cb(cbuf_size / 4);
    vx_event_h read_ev = nullptr;
    RT_CHECK(vx_enqueue_read(queue, zs.data(), depth_buffer, 0, zbuf_size, 0, nullptr, &read_ev));
    RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(read_ev);
    RT_CHECK(vx_enqueue_read(queue, cb.data(), color_buffer, 0, cbuf_size, 0, nullptr, &read_ev));
    RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(read_ev);
    for (uint32_t y = 0; y < dst_height && occ_errors < 16; ++y) {
      for (uint32_t x = 0; x < dst_width && occ_errors < 16; ++x) {
        const uint32_t want_zs = initial_depthstencil(x, y);
        const uint32_t got_zs  = zs[x + y * dst_width];
        const uint32_t got_c   = cb[x + y * dst_width];
        if (got_zs != want_zs || got_c != clear_color) {
          std::cout << "FAILED: a draw writing nothing changed (" << x << "," << y << "): got 0x"
                    << std::hex << got_zs << "/0x" << got_c << ", expected 0x" << want_zs
                    << "/0x" << clear_color << std::dec << std::endl;
          ++occ_errors;
        }
      }
    }
  }

  // ---- save output PNG --------------------------------------------------
  // Vertical flip on save to match reference image orientation.
  if (output_file && strcmp(output_file, "null") != 0) {
    std::vector<uint8_t> dst_pixels(cbuf_size);
    {
      vx_event_h read_ev = nullptr;
      RT_CHECK(vx_enqueue_read(queue, dst_pixels.data(), color_buffer, 0, cbuf_size, 0, nullptr, &read_ev));
      RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
      vx_event_release(read_ev);
    }
    // The image is saved as ARGB8888, so a narrower texel is decoded first.
    const uint32_t bpp = gfx_sw::om_color_bpp(cbuf_format);
    std::vector<uint32_t> argb(dst_width * dst_height);
    for (uint32_t y = 0; y < dst_height; ++y) {
      for (uint32_t x = 0; x < dst_width; ++x) {
        uint32_t raw = 0;
        memcpy(&raw, &dst_pixels[y * cbuf_pitch + x * bpp], bpp);
        argb[y * dst_width + x] = gfx_sw::om_decode_color(cbuf_format, raw);
      }
    }
    const uint32_t argb_pitch = dst_width * 4;
    auto bits = reinterpret_cast<const uint8_t*>(argb.data()) + ((dst_height - 1) * argb_pitch);
    RT_CHECK(SaveImage(output_file, FORMAT_A8R8G8B8, bits, dst_width, dst_height,
                       -(int)argb_pitch));
  }

  // Colour attachment 1, checked against the colour the kernel computes rather
  // than against attachment 0: reading the answer back off the device's own
  // other attachment would pass even if the two shared one buffer.
  int mrt_errors = 0;
  if (mrt_enable) {
    std::vector<uint32_t> rt1(cbuf_size / 4);
    vx_event_h read_ev = nullptr;
    RT_CHECK(vx_enqueue_read(queue, rt1.data(), color_buffer1, 0, cbuf_size, 0, nullptr, &read_ev));
    RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(read_ev);

    for (uint32_t y = 0; y < dst_height && mrt_errors < 16; ++y) {
      for (uint32_t x = 0; x < dst_width && mrt_errors < 16; ++x) {
        uint32_t alpha = blend_enable ? ((y * kernel_arg.a_scale_q16) >> 16) : 0xff;
        uint32_t src   = (alpha << 24)
                       | (((x * kernel_arg.r_scale_q16) >> 16) << 16)
                       | (((y * kernel_arg.g_scale_q16) >> 16) <<  8)
                       |  (((x + y) * kernel_arg.b_scale_q16) >> 16);
        // Attachment 1 does not blend, so the source colour reaches it intact --
        // and only through the channels its own writemask opens.
        uint32_t want = (kRt1Clear & ~kRt1Mask) | (src & kRt1Mask);
        uint32_t got  = rt1[x + y * dst_width];
        if (got != want) {
          std::cout << "FAILED: attachment 1 at (" << x << "," << y << "): got 0x"
                    << std::hex << got << ", expected 0x" << want << std::dec << std::endl;
          ++mrt_errors;
        }
      }
    }
  }

  // -s: every depth/stencil word against the outcome of its own tests. The
  // stencil fails on even rows, where neither depth nor colour may change even
  // when the depth test would pass; it passes on odd rows, where the depth
  // checkerboard decides. Each outcome takes a different stencil op.
  int ds_errors = 0;
  if (stencil_enable || (float_depth && depth_enable)) {
    std::vector<uint32_t> zs(zbuf_size / 4), cb(cbuf_size / 4);
    vx_event_h read_ev = nullptr;
    RT_CHECK(vx_enqueue_read(queue, zs.data(), depth_buffer, 0, zbuf_size, 0, nullptr, &read_ev));
    RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(read_ev);
    RT_CHECK(vx_enqueue_read(queue, cb.data(), color_buffer, 0, cbuf_size, 0, nullptr, &read_ev));
    RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(read_ev);
    const uint32_t zw = stencil_word ? 2 : 1;
    for (uint32_t y = 0; y < dst_height && ds_errors < 16; ++y) {
      for (uint32_t x = 0; x < dst_width && ds_errors < 16; ++x) {
        const uint32_t init = initial_depthstencil(x, y);
        const uint32_t z0 = float_depth ? init : (init & OM_DEPTH_MASK);
        const uint32_t s0 = stencil_word ? initial_stencil(y)
                          : float_depth  ? 0 : (init >> VX_OM_DEPTH_BITS);
        const uint32_t frag_z = kernel_arg.depth;
        // Under -z the expectation comes from the floats themselves, so the
        // check does not share the key's order with the merger it checks.
        const bool spass = !stencil_enable || s0 == (backface ? kStencilBackRef : kStencil.ref);
        const bool dpass = float_depth ? (kFloatFragDepth < kFloatDepths[(x + 3 * y) % 6])
                                       : (frag_z < z0);
        uint32_t s1 = s0;
        if (stencil_enable) {
          if (!spass) {
            s1 = s0 + 1;                // INCR
          } else if (!dpass) {
            s1 = 0;                     // ZERO
          } else {
            s1 = ~s0 & OM_STENCIL_MASK; // INVERT
          }
        }
        const uint32_t z1 = (spass && dpass) ? frag_z : z0;
        const uint32_t src = 0xff000000u
                           | (((x * kernel_arg.r_scale_q16) >> 16) << 16)
                           | (((y * kernel_arg.g_scale_q16) >> 16) <<  8)
                           |  (((x + y) * kernel_arg.b_scale_q16) >> 16);
        const uint32_t want_zs = float_depth ? z1 : ((s1 << VX_OM_DEPTH_BITS) | z1);
        const uint32_t want_s  = stencil_word ? s1 : 0;
        const uint32_t want_c  = (spass && dpass) ? src : clear_color;
        const uint32_t got_zs = zs[(x + y * dst_width) * zw];
        const uint32_t got_s  = stencil_word ? zs[(x + y * dst_width) * zw + 1] : 0;
        const uint32_t got_c  = cb[x + y * dst_width];
        if (got_zs != want_zs || got_s != want_s || got_c != want_c) {
          std::cout << "FAILED: depth/stencil at (" << x << "," << y << "): got 0x"
                    << std::hex << got_zs << "/0x" << got_s << "/0x" << got_c << ", expected 0x"
                    << want_zs << "/0x" << want_s << "/0x" << want_c << std::dec << std::endl;
          ++ds_errors;
        }
      }
    }
  }

  // -y: every D16 texel and its colour against the LESS test of the fragment's
  // 0x8000 with the stored value, compared as the 16-bit values themselves.
  if (depth16) {
    std::vector<uint8_t> zs(zbuf_size);
    std::vector<uint32_t> cb(cbuf_size / 4);
    vx_event_h read_ev = nullptr;
    RT_CHECK(vx_enqueue_read(queue, zs.data(), depth_buffer, 0, zbuf_size, 0, nullptr, &read_ev));
    RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(read_ev);
    RT_CHECK(vx_enqueue_read(queue, cb.data(), color_buffer, 0, cbuf_size, 0, nullptr, &read_ev));
    RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(read_ev);
    for (uint32_t y = 0; y < dst_height && ds_errors < 16; ++y) {
      for (uint32_t x = 0; x < dst_width && ds_errors < 16; ++x) {
        const uint16_t d0 = initial_d16(x, y);
        const bool pass = kD16Frag < d0;
        const uint32_t src = 0xff000000u
                           | (((x * kernel_arg.r_scale_q16) >> 16) << 16)
                           | (((y * kernel_arg.g_scale_q16) >> 16) <<  8)
                           |  (((x + y) * kernel_arg.b_scale_q16) >> 16);
        uint16_t got_z;
        memcpy(&got_z, &zs[y * zbuf_pitch + x * 2], 2);
        const uint16_t want_z = pass ? kD16Frag : d0;
        const uint32_t want_c = pass ? src : clear_color;
        if (got_z != want_z || cb[x + y * dst_width] != want_c) {
          std::cout << "FAILED: D16 at (" << x << "," << y << "): got 0x" << std::hex << got_z
                    << "/0x" << cb[x + y * dst_width] << ", expected 0x" << want_z << "/0x"
                    << want_c << std::dec << std::endl;
          ++ds_errors;
        }
      }
    }
    if (ds_errors == 0) {
      std::cout << "D16: " << dst_width * dst_height << " texels match" << std::endl;
    }
  }

  // -F: each texel against the kernel's colour, blended (-b) with the clear as
  // the attachment stores it, and packed.
  int fmt_errors = 0;
  if (cbuf_format != VX_OM_COLOR_FORMAT_A8R8G8B8) {
    std::vector<uint8_t> cb(cbuf_size);
    vx_event_h read_ev = nullptr;
    RT_CHECK(vx_enqueue_read(queue, cb.data(), color_buffer, 0, cbuf_size, 0, nullptr, &read_ev));
    RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(read_ev);
    const uint32_t dst = gfx_sw::om_decode_color(cbuf_format, gfx_sw::om_encode_color(cbuf_format, clear_color));
    for (uint32_t y = 0; y < dst_height && fmt_errors < 16; ++y) {
      for (uint32_t x = 0; x < dst_width && fmt_errors < 16; ++x) {
        const uint32_t alpha = blend_enable ? ((y * kernel_arg.a_scale_q16) >> 16) : 0xff;
        const uint32_t src = (alpha << 24)
                           | (((x * kernel_arg.r_scale_q16) >> 16) << 16)
                           | (((y * kernel_arg.g_scale_q16) >> 16) <<  8)
                           |  (((x + y) * kernel_arg.b_scale_q16) >> 16);
        uint32_t c = src;
        if (blend_enable) {
          const cocogfx::ColorARGB s(src), d(dst), k(0);
          const auto f = gfx_sw::DoBlendFunc(VX_OM_BLEND_FUNC_ONE, s, d, k);
          const auto g = gfx_sw::DoBlendFunc(VX_OM_BLEND_FUNC_ONE_MINUS_SRC_A, s, d, k);
          const auto rgb = gfx_sw::DoBlendMode(VX_OM_BLEND_MODE_ADD, VX_OM_LOGIC_OP_COPY, s, d, f, g);
          c = cocogfx::ColorARGB(rgb.a, rgb.r, rgb.g, rgb.b).value;
        }
        const uint32_t want = gfx_sw::om_encode_color(cbuf_format, c);
        uint32_t got = 0;
        memcpy(&got, &cb[y * cbuf_pitch + x * cbuf_bpp], cbuf_bpp);
        if (got != want) {
          std::cout << "FAILED: format " << cbuf_format << " texel at (" << x << "," << y
                    << "): got 0x" << std::hex << got << ", expected 0x" << want << std::dec
                    << std::endl;
          ++fmt_errors;
        }
      }
    }
    if (fmt_errors == 0) {
      std::cout << "format " << cbuf_format << ": " << dst_width * dst_height
                << " texels match" << std::endl;
    }
  }

  cleanup();

  if (mrt_errors != 0 || ds_errors != 0 || fmt_errors != 0 || occ_errors != 0) {
    return 1;
  }

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
