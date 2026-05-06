// Minimal TEX smoke test:
//
// 1. Allocate a 1x1 ARGB8 texture in device memory with a known sentinel
//    color.
// 2. Configure TEX stage 0 DCRs to sample from it (1x1, point filter,
//    clamp wrap, ARGB8888).
// 3. Launch a kernel where each thread issues vx_tex(0,0,0,0) and writes
//    the result into a per-thread slot of dst.
// 4. Read dst back; every slot should contain the sentinel.
//
// Goal: verify the SFU TEX agent + cluster-shared TEX unit + tcache
// pipeline returns a sane texel without hanging.

#include <iostream>
#include <vector>
#include <unistd.h>
#include <string.h>
#include <vortex.h>
#include <VX_types.h>
#include "common.h"

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

const char* kernel_file = "kernel.vxbin";

vx_device_h device      = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
vx_buffer_h tex_buffer  = nullptr;
vx_buffer_h dst_buffer  = nullptr;
kernel_arg_t kernel_arg = {};

void cleanup() {
    if (device) {
        if (tex_buffer)  vx_mem_free(tex_buffer);
        if (dst_buffer)  vx_mem_free(dst_buffer);
        if (krnl_buffer) vx_mem_free(krnl_buffer);
        if (args_buffer) vx_mem_free(args_buffer);
        vx_dev_close(device);
    }
}

int main(int argc, char* argv[]) {
    int c;
    while ((c = getopt(argc, argv, "k:h")) != -1) {
        switch (c) {
        case 'k': kernel_file = optarg; break;
        case 'h':
        default:  std::cout << "tex_smoke: -k kernel\n"; return 0;
        }
    }

    std::cout << "open device" << std::endl;
    RT_CHECK(vx_dev_open(&device));

    uint64_t isa_flags;
    RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
    if ((isa_flags & VX_ISA_EXT_TEX) == 0) {
        std::cout << "TEX extension not enabled in this build (build with "
                     "-DEXT_TEX_ENABLE)" << std::endl;
        cleanup();
        return -1;
    }

    uint64_t num_cores, num_warps, num_threads;
    RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_CORES,   &num_cores));
    RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_WARPS,   &num_warps));
    RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads));
    uint32_t total = static_cast<uint32_t>(num_cores * num_warps * num_threads);
    std::cout << "cores=" << num_cores << " warps=" << num_warps
              << " threads=" << num_threads << " total=" << total << std::endl;

    // ---- 1x1 ARGB texture with a sentinel color --------------------------
    constexpr uint32_t kSentinel = 0xCAFEBEEFu;
    constexpr uint32_t kTexBytes = 4;

    RT_CHECK(vx_mem_alloc(device, kTexBytes, VX_MEM_READ, &tex_buffer));
    uint64_t tex_addr;
    RT_CHECK(vx_mem_address(tex_buffer, &tex_addr));
    RT_CHECK(vx_copy_to_dev(tex_buffer, &kSentinel, 0, kTexBytes));
    std::cout << "tex_addr=0x" << std::hex << tex_addr << std::dec << std::endl;

    // ---- destination buffer (one uint32 per thread) ----------------------
    uint32_t dst_count = total;
    uint32_t dst_bytes = dst_count * sizeof(uint32_t);
    RT_CHECK(vx_mem_alloc(device, dst_bytes, VX_MEM_WRITE, &dst_buffer));
    uint64_t dst_addr;
    RT_CHECK(vx_mem_address(dst_buffer, &dst_addr));

    kernel_arg.dst_addr  = dst_addr;
    kernel_arg.dst_count = dst_count;

    // Pre-fill dst with a marker that the kernel must overwrite.
    std::vector<uint32_t> h_dst(dst_count, 0xDEADBEEFu);
    RT_CHECK(vx_copy_to_dev(dst_buffer, h_dst.data(), 0, dst_bytes));

    // ---- configure TEX stage 0 DCRs --------------------------------------
    // ADDR is in 64-byte blocks per skybox convention.
    RT_CHECK(vx_dcr_write(device, VX_DCR_TEX_STAGE,  0));
    RT_CHECK(vx_dcr_write(device, VX_DCR_TEX_LOGDIM, 0));   // 1x1 texture (logwidth=0, logheight=0)
    RT_CHECK(vx_dcr_write(device, VX_DCR_TEX_FORMAT, VX_TEX_FORMAT_A8R8G8B8));
    RT_CHECK(vx_dcr_write(device, VX_DCR_TEX_FILTER, VX_TEX_FILTER_POINT));
    RT_CHECK(vx_dcr_write(device, VX_DCR_TEX_WRAP,   (VX_TEX_WRAP_CLAMP << 16) | VX_TEX_WRAP_CLAMP));
    RT_CHECK(vx_dcr_write(device, VX_DCR_TEX_ADDR,   tex_addr / 64));
    RT_CHECK(vx_dcr_write(device, VX_DCR_TEX_MIPOFF(0), 0));

    // ---- upload kernel + args, launch ------------------------------------
    RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
    RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg), &args_buffer));

    uint32_t grid[1]  = { (dst_count + (uint32_t)num_threads - 1) / (uint32_t)num_threads };
    uint32_t block[1] = { (uint32_t)num_threads };
    std::cout << "launch grid=" << grid[0] << " block=" << block[0] << std::endl;
    RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 1, grid, block, 0));
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

    // ---- verify ----------------------------------------------------------
    RT_CHECK(vx_copy_from_dev(h_dst.data(), dst_buffer, 0, dst_bytes));

    int errors = 0;
    for (uint32_t i = 0; i < dst_count; ++i) {
        if (h_dst[i] != kSentinel) {
            if (errors < 8) {
                printf("*** error: dst[%u] = 0x%08x (expected 0x%08x)\n",
                       i, h_dst[i], kSentinel);
            }
            ++errors;
        }
    }

    cleanup();

    if (errors != 0) {
        std::cout << "FAILED: " << errors << " of " << dst_count
                  << " texels mismatched" << std::endl;
        return 1;
    }
    std::cout << "PASSED!" << std::endl;
    return 0;
}
