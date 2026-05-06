// Minimal OM smoke test:
//
// 1. Allocate cbuf in device memory (cleared to a sentinel pattern).
// 2. Configure OM DCRs: cbuf_addr/pitch, depth/stencil/blend disabled
//    (always pass, no blend → just a passthrough write).
// 3. Launch kernel; each thread issues vx_om(x=tid, y=0, face=0, color)
//    once.
// 4. Read cbuf back. cbuf[tid] should hold the color the kernel wrote.
//
// Goal: validate the SFU OM agent + cluster-shared OM unit + ocache
// path executes without hanging and the writes show up in the
// configured cbuf.

#include <iostream>
#include <vector>
#include <unistd.h>
#include <vortex.h>
#include <VX_types.h>
#include "common.h"

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
vx_buffer_h cbuf_buffer = nullptr;
kernel_arg_t kernel_arg = {};

void cleanup() {
    if (device) {
        if (cbuf_buffer) vx_mem_free(cbuf_buffer);
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
        default:  std::cout << "om_smoke: -k kernel\n"; return 0;
        }
    }

    RT_CHECK(vx_dev_open(&device));

    uint64_t isa_flags;
    RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
    if ((isa_flags & VX_ISA_EXT_OM) == 0) {
        std::cout << "OM extension not enabled (build with "
                     "-DEXT_OM_ENABLE)" << std::endl;
        cleanup();
        return -1;
    }

    uint64_t num_cores, num_warps, num_threads;
    RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_CORES,   &num_cores));
    RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_WARPS,   &num_warps));
    RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads));
    uint32_t total = static_cast<uint32_t>(num_cores * num_warps * num_threads);

    // cbuf: 1-row image of `total` ARGB pixels.
    uint32_t cbuf_count = total;
    uint32_t cbuf_bytes = cbuf_count * sizeof(uint32_t);
    RT_CHECK(vx_mem_alloc(device, cbuf_bytes, VX_MEM_READ_WRITE, &cbuf_buffer));
    uint64_t cbuf_addr;
    RT_CHECK(vx_mem_address(cbuf_buffer, &cbuf_addr));

    std::vector<uint32_t> h_cbuf(cbuf_count, 0xDEADBEEFu);
    RT_CHECK(vx_copy_to_dev(cbuf_buffer, h_cbuf.data(), 0, cbuf_bytes));

    kernel_arg.dst_count = total;

    // Configure OM: cbuf at cbuf_addr (block-aligned in 64B units),
    // pitch = total * 4 bytes, no depth/stencil/blend.
    RT_CHECK(vx_dcr_write(device, VX_DCR_OM_CBUF_ADDR,        cbuf_addr / 64));
    RT_CHECK(vx_dcr_write(device, VX_DCR_OM_CBUF_PITCH,       cbuf_count * 4));
    RT_CHECK(vx_dcr_write(device, VX_DCR_OM_CBUF_WRITEMASK,   0xF));
    RT_CHECK(vx_dcr_write(device, VX_DCR_OM_ZBUF_ADDR,        0));
    RT_CHECK(vx_dcr_write(device, VX_DCR_OM_ZBUF_PITCH,       0));
    RT_CHECK(vx_dcr_write(device, VX_DCR_OM_DEPTH_FUNC,
                          VX_OM_DEPTH_FUNC_ALWAYS));
    RT_CHECK(vx_dcr_write(device, VX_DCR_OM_DEPTH_WRITEMASK,  0));
    RT_CHECK(vx_dcr_write(device, VX_DCR_OM_STENCIL_FUNC,
                          (VX_OM_DEPTH_FUNC_ALWAYS << 16) | VX_OM_DEPTH_FUNC_ALWAYS));
    RT_CHECK(vx_dcr_write(device, VX_DCR_OM_STENCIL_ZPASS,
                          (VX_OM_STENCIL_OP_KEEP << 16) | VX_OM_STENCIL_OP_KEEP));
    RT_CHECK(vx_dcr_write(device, VX_DCR_OM_STENCIL_ZFAIL,
                          (VX_OM_STENCIL_OP_KEEP << 16) | VX_OM_STENCIL_OP_KEEP));
    RT_CHECK(vx_dcr_write(device, VX_DCR_OM_STENCIL_FAIL,
                          (VX_OM_STENCIL_OP_KEEP << 16) | VX_OM_STENCIL_OP_KEEP));
    RT_CHECK(vx_dcr_write(device, VX_DCR_OM_STENCIL_REF,        0));
    RT_CHECK(vx_dcr_write(device, VX_DCR_OM_STENCIL_MASK,       0));
    RT_CHECK(vx_dcr_write(device, VX_DCR_OM_STENCIL_WRITEMASK,  0));
    RT_CHECK(vx_dcr_write(device, VX_DCR_OM_BLEND_MODE,
                          (VX_OM_BLEND_MODE_ADD << 16) | VX_OM_BLEND_MODE_ADD));
    // BLEND_FUNC bit layout (per VX_om_dcr.sv):
    //   [3:0]   src_rgb, [11:8]  src_a
    //   [19:16] dst_rgb, [27:24] dst_a
    // For pure passthrough (result = src), src=ONE, dst=ZERO.
    RT_CHECK(vx_dcr_write(device, VX_DCR_OM_BLEND_FUNC,
                          (VX_OM_BLEND_FUNC_ZERO << 24)   // dst_a   = ZERO
                        | (VX_OM_BLEND_FUNC_ZERO << 16)   // dst_rgb = ZERO
                        | (VX_OM_BLEND_FUNC_ONE  <<  8)   // src_a   = ONE
                        |  VX_OM_BLEND_FUNC_ONE));        // src_rgb = ONE
    RT_CHECK(vx_dcr_write(device, VX_DCR_OM_BLEND_CONST, 0));
    RT_CHECK(vx_dcr_write(device, VX_DCR_OM_LOGIC_OP,    VX_OM_LOGIC_OP_COPY));

    RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
    RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg), &args_buffer));

    uint32_t grid[1]  = { (cbuf_count + (uint32_t)num_threads - 1) / (uint32_t)num_threads };
    uint32_t block[1] = { (uint32_t)num_threads };
    RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 1, grid, block, 0));
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

    RT_CHECK(vx_copy_from_dev(h_cbuf.data(), cbuf_buffer, 0, cbuf_bytes));

    int errors = 0;
    for (uint32_t i = 0; i < cbuf_count; ++i) {
        uint32_t expected = 0x10000000u | i;
        if (h_cbuf[i] != expected) {
            if (errors < 8)
                printf("*** error: cbuf[%u] = 0x%08x (expected 0x%08x)\n",
                       i, h_cbuf[i], expected);
            ++errors;
        }
    }

    cleanup();

    if (errors != 0) {
        std::cout << "FAILED: " << errors << " errors" << std::endl;
        return 1;
    }
    std::cout << "PASSED!" << std::endl;
    return 0;
}
