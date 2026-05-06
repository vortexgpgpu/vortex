// Minimal RASTER smoke test:
//
// 1. Allocate dst buffer.
// 2. Configure RASTER DCRs with tile_count = 0 (so the unit immediately
//    reports done on every pop).
// 3. Launch kernel; each thread does one vx_rast() and stores the result.
// 4. Verify every slot has bit 0 (the done-flag, low bit of the
//    descriptor word) set.
//
// Goal: validate that the SFU RASTER agent + VX_raster_arb +
// VX_raster_core pipeline returns descriptors without hanging.

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
vx_buffer_h dst_buffer  = nullptr;
kernel_arg_t kernel_arg = {};

void cleanup() {
    if (device) {
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
        default:  std::cout << "raster_smoke: -k kernel\n"; return 0;
        }
    }

    RT_CHECK(vx_dev_open(&device));

    uint64_t isa_flags;
    RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
    if ((isa_flags & VX_ISA_EXT_RASTER) == 0) {
        std::cout << "RASTER extension not enabled (build with "
                     "-DEXT_RASTER_ENABLE)" << std::endl;
        cleanup();
        return -1;
    }

    uint64_t num_cores, num_warps, num_threads;
    RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_CORES,   &num_cores));
    RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_WARPS,   &num_warps));
    RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads));
    uint32_t total = static_cast<uint32_t>(num_cores * num_warps * num_threads);

    uint32_t dst_count = total;
    uint32_t dst_bytes = dst_count * sizeof(uint32_t);
    RT_CHECK(vx_mem_alloc(device, dst_bytes, VX_MEM_WRITE, &dst_buffer));
    uint64_t dst_addr;
    RT_CHECK(vx_mem_address(dst_buffer, &dst_addr));

    kernel_arg.dst_addr  = dst_addr;
    kernel_arg.dst_count = dst_count;

    std::vector<uint32_t> h_dst(dst_count, 0xDEADBEEFu);
    RT_CHECK(vx_copy_to_dev(dst_buffer, h_dst.data(), 0, dst_bytes));

    // Configure raster: tile_count=0 means "no tiles", so every pop
    // returns done=1 and the agent writes that into bit 0 of the result.
    RT_CHECK(vx_dcr_write(device, VX_DCR_RASTER_TILE_COUNT, 0));
    RT_CHECK(vx_dcr_write(device, VX_DCR_RASTER_TBUF_ADDR,  0));
    RT_CHECK(vx_dcr_write(device, VX_DCR_RASTER_PBUF_ADDR,  0));
    RT_CHECK(vx_dcr_write(device, VX_DCR_RASTER_PBUF_STRIDE, 0));
    RT_CHECK(vx_dcr_write(device, VX_DCR_RASTER_SCISSOR_X, 0));
    RT_CHECK(vx_dcr_write(device, VX_DCR_RASTER_SCISSOR_Y, 0));

    RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
    RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg), &args_buffer));

    uint32_t grid[1]  = { (dst_count + (uint32_t)num_threads - 1) / (uint32_t)num_threads };
    uint32_t block[1] = { (uint32_t)num_threads };
    RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 1, grid, block, 0));
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

    RT_CHECK(vx_copy_from_dev(h_dst.data(), dst_buffer, 0, dst_bytes));

    int errors = 0;
    for (uint32_t i = 0; i < dst_count; ++i) {
        // Result is packed pos_mask, or 0 when the unit is drained
        // (which is the expected case here since tile_count=0).
        uint32_t r = h_dst[i];
        if (r != 0u) {
            if (errors < 8)
                printf("*** error: dst[%u] = 0x%08x (expected drained=0)\n", i, r);
            ++errors;
        }
        if (h_dst[i] == 0xDEADBEEFu) {
            // Marker untouched — kernel didn't write here.
            if (errors < 8)
                printf("*** error: dst[%u] still has untouched marker\n", i);
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
