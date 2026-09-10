#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <vortex2.h>
#include "common.h"

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);  \
     cleanup();                                                 \
     exit(-1);                                                  \
   } while (false)

const char* kernel_file   = "kernel.vxbin";
uint32_t num_warps        = 4;
uint32_t threads_per_warp = 0; // 0 = use device default (num_threads)

vx_device_h device      = nullptr;
vx_buffer_h dst_buffer  = nullptr;
vx_queue_h  queue       = nullptr;
vx_module_h module_     = nullptr;
vx_kernel_h kernel      = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
    std::cout << "Vortex quad-LOD test." << std::endl;
    std::cout << "Usage: [-k kernel] [-n num_warps] [-t threads_per_warp] [-h help]" << std::endl;
}

static void parse_args(int argc, char** argv) {
    int c;
    while ((c = getopt(argc, argv, "n:t:k:h")) != -1) {
        switch (c) {
        case 'n': num_warps        = atoi(optarg); break;
        case 't': threads_per_warp = atoi(optarg); break;
        case 'k': kernel_file      = optarg;        break;
        case 'h': show_usage(); exit(0);
        default:  show_usage(); exit(-1);
        }
    }
}

void cleanup() {
    if (device) {
        if (dst_buffer) vx_buffer_release(dst_buffer);
        if (kernel)  vx_kernel_release(kernel);
        if (module_) vx_module_release(module_);
        if (queue)   vx_queue_release(queue);
        vx_device_dump_perf(device, stdout);
        vx_device_release(device);
    }
}

int main(int argc, char* argv[]) {
    parse_args(argc, argv);

    RT_CHECK(vx_device_open(0, &device));

    vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
    RT_CHECK(vx_queue_create(device, &qi, &queue));

    uint64_t num_threads;
    RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));

    // use device NT as default when not specified on command line
    if (threads_per_warp == 0) {
        threads_per_warp = (uint32_t)num_threads;
    }

    // wgather requires groups of 4 — skip on configs with NT < 4
    if (num_threads < 4) {
        std::cout << "SKIPPED (num_threads=" << num_threads
                  << " < 4, wgather requires groups of 4)" << std::endl;
        cleanup();
        device = nullptr;
        return 0;
    }

    if (threads_per_warp % 4 != 0) {
        std::cout << "threads_per_warp must be a multiple of 4" << std::endl;
        cleanup();
        return 1;
    }

    if (threads_per_warp > (uint32_t)num_threads) {
        std::cout << "threads_per_warp=" << threads_per_warp
                  << " exceeds device num_threads=" << num_threads << std::endl;
        cleanup();
        return 1;
    }

    uint32_t num_groups_per_warp = threads_per_warp / 4;
    std::cout << "num_warps=" << num_warps
              << " threads_per_warp=" << threads_per_warp
              << " num_groups_per_warp=" << num_groups_per_warp
              << std::endl;

    uint32_t num_threads_total = num_warps * threads_per_warp;
    uint32_t buf_size = num_threads_total * sizeof(uint32_t);

    RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_WRITE, &dst_buffer));
    RT_CHECK(vx_buffer_address(dst_buffer, &kernel_arg.dst_addr));

    kernel_arg.logw = 8;   // 256-texel wide reference texture
    kernel_arg.logh = 8;

    RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
    RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

    // 1D grid: num_warps blocks, threads_per_warp threads each
    vx_event_h launch_ev = nullptr;
    {
        uint32_t grid_dim[1]  = {num_warps};
        uint32_t block_dim[1] = {threads_per_warp};
        vx_launch_info_t li = {};
        li.struct_size  = sizeof(li);
        li.kernel       = kernel;
        li.args_host    = &kernel_arg;
        li.args_size    = sizeof(kernel_arg);
        li.ndim         = 1;
        li.grid_dim[0]  = grid_dim[0];
        li.block_dim[0] = block_dim[0];
        RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
    }

    // Each lane XORs the cross-lane LOD against the single-owner LOD, so every
    // output word must be zero.
    std::vector<uint32_t> h_dst(num_threads_total);
    vx_event_h read_ev = nullptr;
    RT_CHECK(vx_enqueue_read(queue, h_dst.data(), dst_buffer, 0, buf_size, 1, &launch_ev, &read_ev));

    RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(read_ev);
    vx_event_release(launch_ev);

    int errors = 0;
    uint32_t max_lod = 0;
    for (uint32_t w = 0; w < num_warps; ++w) {
        for (uint32_t tid = 0; tid < threads_per_warp; ++tid) {
            uint32_t word = h_dst[w * threads_per_warp + tid];
            uint32_t mismatch = word & 0xffff;
            uint32_t lod      = word >> 16;
            if (lod > max_lod) {
                max_lod = lod;
            }
            if (mismatch != 0) {
                if (errors < 20) {
                    printf("*** quad-LOD mismatch: warp=%u tid=%u lod=%u xor=0x%x\n",
                           w, tid, lod, mismatch);
                }
                ++errors;
            }
        }
    }

    // A sweep that never leaves mip 0 agrees trivially, so it would pass with the
    // cross-lane read broken. Refuse to call that a pass.
    std::cout << "max lod reached: " << max_lod << std::endl;
    if (max_lod == 0) {
        std::cout << "*** the LOD sweep never left mip 0: the test proves nothing" << std::endl;
        ++errors;
    }

    cleanup();

    if (errors != 0) {
        std::cout << "Found " << errors << " errors!" << std::endl;
        std::cout << "FAILED!" << std::endl;
        return 1;
    }

    std::cout << "PASSED!" << std::endl;
    return 0;
}
