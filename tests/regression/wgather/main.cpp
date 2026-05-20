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
vx_buffer_h tp_buffer   = nullptr;
vx_queue_h  queue       = nullptr;
vx_module_h module_     = nullptr;
vx_kernel_h kernel      = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
    std::cout << "Vortex wgather test." << std::endl;
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
        if (tp_buffer)  vx_buffer_release(tp_buffer);
        if (kernel)  vx_kernel_release(kernel);
        if (module_) vx_module_release(module_);
        if (queue)   vx_queue_release(queue);
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
    uint32_t tp_size  = num_threads_total * 4 * sizeof(uint32_t);

    RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_WRITE, &dst_buffer));
    RT_CHECK(vx_buffer_address(dst_buffer, &kernel_arg.dst_addr));

    RT_CHECK(vx_buffer_create(device, tp_size, VX_MEM_WRITE, &tp_buffer));
    RT_CHECK(vx_buffer_address(tp_buffer, &kernel_arg.tp_addr));

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

    // ---- Verify basic wgather (dst_buffer) ----
    // Kernel uses src_offset=0, so within each group of 4 the source is
    // the lane whose lower 2 bits are 0 (i.e. tid & ~3).
    //
    // For warp w, thread tid:
    //   group_base = tid & ~3
    //   offset=0: result = self_val = w*1000 + group_base  (kept)
    //   offset=1: result = v1[src]  = group_base*10 + 1
    //   offset=2: result = v2[src]  = group_base*10 + 2
    //   offset=3: result = v3[src]  = group_base*10 + 3

    std::vector<uint32_t> h_dst(num_threads_total);
    vx_event_h read_ev0 = nullptr;
    RT_CHECK(vx_enqueue_read(queue, h_dst.data(), dst_buffer, 0, buf_size, 1, &launch_ev, &read_ev0));

    // ---- Verify transpose (tp_buffer) ----
    // Kernel sets up per-group 4x4 matrix M where:
    //   M[i][j] = group_base_val + j*4 + i + 1
    //   group_base_val = w*100 + group*16
    //   i = lane within group (0-3), j = column (0-3)
    //
    // After vx_transpose4, lane i holds column i of M:
    //   T[i][j] = M[j][i] = group_base_val + i*4 + j + 1

    std::vector<uint32_t> h_tp(num_threads_total * 4);
    vx_event_h read_ev1 = nullptr;
    RT_CHECK(vx_enqueue_read(queue, h_tp.data(), tp_buffer, 0, tp_size, 1, &read_ev0, &read_ev1));

    // wait for the last read to complete
    RT_CHECK(vx_event_wait_value(read_ev1, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(read_ev1);
    vx_event_release(read_ev0);
    vx_event_release(launch_ev);

    int errors = 0;
    for (uint32_t w = 0; w < num_warps; ++w) {
        for (uint32_t tid = 0; tid < threads_per_warp; ++tid) {
            uint32_t group_base = tid & ~0x3u;
            uint32_t offset     = tid - group_base;
            uint32_t expected;
            if (offset == 0) {
                expected = w * 1000 + group_base;
            } else {
                expected = group_base * 10 + offset;
            }
            uint32_t actual = h_dst[w * threads_per_warp + tid];
            if (actual != expected) {
                if (errors < 20) {
                    printf("*** wgather error: warp=%u tid=%u (group=%u offset=%u)"
                           " expected=%u actual=%u\n",
                           w, tid, group_base, offset, expected, actual);
                }
                ++errors;
            }
        }
    }

    for (uint32_t w = 0; w < num_warps; ++w) {
        for (uint32_t tid = 0; tid < threads_per_warp; ++tid) {
            uint32_t group          = tid >> 2;
            uint32_t group_base     = group << 2;
            uint32_t i              = tid - group_base;
            uint32_t group_base_val = w * 100 + group * 16;
            uint32_t base_idx       = (w * threads_per_warp + tid) * 4;

            for (uint32_t j = 0; j < 4; ++j) {
                uint32_t expected = group_base_val + i * 4 + j + 1;
                uint32_t actual   = h_tp[base_idx + j];
                if (actual != expected) {
                    if (errors < 20) {
                        printf("*** transpose error: warp=%u tid=%u (group=%u i=%u j=%u)"
                               " expected=%u actual=%u\n",
                               w, tid, group, i, j, expected, actual);
                    }
                    ++errors;
                }
            }
        }
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
